"""Unit tests for the evaluation harness.

Everything here is pure: no LLM, no index, no network — so it runs in CI. The
parts that need a model (scoring, generation) are exercised by running an
evaluation from the dashboard or `python evaluate.py`.
"""

import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.evaluation import (  # noqa: E402
    ALL_METRICS,
    METRIC_LABELS,
    REFERENCE_FREE_METRICS,
    _aggregate,
    _build_metrics,
    _clean_line,
    _clean_question,
    _per_row_scores,
    judge_warning,
    parse_testset,
    select_metrics,
)
from backend.evaluation import (
    TestSet as EvalTestSet,
)
from backend.evaluation import (
    testset_path as resolve_testset_path,
)
from backend.providers import ConfigError, ProviderConfig  # noqa: E402
from backend.workspace import Workspace  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── metric selection ─────────────────────────────────────────────────────────


def test_reference_set_gets_every_metric():
    ts = EvalTestSet(name="t", items=[{"question": "q", "ground_truth": "a"}])
    assert ts.has_references
    assert ts.metrics() == ALL_METRICS


def test_set_without_references_skips_reference_metrics():
    ts = EvalTestSet(name="t", items=[{"question": "q"}])
    assert not ts.has_references
    assert ts.metrics() == REFERENCE_FREE_METRICS


def test_partially_referenced_set_is_treated_as_unreferenced():
    """A half-labelled set would average the reference metrics over a different
    row count than the others, which silently skews the scorecard."""
    ts = EvalTestSet(name="t", items=[{"question": "a", "ground_truth": "x"}, {"question": "b"}])
    assert not ts.has_references
    assert ts.metrics() == REFERENCE_FREE_METRICS


def test_empty_set_has_no_references():
    assert not EvalTestSet(name="t", items=[]).has_references


# ── parsing ──────────────────────────────────────────────────────────────────


def test_parse_json_testset():
    ts = parse_testset("t", json.dumps([{"question": "What is X?", "ground_truth": "X."}]))
    assert ts.items == [{"question": "What is X?", "ground_truth": "X."}]


def test_parse_csv_testset():
    ts = parse_testset("t", "question,ground_truth\nWhat is X?,X is a thing\n")
    assert ts.items == [{"question": "What is X?", "ground_truth": "X is a thing"}]


def test_parse_accepts_the_repos_existing_golden_field_name():
    """eval/golden_*.json use `golden_answer`; they must load without editing."""
    ts = parse_testset("t", json.dumps([{"question": "q", "golden_answer": "a"}]))
    assert ts.items[0]["ground_truth"] == "a"
    assert ts.has_references


@pytest.mark.parametrize("path", ["eval/golden_v1_flawed.json", "eval/golden_v2_fixed.json"])
def test_bundled_golden_sets_parse(path):
    with open(os.path.join(PROJECT_ROOT, path), encoding="utf-8") as handle:
        ts = parse_testset("golden", handle.read())
    assert len(ts.items) == 8
    assert ts.has_references


def test_parse_wraps_object_form():
    ts = parse_testset("t", json.dumps({"items": [{"question": "q"}]}))
    assert len(ts.items) == 1


def test_parse_skips_rows_without_a_question():
    ts = parse_testset("t", json.dumps([{"question": "keep"}, {"ground_truth": "drop"}]))
    assert [i["question"] for i in ts.items] == ["keep"]


def test_parse_rejects_empty_input():
    with pytest.raises(ValueError):
        parse_testset("t", "   ")


def test_parse_rejects_input_with_no_questions():
    with pytest.raises(ValueError):
        parse_testset("t", json.dumps([{"answer": "no question here"}]))


def test_parse_enforces_the_size_cap():
    """Each question costs a full RAG query plus a judge call per metric."""
    big = json.dumps([{"question": f"q{i}"} for i in range(500)])
    with pytest.raises(ValueError, match="limit"):
        parse_testset("t", big)


# ── name safety ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["../escape", "a/b", "", "x" * 65, "sneaky/../../etc"])
def test_testset_path_rejects_traversal(name):
    ws = Workspace.for_user("local")
    with pytest.raises(ValueError):
        resolve_testset_path(ws, name)


def test_testset_path_accepts_ordinary_names():
    ws = Workspace.for_user("local")
    path = resolve_testset_path(ws, "my set-1_v2")
    assert path.name == "my set-1_v2.json"
    assert str(path).startswith(str(ws.testsets_dir.resolve()))


# ── aggregation ──────────────────────────────────────────────────────────────


def test_aggregate_averages_only_scored_rows():
    rows = [
        {"scores": {"faithfulness": 1.0}},
        {"scores": {"faithfulness": 0.5}},
        {"scores": {}},  # judge failed on this row
        {"error": "boom"},
    ]
    assert _aggregate(rows, ("faithfulness",)) == {"faithfulness": 0.75}


def test_aggregate_reports_none_when_nothing_scored():
    assert _aggregate([{"scores": {}}], ("faithfulness",)) == {"faithfulness": None}


def test_every_metric_has_a_label():
    assert set(ALL_METRICS) <= set(METRIC_LABELS)


def test_ragas_metric_names_map_back_to_report_keys():
    """RAGAS does not name every metric the way we do — context precision
    reports itself as `llm_context_precision_without_reference`. If the alias map
    misses one, that column silently reads '—' forever instead of failing."""
    metrics, aliases = _build_metrics(ALL_METRICS)
    assert len(metrics) == len(ALL_METRICS)
    for metric, key in zip(metrics, ALL_METRICS):
        assert aliases[metric.name] == key


def test_per_row_scores_translates_ragas_names():
    _metrics, aliases = _build_metrics(ALL_METRICS)
    raw = type("R", (), {"scores": [{"llm_context_precision_without_reference": 0.5}]})()
    assert _per_row_scores(raw, 1, aliases) == [{"context_precision": 0.5}]


def test_per_row_scores_drops_nan_and_pads_missing_rows():
    aliases = {"faithfulness": "faithfulness"}
    raw = type("R", (), {"scores": [{"faithfulness": float("nan")}]})()
    assert _per_row_scores(raw, 2, aliases) == [{}, {}]


# ── generation output cleaning ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Q: What is the roast profile?", "What is the roast profile?"),
        ("**Question:** What is X?", "What is X?"),
        ("1. What is X?", "What is X?"),
        ("- What is X?", "What is X?"),
        ('"What is X?"', "What is X?"),
        ("Here you go:\nWhat is the charge temperature?", "What is the charge temperature?"),
        ("What is X?\nA: because", "What is X?"),
    ],
)
def test_clean_question_strips_model_noise(raw, expected):
    assert _clean_question(raw) == expected


def test_clean_question_returns_empty_when_unusable():
    assert _clean_question("") == ""
    assert _clean_question("no.") == ""


def test_clean_line_takes_the_first_meaningful_line():
    assert _clean_line("\n\nA: The answer is 42 degrees.") == "The answer is 42 degrees."


# ── judge warnings ───────────────────────────────────────────────────────────


def _config(provider: str, model: str) -> ProviderConfig:
    return ProviderConfig(
        llm_provider=provider,
        llm_model=model,
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
    )


@pytest.mark.parametrize("model", ["llama3.2:1b", "qwen3.5:0.8b", "gemma:2b"])
def test_small_local_judges_are_flagged_as_unreliable(model):
    warning = judge_warning(_config("ollama", model))
    assert warning and "not a reliable judge" in warning


def test_larger_local_judges_get_a_softer_caveat():
    warning = judge_warning(_config("ollama", "llama3.3:70b"))
    assert warning and "not a reliable judge" not in warning


def test_cloud_judges_are_not_flagged():
    assert judge_warning(_config("anthropic", "claude-opus-5")) is None


# ── separate judge model ─────────────────────────────────────────────────────


def test_for_evaluation_returns_self_when_no_judge_configured():
    cfg = _config("ollama", "llama3.2:1b")
    assert cfg.for_evaluation() is cfg


def test_for_evaluation_swaps_the_llm_but_keeps_embeddings():
    """Embeddings must not change: answer relevancy compares against the same
    vector space the index was built in."""
    cfg = ProviderConfig(
        llm_provider="ollama",
        llm_model="llama3.2:1b",
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        eval_llm_provider="gemini",
        eval_llm_model="gemini-2.5-flash",
        keys={"GOOGLE_API_KEY": "x"},
    )
    judge = cfg.for_evaluation()
    assert (judge.llm_provider, judge.resolved_llm_model) == ("gemini", "gemini-2.5-flash")
    assert judge.embedding_provider == "ollama"
    assert judge.embedding_model == "nomic-embed-text"
    # the original is untouched — it still answers with the chat model
    assert cfg.llm_provider == "ollama"


def test_a_weak_chat_model_with_a_strong_judge_is_not_warned_about():
    cfg = ProviderConfig(
        llm_provider="ollama",
        llm_model="llama3.2:1b",
        eval_llm_provider="anthropic",
        eval_llm_model="claude-opus-5",
        keys={"ANTHROPIC_API_KEY": "x"},
    )
    assert judge_warning(cfg) is not None  # the chat model on its own is weak
    assert judge_warning(cfg.for_evaluation()) is None  # but it is not the judge


def test_unknown_judge_provider_is_rejected():
    cfg = ProviderConfig(eval_llm_provider="nope")
    with pytest.raises(ConfigError):
        cfg.validate()


def test_judge_provider_missing_its_key_is_rejected_up_front():
    """Caught at save time, not after a run has already burned quota."""
    cfg = ProviderConfig(eval_llm_provider="openai", keys={})
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        cfg.validate()


# ── metric selection ─────────────────────────────────────────────────────────


def test_select_metrics_defaults_to_everything_supported():
    ts = EvalTestSet(name="t", items=[{"question": "q", "ground_truth": "a"}])
    assert select_metrics(ts, None) == ALL_METRICS
    assert select_metrics(ts, []) == ALL_METRICS


def test_select_metrics_honours_a_subset():
    ts = EvalTestSet(name="t", items=[{"question": "q", "ground_truth": "a"}])
    assert select_metrics(ts, ["faithfulness", "context_recall"]) == (
        "faithfulness",
        "context_recall",
    )


def test_select_metrics_drops_ones_the_set_cannot_support():
    """Asking for context recall without references is not worth failing a run
    over — it simply is not computable."""
    ts = EvalTestSet(name="t", items=[{"question": "q"}])
    assert select_metrics(ts, ["faithfulness", "context_recall"]) == ("faithfulness",)


def test_select_metrics_falls_back_when_nothing_requested_is_available():
    ts = EvalTestSet(name="t", items=[{"question": "q"}])
    assert select_metrics(ts, ["factual_correctness"]) == REFERENCE_FREE_METRICS


def test_select_metrics_ignores_unknown_names():
    ts = EvalTestSet(name="t", items=[{"question": "q"}])
    assert select_metrics(ts, ["faithfulness", "not_a_metric"]) == ("faithfulness",)


# ── parallel answering ───────────────────────────────────────────────────────


def test_parallel_answering_preserves_order_and_records_failures(monkeypatch):
    """Rows must line up with the test set regardless of completion order —
    otherwise a scorecard shows one question's score against another's text.
    The slowest question is answered first here, so any append-as-they-finish
    implementation would reverse them."""
    import time

    import backend.evaluation as ev

    questions = ["slow", "medium", "fast", "boom"]
    delays = {"slow": 0.30, "medium": 0.15, "fast": 0.01, "boom": 0.0}

    def fake_run_query(_pipeline, question, **_kw):
        time.sleep(delays[question])
        if question == "boom":
            raise RuntimeError("pipeline exploded")
        return SimpleNamespace(
            answer=f"answer to {question}",
            context_docs=[SimpleNamespace(page_content=f"ctx {question}")],
            route="standard",
            grounded=True,
            latency_ms=1,
        )

    monkeypatch.setattr(ev, "run_query", fake_run_query)
    monkeypatch.setattr(ev, "_score", lambda *a, **k: None)
    monkeypatch.setattr(ev, "EVAL_ANSWER_WORKERS", 4)

    ts = EvalTestSet(name="t", items=[{"question": q} for q in questions])
    report = ev.run_evaluation(
        pipeline={}, config=ProviderConfig(), testset=ts, run_id="r"
    )

    assert report["status"] == "completed"
    assert [r["question"] for r in report["rows"]] == questions
    assert report["rows"][0]["answer"] == "answer to slow"
    assert report["rows"][2]["contexts"] == ["ctx fast"]
    # a failed question is recorded, not fatal, and carries no score
    assert "pipeline exploded" in report["rows"][3]["error"]
    assert report["rows"][3]["answer"] is None


def test_cancelling_before_the_run_starts_answers_nothing(monkeypatch):
    import backend.evaluation as ev

    called = []
    monkeypatch.setattr(
        ev, "run_query", lambda *a, **k: called.append(1) or SimpleNamespace()
    )
    monkeypatch.setattr(ev, "EVAL_ANSWER_WORKERS", 1)

    cancel = threading.Event()
    cancel.set()
    ts = EvalTestSet(name="t", items=[{"question": "q"}])
    report = ev.run_evaluation(
        pipeline={}, config=ProviderConfig(), testset=ts, run_id="r", cancel=cancel
    )

    assert report["status"] == "cancelled"
    assert called == []


def test_run_records_both_the_answering_model_and_the_judge(monkeypatch):
    import backend.evaluation as ev

    monkeypatch.setattr(ev, "_score", lambda *a, **k: None)
    monkeypatch.setattr(
        ev,
        "run_query",
        lambda *a, **k: SimpleNamespace(
            answer="a", context_docs=[], route="simple", grounded=True, latency_ms=1
        ),
    )
    cfg = ProviderConfig(
        llm_provider="ollama",
        llm_model="llama3.2:1b",
        eval_llm_provider="anthropic",
        eval_llm_model="claude-opus-5",
        keys={"ANTHROPIC_API_KEY": "x"},
    )
    ts = EvalTestSet(name="t", items=[{"question": "q"}])
    report = ev.run_evaluation(pipeline={}, config=cfg, testset=ts, run_id="r")

    assert report["llm_model"] == "claude-opus-5"          # who judged
    assert report["answered_by"] == "ollama/llama3.2:1b"   # who answered
    assert report["judge_warning"] is None


# ── stuck / interrupted runs ─────────────────────────────────────────────────


def test_rows_without_context_skip_the_context_metrics(monkeypatch):
    """The expensive-and-meaningless case: retrieval returned nothing, so the
    answer is ungrounded and long, and faithfulness would split it into many
    statements to verify against no context at all."""
    import backend.evaluation as ev

    seen: list[tuple[int, tuple]] = []

    def fake_batch(batch, _config, metric_names):
        seen.append((len(batch), tuple(metric_names)))

    monkeypatch.setattr(ev, "_score_batch", fake_batch)
    rows = [
        {"question": "a", "answer": "x", "contexts": ["c"], "scores": {}, "error": None},
        {"question": "b", "answer": "y", "contexts": [], "scores": {}, "error": None},
    ]
    ev._score(rows, ProviderConfig(), ALL_METRICS)

    by_metrics = {metrics: n for n, metrics in seen}
    assert by_metrics[ALL_METRICS] == 1  # the row that had context
    # the contextless row is graded only on what does not need context
    assert by_metrics[("answer_relevancy", "factual_correctness")] == 1


def test_scoring_stops_at_the_deadline_and_keeps_what_it_had(monkeypatch):
    """Without this a slow judge leaves the run 'running' forever, which also
    trips the one-run-at-a-time guard and locks the workspace."""
    import backend.evaluation as ev

    calls = []
    monkeypatch.setattr(
        ev, "_score_batch", lambda b, c, m: calls.append(len(b))
    )
    monkeypatch.setattr(ev, "EVAL_SCORE_BATCH", 1)

    rows = [
        {"question": str(i), "answer": "x", "contexts": ["c"], "scores": {}, "error": None}
        for i in range(5)
    ]
    logged: list[str] = []
    ev._score(
        rows,
        ProviderConfig(),
        ("faithfulness",),
        deadline=time.monotonic() - 1,  # already expired
        log=logged.append,
    )
    assert calls == []
    assert any("budget exhausted" in m for m in logged)


def test_scoring_can_be_cancelled_between_batches(monkeypatch):
    import backend.evaluation as ev

    calls = []
    cancel = threading.Event()

    def fake_batch(batch, _c, _m):
        calls.append(len(batch))
        cancel.set()  # cancel arrives while the first batch is in flight

    monkeypatch.setattr(ev, "_score_batch", fake_batch)
    monkeypatch.setattr(ev, "EVAL_SCORE_BATCH", 1)
    rows = [
        {"question": str(i), "answer": "x", "contexts": ["c"], "scores": {}, "error": None}
        for i in range(4)
    ]
    ev._score(rows, ProviderConfig(), ("faithfulness",), cancel=cancel, log=lambda _m: None)
    assert calls == [1]  # stopped after the first batch instead of grinding on


def test_run_that_overruns_while_answering_returns_unscored_answers(monkeypatch):
    import backend.evaluation as ev

    monkeypatch.setattr(ev, "EVAL_RUN_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(ev, "EVAL_ANSWER_WORKERS", 1)
    scored = []
    monkeypatch.setattr(ev, "_score", lambda *a, **k: scored.append(1))
    monkeypatch.setattr(
        ev,
        "run_query",
        lambda *a, **k: SimpleNamespace(
            answer="a", context_docs=[], route="simple", grounded=True, latency_ms=1
        ),
    )

    ts = EvalTestSet(name="t", items=[{"question": "q"}])
    report = ev.run_evaluation(pipeline={}, config=ProviderConfig(), testset=ts, run_id="r")

    assert report["status"] == "completed"        # not left "running"
    assert scored == []                            # never started scoring
    assert "Timed out before scoring" in report["error"]
    assert report["rows"][0]["answer"] == "a"      # the work done is kept
