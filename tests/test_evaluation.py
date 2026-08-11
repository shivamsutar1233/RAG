"""Unit tests for the evaluation harness.

Everything here is pure: no LLM, no index, no network — so it runs in CI. The
parts that need a model (scoring, generation) are exercised by running an
evaluation from the dashboard or `python evaluate.py`.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.evaluation import (  # noqa: E402
    ALL_METRICS,
    METRIC_LABELS,
    REFERENCE_FREE_METRICS,
    _aggregate,
    _clean_line,
    _clean_question,
    judge_warning,
    parse_testset,
)
from backend.evaluation import (
    TestSet as EvalTestSet,
)
from backend.evaluation import (
    testset_path as resolve_testset_path,
)
from backend.providers import ProviderConfig  # noqa: E402
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
