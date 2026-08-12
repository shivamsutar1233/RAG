"""Unit tests for live (production-traffic) evaluation.

No LLM and no index: the spool, the sampler, the worker's bookkeeping and the
rolling summary are all pure. Scoring itself is stubbed — it is already covered
in test_evaluation.py.
"""

import json
import os
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.live_eval as live  # noqa: E402
from backend.workspace import Workspace  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """An isolated workspace root, so tests never touch real user data."""
    monkeypatch.setattr("backend.workspace.WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(live, "WORKSPACE_ROOT", tmp_path)
    return Workspace.for_user("tenant-a").ensure()


def _result(answer="an answer", contexts=("some context",), **kw):
    return SimpleNamespace(
        answer=answer,
        context_docs=[SimpleNamespace(page_content=c) for c in contexts],
        route=kw.get("route", "standard"),
        grounded=kw.get("grounded", True),
        latency_ms=kw.get("latency_ms", 1234),
    )


# ── sampling ─────────────────────────────────────────────────────────────────


def test_sample_rate_of_one_takes_everything(monkeypatch):
    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 1.0)
    assert all(live.should_sample() for _ in range(20))


def test_sample_rate_of_zero_disables_the_feature(monkeypatch):
    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 0.0)
    assert not live.enabled()


def test_partial_sample_rate_is_roughly_honoured(monkeypatch):
    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 0.25)
    import random

    random.seed(7)
    taken = sum(live.should_sample() for _ in range(2000))
    assert 350 < taken < 650  # 25% of 2000, generous band


# ── the spool ────────────────────────────────────────────────────────────────


def test_enqueue_writes_one_file_per_turn(workspace, monkeypatch):
    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 1.0)
    assert live.enqueue(workspace, "what is X?", _result())
    assert live.pending_count(workspace) == 1

    sample = json.loads(next(workspace.live_pending_dir.glob("*.json")).read_text())
    assert sample["question"] == "what is X?"
    assert sample["answer"] == "an answer"
    assert sample["contexts"] == ["some context"]
    assert sample["latency_ms"] == 1234


def test_enqueue_skips_when_not_sampled(workspace, monkeypatch):
    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 0.0)
    assert not live.enqueue(workspace, "q", _result())
    assert live.pending_count(workspace) == 0


def test_enqueue_skips_an_empty_answer(workspace, monkeypatch):
    """A failed turn has nothing to grade."""
    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 1.0)
    assert not live.enqueue(workspace, "q", _result(answer=""))
    assert live.pending_count(workspace) == 0


def test_enqueue_never_raises_into_the_chat_path(workspace, monkeypatch):
    """The whole point: evaluation must not be able to break a reply."""
    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 1.0)

    class Hostile:
        @property
        def answer(self):
            raise RuntimeError("boom")

    assert live.enqueue(workspace, "q", Hostile()) is False


def test_enqueue_truncates_oversized_contexts(workspace, monkeypatch):
    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 1.0)
    live.enqueue(workspace, "q", _result(contexts=["x" * 50_000] * 20))
    sample = json.loads(next(workspace.live_pending_dir.glob("*.json")).read_text())
    assert len(sample["contexts"]) <= live._MAX_CONTEXTS
    assert all(len(c) <= live._MAX_CONTEXT_CHARS for c in sample["contexts"])


# ── draining ─────────────────────────────────────────────────────────────────


def _stub_scoring(monkeypatch, score=0.9):
    def fake_score(rows, _config, metric_names, **_kw):
        for row in rows:
            row["scores"] = {m: score for m in metric_names}

    monkeypatch.setattr("backend.evaluation._score", fake_score)
    monkeypatch.setattr(
        "backend.user_config.load_user_config",
        lambda *a, **k: SimpleNamespace(
            for_evaluation=lambda: SimpleNamespace(
                llm_provider="test", resolved_llm_model="judge-1"
            )
        ),
    )


def test_drain_scores_and_clears_the_spool(workspace, monkeypatch):
    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 1.0)
    monkeypatch.setattr(live, "LIVE_METRICS", ("faithfulness",))
    _stub_scoring(monkeypatch)

    for i in range(3):
        live.enqueue(workspace, f"q{i}", _result())
    assert live.pending_count(workspace) == 3

    assert live.drain_once(workspace) == 3
    assert live.pending_count(workspace) == 0

    scored = live.read_scored(workspace)
    assert len(scored) == 3
    assert scored[0]["scores"] == {"faithfulness": 0.9}
    assert scored[0]["judge"] == "test/judge-1"


def test_drain_respects_the_batch_limit(workspace, monkeypatch):
    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 1.0)
    monkeypatch.setattr(live, "LIVE_METRICS", ("faithfulness",))
    _stub_scoring(monkeypatch)
    for i in range(5):
        live.enqueue(workspace, f"q{i}", _result())

    assert live.drain_once(workspace, limit=2) == 2
    assert live.pending_count(workspace) == 3


def test_drain_yields_to_a_running_batch_evaluation(workspace, monkeypatch):
    """Both draw on the same rate limit, and a batch run has someone watching it."""
    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 1.0)
    _stub_scoring(monkeypatch)
    live.enqueue(workspace, "q", _result())

    (workspace.evals_dir / "run.json").write_text(
        json.dumps({"id": "r", "status": "running"}), encoding="utf-8"
    )
    assert live.drain_once(workspace) == 0
    assert live.pending_count(workspace) == 1  # still queued, not lost


def test_drain_discards_turns_older_than_the_max_age(workspace, monkeypatch):
    """A workspace whose judge is misconfigured must not build a spool that
    floods the moment the key is fixed."""
    import time

    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 1.0)
    monkeypatch.setattr(live, "LIVE_MAX_AGE_SECONDS", 3600)
    _stub_scoring(monkeypatch)
    live.enqueue(workspace, "q", _result())

    # Age the spooled file explicitly rather than leaning on a zero max-age and
    # the clock's resolution, which is coarse enough on Windows to be flaky.
    stale = next(workspace.live_pending_dir.glob("*.json"))
    old = time.time() - 7200
    os.utime(stale, (old, old))

    assert live.drain_once(workspace) == 0
    assert live.pending_count(workspace) == 0  # dropped, not retried forever


def test_scored_log_is_trimmed_to_the_history_limit(workspace, monkeypatch):
    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 1.0)
    monkeypatch.setattr(live, "LIVE_METRICS", ("faithfulness",))
    monkeypatch.setattr(live, "LIVE_HISTORY_LIMIT", 4)
    _stub_scoring(monkeypatch)

    for i in range(6):
        live.enqueue(workspace, f"q{i}", _result())
        live.drain_once(workspace)

    assert len(live.read_scored(workspace)) == 4


def test_tenants_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.workspace.WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(live, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 1.0)

    a = Workspace.for_user("tenant-a").ensure()
    b = Workspace.for_user("tenant-b").ensure()
    live.enqueue(a, "only-a", _result())

    assert live.pending_count(a) == 1
    assert live.pending_count(b) == 0
    assert [w.user_id for w in live._workspaces_with_pending()] == ["tenant-a"]


# ── summary ──────────────────────────────────────────────────────────────────


def test_summary_averages_only_scored_turns(workspace, monkeypatch):
    monkeypatch.setattr(live, "LIVE_METRICS", ("faithfulness",))
    live._append_scored(
        workspace,
        [
            {"id": "1", "at": "t", "scores": {"faithfulness": 1.0}, "grounded": True,
             "latency_ms": 100},
            {"id": "2", "at": "t", "scores": {"faithfulness": 0.5}, "grounded": False,
             "latency_ms": 300},
            {"id": "3", "at": "t", "scores": {}, "grounded": True, "latency_ms": 200},
        ],
    )
    summary = live.summarise(workspace)
    assert summary["averages"]["faithfulness"] == 0.75  # the unscored row is excluded
    assert summary["scored_total"] == 3
    assert summary["grounded_rate"] == round(2 / 3, 4)
    assert summary["median_latency_ms"] == 200


def test_summary_is_empty_but_valid_before_any_traffic(workspace):
    summary = live.summarise(workspace)
    assert summary["scored_total"] == 0
    assert summary["pending"] == 0
    assert summary["recent"] == []
    assert all(v is None for v in summary["averages"].values())


def test_summary_returns_newest_turns_first(workspace, monkeypatch):
    monkeypatch.setattr(live, "LIVE_METRICS", ("faithfulness",))
    live._append_scored(
        workspace, [{"id": str(i), "at": "t", "scores": {}} for i in range(5)]
    )
    assert [r["id"] for r in live.summarise(workspace, recent=3)["recent"]] == ["4", "3", "2"]


# ── worker ───────────────────────────────────────────────────────────────────


def test_worker_does_not_start_when_disabled(monkeypatch):
    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 0.0)
    worker = live.LiveEvalWorker()
    worker.start()
    assert worker._thread is None


def test_worker_survives_a_failing_drain(workspace, monkeypatch):
    """A broken judge must not kill the worker for every other tenant."""
    monkeypatch.setattr(live, "LIVE_SAMPLE_RATE", 1.0)
    live.enqueue(workspace, "q", _result())

    calls = []

    def exploding_drain(_ws, *a, **k):
        calls.append(1)
        raise RuntimeError("judge is down")

    monkeypatch.setattr(live, "drain_once", exploding_drain)
    worker = live.LiveEvalWorker(poll_seconds=5)
    worker._stop = threading.Event()

    thread = threading.Thread(target=worker._loop, daemon=True)
    thread.start()
    threading.Event().wait(0.3)
    worker.stop()
    thread.join(timeout=2)

    assert calls, "worker should have attempted a drain"
    assert not thread.is_alive(), "worker thread should exit cleanly on stop"
