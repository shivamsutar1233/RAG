"""
live_eval.py
────────────
Scoring real chat traffic, off the request path.

Batch evaluation answers "how would this pipeline do on a fixed set of
questions". This answers "how is it actually doing, on what people really ask" —
and it is much cheaper than it sounds. A batch run pays for the RAG query *and*
the judge; here the query already happened to serve the user, so a scored turn
costs only the judge calls. Nothing is re-run.

The chat path does exactly one thing: drop a small JSON file in a spool
directory. That write is a few hundred microseconds, happens after the response
is built, and can never raise into the request — an evaluation feature that
degrades chat is worse than no evaluation feature.

A background worker drains the spool. It deliberately does *not* load a
pipeline: judging needs the answer, the question and the retrieved contexts,
all of which the chat request already computed and handed over. So a worker
costs a judge client and nothing else.

The spool is on disk rather than in memory because this deployment restarts —
that is how the ghost-run bug happened. An in-memory queue would silently lose
every unscored turn on each deploy, in a feature whose entire job is to be
trustworthy about quality.
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .workspace import WORKSPACE_ROOT, Workspace

# Fraction of chat turns that get scored. Every turn is eligible; this decides
# how many are actually judged. Scoring a turn costs several model calls against
# the same quota serving live chat, so the default samples rather than scoring
# everything. 0 disables live evaluation entirely.
LIVE_SAMPLE_RATE = float(os.getenv("EVAL_LIVE_SAMPLE_RATE", "0.2"))

# Chat has no reference answers, so only the context-relative metrics are
# computable. Faithfulness catches hallucination and context precision grades the
# retriever — the two you would actually act on.
LIVE_METRICS = tuple(
    m.strip()
    for m in os.getenv("EVAL_LIVE_METRICS", "faithfulness,context_precision").split(",")
    if m.strip()
)

# Turns per scoring pass, and how long the worker sleeps when the spool is empty.
LIVE_BATCH = max(1, int(os.getenv("EVAL_LIVE_BATCH", "5")))
LIVE_POLL_SECONDS = max(5, int(os.getenv("EVAL_LIVE_POLL_SECONDS", "30")))

# Scored turns kept per workspace. The log is trimmed to this on write so it
# cannot grow without bound on a busy deployment.
LIVE_HISTORY_LIMIT = max(50, int(os.getenv("EVAL_LIVE_HISTORY", "500")))

# A turn nobody scored within this long is dropped rather than kept forever. It
# exists so a workspace whose judge is misconfigured does not accumulate a spool
# that floods the moment the key is fixed.
LIVE_MAX_AGE_SECONDS = int(os.getenv("EVAL_LIVE_MAX_AGE_SECONDS", str(24 * 3600)))

# Contexts are truncated before spooling. The judge reads them, so they must be
# real text, but a whole chunk per turn on disk adds up quickly.
_MAX_CONTEXT_CHARS = 2000
_MAX_CONTEXTS = 6


def enabled() -> bool:
    return LIVE_SAMPLE_RATE > 0 and bool(LIVE_METRICS)


def should_sample() -> bool:
    """Whether this turn is one of the sampled ones."""
    if LIVE_SAMPLE_RATE >= 1:
        return True
    return random.random() < LIVE_SAMPLE_RATE


def enqueue(workspace: Workspace, question: str, result: Any) -> bool:
    """Spool one chat turn for later scoring. Returns whether it was queued.

    Called from the chat request. Every failure mode is swallowed: a full disk
    or a permissions problem must degrade evaluation, never the reply the user
    is waiting for.
    """
    if not enabled() or not should_sample():
        return False
    try:
        answer = (getattr(result, "answer", None) or "").strip()
        if not answer:
            return False

        contexts = [
            doc.page_content[:_MAX_CONTEXT_CHARS]
            for doc in (getattr(result, "context_docs", None) or [])[:_MAX_CONTEXTS]
            if getattr(doc, "page_content", "").strip()
        ]
        sample = {
            "id": uuid.uuid4().hex,
            "at": datetime.now(timezone.utc).isoformat(),
            "question": question.strip()[:2000],
            "answer": answer[:8000],
            "contexts": contexts,
            "route": getattr(result, "route", None),
            "grounded": getattr(result, "grounded", None),
            "latency_ms": getattr(result, "latency_ms", 0),
        }

        pending = workspace.live_pending_dir
        pending.mkdir(parents=True, exist_ok=True)
        # Timestamp-prefixed so the worker drains oldest-first without parsing.
        path = pending / f"{int(time.time() * 1000)}-{sample['id']}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sample), encoding="utf-8")
        tmp.replace(path)
        return True
    except Exception as exc:  # noqa: BLE001 - never surface into a chat reply
        print(f"[live-eval] Could not queue a turn: {exc}", file=sys.stderr)
        return False


def _pending_files(workspace: Workspace) -> list[Path]:
    directory = workspace.live_pending_dir
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.json") if p.is_file())


def pending_count(workspace: Workspace) -> int:
    return len(_pending_files(workspace))


def _batch_is_running(workspace: Workspace) -> bool:
    """True while a batch evaluation is in progress for this workspace.

    Live scoring yields to it: both draw on the same rate limit, and a batch run
    is something a person is sitting and watching.
    """
    if not workspace.evals_dir.is_dir():
        return False
    for path in workspace.evals_dir.glob("*.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("status") == "running":
                return True
        except Exception:
            continue
    return False


def _append_scored(workspace: Workspace, rows: Iterable[dict]) -> None:
    """Append scored turns, trimming the log to the history limit."""
    path = workspace.live_scored_path
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[str] = []
    if path.is_file():
        existing = path.read_text(encoding="utf-8").splitlines()
    lines = existing + [json.dumps(r) for r in rows]
    if len(lines) > LIVE_HISTORY_LIMIT:
        lines = lines[-LIVE_HISTORY_LIMIT:]

    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def drain_once(workspace: Workspace, limit: int = LIVE_BATCH) -> int:
    """Score up to *limit* spooled turns. Returns how many were scored."""
    from .evaluation import _score
    from .user_config import load_user_config

    files = _pending_files(workspace)
    if not files:
        return 0
    if _batch_is_running(workspace):
        return 0

    cutoff = time.time() - LIVE_MAX_AGE_SECONDS
    fresh: list[tuple[Path, dict]] = []
    for path in files:
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                continue
            fresh.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            path.unlink(missing_ok=True)
        if len(fresh) >= limit:
            break

    if not fresh:
        return 0

    config = load_user_config(workspace).for_evaluation()
    rows = [
        {
            "question": sample["question"],
            "answer": sample["answer"],
            "contexts": sample.get("contexts") or [],
            "ground_truth": None,
            "scores": {},
            "error": None,
        }
        for _path, sample in fresh
    ]

    # _score already skips context-relative metrics for turns that retrieved
    # nothing, and never raises on a judge failure.
    _score(rows, config, LIVE_METRICS)

    scored = [
        {
            "id": sample["id"],
            "at": sample["at"],
            "question": sample["question"],
            "answer": sample["answer"][:1200],
            "route": sample.get("route"),
            "grounded": sample.get("grounded"),
            "latency_ms": sample.get("latency_ms", 0),
            "contexts": len(sample.get("contexts") or []),
            "scores": row["scores"],
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "judge": f"{config.llm_provider}/{config.resolved_llm_model}",
        }
        for (_path, sample), row in zip(fresh, rows)
    ]
    _append_scored(workspace, scored)

    for path, _sample in fresh:
        path.unlink(missing_ok=True)
    return len(scored)


def _workspaces_with_pending() -> list[Workspace]:
    """Every workspace holding spooled turns. One worker serves all tenants."""
    out: list[Workspace] = []
    if not WORKSPACE_ROOT.is_dir():
        return out
    for entry in WORKSPACE_ROOT.iterdir():
        if not entry.is_dir():
            continue
        try:
            workspace = Workspace.for_user(entry.name)
        except Exception:
            continue
        if _pending_files(workspace):
            out.append(workspace)
    return out


def read_scored(workspace: Workspace, limit: int = LIVE_HISTORY_LIMIT) -> list[dict]:
    """Scored turns, newest first."""
    path = workspace.live_scored_path
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows.reverse()
    return rows[:limit]


def summarise(workspace: Workspace, recent: int = 25) -> dict:
    """Rolling quality for the dashboard.

    Averages cover the whole retained window; ``recent`` only limits how many
    individual turns are returned for the feed.
    """
    rows = read_scored(workspace)
    averages: dict[str, Optional[float]] = {}
    for metric in LIVE_METRICS:
        values = [
            r["scores"][metric]
            for r in rows
            if isinstance(r.get("scores"), dict)
            and isinstance(r["scores"].get(metric), (int, float))
        ]
        averages[metric] = round(sum(values) / len(values), 4) if values else None

    latencies = sorted(r.get("latency_ms", 0) for r in rows if r.get("latency_ms"))
    grounded = [r for r in rows if r.get("grounded") is not None]

    return {
        "enabled": enabled(),
        "sample_rate": LIVE_SAMPLE_RATE,
        "metrics": list(LIVE_METRICS),
        "scored_total": len(rows),
        "pending": pending_count(workspace),
        "averages": averages,
        "grounded_rate": (
            round(sum(1 for r in grounded if r["grounded"]) / len(grounded), 4)
            if grounded
            else None
        ),
        "median_latency_ms": latencies[len(latencies) // 2] if latencies else None,
        "judge": rows[0].get("judge") if rows else None,
        "recent": rows[:recent],
    }


class LiveEvalWorker:
    """One background thread per process, draining every tenant's spool."""

    def __init__(self, poll_seconds: int = LIVE_POLL_SECONDS):
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not enabled() or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="live-eval-worker", daemon=True
        )
        self._thread.start()
        print(
            f"[live-eval] Worker started — sampling {LIVE_SAMPLE_RATE:.0%} of turns, "
            f"metrics: {', '.join(LIVE_METRICS)}.",
            file=sys.stderr,
        )

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                scored = 0
                for workspace in _workspaces_with_pending():
                    if self._stop.is_set():
                        break
                    scored += drain_once(workspace)
                # Only idle when there was nothing to do; a backlog is worked
                # through without waiting a full poll interval between batches.
                if scored == 0:
                    self._stop.wait(self.poll_seconds)
            except Exception as exc:  # noqa: BLE001 - the worker must not die
                print(f"[live-eval] Worker error: {exc}", file=sys.stderr)
                self._stop.wait(self.poll_seconds)
