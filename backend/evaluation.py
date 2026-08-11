"""
evaluation.py
─────────────
RAGAS scoring for a workspace's own pipeline.

The dashboard needs to answer "is this RAG any good?" with numbers an outsider
would accept, over the tenant's *own* documents. That rules out the previous
word-overlap proxy in ``evaluate.py``, and it rules out a hosted evaluation
service — scores are computed in-process with the tenant's configured LLM and
embeddings, so nothing leaves the deployment and every tenant is scored by the
models they actually run.

Two kinds of test set exist, and they get different metrics:

* **With references** (generated from the corpus, or uploaded) — every metric,
  including the two that compare the answer against a known-good one.
* **Without references** (questions lifted from real chat history) — only the
  three metrics that judge an answer against its own retrieved context.

A run is deliberately scored against ``run_query(..., web_fallback=False)``.
With the fallback on, a question the index cannot answer would be silently
answered from DuckDuckGo and score *well*, which would measure the search engine
rather than the user's index.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .providers import ProviderConfig, get_embeddings, get_llm
from .query_run import run_query
from .workspace import Workspace


def _install_vertexai_shim() -> None:
    """Let ragas import against a langchain-community that dropped Vertex AI.

    ``ragas.llms.base`` imports ``langchain_community.chat_models.vertexai`` at
    module scope, but langchain-community 0.4 (now sunsetting) removed that
    module. The import is only used to build a list of classes checked with
    ``isinstance`` to decide whether an LLM supports n-completions — so an empty
    placeholder class is not merely a workaround, it gives the correct answer:
    our LLM is never a Vertex AI one.

    Revisit if ragas drops the import; the shim is harmless either way because
    it only registers a name that would otherwise raise ImportError.
    """
    import sys as _sys
    import types

    name = "langchain_community.chat_models.vertexai"
    if name in _sys.modules:
        return
    stub = types.ModuleType(name)

    class ChatVertexAI:  # noqa: D401 - placeholder, never instantiated
        """Placeholder for the removed langchain-community Vertex AI model."""

    stub.ChatVertexAI = ChatVertexAI
    _sys.modules[name] = stub


# ── metric registry ──────────────────────────────────────────────────────────
#
# Keyed by the name that goes into the report and the dashboard, so adding a
# metric is one entry here plus a label on the frontend.

REFERENCE_FREE_METRICS = ("faithfulness", "answer_relevancy", "context_precision")
REFERENCE_ONLY_METRICS = ("context_recall", "factual_correctness")
ALL_METRICS = REFERENCE_FREE_METRICS + REFERENCE_ONLY_METRICS

# Metrics that grade the answer against its retrieved context. When retrieval
# came back empty there is nothing for them to grade, and asking anyway is the
# worst case for cost: an ungrounded answer is long, faithfulness splits it into
# many statements, and every one is an LLM call to verify against nothing.
CONTEXT_DEPENDENT_METRICS = frozenset(
    {"faithfulness", "context_precision", "context_recall"}
)

METRIC_LABELS: dict[str, str] = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "factual_correctness": "Factual correctness",
}

# Scoring runs concurrently inside ragas. The ceiling is low by default because
# the judge shares the tenant's API key (and rate limit) with live chat traffic,
# and because a local Ollama server serialises requests anyway.
EVAL_MAX_WORKERS = int(os.getenv("EVAL_MAX_WORKERS", "4"))

# Each question costs one full RAG query plus roughly one judge call per metric,
# so an unbounded test set is an unbounded bill.
MAX_TESTSET_SIZE = int(os.getenv("MAX_TESTSET_SIZE", "50"))

# Per-judge-call ceiling. RAGAS defaults to 180s, which a local model on CPU
# routinely blows through — the call is not wedged, just slow, and timing it out
# silently drops that row's score. Generous by default; a cloud judge never gets
# near it.
EVAL_TIMEOUT_SECONDS = int(os.getenv("EVAL_TIMEOUT_SECONDS", "900"))

# How many questions are answered at once. Answering is independent per question
# and entirely I/O-bound, so this is the cheapest speed-up available — but it
# multiplies the request rate, so it is capped by the same worker budget as
# scoring and should stay at 1 on a rate-limited free tier.
EVAL_ANSWER_WORKERS = int(os.getenv("EVAL_ANSWER_WORKERS", str(EVAL_MAX_WORKERS)))

# Reverse-generated questions per answer for the answer-relevancy metric, one LLM
# call each. RAGAS defaults to 3; 1 is usually indistinguishable in the aggregate
# and cuts a third of that metric's cost.
ANSWER_RELEVANCY_STRICTNESS = int(os.getenv("ANSWER_RELEVANCY_STRICTNESS", "1"))

# Rows per ragas call. Scoring everything in one call emits no progress until it
# finishes and cannot be interrupted, which is indistinguishable from a hang.
EVAL_SCORE_BATCH = max(1, int(os.getenv("EVAL_SCORE_BATCH", "5")))

# Wall-clock ceiling for a whole run. Without one, a slow judge leaves a run in
# "running" forever: the dashboard spins and the one-run-at-a-time guard locks
# the workspace out of starting another. Hitting it keeps whatever was scored.
EVAL_RUN_TIMEOUT_SECONDS = int(os.getenv("EVAL_RUN_TIMEOUT_SECONDS", "3600"))

# Ollama's default is a 1B model. It cannot reliably produce the structured
# judgements RAGAS asks for, and silently returns near-random scores rather than
# failing, so the dashboard says so instead of presenting the numbers as fact.
_SMALL_MODEL = re.compile(r"[^\d](0\.\d+|[0-3])\s*b\b", re.IGNORECASE)


def judge_warning(config: ProviderConfig) -> Optional[str]:
    """A caveat about the judge model, or None when it is fit for the job."""
    model = config.resolved_llm_model or ""
    if config.llm_provider == "ollama" and _SMALL_MODEL.search(model):
        return (
            f"Scored by '{model}', a small local model. It is not a reliable judge — "
            f"expect noisy scores and skipped rows. Switch to a frontier model in "
            f"Settings for numbers worth quoting."
        )
    if config.llm_provider == "ollama":
        return (
            f"Scored by the local model '{model}'. Local judges are weaker than "
            f"frontier models; treat these scores as indicative."
        )
    return None


# ── test sets ────────────────────────────────────────────────────────────────


@dataclass
class TestSet:
    """A named list of questions, each optionally with a reference answer."""

    name: str
    items: list[dict] = field(default_factory=list)
    source: str = "uploaded"  # uploaded | generated | chat | sample

    @property
    def has_references(self) -> bool:
        """True only when *every* item has one — a partial set would make the
        reference-based metrics average over a different row count than the
        others, which is exactly the kind of silent skew that makes a scorecard
        untrustworthy."""
        return bool(self.items) and all(i.get("ground_truth") for i in self.items)

    def metrics(self) -> tuple[str, ...]:
        return ALL_METRICS if self.has_references else REFERENCE_FREE_METRICS

    def to_json(self) -> dict:
        return {"name": self.name, "source": self.source, "items": self.items}


def parse_testset(name: str, raw: str, source: str = "uploaded") -> TestSet:
    """Read a test set from JSON or CSV text.

    Accepts the field names this project has already used in ``eval/*.json``
    (``golden_answer``) alongside the RAGAS-style ``ground_truth``/``reference``,
    so the existing golden sets load without editing.
    """
    text = raw.strip()
    if not text:
        raise ValueError("Test set is empty.")

    rows: list[dict]
    if text[0] in "[{":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("items", data.get("questions", []))
        if not isinstance(data, list):
            raise ValueError("JSON test set must be a list of objects.")
        rows = [r for r in data if isinstance(r, dict)]
    else:
        rows = list(csv.DictReader(io.StringIO(text)))

    items: list[dict] = []
    for row in rows:
        question = _first(row, "question", "user_input", "query", "input")
        if not question:
            continue
        reference = _first(row, "ground_truth", "reference", "golden_answer", "answer")
        item: dict = {"question": question.strip()}
        if reference and reference.strip():
            item["ground_truth"] = reference.strip()
        items.append(item)

    if not items:
        raise ValueError("No questions found. Expected a 'question' field per row.")
    if len(items) > MAX_TESTSET_SIZE:
        raise ValueError(
            f"Test set has {len(items)} questions; the limit is {MAX_TESTSET_SIZE}. "
            f"Each question costs a full RAG query plus a judge call per metric."
        )
    return TestSet(name=name, items=items, source=source)


def _first(row: dict, *keys: str) -> Optional[str]:
    """First non-empty value among *keys*, matched case-insensitively."""
    lowered = {str(k).strip().lower(): v for k, v in row.items() if k}
    for key in keys:
        value = lowered.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


_SAFE_NAME = re.compile(r"^[A-Za-z0-9 _-]{1,64}$")


def testset_path(workspace: Workspace, name: str) -> Path:
    """Resolve a test set file, rejecting anything that could escape the dir."""
    if not _SAFE_NAME.match(name or ""):
        raise ValueError(
            "Test set names may use letters, numbers, spaces, hyphens and underscores."
        )
    path = (workspace.testsets_dir / f"{name}.json").resolve()
    if not str(path).startswith(str(workspace.testsets_dir.resolve())):
        raise ValueError("Refusing to write outside the workspace.")
    return path


def save_testset(workspace: Workspace, testset: TestSet) -> Path:
    workspace.testsets_dir.mkdir(parents=True, exist_ok=True)
    path = testset_path(workspace, testset.name)
    path.write_text(json.dumps(testset.to_json(), indent=2), encoding="utf-8")
    return path


def load_testset(workspace: Workspace, name: str) -> TestSet:
    path = testset_path(workspace, name)
    if not path.is_file():
        raise FileNotFoundError(f"No test set named '{name}'.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return TestSet(
        name=data.get("name", name),
        items=data.get("items", []),
        source=data.get("source", "uploaded"),
    )


def list_testsets(workspace: Workspace) -> list[dict]:
    """Summaries for the dashboard's test set picker."""
    out: list[dict] = []
    if not workspace.testsets_dir.is_dir():
        return out
    for path in sorted(workspace.testsets_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items = data.get("items", [])
            ts = TestSet(name=data.get("name", path.stem), items=items)
            out.append({
                "name": ts.name,
                "source": data.get("source", "uploaded"),
                "size": len(items),
                "has_references": ts.has_references,
                "metrics": list(ts.metrics()),
            })
        except Exception as exc:
            print(f"[eval] Skipping unreadable test set {path.name}: {exc}", file=sys.stderr)
    return out


# ── test set generation ──────────────────────────────────────────────────────

# Asking one model call for a question *and* its answer in a fixed two-line
# format is the obvious approach, and it fails on small models — llama3.2:1b
# writes a good question, then ignores the format and omits the answer entirely.
# Two single-purpose calls are far more robust, and answering a known question
# from a passage is a task even weak models handle. The extra call buys
# generation that works on the default local setup instead of erroring out.

_QUESTION_PROMPT = ChatPromptTemplate.from_template(
    "Read the passage and write ONE specific question that it fully answers.\n\n"
    "Rules:\n"
    "- The question must be answerable from this passage alone.\n"
    "- Ask about a concrete detail: a name, number, step, or definition.\n"
    "- Do not mention 'the passage', 'the text' or 'the document'.\n"
    "- Reply with the question only — no preamble, no answer.\n\n"
    "Passage:\n{chunk}"
)

_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only the passage below.\n\n"
    "Rules:\n"
    "- One or two sentences, stated as fact.\n"
    "- Use the passage's own wording where you can.\n"
    "- Reply with the answer only.\n\n"
    "Passage:\n{chunk}\n\nQuestion: {question}"
)

# Leading noise models add despite being told not to: "Q:", "Question:", markdown
# emphasis, list bullets and numbering.
_PREFIX = re.compile(
    r"^\s*(?:[*_`#>\-\d.)\s]*)(?:\**\s*(?:q|question|answer|a)\s*\**\s*[:.\-]\s*)?",
    re.IGNORECASE,
)


def _clean_line(text: str) -> str:
    """First meaningful line of a generation, stripped of formatting noise."""
    for raw in (text or "").splitlines():
        line = _PREFIX.sub("", raw).strip().strip("*_`").strip()
        line = line.strip('"').strip("'").strip()
        if len(line) >= 8:
            return line
    return ""


def _clean_question(text: str) -> str:
    """A generated question, or '' when the model did not produce one.

    Preference goes to a line that is actually a question; some models emit a
    sentence of preamble before it.
    """
    candidates = [
        _PREFIX.sub("", raw).strip().strip("*_`\"'").strip()
        for raw in (text or "").splitlines()
    ]
    for line in candidates:
        if line.endswith("?") and len(line) >= 10:
            return line
    first = _clean_line(text)
    return first if len(first) >= 10 else ""


def generate_testset(
    pipeline: dict,
    config: ProviderConfig,
    size: int = 10,
    log: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Write a reference-bearing test set from the workspace's own chunks.

    Chunks come from the FAISS docstore rather than re-parsing the source files.
    Multi-representation indexing means what is stored there is an LLM *summary*
    of each chunk, with the real text kept in ``metadata['original_content']`` —
    generating questions from the summaries would produce a test set that does
    not match the corpus, so the originals are restored first.
    """
    from .multi_rep_utils import restore_original_content

    say = log or (lambda _m: None)
    store = pipeline["vectorstore"].docstore._dict
    docs = restore_original_content(list(store.values()))

    # Long chunks carry enough detail to ask something specific about; a
    # one-line heading does not.
    docs = [d for d in docs if len(d.page_content.strip()) > 200]
    if not docs:
        raise ValueError(
            "No chunks are long enough to generate questions from. "
            "Ingest more documents first."
        )

    docs.sort(key=lambda d: len(d.page_content), reverse=True)
    chosen = docs[: min(size, MAX_TESTSET_SIZE)]
    say(f"Generating {len(chosen)} question(s) from {len(docs)} eligible chunk(s)...")

    llm = get_llm(config)
    batch_config = {"max_concurrency": min(EVAL_MAX_WORKERS, 5)}
    passages = [d.page_content[:4000] for d in chosen]

    question_chain = _QUESTION_PROMPT | llm | StrOutputParser()
    raw_questions = question_chain.batch(
        [{"chunk": p} for p in passages], config=batch_config
    )

    # Only pay for answers to questions that actually parsed.
    pending = [
        (doc, passage, _clean_question(raw))
        for doc, passage, raw in zip(chosen, passages, raw_questions)
    ]
    pending = [p for p in pending if p[2]]
    skipped = len(chosen) - len(pending)
    if skipped:
        say(f"  {skipped} chunk(s) produced no usable question")
    if not pending:
        raise ValueError(
            "The model did not return any usable questions. "
            "A larger model usually fixes this."
        )

    answer_chain = _ANSWER_PROMPT | llm | StrOutputParser()
    raw_answers = answer_chain.batch(
        [{"chunk": passage, "question": q} for _doc, passage, q in pending],
        config=batch_config,
    )

    items: list[dict] = []
    for (doc, _passage, question), raw_answer in zip(pending, raw_answers):
        answer = _clean_line(raw_answer)
        if not answer:
            say("  dropped a question — model returned no answer for it")
            continue
        items.append({
            "question": question,
            "ground_truth": answer,
            "source": doc.metadata.get("title") or doc.metadata.get("source", ""),
        })

    if not items:
        raise ValueError(
            "The model produced questions but no answers for them. "
            "A larger model usually fixes this."
        )
    say(f"Generated {len(items)} question(s).")
    return items


# ── scoring ──────────────────────────────────────────────────────────────────


def _build_metrics(names: tuple[str, ...]) -> tuple[list[Any], dict[str, str]]:
    """Instantiate the requested metrics, plus a map from RAGAS's own metric
    name back to our report key.

    The two are not always the same — ``LLMContextPrecisionWithoutReference``
    reports itself as ``llm_context_precision_without_reference``. Deriving the
    map from the constructed objects rather than hardcoding it means a rename in
    a future ragas release cannot silently blank a column: the key is whatever
    the object says it is.
    """
    _install_vertexai_shim()

    from ragas.metrics import (
        FactualCorrectness,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    builders: dict[str, Callable[[], Any]] = {
        "faithfulness": Faithfulness,
        # strictness is how many questions it reverse-generates from the answer to
        # compare against the original — and it is one LLM call each. The default
        # of 3 makes this the second most expensive metric for a averaging effect
        # that rarely changes the verdict, so it is configurable and defaults low.
        "answer_relevancy": lambda: ResponseRelevancy(strictness=ANSWER_RELEVANCY_STRICTNESS),
        "context_precision": LLMContextPrecisionWithoutReference,
        "context_recall": LLMContextRecall,
        "factual_correctness": FactualCorrectness,
    }
    metrics: list[Any] = []
    aliases: dict[str, str] = {}
    for name in names:
        builder = builders.get(name)
        if builder is None:
            continue
        metric = builder()
        metrics.append(metric)
        aliases[getattr(metric, "name", name)] = name
        aliases[name] = name
    return metrics, aliases


def _score_batch(
    batch: list[dict], config: ProviderConfig, metric_names: tuple[str, ...]
) -> None:
    """Score one batch of rows in place with *metric_names*."""
    if not batch or not metric_names:
        return

    _install_vertexai_shim()

    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["contexts"] or [""],
            response=r["answer"],
            reference=r.get("ground_truth"),
        )
        for r in batch
    ]

    metrics, aliases = _build_metrics(metric_names)
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=LangchainLLMWrapper(get_llm(config)),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings(config)),
        run_config=RunConfig(
            max_workers=EVAL_MAX_WORKERS, timeout=EVAL_TIMEOUT_SECONDS
        ),
        # A judge that fails on one row must not abandon the whole run; the row
        # simply carries no score and is excluded from the average.
        raise_exceptions=False,
        show_progress=False,
    )

    for row, scores in zip(batch, _per_row_scores(result, len(batch), aliases)):
        row["scores"].update(scores)


def _score(
    rows: list[dict],
    config: ProviderConfig,
    metric_names: tuple[str, ...],
    *,
    cancel: Optional[threading.Event] = None,
    deadline: Optional[float] = None,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """Attach per-row scores to *rows* in place.

    Scored in batches rather than one giant ragas call. A single call is opaque:
    it emits nothing until every row is done, cannot be cancelled, and cannot be
    stopped when a run has overrun its budget — which is exactly how a run ends up
    apparently stuck with a silent log. Batching gives progress, a cancellation
    point, and a deadline check between batches.

    Rows the pipeline failed on are skipped: asking a judge to grade an error
    message wastes calls and drags the average down for a reason that has nothing
    to do with retrieval quality. Rows that retrieved *nothing* are skipped for
    the context-relative metrics only — there is no context to grade them
    against, and an ungrounded answer is the most expensive thing to ask about.
    """
    say = log or (lambda _m: None)
    scorable = [r for r in rows if not r.get("error") and r.get("answer")]
    if not scorable:
        say("Nothing to score — every question failed.")
        return

    with_context = [r for r in scorable if r.get("contexts")]
    without_context = [r for r in scorable if not r.get("contexts")]
    context_free = tuple(m for m in metric_names if m not in CONTEXT_DEPENDENT_METRICS)

    if without_context:
        skipped = ", ".join(m for m in metric_names if m in CONTEXT_DEPENDENT_METRICS)
        say(
            f"{len(without_context)} question(s) retrieved no context — "
            f"skipping {skipped or 'context metrics'} for those."
        )

    groups = [(with_context, metric_names), (without_context, context_free)]
    total = sum(len(g) for g, m in groups if m)
    done = 0

    for group, metrics in groups:
        if not metrics:
            continue
        for start in range(0, len(group), EVAL_SCORE_BATCH):
            if cancel and cancel.is_set():
                say("Cancelled during scoring — partial scores kept.")
                return
            if deadline is not None and time.monotonic() >= deadline:
                say(
                    f"Scoring budget exhausted after {done}/{total} question(s). "
                    f"Raise EVAL_RUN_TIMEOUT_SECONDS to score the rest."
                )
                return
            batch = group[start : start + EVAL_SCORE_BATCH]
            _score_batch(batch, config, metrics)
            done += len(batch)
            say(f"  scored {done}/{total}")


def _per_row_scores(result: Any, expected: int, aliases: dict[str, str]) -> list[dict]:
    """Per-row scores from a ragas EvaluationResult, as plain JSON-safe dicts.

    ``EvaluationResult.scores`` is the documented per-sample list, but the shape
    has moved between ragas versions, so fall back to the dataframe and finally
    to empty dicts rather than losing a completed run to an attribute error.
    """
    raw: Any = getattr(result, "scores", None)
    rows: list[dict] = []
    try:
        if raw is not None:
            rows = [dict(entry) for entry in list(raw)]
        elif hasattr(result, "to_pandas"):
            rows = result.to_pandas().to_dict(orient="records")
    except Exception as exc:
        print(f"[eval] Could not read per-row scores: {exc}", file=sys.stderr)
        rows = []

    out: list[dict] = []
    for i in range(expected):
        entry = rows[i] if i < len(rows) else {}
        scores: dict = {}
        for key, value in entry.items():
            report_key = aliases.get(key)
            if report_key is None or not isinstance(value, (int, float)):
                continue
            if isinstance(value, bool) or _is_nan(value):
                continue
            scores[report_key] = round(float(value), 4)
        out.append(scores)
    return out


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and value != value


def _aggregate(rows: list[dict], metric_names: tuple[str, ...]) -> dict:
    """Mean of each metric over the rows that actually produced a score."""
    out: dict = {}
    for name in metric_names:
        values = [
            r["scores"][name]
            for r in rows
            if isinstance(r.get("scores"), dict) and name in r["scores"]
        ]
        out[name] = round(statistics.fmean(values), 4) if values else None
    return out


# ── the run ──────────────────────────────────────────────────────────────────


def select_metrics(testset: TestSet, requested: Optional[Sequence[str]]) -> tuple[str, ...]:
    """Metrics to run: what was asked for, restricted to what the set supports.

    Asking for context recall on a set with no reference answers is not an error
    worth failing a run over — it is simply not computable, so it is dropped.
    """
    allowed = testset.metrics()
    if not requested:
        return allowed
    chosen = tuple(m for m in allowed if m in set(requested))
    return chosen or allowed


def run_evaluation(
    pipeline: dict,
    config: ProviderConfig,
    testset: TestSet,
    *,
    run_id: str,
    metrics: Optional[Sequence[str]] = None,
    cancel: Optional[threading.Event] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    """Answer every question in *testset*, score the answers, return the report.

    Questions are answered concurrently — they are independent and entirely
    I/O-bound, so this is the cheapest speed-up available — then scored in a
    single batched ragas call, which applies its own concurrency.

    Scoring uses ``config.for_evaluation()``, so a workspace can chat with one
    model and be graded by another.
    """
    say = log or (lambda _m: None)
    metric_names = select_metrics(testset, metrics)
    judge = config.for_evaluation()
    warning = judge_warning(judge)
    deadline = time.monotonic() + EVAL_RUN_TIMEOUT_SECONDS

    report: dict = {
        "id": run_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "testset": testset.name,
        "testset_source": testset.source,
        "testset_size": len(testset.items),
        "has_references": testset.has_references,
        "metrics": list(metric_names),
        # Recorded per run because a score is only comparable to another score
        # produced by the same judge. Without this the trend chart would happily
        # plot a model change as a quality improvement.
        "llm_provider": judge.llm_provider,
        "llm_model": judge.resolved_llm_model,
        "embedding_provider": judge.embedding_provider,
        "embedding_model": judge.resolved_embedding_model,
        "answered_by": f"{config.llm_provider}/{config.resolved_llm_model}",
        "judge_warning": warning,
        "scores": {},
        "rows": [],
        "error": None,
    }

    if warning:
        say(f"[warning] {warning}")
    say(f"Evaluating '{testset.name}' — {len(testset.items)} question(s).")
    say(f"Metrics: {', '.join(METRIC_LABELS[m] for m in metric_names)}")
    if not testset.has_references:
        say("No reference answers in this set — reference-based metrics are skipped.")

    rows: list[dict] = [
        {
            "question": item["question"],
            "ground_truth": item.get("ground_truth"),
            "answer": None,
            "contexts": [],
            "route": None,
            "grounded": None,
            "latency_ms": 0,
            "scores": {},
            "error": None,
        }
        for item in testset.items
    ]

    def answer(index: int) -> None:
        """Fill one row in place. Runs on a worker thread."""
        if cancel and cancel.is_set():
            return
        row = rows[index]
        try:
            result = run_query(pipeline, row["question"], web_fallback=False)
            row.update(
                answer=result.answer or "",
                contexts=[d.page_content for d in result.context_docs],
                route=result.route,
                grounded=result.grounded,
                latency_ms=result.latency_ms,
            )
            say(
                f"[{index + 1}/{len(rows)}] {row['question'][:70]} — "
                f"route={result.route} contexts={len(result.context_docs)} "
                f"{result.latency_ms}ms"
            )
        except Exception as exc:
            row["error"] = str(exc)[:300]
            say(f"[{index + 1}/{len(rows)}] FAILED: {exc}")

    workers = max(1, min(EVAL_ANSWER_WORKERS, len(rows)))
    say(f"Answering {len(rows)} question(s), {workers} at a time...")
    if workers == 1:
        for i in range(len(rows)):
            if cancel and cancel.is_set():
                break
            answer(i)
    else:
        # Cancellation cannot interrupt a query already in flight, so a cancelled
        # run finishes what is running and skips the rest — bounded by one query,
        # not by the whole test set.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(answer, range(len(rows))))

    if cancel and cancel.is_set():
        say("Cancelled.")
        report["status"] = "cancelled"
        report["rows"] = rows
        return report

    if time.monotonic() >= deadline:
        # Answering alone blew the budget. Return what we have rather than
        # starting a scoring phase that cannot finish either.
        say("Run budget exhausted while answering — returning unscored answers.")
        report["rows"] = rows
        report["scores"] = _aggregate(rows, metric_names)
        report["status"] = "completed"
        report["error"] = (
            "Timed out before scoring. Answers were kept. Use a faster judge, "
            "fewer questions, or raise EVAL_RUN_TIMEOUT_SECONDS."
        )
        return report

    say(f"Scoring with RAGAS ({', '.join(METRIC_LABELS[m] for m in metric_names)})...")
    try:
        _score(rows, judge, metric_names, cancel=cancel, deadline=deadline, log=say)
    except Exception as exc:
        # The answers are still worth keeping: the per-question table, routes and
        # latencies are useful on their own, so a scoring failure degrades the
        # run rather than discarding it.
        print(f"[eval] Scoring failed: {exc}", file=sys.stderr)
        say(f"Scoring failed: {exc}")
        report["error"] = f"Scoring failed: {str(exc)[:300]}"

    report["rows"] = rows
    report["scores"] = _aggregate(rows, metric_names)
    report["status"] = "completed"

    say("")
    for name in metric_names:
        value = report["scores"].get(name)
        say(f"  {METRIC_LABELS[name]:<22} {'—' if value is None else f'{value:.3f}'}")
    failed = sum(1 for r in rows if r.get("error"))
    if failed:
        say(f"\n{failed} question(s) failed to run.")
    return report
