"""
query_run.py
────────────
One implementation of "ask the pipeline a question".

Both chat endpoints and the evaluation harness need the same thing: take a
message, route it, retrieve context, and produce an answer.  That logic used to
exist three times — in ``chat_endpoint``, in ``_retrieve_for_streaming``, and
again in ``evaluate.py`` — and the three copies had already drifted apart.  The
evaluation harness is what forced the issue: RAGAS scores *retrieved contexts*,
not just answers, and neither endpoint handed the contexts back to its caller.

``run_query`` is that single implementation.  It returns the retrieved
documents, so a caller that wants to grade retrieval quality can, while the
endpoints keep using only the fields they already used.

Streaming is the one real fork.  Routes that end in a plain LLM call can have
their final generation streamed token-by-token; the graph routes (decomposition,
agentic) compute a whole answer internally and cannot.  ``stream=True`` asks for
the streamable case to be left un-generated so the caller can stream it, and
``QueryResult.streamable`` reports whether that actually happened.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

from ddgs import DDGS
from langchain_core.documents import Document as LangDocument
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .main import post_filter_documents
from .multi_rep_utils import restore_original_content

# Below this cross-encoder score the retrieved chunks are not about the question at
# all. Measured against a coffee-handbook index: "what is the capital of France?"
# scored 0.0013, while genuine hits scored 0.79-0.99.
#
# The floor is deliberately low. Cross-encoders are unreliable in the *other*
# direction — "explain quantum entanglement" scored 0.988 against an unrelated
# chunk — so a high score proves nothing and only a very low one is trustworthy.
# Catching the obvious misses is all this is for; the prompt handles the rest by
# letting the model read the context and judge for itself.
RELEVANCE_FLOOR = float(os.getenv("RELEVANCE_FLOOR", "0.02"))

# Routes that finish with a single LLM call over retrieved context, and so can
# have that call streamed. Everything else runs a graph that computes the whole
# answer internally before returning.
STREAMABLE_ROUTES = {"standard", "multi_query", "rag_fusion", "step_back", "hyde"}

# Keywords that promote a "standard" query to the agentic (CRAG/Self-RAG) graph.
_AGENTIC_KEYWORDS = (
    "compare",
    "versus",
    "difference",
    "evaluate",
    "analyse",
    "analyze",
    "pros and cons",
    "tradeoff",
    "contrast",
)


class HistoryMessage(Protocol):
    """Anything with a role and content — the API's ChatMessage satisfies this."""

    role: str
    content: str


@dataclass
class QueryResult:
    """Everything one pass through the pipeline produced.

    ``answer`` is ``None`` exactly when ``streamable`` is True, which only
    happens if the caller asked for ``stream=True``; the caller is then
    responsible for running ``question_answer_chain`` itself.
    """

    query_text: str  # the standalone question actually retrieved against
    route: str
    context_docs: list = field(default_factory=list)
    chat_history: list[BaseMessage] = field(default_factory=list)
    answer: Optional[str] = None
    streamable: bool = False
    web_results: list[dict] = field(default_factory=list)
    grounded: bool = True
    latency_ms: int = 0


def context_is_relevant(docs: list) -> bool:
    """True when retrieval produced context plausibly about the question."""
    if not docs:
        return False
    scores = [
        doc.metadata.get("relevance_score")
        for doc in docs
        if doc.metadata.get("relevance_score") is not None
    ]
    if not scores:
        # No reranker score available (the fast path skips reranking), so we cannot
        # judge — assume grounded rather than mislabel a good answer.
        return True
    return max(float(s) for s in scores) >= RELEVANCE_FLOOR


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Run a DuckDuckGo text search and return the top results.

    Returns an empty list on any failure so callers can treat it as
    an optional enrichment rather than a critical path dependency.
    """
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        print(f"[DDG] Search failed: {exc}", file=sys.stderr)
        return []


def ddg_results_to_docs(results: list[dict]) -> list:
    """Convert DuckDuckGo result dicts to LangChain Document objects."""
    return [
        LangDocument(
            page_content=r.get("body", ""),
            metadata={
                "source": r.get("href", ""),
                "title": r.get("title", "Web Result"),
                "data_source": "web_search",
                # Use a mid-range score so context_is_relevant passes, but we
                # still mark grounded=False via the web_results sentinel.
                "relevance_score": 0.5,
            },
        )
        for r in results
    ]


def to_chat_messages(history: Optional[Sequence[HistoryMessage]]) -> list[BaseMessage]:
    """Convert the API's role/content pairs to LangChain messages."""
    out: list[BaseMessage] = []
    for msg in history or []:
        if msg.role == "user":
            out.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            out.append(AIMessage(content=msg.content))
    return out


def collect_sources(
    context_docs: list, web_results: list[dict], grounded: bool
) -> list[dict]:
    """Citation dicts for the UI: deduped document sources, then any web hits.

    Document sources are only cited when the answer is grounded — citing chunks
    the model was told to ignore would be a lie. Web results are always cited,
    because when they are present they are what the answer was actually built on.
    """
    out: list[dict] = []
    seen: set = set()
    if grounded:
        for doc in context_docs:
            source_url = doc.metadata.get("source", "Unknown Source")
            title = doc.metadata.get("title", os.path.basename(source_url))
            page = doc.metadata.get("page")
            snippet = doc.page_content[:250].strip()
            key = (title, page, snippet[:50])
            if key in seen:
                continue
            seen.add(key)
            out.append({"title": title, "source": source_url, "page": page, "snippet": snippet})

    out.extend(
        {
            "title": r.get("title", "Web Result"),
            "source": r.get("href", ""),
            "page": None,
            "snippet": r.get("body", "")[:250].strip(),
        }
        for r in web_results
    )
    return out


def run_query(
    pipeline: dict,
    message: str,
    history: Optional[Sequence[HistoryMessage]] = None,
    *,
    stream: bool = False,
    web_fallback: bool = True,
) -> QueryResult:
    """Answer *message* with *pipeline*.

    Args:
        pipeline: a workspace pipeline from ``setup_pipeline``.
        message: the raw user question.
        history: prior turns, used to rewrite *message* into a standalone question.
        stream: when True, streamable routes return ``answer=None`` and
            ``streamable=True`` so the caller can stream the generation itself.
        web_fallback: when False, skip the DuckDuckGo fallback entirely. Evaluation
            runs set this — scoring retrieval against web results measures the
            search engine, not the user's index.
    """
    started = time.perf_counter()
    query = message.strip()
    chat_history = to_chat_messages(history)

    llm = pipeline["llm"]

    # 1. Fold history into a self-contained question, so retrieval never has to
    #    resolve "what about the second one?" on its own.
    if chat_history:
        contextualize_chain = pipeline["contextualize_q_prompt"] | llm
        standalone_q = contextualize_chain.invoke(
            {"input": query, "chat_history": chat_history}
        ).text.strip()
    else:
        standalone_q = query

    routing_retriever = pipeline["routing_retriever"]
    route, _ = routing_retriever.determine_route(standalone_q)

    def finish(
        *,
        query_text: str,
        context_docs: list,
        answer: Optional[str],
        streamable: bool,
        web_results: list[dict],
    ) -> QueryResult:
        # An answer is grounded in the user's documents only when the reranked
        # context cleared the relevance floor AND we did not fall back to the web.
        # Web-search answers keep grounded=False so the UI can badge them.
        return QueryResult(
            query_text=query_text,
            route=route,
            context_docs=context_docs,
            chat_history=chat_history,
            answer=answer,
            streamable=streamable,
            web_results=web_results,
            grounded=context_is_relevant(context_docs) and not web_results,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    # 2. Fast path: conversational or trivial questions skip retrieval entirely.
    if route == "simple":
        fast_result = pipeline["fast_rag_chain"].invoke(
            {"input": standalone_q, "chat_history": chat_history}
        )
        return finish(
            query_text=standalone_q,
            context_docs=fast_result.get("context", []),
            answer=fast_result["answer"],
            streamable=False,
            web_results=[],
        )

    # 3. Heavy path. Extract structured filters, then dispatch on route.
    structured_query = pipeline["query_analyzer"].analyze(standalone_q)
    search_text = structured_query.content_search

    db_filters: dict = {}
    if structured_query.file_type:
        db_filters["file_type"] = structured_query.file_type
    if structured_query.publish_year:
        db_filters["year"] = structured_query.publish_year
    if structured_query.page_number:
        db_filters["page"] = structured_query.page_number
    if structured_query.data_source:
        db_filters["data_source"] = structured_query.data_source
    pipeline["vector_retriever"].search_kwargs["filter"] = db_filters or None

    if route == "decomposition":
        from .decomposition_graph import create_decomposition_graph

        graph = create_decomposition_graph(pipeline["compression_retriever"], llm)
        state = graph.invoke(
            {
                "main_question": search_text,
                "sub_questions": [],
                "current_index": 0,
                "sub_answers": [],
                "retrieved_docs": [],
                "final_answer": "",
            }
        )
        return finish(
            query_text=search_text,
            context_docs=restore_original_content(state["retrieved_docs"]),
            answer=state["final_answer"],
            streamable=False,
            web_results=[],
        )

    if route == "standard" and any(kw in search_text.lower() for kw in _AGENTIC_KEYWORDS):
        from .agentic_graph import create_agentic_graph

        agentic = create_agentic_graph(pipeline["compression_retriever"], llm)
        state = agentic.invoke(
            {
                "question": search_text,
                "rewritten_question": "",
                "retrieved_docs": [],
                "relevant_docs": [],
                "answer": "",
                "reflection_passed": False,
                "answer_relevant": False,
                "retry_count": 0,
            }
        )
        return finish(
            query_text=search_text,
            context_docs=restore_original_content(
                state["relevant_docs"] or state["retrieved_docs"]
            ),
            answer=state["answer"],
            streamable=False,
            web_results=[],
        )

    # 4. Streamable routes: standard / multi_query / rag_fusion / step_back / hyde.
    context_docs = routing_retriever.retrieve_for_route(search_text, route)
    context_docs = post_filter_documents(context_docs, structured_query)
    context_docs = restore_original_content(context_docs)

    # When retrieval finds nothing relevant in the user's documents, search the
    # web before giving up. Deliberately searches the *original* message rather
    # than the rewritten one — a search engine handles raw phrasing better than
    # the analyzer's normalised form.
    web_results: list[dict] = []
    if web_fallback and not context_is_relevant(context_docs):
        web_results = web_search(message)
        if web_results:
            context_docs = ddg_results_to_docs(web_results)
            print(f"[DDG] Falling back to web search for: '{message[:60]}'", file=sys.stderr)

    if stream:
        # Leave generation to the caller so it can stream tokens.
        return finish(
            query_text=search_text,
            context_docs=context_docs,
            answer=None,
            streamable=True,
            web_results=web_results,
        )

    answer = pipeline["question_answer_chain"].invoke(
        {"context": context_docs, "input": search_text, "chat_history": chat_history}
    )
    return finish(
        query_text=search_text,
        context_docs=context_docs,
        answer=answer,
        streamable=False,
        web_results=web_results,
    )
