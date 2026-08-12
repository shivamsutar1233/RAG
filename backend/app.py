import asyncio
import json
import os
import subprocess
import sys
import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import pipeline components from main.py
from .auth import AUTH_ENABLED, CurrentUser, auth_status, get_current_user
from . import live_eval
from .main import setup_pipeline
from .providers import ConfigError, provider_catalog
from .query_run import collect_sources, run_query, web_search
from .storage import get_storage_client
from .user_config import embedding_changed, load_user_config, save_user_config
from .workspace import InvalidUserIdError, Workspace

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Track active ingestion sub-processes by build_id to allow cancellation
ACTIVE_BUILDS: Dict[str, subprocess.Popen] = {}


def project_python() -> str:
    """Interpreter to run the ingest subprocess with.

    `sys.executable` is not reliably right. Depending on how the server was
    launched it can point at a base interpreter that has the project on its path
    but none of its dependencies installed — the subprocess then dies immediately
    with ModuleNotFoundError and ingestion "fails" for no visible reason.

    Prefer, in order: the venv we are running inside, a `.venv` beside the
    project, then whatever launched us.
    """
    exe = "python.exe" if os.name == "nt" else "python"
    bindir = "Scripts" if os.name == "nt" else "bin"

    if sys.prefix != sys.base_prefix:
        candidate = Path(sys.prefix) / bindir / exe
        if candidate.is_file():
            return str(candidate)

    candidate = PROJECT_ROOT / ".venv" / bindir / exe
    if candidate.is_file():
        return str(candidate)

    return sys.executable


# How many users' pipelines stay resident. Each entry holds a FAISS index plus a
# BM25 index over every chunk, so this bound is what stops memory growing with the
# user count. The Flashrank model is shared separately (see main.get_reranker).
MAX_RESIDENT_PIPELINES = int(os.getenv("MAX_RESIDENT_PIPELINES", "4"))

# Ingestion makes one model call per chunk. On CPU-bound local models that is slow,
# so the ceiling is generous — it exists to stop a wedged run holding the lock forever.
INGEST_TIMEOUT_SECONDS = int(os.getenv("INGEST_TIMEOUT_SECONDS", "3600"))

class PipelineCache:
    """LRU of per-workspace pipelines, plus a per-workspace ingest lock.

    Replaces the previous single global pipeline. Lookups are keyed by the
    *verified* user id, so one user can never be handed another's retriever.
    """

    def __init__(self, capacity: int = MAX_RESIDENT_PIPELINES):
        self.capacity = max(1, capacity)
        self._entries: "OrderedDict[str, dict]" = OrderedDict()
        self._ingest_locks: dict[str, asyncio.Lock] = {}
        # Loads happen on FastAPI's threadpool (see chat_endpoint), so load
        # coordination uses threading primitives; ingestion is driven from the
        # event loop and uses asyncio ones. Different mechanisms, different callers.
        self._load_locks: dict[str, threading.Lock] = {}
        self._registry_guard = threading.Lock()

    def lock_for(self, user_id: str) -> asyncio.Lock:
        """Serialise ingestion per user so two builds cannot clobber one index."""
        return self._ingest_locks.setdefault(user_id, asyncio.Lock())

    def peek(self, user_id: str) -> Optional[dict]:
        """Return a loaded pipeline without building one."""
        entry = self._entries.get(user_id)
        if entry is not None:
            self._entries.move_to_end(user_id)
        return entry

    def _store(self, user_id: str, pipeline: dict) -> dict:
        self._entries[user_id] = pipeline
        self._entries.move_to_end(user_id)
        while len(self._entries) > self.capacity:
            evicted, _ = self._entries.popitem(last=False)
            print(f"♻️  [Cache] Evicted pipeline for workspace '{evicted}'.")
        return pipeline

    def get(self, workspace: Workspace) -> dict:
        """Return the workspace's pipeline, loading it on first use.

        Blocking by design: `setup_pipeline` reads FAISS off disk, builds a BM25
        index over every chunk and embeds the router's reference samples. Callers
        must be on a worker thread (a sync FastAPI handler, startup, or the CLI) —
        never inside a coroutine, or the event loop stalls for every other user.

        The per-workspace lock stops two concurrent first requests both paying the
        load cost; the registry guard protects the lock dict itself.
        """
        cached = self.peek(workspace.user_id)
        if cached is not None:
            return cached

        with self._registry_guard:
            lock = self._load_locks.setdefault(workspace.user_id, threading.Lock())

        with lock:
            # Another thread may have loaded it while we queued for the lock.
            cached = self.peek(workspace.user_id)
            if cached is not None:
                return cached
            return self._store(workspace.user_id, setup_pipeline(workspace))

    def invalidate(self, user_id: str) -> None:
        """Drop a workspace's pipeline so the next request reloads it."""
        self._entries.pop(user_id, None)


pipelines = PipelineCache()


def current_workspace(user: CurrentUser) -> Workspace:
    """Workspace for a verified caller.

    The id comes from the JWT's `sub` claim and nowhere else — never from a body,
    query string or header the client controls. In single-user mode (no Supabase
    configured) `user.id` is the shared local sentinel.
    """
    try:
        return Workspace.for_user(user.id).ensure()
    except InvalidUserIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def require_pipeline(workspace: Workspace, token: str | None = None) -> dict:
    """Fetch a workspace's pipeline or explain precisely why it is unavailable.

    On a cache miss — local index directory is empty — tries to pull the index
    from Supabase Storage before giving up.  This is what lets a user sign in
    from a fresh server and immediately query their existing documents.
    """
    try:
        return pipelines.get(workspace)
    except FileNotFoundError:
        # Cache miss: try recovering from Storage before failing.
        if workspace.sync_index_from_storage(token):
            try:
                return pipelines.get(workspace)
            except Exception as exc:
                print(
                    f"[API ERROR] Pipeline load failed after Storage recovery "
                    f"for '{workspace.user_id}': {exc}",
                    file=sys.stderr,
                )
                raise HTTPException(status_code=503, detail=f"Pipeline unavailable: {exc}") from exc
        raise HTTPException(
            status_code=503,
            detail=(
                f"FAISS index for workspace '{workspace.user_id}' is empty or does not exist. "
                f"Upload documents and build the index first."
            ),
        )
    except Exception as exc:
        print(f"[API ERROR] Pipeline load failed for '{workspace.user_id}': {exc}", file=sys.stderr)
        raise HTTPException(status_code=503, detail=f"Pipeline unavailable: {exc}") from exc


# Modern Lifespan Manager replacing deprecated startup event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ingestion runs as a subprocess, so surface which interpreter it will use.
    # When these two differ, `sys.executable` could not have imported the project.
    resolved = project_python()
    print(f"🐍 [Ingest] interpreter: {resolved}")
    if resolved != sys.executable:
        print(f"   (server is running under {sys.executable})")
    print(f"🔐 [Auth] {'enabled — Supabase' if AUTH_ENABLED else 'disabled — single-user mode'}")

    # Warm the default workspace so a single-user install is ready on first request.
    # A missing index is normal on a fresh install and must not block startup.
    try:
        pipelines.get(Workspace.for_user(None).ensure())
        print("💡 [API] RAG Pipeline loaded successfully.")
    except Exception as e:
        print(f"❌ [API] Pipeline not ready yet: {e}", file=sys.stderr)

    # Drains the live-evaluation spool in the background. Started after the
    # pipeline warm-up because it needs neither the pipeline nor a request — it
    # scores answers the chat path already produced.
    live_worker = live_eval.LiveEvalWorker()
    live_worker.start()
    try:
        yield
    finally:
        live_worker.stop()


# Initialize FastAPI application
app = FastAPI(
    title="Conversational RAG API",
    description="Backend API serving the Advanced Local Conversational RAG pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/api/auth/config")
def get_auth_config():
    """Public: lets the dashboard decide whether to show a sign-in screen.

    Deliberately unauthenticated — the client cannot know to send a token until
    it knows auth is switched on. Returns no secrets.
    """
    return auth_status()


@app.get("/health")
def health_check():
    """Public: unauthenticated health check for Railway/Docker container orchestration."""
    return {"status": "ok"}


# Enable CORS for frontend integration (CORS origins configurable via env)
allowed_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=bool(allowed_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


class SourceDocument(BaseModel):
    title: str
    source: str
    page: Optional[int] = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    route: str
    sources: List[SourceDocument]
    grounded: bool = True
    """False when retrieval found nothing usable, so the answer came from the
    model's general knowledge rather than the user's documents. The dashboard
    badges these — an ungrounded answer that looks grounded is worse than no
    answer at all."""


class TestSetSummary(BaseModel):
    name: str
    source: str
    size: int
    has_references: bool
    metrics: List[str]


class TestSetCreateRequest(BaseModel):
    name: str
    # Either raw JSON/CSV text (an upload or a paste)...
    content: Optional[str] = None
    # ...or a plain list of questions, which is how the dashboard sends the ones
    # a user picked out of their chat history. Those have no reference answers,
    # so such a set is scored on the reference-free metrics only.
    questions: Optional[List[str]] = None
    source: str = "uploaded"


class GenerateTestSetRequest(BaseModel):
    name: str
    size: int = 10


class EvalRunRequest(BaseModel):
    testset: str
    # Omitted means "every metric this test set supports". Naming a subset is the
    # main cost lever a user has: each metric is several judge calls per question.
    metrics: Optional[List[str]] = None


class EvalRow(BaseModel):
    question: str
    ground_truth: Optional[str] = None
    answer: Optional[str] = None
    contexts: List[str] = []
    route: Optional[str] = None
    grounded: Optional[bool] = None
    latency_ms: int = 0
    scores: dict = {}
    error: Optional[str] = None


class EvalRun(BaseModel):
    """One evaluation run. Also used for test set generation jobs, which share
    the run machinery but carry `kind="generate"` and no scores."""

    id: str
    kind: str = "eval"
    status: str
    started_at: str
    completed_at: Optional[str] = None
    testset: Optional[str] = None
    testset_source: Optional[str] = None
    testset_size: int = 0
    has_references: bool = False
    metrics: List[str] = []
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    # Which model produced the answers, when a separate judge graded them. Kept
    # distinct from llm_model (the judge) so a scorecard says who did what.
    answered_by: Optional[str] = None
    judge_warning: Optional[str] = None
    scores: dict = {}
    error: Optional[str] = None


class EvalRunDetail(EvalRun):
    rows: List[EvalRow] = []


class LiveTurn(BaseModel):
    """One scored turn of real chat traffic."""

    id: str
    at: str
    question: str
    answer: str
    route: Optional[str] = None
    grounded: Optional[bool] = None
    latency_ms: int = 0
    contexts: int = 0
    scores: dict = {}
    scored_at: Optional[str] = None
    judge: Optional[str] = None


class LiveQuality(BaseModel):
    """Rolling quality over recent chat traffic, plus the worker's backlog."""

    enabled: bool
    sample_rate: float
    metrics: List[str] = []
    scored_total: int = 0
    pending: int = 0
    averages: dict = {}
    grounded_rate: Optional[float] = None
    median_latency_ms: Optional[int] = None
    judge: Optional[str] = None
    recent: List[LiveTurn] = []


class ConfigUpdateRequest(BaseModel):
    routing_method: str
    reranker_provider: str
    # Runtime provider selection. Omitted fields keep their current value.
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    # Optional separate judge for evaluation runs. An empty string clears it back
    # to "use the chat model", which None cannot express.
    eval_llm_provider: Optional[str] = None
    eval_llm_model: Optional[str] = None
    # Credentials, applied to the process environment when supplied.
    openai_key: Optional[str] = None
    anthropic_key: Optional[str] = None
    google_key: Optional[str] = None
    xai_key: Optional[str] = None
    cohere_key: Optional[str] = None


# The Next.js dashboard builds to a static export, which this server mounts.
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "out")


@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serves the dashboard shell, or build instructions if it has not been built."""
    try:
        with open(os.path.join(FRONTEND_DIR, "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        pass
    return (
        "<html><head><title>Aether AI</title></head>"
        '<body style="background:#0a0f0d;color:#e7efeb;font-family:system-ui;'
        'display:flex;align-items:center;justify-content:center;height:100vh;">'
        "<div style='max-width:34rem'>"
        "<h2>Dashboard not built</h2>"
        "<p>Build the frontend, then reload:</p>"
        "<pre style='background:#121a17;padding:1rem;border-radius:.5rem'>"
        "cd frontend\nnpm install\nnpm run build</pre>"
        "<p>The API itself is running — see <code>/docs</code>.</p>"
        "</div></body></html>"
    )


@app.get("/api/status")
def get_status(user: CurrentUser = Depends(get_current_user)):
    """Database status, active configuration and staged files for the caller's workspace."""
    workspace = current_workspace(user)

    # Report on an already-loaded pipeline only. Building one here would make a
    # polling endpoint pay the FAISS + BM25 load cost.
    loaded = pipelines.peek(workspace.user_id)
    doc_count = 0
    if loaded:
        try:
            doc_count = len(loaded["vector_retriever"].vectorstore.docstore._dict)
        except Exception:
            pass

    return {
        "status": "ready" if loaded else "error",
        "workspace": workspace.user_id,
        "user_email": user.email,
        **auth_status(),
        "storage_enabled": AUTH_ENABLED,
        "database_loaded": workspace.has_index,
        "document_chunks": doc_count,
        "langsmith_tracing": os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        "staged_files": workspace.staged_files(),
        # Provider/model/routing all come from *this user's* saved config.
        **load_user_config(workspace).summary(),
    }


@app.get("/api/providers")
def get_providers(user: CurrentUser = Depends(get_current_user)):
    """Catalog of selectable providers, scoped to which keys this user has saved."""
    return provider_catalog(load_user_config(current_workspace(user)))


@app.post("/api/config")
async def update_config(
    config: ConfigUpdateRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(get_current_user),
):
    """Save this caller's provider settings and rebuild their pipeline.

    Nothing here touches `os.environ`. Settings are written to the caller's own
    workspace, so one user changing provider or key affects only themselves —
    which is the whole point of this endpoint after multi-tenancy.
    """
    workspace = current_workspace(user)
    previous = load_user_config(workspace)

    # Merge the request onto the stored config. Omitted credentials keep their
    # saved value rather than being wiped, so the UI can leave fields blank.
    try:
        updated = previous.with_updates(
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            routing_method=config.routing_method,
            reranker_provider=config.reranker_provider,
            # with_updates drops None, so omitting these keeps the saved judge
            # while sending "" clears it back to the chat model.
            eval_llm_provider=config.eval_llm_provider,
            eval_llm_model=config.eval_llm_model,
            keys={
                "OPENAI_API_KEY": config.openai_key,
                "ANTHROPIC_API_KEY": config.anthropic_key,
                "GOOGLE_API_KEY": config.google_key,
                "XAI_API_KEY": config.xai_key,
                "COHERE_API_KEY": config.cohere_key,
            },
        )
        updated.validate()
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Changing how documents are embedded invalidates the existing index: the stored
    # vectors belong to the old model's space.
    index_stale = embedding_changed(previous, updated)

    # Build the pipeline *before* persisting, so a configuration that cannot
    # actually serve requests is rejected instead of being saved and left broken.
    pipelines.invalidate(workspace.user_id)
    try:
        setup_pipeline(workspace, updated)
    except FileNotFoundError:
        # No index yet — legitimate for a new user. The config itself is fine.
        pass
    except Exception as exc:
        print(f"[API ERROR] Config apply failed: {exc}", file=sys.stderr)
        detail = "An internal server error occurred while applying the configuration. Please check server logs."
        if index_stale:
            detail = (
                f"{detail} Note: changing the embedding model invalidates the existing index. "
                f"Re-ingest your documents, then apply this change again."
            )
        raise HTTPException(status_code=400, detail=detail) from exc

    save_user_config(workspace, updated, token=user.token, background_tasks=background_tasks)
    pipelines.invalidate(workspace.user_id)

    message = "Configuration saved."
    if index_stale:
        message += (
            " The embedding model changed, so your existing index is stale — "
            "re-ingest your documents before querying."
        )
    return {
        "status": "success",
        "message": message,
        "embedding_changed": index_stale,
        **updated.summary(),
    }


# Deliberately a sync `def`, not `async def`. Everything below — retrieval, the
# LangGraph agents, every LLM call — is synchronous LangChain code with no awaitable
# I/O. Declared `async` it would run on the event loop and block every other request
# for the duration of a query; as a sync handler FastAPI runs it on its threadpool,
# so concurrent users are actually served concurrently.
@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
):
    workspace = current_workspace(user)
    pipeline = require_pipeline(workspace, token=user.token)

    try:
        result = run_query(pipeline, request.message, request.history)
        sources = collect_sources(result.context_docs, result.web_results, result.grounded)

        # Spool this turn for background scoring. A small file write, after the
        # answer is already in hand, and it swallows its own failures — live
        # evaluation must never cost the user a reply.
        #
        # The *original* message, not the rewritten query the analyzer produced:
        # a live feed showing "coffee quality" when someone asked "what scale
        # grades coffee quality?" is not showing real traffic, and relevance
        # judged against the rewrite would grade the pipeline on its own
        # paraphrase rather than on what the person actually wanted.
        live_eval.enqueue(workspace, request.message, result)

        return ChatResponse(
            answer=result.answer or "",
            route=result.route,
            sources=[SourceDocument(**s) for s in sources[:5]],
            grounded=result.grounded,
        )
    except Exception as e:
        print(f"[API ERROR] Chat execution failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=(
                "An internal server error occurred while processing your chat request. "
                "Please check server logs."
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Streaming chat  (Server-Sent Events)
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/api/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """SSE streaming variant of /api/chat.

    Retrieval runs synchronously on the thread pool so the event loop is not
    blocked. Only the final LLM generation step is streamed token-by-token.
    Graph routes (decomposition, agentic) return a pre-computed answer and are
    yielded as a single chunk for consistency.

    Event format::

        data: {"token": "Hello", "done": false}\n\n   # partial token
        data: {"done": true, "route": "standard",   # final metadata
               "sources": [...], "grounded": true}\n\n
        data: {"error": "...", "done": true}\n\n     # on failure
    """
    workspace = current_workspace(user)
    pipeline = require_pipeline(workspace, token=user.token)

    async def event_stream():
        loop = asyncio.get_running_loop()

        # ── Phase 1: Retrieval (sync, runs on thread pool) ────────────────
        # run_query performs the DuckDuckGo fallback internally, so that too
        # stays off the event loop instead of needing its own executor hop.
        try:
            result = await loop.run_in_executor(
                None,
                lambda: run_query(pipeline, request.message, request.history, stream=True),
            )
        except Exception as exc:
            print(f"[STREAM] Retrieval failed: {exc}", file=sys.stderr)
            yield f"data: {json.dumps({'error': 'An internal server error occurred while processing your request. Please check server logs.', 'done': True})}\n\n"
            return

        # ── Phase 2: Token streaming ───────────────────────────────────────
        # Tokens are accumulated as they go out so the finished answer can be
        # spooled for live scoring; the streamed reply itself is unaffected.
        streamed: list[str] = []
        if not result.streamable:
            # Graph routes already have the answer — yield it as one block.
            yield f"data: {json.dumps({'token': result.answer or '', 'done': False})}\n\n"
        else:
            question_answer_chain = pipeline["question_answer_chain"]
            token_q: asyncio.Queue = asyncio.Queue()

            def _streamer() -> None:
                try:
                    for chunk in question_answer_chain.stream({
                        "context": result.context_docs,
                        "input": result.query_text,
                        "chat_history": result.chat_history,
                    }):
                        asyncio.run_coroutine_threadsafe(token_q.put(chunk), loop)
                except Exception as exc:  # noqa: BLE001
                    asyncio.run_coroutine_threadsafe(token_q.put(exc), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(token_q.put(None), loop)

            threading.Thread(target=_streamer, daemon=True).start()

            while True:
                item = await token_q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    yield f"data: {json.dumps({'error': 'An internal server error occurred while processing your request. Please check server logs.', 'done': True})}\n\n"
                    return
                streamed.append(str(item))
                yield f"data: {json.dumps({'token': item, 'done': False})}\n\n"

        # ── Phase 3: Final metadata event ─────────────────────────────────
        # Spool after the last token has gone out, so scoring never sits between
        # the user and their reply.
        if streamed:
            result.answer = "".join(streamed)
        live_eval.enqueue(workspace, request.message, result)

        final_event = json.dumps({
            "done": True,
            "route": result.route,
            "sources": collect_sources(
                result.context_docs, result.web_results, result.grounded
            )[:5],
            "grounded": result.grounded,
        })
        yield f"data: {final_event}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/search")
def web_search_endpoint(
    q: str,
    user: CurrentUser = Depends(get_current_user),
):
    """DuckDuckGo search proxy — returns up to 8 results for the given query."""
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required.")
    results = web_search(q.strip(), max_results=8)
    return {"results": results}


@app.post("/api/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    """Stage files into the caller's workspace, sanitising every filename.

    Each file is saved to local disk *and* uploaded to Supabase Storage (when
    configured).  The Storage upload is best-effort — a failure is logged but
    does not fail the request.
    """
    workspace = current_workspace(user)
    storage = get_storage_client(workspace.user_id, user.token)
    saved_files = []
    
    ALLOWED_MIME_TYPES = {
        "text/plain", 
        "application/pdf", 
        "application/msword", 
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/markdown", 
        "text/csv"
    }

    for file in files:
        if file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
            
        # Sanitise file name to prevent path traversal (CWE-22 / CWE-23). The
        # workspace root is derived from a verified id, but a crafted filename
        # could still climb out of it.
        safe_filename = os.path.basename(file.filename or "")
        safe_filename = safe_filename.replace("\0", "").replace("/", "").replace("\\", "")
        if not safe_filename:
            continue

        file_bytes = await file.read()
        (workspace.documents_dir / safe_filename).write_bytes(file_bytes)
        saved_files.append(safe_filename)

        # Push to Supabase Storage (best-effort, in background)
        if storage:
            background_tasks.add_task(storage.upload_document, safe_filename, file_bytes)

    return {"status": "success", "uploaded_files": saved_files}


@app.delete("/api/documents/{filename}")
async def delete_document(
    filename: str,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(get_current_user),
):
    """Delete a staged document from both local disk and Supabase Storage."""
    workspace = current_workspace(user)

    # Sanitise — same rules as upload
    safe = os.path.basename(filename or "")
    safe = safe.replace("\0", "").replace("/", "").replace("\\", "")
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    local_deleted = workspace.delete_document(safe)

    # Remove from Supabase Storage too (in background)
    storage = get_storage_client(workspace.user_id, user.token)
    if storage:
        background_tasks.add_task(storage.delete_document, safe)

    if not local_deleted and not storage:
        raise HTTPException(status_code=404, detail=f"File '{safe}' not found.")

    return {"status": "success", "deleted": safe}


def _run_ingestion_background(workspace: Workspace, raptor: bool, build_id: str, token: str):
    log_file = workspace.builds_dir / f"{build_id}.log"
    meta_file = workspace.builds_dir / f"{build_id}.json"

    meta = {
        "id": build_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "raptor": raptor,
        "error": None
    }
    meta_file.write_text(json.dumps(meta))

    cmd = [project_python(), "-m", "backend.ingest", "--user", workspace.user_id]
    if raptor:
        cmd.append("--raptor")

    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"

    try:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Starting build {build_id}...\n")

            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=f,
                stderr=subprocess.STDOUT,
                env=child_env
            )
            ACTIVE_BUILDS[build_id] = process

            # Wait for it to finish in this thread
            returncode = process.wait()
            ACTIVE_BUILDS.pop(build_id, None)

        if returncode != 0:
            meta["status"] = "failed"
            meta["error"] = f"exit code {returncode}"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n[API ERROR] Ingestion failed with exit code {returncode}\n")
        else:
            meta["status"] = "completed"

            try:
                pipelines.invalidate(workspace.user_id)
                pipelines.get(workspace)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write("\n[API] Pipeline reloaded successfully.\n")
            except Exception as exc:
                meta["error"] = f"Reload failed: {exc}"
                meta["status"] = "failed"
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n[API ERROR] Reload failed: {exc}\n")

            if meta["status"] == "completed":
                try:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("[API] Syncing to cloud storage...\n")
                    workspace.sync_to_storage(token)
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("[API] Cloud sync complete.\n")
                except Exception as exc:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"[API WARN] Cloud sync failed: {exc}\n")

    except Exception as exc:
        meta["status"] = "failed"
        meta["error"] = "An internal server error occurred while building the index. Please check server logs."
        ACTIVE_BUILDS.pop(build_id, None)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[API ERROR] Subprocess error: {exc}\n")
    finally:
        meta["completed_at"] = datetime.now(timezone.utc).isoformat()

        # If the build was cancelled, override the status unless another process
        # has already set it.
        if meta["status"] == "running":
             # This handles the ghost build issue when the server restarted mid-build
             meta["status"] = "failed"
             meta["error"] = "Cancelled or server restarted"

        meta_file.write_text(json.dumps(meta))
        ACTIVE_BUILDS.pop(build_id, None)


@app.post("/api/builds/{build_id}/cancel")
def cancel_build(build_id: str, user: CurrentUser = Depends(get_current_user)):
    """Cancel a running build."""
    workspace = current_workspace(user)

    # 1. Kill the process if it's currently actively tracked in memory
    process = ACTIVE_BUILDS.get(build_id)
    if process:
        try:
            process.terminate()  # Sends SIGTERM
            ACTIVE_BUILDS.pop(build_id, None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to kill process: {e}")

    # 2. Even if it's not in memory (e.g., server restarted leaving a 'ghost' build),
    # explicitly mark it as cancelled in the JSON file so the user is unblocked.
    meta_file = workspace.builds_dir / f"{build_id}.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
            if meta.get("status") == "running":
                meta["status"] = "cancelled"
                meta["completed_at"] = datetime.now(timezone.utc).isoformat()
                meta["error"] = "Build cancelled by user"
                meta_file.write_text(json.dumps(meta))

                # Append to logs
                log_file = workspace.builds_dir / f"{build_id}.log"
                if log_file.exists():
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("\n[API] Build cancelled by user.\n")
        except Exception:
            pass

    return {"status": "cancelled"}


@app.post("/api/ingest")
async def trigger_ingestion(
    background_tasks: BackgroundTasks,
    raptor: bool = False,
    user: CurrentUser = Depends(get_current_user),
):
    """Start an ingestion background job and return a build_id."""
    workspace = current_workspace(user)

    # Check if a build is already running
    if workspace.builds_dir.exists():
        for p in workspace.builds_dir.glob("*.json"):
            try:
                m = json.loads(p.read_text())
                if m.get("status") == "running":
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "An ingestion is already running for this workspace. "
                            "Wait for it to finish."
                        ),
                    )
            except Exception:
                continue

    build_id = str(uuid.uuid4())
    background_tasks.add_task(
        _run_ingestion_background,
        workspace,
        raptor,
        build_id,
        user.token
    )

    return {"status": "success", "build_id": build_id}


@app.get("/api/builds")
def list_builds(user: CurrentUser = Depends(get_current_user)):
    workspace = current_workspace(user)

    # Ensure local cache is up-to-date with cloud
    workspace.sync_builds_from_storage(user.token)

    builds = []
    if workspace.builds_dir.exists():
        for p in workspace.builds_dir.glob("*.json"):
            try:
                builds.append(json.loads(p.read_text()))
            except Exception:
                pass
    builds.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    return {"builds": builds}


def _stream_log(log_file: Path, meta_file: Path) -> StreamingResponse:
    """Tail a job's log over SSE until its status JSON leaves 'running'.

    Shared by build and evaluation runs: both write a log beside a status file
    and both are followed live by the dashboard.
    """
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="Run not found.")

    async def log_generator():
        # Open file in read mode. It might not exist immediately if the thread hasn't opened it yet.
        for _ in range(10):
            if log_file.exists():
                break
            await asyncio.sleep(0.2)

        if not log_file.exists():
            yield "event: close\ndata: \n\n"
            return

        with open(log_file, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if line:
                    yield f"data: {json.dumps({'text': line})}\n\n"
                else:
                    # Reached EOF, check if build is done
                    try:
                        meta = json.loads(meta_file.read_text())
                        if meta.get("status") != "running":
                            yield "event: close\ndata: \n\n"
                            break
                    except Exception:
                        pass
                    # Yield a keep-alive comment so the connection doesn't drop
                    yield ": keep-alive\n\n"
                    await asyncio.sleep(0.5)

    return StreamingResponse(log_generator(), media_type="text/event-stream")


@app.get("/api/builds/{build_id}/stream")
async def stream_build_logs(build_id: str, user: CurrentUser = Depends(get_current_user)):
    workspace = current_workspace(user)
    safe_id = os.path.basename(build_id)
    return _stream_log(
        workspace.builds_dir / f"{safe_id}.log",
        workspace.builds_dir / f"{safe_id}.json",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
#
# Runs reuse the build-job shape — a status JSON plus a log file per run, polled
# and tailed by the dashboard — with two deliberate differences. An eval runs
# in-process rather than as a subprocess, because the whole point is to score the
# *live* cached pipeline and a subprocess would reload FAISS, BM25 and Flashrank
# from scratch. And because there is no child process to signal, cancellation is
# a threading.Event the run checks between questions.
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_EVALS: Dict[str, threading.Event] = {}

# Runs live in this process, so a restart abandons any that were in flight. Their
# status JSON still says "running", which would spin the dashboard forever and —
# worse — trip the one-run-at-a-time guard, locking the workspace out of ever
# starting another. Anything that claims to be running but began before this
# process did is therefore a ghost, and gets reaped on sight.
PROCESS_STARTED_AT = datetime.now(timezone.utc)


def _reap_ghost_runs(evals_dir: Path) -> None:
    """Mark runs abandoned by a restart as failed. Cheap and idempotent."""
    if not evals_dir.is_dir():
        return
    for path in evals_dir.glob("*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("status") != "running" or meta.get("id") in ACTIVE_EVALS:
            continue
        try:
            started = datetime.fromisoformat(meta["started_at"])
        except (KeyError, TypeError, ValueError):
            started = PROCESS_STARTED_AT  # unparseable: treat as a ghost
        if started >= PROCESS_STARTED_AT:
            continue  # started by this process and still tracked elsewhere
        meta["status"] = "failed"
        meta["completed_at"] = datetime.now(timezone.utc).isoformat()
        meta["error"] = "Interrupted by a server restart."
        _write_run_json(path, meta)
        print(f"[API] Reaped abandoned evaluation run {meta.get('id')}.", file=sys.stderr)


def _write_run_json(path: Path, data: dict) -> None:
    """Write run metadata atomically.

    The dashboard polls these files every few seconds while they are being
    rewritten. A plain write can be read back half-finished, which surfaces as a
    JSON parse error in the UI for no reason the user can act on.
    """
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def _running_run(evals_dir: Path) -> Optional[str]:
    """Id of the run currently in progress for this workspace, if any."""
    if not evals_dir.is_dir():
        return None
    _reap_ghost_runs(evals_dir)
    for path in evals_dir.glob("*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("status") == "running":
            return meta.get("id")
    return None


def _run_eval_job(
    workspace: Workspace,
    run_id: str,
    token: Optional[str],
    kind: str,
    body,
    **meta_extra,
) -> None:
    """Scaffolding shared by evaluation runs and test set generation.

    Owns the log file, the status JSON, the cancel token and the storage sync;
    *body* does the actual work and returns fields to merge into the report.
    """
    workspace.ensure()
    log_file = workspace.evals_dir / f"{run_id}.log"
    meta_file = workspace.evals_dir / f"{run_id}.json"

    cancel = threading.Event()
    ACTIVE_EVALS[run_id] = cancel

    def say(message: str) -> None:
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")

    meta: dict = {
        "id": run_id,
        "kind": kind,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "error": None,
        **meta_extra,
    }
    log_file.write_text("", encoding="utf-8")
    _write_run_json(meta_file, meta)

    try:
        meta.update(body(say, cancel))
    except Exception as exc:
        print(f"[API ERROR] {kind} run {run_id} failed: {exc}", file=sys.stderr)
        meta["status"] = "failed"
        meta["error"] = str(exc)[:300]
        say(f"[ERROR] {exc}")
    finally:
        meta["completed_at"] = datetime.now(timezone.utc).isoformat()
        if meta.get("status") == "running":
            # Reached only if body() returned without setting a terminal status,
            # or the process was restarted mid-run.
            meta["status"] = "failed"
            meta["error"] = meta.get("error") or "Interrupted"

        # Log first, status second. The SSE tailer closes the stream as soon as
        # the status leaves "running", so anything written after that write can
        # be lost from the live view.
        say("")
        say(f"[done] {meta['status']}")
        _write_run_json(meta_file, meta)
        ACTIVE_EVALS.pop(run_id, None)
        try:
            workspace.sync_evals_to_storage(token)
        except Exception as exc:
            print(f"[API WARN] Eval storage sync failed: {exc}", file=sys.stderr)


def _run_evaluation_background(
    workspace: Workspace,
    testset_name: str,
    run_id: str,
    token: Optional[str],
    metrics: Optional[List[str]] = None,
) -> None:
    def body(say, cancel) -> dict:
        from .evaluation import load_testset, run_evaluation

        testset = load_testset(workspace, testset_name)
        config = load_user_config(workspace, token)
        pipeline = pipelines.get(workspace)
        return run_evaluation(
            pipeline,
            config,
            testset,
            run_id=run_id,
            metrics=metrics,
            cancel=cancel,
            log=say,
        )

    _run_eval_job(workspace, run_id, token, "eval", body, testset=testset_name)


def _run_generation_background(
    workspace: Workspace, name: str, size: int, run_id: str, token: Optional[str]
) -> None:
    def body(say, cancel) -> dict:
        from .evaluation import TestSet, generate_testset, save_testset

        config = load_user_config(workspace, token)
        pipeline = pipelines.get(workspace)
        items = generate_testset(pipeline, config, size=size, log=say)
        if cancel.is_set():
            return {"status": "cancelled"}
        save_testset(workspace, TestSet(name=name, items=items, source="generated"))
        say(f"Saved test set '{name}'.")
        return {"status": "completed", "testset_size": len(items)}

    _run_eval_job(workspace, run_id, token, "generate", body, testset=name)


@app.get("/api/eval/testsets", response_model=List[TestSetSummary])
def list_eval_testsets(user: CurrentUser = Depends(get_current_user)):
    from .evaluation import list_testsets

    workspace = current_workspace(user)
    workspace.sync_evals_from_storage(user.token)
    return list_testsets(workspace)


@app.post("/api/eval/testsets", response_model=TestSetSummary)
def create_eval_testset(
    request: TestSetCreateRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """Save an uploaded/pasted test set, or one built from chat history."""
    from .evaluation import TestSet, parse_testset, save_testset

    workspace = current_workspace(user)
    try:
        if request.questions:
            items = [{"question": q.strip()} for q in request.questions if q.strip()]
            if not items:
                raise ValueError("No questions supplied.")
            testset = TestSet(name=request.name, items=items, source=request.source)
        elif request.content:
            testset = parse_testset(request.name, request.content, request.source)
        else:
            raise ValueError("Provide either 'content' or 'questions'.")
        save_testset(workspace, testset)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    workspace.sync_evals_to_storage(user.token)
    return TestSetSummary(
        name=testset.name,
        source=testset.source,
        size=len(testset.items),
        has_references=testset.has_references,
        metrics=list(testset.metrics()),
    )


@app.delete("/api/eval/testsets/{name}")
def delete_eval_testset(name: str, user: CurrentUser = Depends(get_current_user)):
    from .evaluation import testset_path

    workspace = current_workspace(user)
    try:
        path = testset_path(workspace, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No test set named '{name}'.")
    path.unlink()
    return {"status": "success", "deleted": name}


@app.post("/api/eval/testsets/generate")
def generate_eval_testset(
    request: GenerateTestSetRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(get_current_user),
):
    """Write a test set from the workspace's own chunks. Returns a run id."""
    from .evaluation import MAX_TESTSET_SIZE, testset_path

    workspace = current_workspace(user)
    try:
        testset_path(workspace, request.name)  # validates the name
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not 1 <= request.size <= MAX_TESTSET_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Size must be between 1 and {MAX_TESTSET_SIZE}.",
        )

    # Sync so the caller learns the index is missing now, rather than from a
    # failed job thirty seconds later.
    require_pipeline(workspace, token=user.token)

    if (active := _running_run(workspace.evals_dir)) is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Run {active} is already in progress for this workspace.",
        )

    run_id = str(uuid.uuid4())
    background_tasks.add_task(
        _run_generation_background, workspace, request.name, request.size, run_id, user.token
    )
    return {"status": "success", "run_id": run_id}


# Sync `def` for the same reason as chat_endpoint: it calls require_pipeline,
# which loads FAISS off disk. On the event loop that would stall every other
# request; on the threadpool it only costs this one.
@app.post("/api/eval/run")
def trigger_evaluation(
    request: EvalRunRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(get_current_user),
):
    """Start an evaluation run against a saved test set. Returns a run id."""
    from .evaluation import load_testset

    workspace = current_workspace(user)
    try:
        load_testset(workspace, request.testset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    require_pipeline(workspace, token=user.token)

    if (active := _running_run(workspace.evals_dir)) is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Run {active} is already in progress for this workspace.",
        )

    run_id = str(uuid.uuid4())
    background_tasks.add_task(
        _run_evaluation_background,
        workspace,
        request.testset,
        run_id,
        user.token,
        request.metrics,
    )
    return {"status": "success", "run_id": run_id}


@app.get("/api/eval/live", response_model=LiveQuality)
def get_live_quality(
    recent: int = 25,
    user: CurrentUser = Depends(get_current_user),
):
    """Rolling quality of real chat traffic for this workspace.

    Reads only what the worker has already written — it never scores on the
    request, so this stays a cheap poll.
    """
    workspace = current_workspace(user)
    return live_eval.summarise(workspace, recent=max(1, min(recent, 100)))


@app.get("/api/eval/runs", response_model=List[EvalRun])
def list_eval_runs(user: CurrentUser = Depends(get_current_user)):
    """Run summaries, newest first. Rows are omitted — the trend chart and the
    history list only need the aggregates, and a run's rows carry every retrieved
    chunk, which would make this response enormous."""
    workspace = current_workspace(user)
    workspace.sync_evals_from_storage(user.token)
    _reap_ghost_runs(workspace.evals_dir)

    runs = []
    if workspace.evals_dir.is_dir():
        for path in workspace.evals_dir.glob("*.json"):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                meta.pop("rows", None)
                runs.append(meta)
            except Exception:
                continue
    runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)
    return runs


@app.get("/api/eval/runs/{run_id}", response_model=EvalRunDetail)
def get_eval_run(run_id: str, user: CurrentUser = Depends(get_current_user)):
    workspace = current_workspace(user)
    path = workspace.evals_dir / f"{os.path.basename(run_id)}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Run file is corrupt.") from exc


@app.post("/api/eval/runs/{run_id}/cancel")
def cancel_eval_run(run_id: str, user: CurrentUser = Depends(get_current_user)):
    """Ask a run to stop. It finishes the question in flight, then exits."""
    workspace = current_workspace(user)
    safe_id = os.path.basename(run_id)

    if (event := ACTIVE_EVALS.get(safe_id)) is not None:
        event.set()

    # Also flip a run left 'running' by a server restart, so the workspace is
    # not blocked forever by a job no thread is working on.
    meta_file = workspace.evals_dir / f"{safe_id}.json"
    if meta_file.is_file() and event is None:
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("status") == "running":
                meta["status"] = "cancelled"
                meta["completed_at"] = datetime.now(timezone.utc).isoformat()
                meta["error"] = "Cancelled by user"
                _write_run_json(meta_file, meta)
        except Exception:
            pass

    return {"status": "cancelled"}


@app.get("/api/eval/runs/{run_id}/stream")
async def stream_eval_logs(run_id: str, user: CurrentUser = Depends(get_current_user)):
    workspace = current_workspace(user)
    safe_id = os.path.basename(run_id)
    return _stream_log(
        workspace.evals_dir / f"{safe_id}.log",
        workspace.evals_dir / f"{safe_id}.json",
    )


# Mounted last: a mount at "/" matches every path, so it must be registered after
# every API route or it would shadow them. `html=True` resolves directory paths to
# their index.html, which is what `trailingSlash: true` in next.config.ts emits.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="dashboard")
else:
    print(
        f"[API] No frontend build at {FRONTEND_DIR} — run 'npm run build' in ./frontend.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
