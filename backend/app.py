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
    yield


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


class ConfigUpdateRequest(BaseModel):
    routing_method: str
    reranker_provider: str
    # Runtime provider selection. Omitted fields keep their current value.
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
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
                yield f"data: {json.dumps({'token': item, 'done': False})}\n\n"

        # ── Phase 3: Final metadata event ─────────────────────────────────
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


@app.get("/api/builds/{build_id}/stream")
async def stream_build_logs(build_id: str, user: CurrentUser = Depends(get_current_user)):
    workspace = current_workspace(user)
    log_file = workspace.builds_dir / f"{build_id}.log"
    meta_file = workspace.builds_dir / f"{build_id}.json"

    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="Build not found.")

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
