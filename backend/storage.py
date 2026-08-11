"""
storage.py
──────────
Supabase Storage wrapper for durable, per-user file persistence.

Every call is made **as the signed-in user** (their JWT is forwarded to the
Supabase Python SDK), so row-level security — not a service-role key — is what
isolates tenants.  There is no privileged key anywhere in this app.

When Supabase is not configured (``AUTH_ENABLED is False``) every public method
is a silent no-op, so single-user installs keep working unchanged.

Path convention inside the ``workspaces`` bucket::

    {user_id}/documents/{filename}
    {user_id}/faiss_db/{filename}
    {user_id}/config.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from .auth import AUTH_ENABLED, SUPABASE_PUBLISHABLE_KEY, SUPABASE_URL

BUCKET = "workspaces"

# 50 MB — matches the bucket-level limit set in the migration.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class StorageError(Exception):
    """Non-fatal: logged but never crashes a request."""


class StorageClient:
    """Thin wrapper around ``supabase.storage.from_(BUCKET)`` scoped to one user.

    Instantiated per-request with the caller's JWT so the Supabase client
    authenticates *as them* and RLS applies.
    """

    def __init__(self, user_id: str, access_token: str):
        self.user_id = user_id
        self._token = access_token
        self._client = self._build_client()

    def _build_client(self):
        """Build a Supabase client authenticated with the user's JWT."""
        try:
            from supabase import create_client
        except ImportError:
            print(
                "[storage] supabase package not installed — cloud sync disabled. "
                "Install with: pip install supabase",
                file=sys.stderr,
            )
            return None

        client = create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)
        # Override the session so every request goes out with *this* user's JWT,
        # not the anonymous publishable key.  The postgrest and storage clients
        # both read from the shared headers dict.
        client.auth.set_session(self._token, self._token)
        # Set the auth header directly on the storage client so uploads/downloads
        # carry the user's JWT instead of the publishable key.
        client.storage._client.headers.update(
            {"Authorization": f"Bearer {self._token}"}
        )
        return client

    @property
    def _storage(self):
        if self._client is None:
            return None
        return self._client.storage.from_(BUCKET)

    # ── helpers ────────────────────────────────────────────────────────────

    def _remote_path(self, *segments: str) -> str:
        """Build ``user_id/segment/segment`` path."""
        return "/".join([self.user_id, *segments])

    # ── documents ─────────────────────────────────────────────────────────

    def upload_document(self, filename: str, data: bytes) -> bool:
        """Upload a staged document to Storage.  Returns True on success."""
        if not self._storage:
            return False
        if len(data) > MAX_UPLOAD_BYTES:
            print(
                f"[storage] Skipping '{filename}' — {len(data)} bytes exceeds "
                f"the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                file=sys.stderr,
            )
            return False
        path = self._remote_path("documents", filename)
        try:
            self._storage.upload(
                path=path,
                file=data,
                file_options={"upsert": "true"},
            )
            return True
        except Exception as exc:
            print(f"[storage] upload_document '{path}' failed: {exc}", file=sys.stderr)
            return False

    def download_document(self, filename: str) -> Optional[bytes]:
        """Download a document from Storage.  Returns bytes or None."""
        if not self._storage:
            return None
        path = self._remote_path("documents", filename)
        try:
            return self._storage.download(path)
        except Exception as exc:
            print(f"[storage] download_document '{path}' failed: {exc}", file=sys.stderr)
            return None

    def list_documents(self) -> list[dict]:
        """List documents in this user's Storage folder."""
        if not self._storage:
            return []
        prefix = self._remote_path("documents")
        try:
            return self._storage.list(prefix)
        except Exception as exc:
            print(f"[storage] list_documents failed: {exc}", file=sys.stderr)
            return []

    def delete_document(self, filename: str) -> bool:
        """Delete a document from Storage.  Returns True on success."""
        if not self._storage:
            return False
        path = self._remote_path("documents", filename)
        try:
            self._storage.remove([path])
            return True
        except Exception as exc:
            print(f"[storage] delete_document '{path}' failed: {exc}", file=sys.stderr)
            return False

    # ── FAISS index ───────────────────────────────────────────────────────

    def upload_index_file(self, filename: str, data: bytes) -> bool:
        """Upload one file of the FAISS index to Storage."""
        if not self._storage:
            return False
        if len(data) > MAX_UPLOAD_BYTES:
            print(
                f"[storage] Skipping index file '{filename}' — {len(data)} bytes "
                f"exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                file=sys.stderr,
            )
            return False
        path = self._remote_path("faiss_db", filename)
        try:
            self._storage.upload(
                path=path,
                file=data,
                file_options={"upsert": "true"},
            )
            return True
        except Exception as exc:
            print(f"[storage] upload_index_file '{path}' failed: {exc}", file=sys.stderr)
            return False

    def download_index_file(self, filename: str) -> Optional[bytes]:
        """Download one file of the FAISS index from Storage."""
        if not self._storage:
            return None
        path = self._remote_path("faiss_db", filename)
        try:
            return self._storage.download(path)
        except Exception as exc:
            print(f"[storage] download_index_file '{path}' failed: {exc}", file=sys.stderr)
            return None

    def list_index_files(self) -> list[dict]:
        """List files in this user's faiss_db folder in Storage."""
        if not self._storage:
            return []
        prefix = self._remote_path("faiss_db")
        try:
            return self._storage.list(prefix)
        except Exception as exc:
            print(f"[storage] list_index_files failed: {exc}", file=sys.stderr)
            return []

    # ── job files (builds, evals) ─────────────────────────────────────────
    #
    # Builds and evaluation runs store the same shape of thing — a status JSON
    # and a log, one pair per run — so they share one set of accessors keyed by
    # folder rather than a copy each.

    def upload_job_file(self, folder: str, filename: str, data: bytes) -> bool:
        """Upload a run's log or json to *folder* in Storage."""
        if not self._storage:
            return False
        if len(data) > MAX_UPLOAD_BYTES:
            print(
                f"[storage] Skipping {folder} file '{filename}' — {len(data)} bytes "
                f"exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                file=sys.stderr,
            )
            return False
        path = self._remote_path(folder, filename)
        try:
            self._storage.upload(
                path=path,
                file=data,
                file_options={"upsert": "true"},
            )
            return True
        except Exception as exc:
            print(f"[storage] upload_job_file '{path}' failed: {exc}", file=sys.stderr)
            return False

    def download_job_file(self, folder: str, filename: str) -> Optional[bytes]:
        """Download a run file from *folder* in Storage."""
        if not self._storage:
            return None
        path = self._remote_path(folder, filename)
        try:
            return self._storage.download(path)
        except Exception as exc:
            print(f"[storage] download_job_file '{path}' failed: {exc}", file=sys.stderr)
            return None

    def list_job_files(self, folder: str) -> list[dict]:
        """List files in this user's *folder* in Storage."""
        if not self._storage:
            return []
        prefix = self._remote_path(folder)
        try:
            return self._storage.list(prefix)
        except Exception as exc:
            print(f"[storage] list_job_files '{prefix}' failed: {exc}", file=sys.stderr)
            return []

    def upload_build_file(self, filename: str, data: bytes) -> bool:
        return self.upload_job_file("builds", filename, data)

    def download_build_file(self, filename: str) -> Optional[bytes]:
        return self.download_job_file("builds", filename)

    def list_build_files(self) -> list[dict]:
        return self.list_job_files("builds")

    # ── config ────────────────────────────────────────────────────────────

    def upload_config(self, data: bytes) -> bool:
        """Upload the user's config.json to Storage."""
        if not self._storage:
            return False
        path = self._remote_path("config.json")
        try:
            self._storage.upload(
                path=path,
                file=data,
                file_options={"content-type": "application/json", "upsert": "true"},
            )
            return True
        except Exception as exc:
            print(f"[storage] upload_config failed: {exc}", file=sys.stderr)
            return False

    def download_config(self) -> Optional[bytes]:
        """Download the user's config.json from Storage."""
        if not self._storage:
            return None
        path = self._remote_path("config.json")
        try:
            return self._storage.download(path)
        except Exception as exc:
            print(f"[storage] download_config failed: {exc}", file=sys.stderr)
            return None


def get_storage_client(user_id: str, token: Optional[str]) -> Optional[StorageClient]:
    """Build a StorageClient if Supabase is configured and a token is available.

    Returns None when running single-user or when the caller is anonymous — in
    both cases Storage calls should be skipped silently.
    """
    if not AUTH_ENABLED or not token:
        return None
    return StorageClient(user_id=user_id, access_token=token)


def sync_documents_to_storage(
    storage: Optional[StorageClient],
    documents_dir: Path,
) -> int:
    """Upload every file in *documents_dir* to Storage.  Returns upload count."""
    if storage is None or not documents_dir.is_dir():
        return 0
    count = 0
    for entry in sorted(documents_dir.iterdir()):
        if not entry.is_file():
            continue
        data = entry.read_bytes()
        if storage.upload_document(entry.name, data):
            count += 1
    return count


def sync_index_to_storage(
    storage: Optional[StorageClient],
    index_dir: Path,
) -> int:
    """Upload every file in *index_dir* to Storage.  Returns upload count."""
    if storage is None or not index_dir.is_dir():
        return 0
    count = 0
    for entry in sorted(index_dir.iterdir()):
        if not entry.is_file():
            continue
        data = entry.read_bytes()
        if storage.upload_index_file(entry.name, data):
            count += 1
    return count


def sync_index_from_storage(
    storage: Optional[StorageClient],
    index_dir: Path,
) -> bool:
    """Pull index files from Storage into *index_dir*.  Returns True if anything
    was downloaded (i.e. a cache-miss recovery succeeded)."""
    if storage is None:
        return False
    remote_files = storage.list_index_files()
    if not remote_files:
        return False

    index_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for entry in remote_files:
        name = entry.get("name", "")
        if not name or name.startswith("."):
            continue
        data = storage.download_index_file(name)
        if data:
            (index_dir / name).write_bytes(data)
            downloaded += 1

    if downloaded:
        print(f"☁️  [Storage] Pulled {downloaded} index file(s) from cloud.")
    return downloaded > 0


def sync_documents_from_storage(
    storage: Optional[StorageClient],
    documents_dir: Path,
) -> bool:
    """Pull documents from Storage into *documents_dir*."""
    if storage is None:
        return False
    remote_files = storage.list_documents()
    if not remote_files:
        return False

    documents_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for entry in remote_files:
        name = entry.get("name", "")
        if not name or name.startswith("."):
            continue
        local_path = documents_dir / name
        if local_path.exists():
            continue  # Don't re-download files that already exist locally
        data = storage.download_document(name)
        if data:
            local_path.write_bytes(data)
            downloaded += 1

    if downloaded:
        print(f"☁️  [Storage] Pulled {downloaded} document(s) from cloud.")
    return downloaded > 0


def sync_job_dir_to_storage(
    storage: Optional[StorageClient],
    local_dir: Path,
    folder: str,
) -> int:
    """Upload every file in *local_dir* to *folder* in Storage.

    Only the top level is walked: run metadata and logs are flat files, and
    nested directories (eval test sets) are synced separately.
    """
    if storage is None or not local_dir.is_dir():
        return 0
    count = 0
    for entry in sorted(local_dir.iterdir()):
        if not entry.is_file():
            continue
        if storage.upload_job_file(folder, entry.name, entry.read_bytes()):
            count += 1
    return count


def sync_job_dir_from_storage(
    storage: Optional[StorageClient],
    local_dir: Path,
    folder: str,
) -> bool:
    """Pull run files from *folder* in Storage into *local_dir*."""
    if storage is None:
        return False
    remote_files = storage.list_job_files(folder)
    if not remote_files:
        return False

    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for entry in remote_files:
        name = entry.get("name", "")
        if not name or name.startswith("."):
            continue
        local_path = local_dir / name
        if local_path.exists():
            continue
        data = storage.download_job_file(folder, name)
        if data:
            local_path.write_bytes(data)
            downloaded += 1

    if downloaded:
        print(f"☁️  [Storage] Pulled {downloaded} {folder} file(s) from cloud.")
    return downloaded > 0


def sync_builds_to_storage(storage: Optional[StorageClient], builds_dir: Path) -> int:
    return sync_job_dir_to_storage(storage, builds_dir, "builds")


def sync_builds_from_storage(storage: Optional[StorageClient], builds_dir: Path) -> bool:
    return sync_job_dir_from_storage(storage, builds_dir, "builds")


def sync_evals_to_storage(storage: Optional[StorageClient], evals_dir: Path) -> int:
    """Upload eval run metadata/logs plus the saved test sets."""
    count = sync_job_dir_to_storage(storage, evals_dir, "evals")
    count += sync_job_dir_to_storage(storage, evals_dir / "testsets", "evals/testsets")
    return count


def sync_evals_from_storage(storage: Optional[StorageClient], evals_dir: Path) -> bool:
    runs = sync_job_dir_from_storage(storage, evals_dir, "evals")
    sets = sync_job_dir_from_storage(storage, evals_dir / "testsets", "evals/testsets")
    return runs or sets


def sync_config_from_storage(
    storage: Optional[StorageClient],
    config_path: Path,
) -> bool:
    """Pull config.json from Storage if it does not exist locally."""
    if storage is None or config_path.exists():
        return False
    data = storage.download_config()
    if data:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_bytes(data)
        print("☁️  [Storage] Pulled config from cloud.")
        return True
    return False
