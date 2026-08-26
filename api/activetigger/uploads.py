"""
Staged (chunked) uploads.

Large files are sent by the frontend as a sequence of small requests so that
no single HTTP request outlives reverse-proxy timeouts

Sessions live under {data_path}/uploads/{user}/{upload_id}/ and are removed
when a consumer claims the file, when the client cancels, or by the
orchestrator's periodic cleanup once they are older than SESSION_TTL.
"""

import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException

from activetigger.config import config
from activetigger.functions import safe_upload_path, sanitize_uploaded_filename

# union of the extensions accepted by the consumer endpoints; each consumer
# re-checks against its own, stricter list when claiming the file
ALLOWED_STAGED_EXTENSIONS = (".csv", ".parquet", ".xlsx", ".zip")

MAX_CHUNK_BYTES = 64 * 1024 * 1024
MAX_TOTAL_CHUNKS = 20_000
SESSION_TTL_SECONDS = 6 * 3600

META_FILE = "meta.json"
PARTS_DIR = "parts"


class UploadStaging:
    """
    Filesystem-backed staging area for chunked uploads.

    The staging root lives under data_path so that claiming a finished file
    into a project directory is a same-filesystem rename, not a copy.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _user_dir(self, username: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", username) or "user"
        return self.root / safe

    def _session_dir(self, username: str, upload_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
            raise HTTPException(status_code=400, detail="Invalid upload id")
        return self._user_dir(username) / upload_id

    def _read_meta(self, username: str, upload_id: str) -> tuple[Path, dict]:
        session_dir = self._session_dir(username, upload_id)
        meta_path = session_dir / META_FILE
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="Unknown upload session")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        # sanitized user dirs can collide; the metadata holds the exact owner
        if meta.get("username") != username:
            raise HTTPException(status_code=403, detail="Not your upload session")
        return session_dir, meta

    def _write_meta(self, session_dir: Path, meta: dict) -> None:
        with open(session_dir / META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def start(
        self, username: str, filename: str, total_size: int, total_chunks: int
    ) -> tuple[str, str]:
        """
        Open a session. Returns (upload_id, sanitized filename).
        """
        safe_name = sanitize_uploaded_filename(filename)
        if not safe_name.lower().endswith(ALLOWED_STAGED_EXTENSIONS):
            allowed = ", ".join(ext.lstrip(".") for ext in ALLOWED_STAGED_EXTENSIONS)
            raise HTTPException(status_code=400, detail=f"Only {allowed} files are allowed")
        if total_size <= 0:
            raise HTTPException(status_code=400, detail="Invalid total size")
        if not 0 < total_chunks <= MAX_TOTAL_CHUNKS:
            raise HTTPException(status_code=400, detail="Invalid number of chunks")

        upload_id = uuid.uuid4().hex
        session_dir = self._session_dir(username, upload_id)
        (session_dir / PARTS_DIR).mkdir(parents=True)
        self._write_meta(
            session_dir,
            {
                "username": username,
                "filename": safe_name,
                "total_size": total_size,
                "total_chunks": total_chunks,
                "created_at": time.time(),
                "finished": False,
            },
        )
        return upload_id, safe_name

    def write_chunk(self, username: str, upload_id: str, index: int, stream: BinaryIO) -> None:
        session_dir, meta = self._read_meta(username, upload_id)
        if meta["finished"]:
            raise HTTPException(status_code=400, detail="Upload already finished")
        if not 0 <= index < meta["total_chunks"]:
            raise HTTPException(status_code=400, detail="Chunk index out of range")

        # write to a temp name first so a retried chunk never reads half-written
        part_path = session_dir / PARTS_DIR / f"{index:06d}.part"
        tmp_path = part_path.with_suffix(".tmp")
        written = 0
        with open(tmp_path, "wb") as out_file:
            while chunk := stream.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_CHUNK_BYTES:
                    out_file.close()
                    tmp_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="Chunk too large")
                out_file.write(chunk)
        if written == 0:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Empty chunk")
        tmp_path.replace(part_path)

    def finish(self, username: str, upload_id: str) -> tuple[str, int]:
        """
        Assemble the parts into the final file. Returns (filename, size).
        """
        session_dir, meta = self._read_meta(username, upload_id)
        if meta["finished"]:
            return meta["filename"], meta["total_size"]

        parts_dir = session_dir / PARTS_DIR
        expected = [parts_dir / f"{i:06d}.part" for i in range(meta["total_chunks"])]
        missing = [p.name for p in expected if not p.exists()]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Upload incomplete: {len(missing)} chunk(s) missing",
            )

        target = session_dir / meta["filename"]
        size = 0
        with open(target, "wb") as out_file:
            for part in expected:
                with open(part, "rb") as in_file:
                    while chunk := in_file.read(1024 * 1024):
                        size += len(chunk)
                        out_file.write(chunk)
        if size != meta["total_size"]:
            target.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"Size mismatch: expected {meta['total_size']} bytes, got {size}",
            )

        shutil.rmtree(parts_dir, ignore_errors=True)
        meta["finished"] = True
        self._write_meta(session_dir, meta)
        return meta["filename"], size

    def claim(
        self, username: str, upload_id: str, allowed_extensions: tuple[str, ...]
    ) -> tuple[Path, str]:
        """
        Return (path, filename) of a finished staged file, without consuming
        the session. The caller must call discard() once done with the file.
        """
        session_dir, meta = self._read_meta(username, upload_id)
        if not meta["finished"]:
            raise HTTPException(status_code=400, detail="Upload not finished")
        if not meta["filename"].lower().endswith(allowed_extensions):
            allowed = ", ".join(ext.lstrip(".") for ext in allowed_extensions)
            raise HTTPException(status_code=400, detail=f"Only {allowed} files are allowed")
        path = session_dir / meta["filename"]
        if not path.exists():
            raise HTTPException(status_code=410, detail="Staged file no longer available")
        return path, meta["filename"]

    def move_to(
        self,
        username: str,
        upload_id: str,
        directory: Path,
        allowed_extensions: tuple[str, ...],
    ) -> Path:
        """
        Move a finished staged file into `directory` and drop the session.
        Returns the final path.
        """
        source, filename = self.claim(username, upload_id, allowed_extensions)
        target = safe_upload_path(directory, filename, allowed_extensions)
        shutil.move(str(source), target)
        self.discard(username, upload_id)
        return target

    def discard(self, username: str, upload_id: str) -> None:
        session_dir, _ = self._read_meta(username, upload_id)
        shutil.rmtree(session_dir, ignore_errors=True)

    def clean_old(self, ttl_seconds: int = SESSION_TTL_SECONDS) -> None:
        """
        Remove staging sessions older than ttl_seconds (abandoned uploads).
        """
        now = time.time()
        for meta_path in self.root.glob(f"*/*/{META_FILE}"):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    created_at = json.load(f).get("created_at", 0)
                if now - created_at > ttl_seconds:
                    shutil.rmtree(meta_path.parent, ignore_errors=True)
            except Exception:
                continue


_staging: UploadStaging | None = None


def get_upload_staging() -> UploadStaging:
    """
    Create a handler for the upload component
    """
    global _staging
    if _staging is None:
        _staging = UploadStaging(Path(config.data_path) / "uploads")
    return _staging
