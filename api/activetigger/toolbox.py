import io
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path

import docx
import pandas as pd
from celery.result import AsyncResult
from celery.states import FAILURE, REVOKED
from fastapi import HTTPException
from fastapi.responses import FileResponse
from task_manager.auto_callback_task import QueueName
from task_manager.celery import celery_app
from task_manager.tasks.prepare_dataset_task import PrepareDatasetTaskInput, prepare_dataset_task

from activetigger.config import config
from activetigger.datamodels import (
    PrepareSessionModel,
    PrepareSplitModel,
    PrepareStatusModel,
    PrepareTaskModel,
)
from activetigger.functions import safe_upload_path, slugify
from activetigger.tasks.task_manager_client import enqueue_celery_task, stop_celery_task


class Toolbox:
    """
    Standalone dataset utilities available outside any project
    """

    ALLOWED_EXTENSIONS = (".csv", ".parquet", ".xlsx", ".zip")
    ZIP_DOCUMENT_EXTENSIONS = (".txt", ".docx")
    MAX_ZIP_BYTES = 200 * 1024 * 1024
    MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
    SESSION_MAX_AGE_SECONDS = 24 * 3600
    PREVIEW_ROWS = 5
    # file in the session directory holding the id of its split task
    TASK_FILE = "task"

    def _user_dir(self, username: str) -> Path:
        return Path(config.data_path).joinpath("prepare", slugify(username))

    def _session_dir(self, username: str, session_id: str) -> Path:
        """
        Resolve an existing session directory, rejecting malformed ids
        """
        if not re.fullmatch(r"[a-f0-9-]{36}", session_id):
            raise HTTPException(status_code=400, detail="Invalid session id")
        path = self._user_dir(username).joinpath(session_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Session not found, please upload a file")
        return path

    def _session_task_id(self, session_dir: Path) -> str | None:
        try:
            return session_dir.joinpath(self.TASK_FILE).read_text().strip() or None
        except OSError:
            return None

    def _clean_old_sessions(self, user_dir: Path) -> None:
        if not user_dir.exists():
            return
        for session in user_dir.iterdir():
            try:
                if (
                    session.is_dir()
                    and time.time() - session.stat().st_mtime > self.SESSION_MAX_AGE_SECONDS
                ):
                    shutil.rmtree(session)
            except OSError:
                continue

    def _read_tabular(self, path: Path) -> pd.DataFrame:
        if str(path).endswith(".csv"):
            try:
                return pd.read_csv(
                    path, sep=None, low_memory=False, on_bad_lines="skip", engine="python"
                )
            except Exception:
                return pd.read_csv(path, sep=None, on_bad_lines="skip", engine="python")
        if str(path).endswith(".parquet"):
            return pd.read_parquet(path)
        if str(path).endswith(".xlsx"):
            return pd.read_excel(path)
        raise HTTPException(status_code=400, detail="File format not supported")

    def _read_zip_documents(self, zip_path: Path) -> pd.DataFrame:
        """
        Build a dataframe (id = file name, text = content) from a zip of
        .txt / .docx documents
        """
        if zip_path.stat().st_size > self.MAX_ZIP_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Zip file too large (max {self.MAX_ZIP_BYTES // (1024 * 1024)} MB)",
            )
        rows: list[dict] = []
        seen_ids: set[str] = set()
        with zipfile.ZipFile(zip_path, "r") as zf:
            infos = [
                i
                for i in zf.infolist()
                if not i.is_dir()
                and Path(i.filename).suffix.lower() in self.ZIP_DOCUMENT_EXTENSIONS
                and not Path(i.filename).name.startswith("._")
                and "__MACOSX" not in Path(i.filename).parts
            ]
            if len(infos) == 0:
                raise HTTPException(
                    status_code=400, detail="Zip does not contain any .txt or .docx file"
                )
            for info in infos:
                if info.file_size > self.MAX_DOCUMENT_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Document '{info.filename}' exceeds per-file cap of "
                        f"{self.MAX_DOCUMENT_BYTES // (1024 * 1024)} MB",
                    )
                name = Path(info.filename)
                if name.suffix.lower() == ".txt":
                    text = zf.read(info).decode("utf-8", errors="replace")
                else:
                    try:
                        document = docx.Document(io.BytesIO(zf.read(info)))
                    except Exception:
                        raise HTTPException(
                            status_code=400, detail=f"Could not read docx file '{info.filename}'"
                        )
                    text = "\n\n".join(p.text for p in document.paragraphs if p.text.strip())
                # disambiguate duplicated file names across zip folders
                element_id = name.stem
                suffix = 2
                while element_id in seen_ids:
                    element_id = f"{name.stem}-{suffix}"
                    suffix += 1
                seen_ids.add(element_id)
                rows.append({"id": element_id, "text": text})
        return pd.DataFrame(rows)

    def _preview(self, df: pd.DataFrame) -> list[dict]:
        return df.head(self.PREVIEW_ROWS).fillna("").astype(str).to_dict("records")

    def upload(self, username: str, staged_path: Path, filename: str) -> PrepareSessionModel:
        """
        Ingest a staged upload (chunked-upload protocol, see
        activetigger.uploads): move the file (same filesystem) into a new
        session directory, normalize it as raw.parquet and return its
        columns and a preview. Accepts csv/xlsx/parquet or zip of txt/docx.
        """
        user_dir = self._user_dir(username)
        self._clean_old_sessions(user_dir)

        session_id = str(uuid.uuid4())
        session_dir = user_dir.joinpath(session_id)
        session_dir.mkdir(parents=True)
        try:
            target = safe_upload_path(session_dir, filename, self.ALLOWED_EXTENSIONS)
            shutil.move(str(staged_path), target)

            if str(target).lower().endswith(".zip"):
                df = self._read_zip_documents(target)
            else:
                df = self._read_tabular(target)
            if len(df) == 0:
                raise HTTPException(status_code=400, detail="The file contains no data")
            df.columns = [str(c) for c in df.columns]
            df.to_parquet(session_dir.joinpath("raw.parquet"), index=False)
            target.unlink(missing_ok=True)

            return PrepareSessionModel(
                session_id=session_id,
                filename=filename,
                columns=list(df.columns),
                n_rows=len(df),
                preview=self._preview(df),
            )
        except HTTPException:
            shutil.rmtree(session_dir, ignore_errors=True)
            raise
        except Exception as e:
            shutil.rmtree(session_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=str(e))

    def split(self, username: str, params: PrepareSplitModel) -> PrepareTaskModel:
        """
        Launch the split task on a celery worker (GPU queue for wtpsplit)
        """
        session_dir = self._session_dir(username, params.session_id)
        if not session_dir.joinpath("raw.parquet").exists():
            raise HTTPException(status_code=404, detail="Session data not found")

        # validate the parameters
        columns = list(pd.read_parquet(session_dir.joinpath("raw.parquet")).columns)
        if len(params.cols_text) == 0:
            raise HTTPException(status_code=400, detail="Select at least one text column")
        missing = [c for c in params.cols_text if c not in columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Unknown text columns: {missing}")
        if params.method == "chunk" and (not params.chunk_size or params.chunk_size <= 0):
            raise HTTPException(status_code=400, detail="A positive chunk size is required")
        if params.method == "regex":
            if not params.regex_pattern:
                raise HTTPException(status_code=400, detail="A regex pattern is required")
            try:
                re.compile(params.regex_pattern)
            except re.error as e:
                raise HTTPException(status_code=400, detail=f"Invalid regex pattern: {e}")
        if params.method == "wtpsplit" and params.granularity is None:
            raise HTTPException(status_code=400, detail="Select sentence or paragraph granularity")
        if params.min_chars < 0:
            raise HTTPException(status_code=400, detail="Minimum length must be positive")

        # remove results of a previous run
        session_dir.joinpath("result.parquet").unlink(missing_ok=True)
        session_dir.joinpath("progress").unlink(missing_ok=True)

        try:
            (task_id, _) = enqueue_celery_task(
                prepare_dataset_task,
                PrepareDatasetTaskInput(
                    username=username,
                    session_id=params.session_id,
                    path_session=session_dir,
                    params=params,
                ),
                queue=QueueName.GPU if params.method == "wtpsplit" else None,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        # remember the task so status/stop can check it belongs to this session
        session_dir.joinpath(self.TASK_FILE).write_text(task_id)
        return PrepareTaskModel(task_id=task_id)

    def stop(self, username: str, task_id: str) -> None:
        """
        Stop a running split task of the user
        """
        user_dir = self._user_dir(username)
        owned = user_dir.exists() and any(
            session.is_dir() and self._session_task_id(session) == task_id
            for session in user_dir.iterdir()
        )
        if not owned:
            raise HTTPException(status_code=404, detail="Task not found")
        stop_celery_task(task_id, abortable=True)

    def status(self, username: str, session_id: str, task_id: str) -> PrepareStatusModel:
        """
        Status of a split task; when done, add the number of rows and a preview
        """
        session_dir = self._session_dir(username, session_id)
        path_result = session_dir.joinpath("result.parquet")

        # the result is only written at the very end and removed when a new split starts
        if path_result.exists():
            df = pd.read_parquet(path_result)
            return PrepareStatusModel(status="done", n_rows=len(df), preview=self._preview(df))
        if self._session_task_id(session_dir) != task_id:
            return PrepareStatusModel(status="not found")

        result = AsyncResult(task_id, app=celery_app)
        state = result.state
        if state in (REVOKED, "ABORTED"):
            return PrepareStatusModel(status="failed", error="Task cancelled")
        if state == FAILURE:
            error = str(result.result) if result.result is not None else "Task failed"
            if "interrupted by user" in error:
                error = "Task cancelled"
            return PrepareStatusModel(status="failed", error=error)

        # queued or running: the worker writes the progress file once it starts
        progress = None
        try:
            progress = float(session_dir.joinpath("progress").read_text())
        except (OSError, ValueError):
            pass
        status = "running" if progress is not None else "pending"
        return PrepareStatusModel(status=status, progress=progress)

    def export(self, username: str, session_id: str, format: str) -> FileResponse:
        """
        Return the prepared dataset as a file in the requested format
        """
        if format not in ["csv", "xlsx", "parquet"]:
            raise HTTPException(status_code=400, detail="Format not supported")
        session_dir = self._session_dir(username, session_id)
        path_result = session_dir.joinpath("result.parquet")
        if not path_result.exists():
            raise HTTPException(status_code=404, detail="No prepared dataset available")

        file_name = f"prepared_dataset.{format}"
        path_export = session_dir.joinpath(file_name)
        if format == "parquet":
            path_export = path_result
        else:
            df = pd.read_parquet(path_result)
            if format == "csv":
                df.to_csv(path_export, index=False)
            else:
                for col in df.select_dtypes(include=["datetimetz"]).columns:
                    df[col] = df[col].dt.tz_localize(None)
                df.to_excel(path_export, index=False)
        return FileResponse(path_export, filename=file_name)
