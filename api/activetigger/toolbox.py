import io
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path

import docx
import pandas as pd
from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse

from activetigger.config import config
from activetigger.datamodels import (
    PrepareSessionModel,
    PrepareSplitModel,
    PrepareStatusModel,
    PrepareTaskModel,
)
from activetigger.functions import safe_upload_path, slugify
from activetigger.queue_manager import Queue
from activetigger.tasks.prepare_dataset import PrepareDataset


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

    queue: Queue

    def __init__(self, queue: Queue) -> None:
        self.queue = queue

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

    def _task_slug(self, username: str) -> str:
        """
        Pseudo project slug identifying the user's toolbox tasks in the queue
        """
        return f"prepare-{username}"

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

    def upload(self, username: str, file: UploadFile) -> PrepareSessionModel:
        """
        Save an uploaded file (csv/xlsx/parquet or zip of txt/docx),
        normalize it as raw.parquet in a new session directory and return
        its columns and a preview.
        """
        user_dir = self._user_dir(username)
        self._clean_old_sessions(user_dir)

        session_id = str(uuid.uuid4())
        session_dir = user_dir.joinpath(session_id)
        session_dir.mkdir(parents=True)
        try:
            target = safe_upload_path(session_dir, file.filename, self.ALLOWED_EXTENSIONS)
            with open(target, "wb") as out_file:
                while chunk := file.file.read(1024 * 1024):
                    out_file.write(chunk)

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
                filename=file.filename or "file",
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
        Launch the split task on the queue (gpu queue for wtpsplit)
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

        task = PrepareDataset(session_dir, params)
        try:
            task_id = self.queue.add_task(
                "prepare_dataset",
                self._task_slug(username),
                task,
                queue="gpu" if params.method == "wtpsplit" else "cpu",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return PrepareTaskModel(task_id=task_id)

    def stop(self, username: str, task_id: str) -> None:
        """
        Stop a running split task of the user
        """
        element = self.queue.get(task_id)
        if element is None or element.project_slug != self._task_slug(username):
            raise HTTPException(status_code=404, detail="Task not found")
        self.queue.kill(task_id)

    def status(self, username: str, session_id: str, task_id: str) -> PrepareStatusModel:
        """
        Status of a split task; when done, add the number of rows and a preview
        """
        session_dir = self._session_dir(username, session_id)
        path_result = session_dir.joinpath("result.parquet")

        element = self.queue.get(task_id)
        if element is None or element.project_slug != self._task_slug(username):
            # the queue may already have been cleaned
            if path_result.exists():
                df = pd.read_parquet(path_result)
                return PrepareStatusModel(status="done", n_rows=len(df), preview=self._preview(df))
            return PrepareStatusModel(status="not found")

        if element.state in ["pending", "running"]:
            progress = None
            try:
                progress = float(session_dir.joinpath("progress").read_text())
            except (OSError, ValueError):
                pass
            return PrepareStatusModel(status=element.state, progress=progress)

        if element.state == "done" and path_result.exists():
            df = pd.read_parquet(path_result)
            return PrepareStatusModel(status="done", n_rows=len(df), preview=self._preview(df))

        error = "Task failed"
        if element.state == "cancelled":
            error = "Task cancelled"
        elif element.future is not None and element.future.done() and element.future.exception():
            error = str(element.future.exception())
        return PrepareStatusModel(status="failed", error=error)

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
