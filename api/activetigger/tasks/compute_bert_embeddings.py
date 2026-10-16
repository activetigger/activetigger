import multiprocessing
import multiprocessing.synchronize
from pathlib import Path
from typing import Optional

from pandas import DataFrame, Series
from task_manager.tasks.compute_bert_embeddings import encode_bert_embeddings

from activetigger.tasks.base_task import BaseTask


# TODO: deprecated in-process wrapper, remove with the other BaseTask wrappers
class ComputeBertEmbeddings(BaseTask):
    """
    In-process BERT embeddings; the celery task and this wrapper share encode_bert_embeddings.
    """

    kind = "compute_feature_bert_embeddings"

    def __init__(
        self,
        texts: Series,
        path_process: Path,
        model_path: Path,
        pooling: str = "mean",
        batch_size: int = 32,
        max_tokens: int = 512,
        min_gpu: int = 6,
        path_progress: Path | None = None,
        event: Optional[multiprocessing.synchronize.Event] = None,
    ):
        super().__init__()
        self.texts = texts
        self.model_path = Path(model_path)
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        self.pooling = pooling
        self.batch_size = int(batch_size)
        self.max_tokens = int(max_tokens)
        self.min_gpu = min_gpu
        self.path_process = path_process
        self.event = event
        if path_progress:
            self.progress_file_temporary = False
            self.path_progress = path_progress
        else:
            self.path_progress = self.path_process.joinpath(self.unique_id)
            self.progress_file_temporary = True

    def __call__(self) -> DataFrame:
        if self.progress_file_temporary:
            self.path_progress = self.path_process.joinpath(self.unique_id)
        try:
            return encode_bert_embeddings(
                self.texts,
                model_path=self.model_path,
                path_progress=self.path_progress,
                pooling=self.pooling,
                batch_size=self.batch_size,
                max_tokens=self.max_tokens,
                min_gpu=self.min_gpu,
                is_aborted=lambda: self.event is not None and self.event.is_set(),
            )
        finally:
            if self.progress_file_temporary:
                self.path_progress.unlink(missing_ok=True)
            del self.texts
