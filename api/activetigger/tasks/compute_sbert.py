import multiprocessing
import multiprocessing.synchronize
from pathlib import Path
from typing import Optional

from pandas import DataFrame, Series
from task_manager.tasks.compute_sbert import encode_sbert

from activetigger.tasks.base_task import BaseTask


class ComputeSbert(BaseTask):
    """
    In-process sentence embeddings for composite tasks (bertopic, feature
    recomputation). Standalone feature computation goes through the celery
    task; both share encode_sbert.
    """

    kind = "compute_feature_embeddings"

    def __init__(
        self,
        texts: Series,
        path_process: Path,
        model: str,
        batch_size: int = 32,
        min_gpu: int = 6,
        max_tokens: int = 1024,
        path_progress: Path | None = None,
        event: Optional[multiprocessing.synchronize.Event] = None,
    ):
        super().__init__()
        self.texts = texts
        self.model = model
        self.batch_size = batch_size
        self.min_gpu = min_gpu
        self.path_process = path_process
        self.max_tokens = int(max_tokens)
        self.event = event
        if path_progress:
            self.progress_file_temporary = False
            self.path_progress = path_progress
        else:
            self.path_progress = self.path_process.joinpath(self.unique_id)
            self.progress_file_temporary = True

    def __call__(self) -> DataFrame:
        # update temporary file with the current ID
        if self.progress_file_temporary:
            self.path_progress = self.path_process.joinpath(self.unique_id)
        try:
            return encode_sbert(
                self.texts,
                model=self.model,
                path_progress=self.path_progress,
                batch_size=self.batch_size,
                min_gpu=self.min_gpu,
                max_tokens=self.max_tokens,
                is_aborted=lambda: self.event is not None and self.event.is_set(),
            )
        finally:
            if self.progress_file_temporary:
                self.path_progress.unlink(missing_ok=True)
            del self.texts
