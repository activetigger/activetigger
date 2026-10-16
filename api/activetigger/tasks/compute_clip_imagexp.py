# Experimental image projects — see docs/image-projects-strategy.md
import multiprocessing
import multiprocessing.synchronize
from pathlib import Path
from typing import Optional

from pandas import DataFrame, Series
from task_manager.tasks.compute_clip_imagexp import encode_clip

from activetigger.tasks.base_task import BaseTask


class ComputeClipImagexp(BaseTask):
    """
    In-process CLIP image embeddings for composite tasks (feature
    recomputation). Standalone feature computation goes through the celery
    task; both share encode_clip.
    """

    kind = "compute_feature_image_embeddings"

    def __init__(
        self,
        paths: Series,
        path_process: Path,
        model: str,
        pretrained: str = "openai",
        batch_size: int = 16,
        path_progress: Path | None = None,
        event: Optional[multiprocessing.synchronize.Event] = None,
    ):
        super().__init__()
        self.paths = paths
        self.model = model
        self.pretrained = pretrained
        self.batch_size = int(batch_size)
        self.path_process = path_process
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
            return encode_clip(
                self.paths,
                model=self.model,
                pretrained=self.pretrained,
                path_progress=self.path_progress,
                batch_size=self.batch_size,
                is_aborted=lambda: self.event is not None and self.event.is_set(),
            )
        finally:
            if self.progress_file_temporary:
                self.path_progress.unlink(missing_ok=True)
            del self.paths
