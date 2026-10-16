from pathlib import Path

from pandas import DataFrame, Series
from task_manager.tasks.compute_fasttext import encode_fasttext

from activetigger.tasks.base_task import BaseTask


class ComputeFasttext(BaseTask):
    """
    In-process fasttext embeddings for composite tasks (feature recomputation).
    Standalone feature computation goes through the celery task; both share encode_fasttext.
    """

    kind = "compute_feature_fastext"

    def __init__(
        self,
        texts: Series,
        language: str,
        path_process: Path,
        path_models: Path,
        model: str = "",
    ):
        super().__init__()
        self.texts = texts
        self.path_process = path_process
        self.path_models = path_models
        self.language = language
        self.model = model
        self.event = None

    def __call__(self) -> DataFrame:
        # callers watch path_process/unique_id for progress
        progress_path = self.path_process.joinpath(self.unique_id)
        try:
            return encode_fasttext(
                self.texts,
                language=self.language,
                path_models=self.path_models,
                path_progress=progress_path,
                model=self.model,
                is_aborted=lambda: self.event is not None and self.event.is_set(),
            )
        finally:
            progress_path.unlink(missing_ok=True)
