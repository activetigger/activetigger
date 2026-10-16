from os import unlink
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from pydantic import BaseModel

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.compute_dfm import compute_dfm


class ComputeDfmTaskInput(BaseModel):
    feature_name: str
    username: str
    project_slug: str
    texts_path: Path
    path_process: Path
    # stored as the feature parameters once registered
    parameters: dict[str, Any]
    tfidf: bool = False
    ngrams: int = 1
    min_term_freq: int = 5
    max_term_freq: int | float = 1.0
    max_features: int | None = None
    log: bool = False
    language: str = "en"
    norm: str | None = None


# Return type must be a dict, a BaseModel would not be serialized by Celery
class ComputeDfmTaskResult(TypedDict):
    username: str
    project_slug: str
    features_path: str
    parameters: dict[str, Any]


class ComputeDfmTask(AbortableAutoCallbackTask):
    name = "compute dfm"
    # sklearn vectorizer, never uses the GPU
    queue = QueueName.CPU


@celery_app.task(
    bind=True,
    name=ComputeDfmTask.name,
    queue=ComputeDfmTask.queue,
    base=ComputeDfmTask,
    pydantic=True,
)
def compute_dfm_task(self: ComputeDfmTask, inputs: ComputeDfmTaskInput) -> ComputeDfmTaskResult:
    texts = pd.read_pickle(inputs.texts_path)
    try:
        # one-shot vectoriser: no progress, abort only before starting
        if self.is_aborted():
            raise Exception("Process interrupted by user")
        dtm = compute_dfm(
            texts,
            tfidf=inputs.tfidf,
            ngrams=inputs.ngrams,
            min_term_freq=inputs.min_term_freq,
            max_term_freq=inputs.max_term_freq,
            max_features=inputs.max_features,
            log=inputs.log,
            language=inputs.language,
            norm=inputs.norm,
        )
    finally:
        unlink(inputs.texts_path)
    features_path = inputs.path_process.joinpath(f"dfm_{self.request.id}.parquet")
    dtm.to_parquet(features_path)
    return {
        "project_slug": inputs.project_slug,
        "username": inputs.username,
        "features_path": str(features_path),
        "parameters": inputs.parameters,
    }
