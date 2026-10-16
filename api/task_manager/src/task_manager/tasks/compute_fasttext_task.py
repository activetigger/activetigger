from os import unlink
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from pydantic import BaseModel

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.compute_fasttext import encode_fasttext


class ComputeFasttextTaskInput(BaseModel):
    feature_name: str
    username: str
    project_slug: str
    texts_path: Path
    path_process: Path
    path_models: Path
    language: str
    model: str | None = ""


# Return type must be a dict, a BaseModel would not be serialized by Celery
class ComputeFasttextTaskResult(TypedDict):
    username: str
    project_slug: str
    features_path: str
    parameters: dict[str, Any]


class ComputeFasttextTask(AbortableAutoCallbackTask):
    name = "compute fasttext"
    # fasttext runs on CPU only
    queue = QueueName.CPU


@celery_app.task(
    bind=True,
    name=ComputeFasttextTask.name,
    queue=ComputeFasttextTask.queue,
    base=ComputeFasttextTask,
    pydantic=True,
)
def compute_fasttext(
    self: ComputeFasttextTask, inputs: ComputeFasttextTaskInput
) -> ComputeFasttextTaskResult:
    # the API reads the progress file named after the task id
    path_progress = inputs.path_process.joinpath(self.request.id)
    texts = pd.read_pickle(inputs.texts_path)
    try:
        emb = encode_fasttext(
            texts,
            language=inputs.language,
            path_models=inputs.path_models,
            path_progress=path_progress,
            model=inputs.model,
            is_aborted=self.is_aborted,
        )
    finally:
        path_progress.unlink(missing_ok=True)
        unlink(inputs.texts_path)
    features_path = inputs.path_process.joinpath(f"fasttext_{self.request.id}.parquet")
    emb.to_parquet(features_path)
    return {
        "project_slug": inputs.project_slug,
        "username": inputs.username,
        "features_path": str(features_path),
        "parameters": {
            "model": inputs.model,
            "name": inputs.feature_name,
            "kind": "fasttext",
            "username": inputs.username,
        },
    }
