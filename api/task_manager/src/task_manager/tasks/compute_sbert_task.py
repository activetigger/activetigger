from os import unlink
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from pydantic import BaseModel

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.compute_sbert import encode_sbert


class ComputeSbertTaskInput(BaseModel):
    feature_name: str
    username: str
    project_slug: str
    texts_path: Path
    path_process: Path
    model: str
    batch_size: int = 32
    max_tokens: int = 1024
    min_gpu: int = 6


# Return type must be a dict, a BaseModel would not be serialized by Celery
class ComputeSbertTaskResult(TypedDict):
    username: str
    project_slug: str
    embeddings_path: str
    parameters: dict[str, Any]


class ComputeSbertTask(AbortableAutoCallbackTask):
    name = "compute sbert"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU


@celery_app.task(
    bind=True,
    name=ComputeSbertTask.name,
    queue=ComputeSbertTask.queue,
    base=ComputeSbertTask,
    pydantic=True,
)
def compute_sbert(self: ComputeSbertTask, inputs: ComputeSbertTaskInput) -> ComputeSbertTaskResult:
    # the API reads the progress file named after the task id
    path_progress = inputs.path_process.joinpath(self.request.id)
    texts = pd.read_pickle(inputs.texts_path)
    try:
        emb = encode_sbert(
            texts,
            model=inputs.model,
            path_progress=path_progress,
            batch_size=inputs.batch_size,
            min_gpu=inputs.min_gpu,
            max_tokens=inputs.max_tokens,
            is_aborted=self.is_aborted,
        )
    finally:
        path_progress.unlink(missing_ok=True)
        unlink(inputs.texts_path)
    embeddings_path = inputs.path_process.joinpath(f"sbert_{self.request.id}.parquet")
    emb.to_parquet(embeddings_path)
    return {
        "project_slug": inputs.project_slug,
        "username": inputs.username,
        "embeddings_path": str(embeddings_path),
        "parameters": {
            "model": inputs.model,
            "name": inputs.feature_name,
            "kind": "sentence-embeddings",
            "username": inputs.username,
            "max_length_tokens": inputs.max_tokens,
        },
    }
