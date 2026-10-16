from os import unlink
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from pydantic import BaseModel

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.compute_clip_imagexp import encode_clip


class ComputeClipImagexpTaskInput(BaseModel):
    feature_name: str
    username: str
    project_slug: str
    paths_path: Path
    path_process: Path
    # UI label stored in the feature parameters
    model: str
    # open_clip model name and pretrained tag
    clip_model: str
    pretrained: str = "openai"
    batch_size: int = 16


# Return type must be a dict, a BaseModel would not be serialized by Celery
class ComputeClipImagexpTaskResult(TypedDict):
    username: str
    project_slug: str
    embeddings_path: str
    parameters: dict[str, Any]


class ComputeClipImagexpTask(AbortableAutoCallbackTask):
    name = "compute clip imagexp"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU


@celery_app.task(
    bind=True,
    name=ComputeClipImagexpTask.name,
    queue=ComputeClipImagexpTask.queue,
    base=ComputeClipImagexpTask,
    pydantic=True,
)
def compute_clip_imagexp(
    self: ComputeClipImagexpTask, inputs: ComputeClipImagexpTaskInput
) -> ComputeClipImagexpTaskResult:
    # the API reads the progress file named after the task id
    path_progress = inputs.path_process.joinpath(self.request.id)
    paths = pd.read_pickle(inputs.paths_path)
    try:
        emb = encode_clip(
            paths,
            model=inputs.clip_model,
            pretrained=inputs.pretrained,
            path_progress=path_progress,
            batch_size=inputs.batch_size,
            is_aborted=self.is_aborted,
        )
    finally:
        path_progress.unlink(missing_ok=True)
        unlink(inputs.paths_path)
    embeddings_path = inputs.path_process.joinpath(f"clip_{self.request.id}.parquet")
    emb.to_parquet(embeddings_path)
    return {
        "project_slug": inputs.project_slug,
        "username": inputs.username,
        "embeddings_path": str(embeddings_path),
        "parameters": {
            "model": inputs.model,
            "name": inputs.feature_name,
            "kind": "image-embeddings",
            "username": inputs.username,
            "batch_size": inputs.batch_size,
        },
    }
