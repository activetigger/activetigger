from os import unlink
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from pydantic import BaseModel

from task_manager.active_tigger_task import AbortableActiveTiggerTask, QueueName
from task_manager.celery import celery_app
from task_manager.task_input import TaskInputs
from task_manager.task_names import TaskNames
from task_manager.tasks.compute_clip_imagexp import encode_clip


class ComputeClipImagexpTaskInputs(TaskInputs):
    feature_name: str
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


class ComputeClipImagexpTask(AbortableActiveTiggerTask):
    name: TaskNames = "compute_clip_imagexp"
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
    self: ComputeClipImagexpTask, inputs: ComputeClipImagexpTaskInputs
) -> ComputeClipImagexpTaskResult:
    paths = pd.read_pickle(inputs.paths_path)
    try:
        emb = encode_clip(
            paths,
            model=inputs.clip_model,
            pretrained=inputs.pretrained,
            update_progress=self.update_progress,
            batch_size=inputs.batch_size,
            is_aborted=self.is_aborted,
        )
    finally:
        unlink(inputs.paths_path)
    embeddings_path = inputs.path_process.joinpath(f"clip_{self.request.id}.parquet")
    emb.to_parquet(embeddings_path)
    return {
        "project_slug": inputs.project_slug,
        "username": inputs.user,
        "embeddings_path": str(embeddings_path),
        "parameters": {
            "model": inputs.model,
            "name": inputs.feature_name,
            "kind": "image-embeddings",
            "username": inputs.user,
            "batch_size": inputs.batch_size,
        },
    }
