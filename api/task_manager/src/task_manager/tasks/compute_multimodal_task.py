from os import unlink
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from pydantic import BaseModel

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.compute_multimodal import encode_multimodal


class ComputeMultimodalTaskInput(BaseModel):
    feature_name: str
    username: str
    project_slug: str
    paths_path: Path
    path_process: Path
    # UI label stored in the feature parameters
    model: str
    # full HF name, reloaded by the Prompts layer for prompt text
    hf_name: str
    batch_size: int = 8


# Return type must be a dict, a BaseModel would not be serialized by Celery
class ComputeMultimodalTaskResult(TypedDict):
    username: str
    project_slug: str
    embeddings_path: str
    parameters: dict[str, Any]


class ComputeMultimodalTask(AbortableAutoCallbackTask):
    name = "compute multimodal"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU


@celery_app.task(
    bind=True,
    name=ComputeMultimodalTask.name,
    queue=ComputeMultimodalTask.queue,
    base=ComputeMultimodalTask,
    pydantic=True,
)
def compute_multimodal(
    self: ComputeMultimodalTask, inputs: ComputeMultimodalTaskInput
) -> ComputeMultimodalTaskResult:
    # the API reads the progress file named after the task id
    path_progress = inputs.path_process.joinpath(self.request.id)
    paths = pd.read_pickle(inputs.paths_path)
    try:
        emb = encode_multimodal(
            paths,
            model_name=inputs.hf_name,
            path_progress=path_progress,
            batch_size=inputs.batch_size,
            is_aborted=self.is_aborted,
        )
    finally:
        path_progress.unlink(missing_ok=True)
        unlink(inputs.paths_path)
    embeddings_path = inputs.path_process.joinpath(f"multimodal_{self.request.id}.parquet")
    emb.to_parquet(embeddings_path)
    return {
        "project_slug": inputs.project_slug,
        "username": inputs.username,
        "embeddings_path": str(embeddings_path),
        "parameters": {
            "model": inputs.model,
            "hf_name": inputs.hf_name,
            "name": inputs.feature_name,
            "kind": "multimodal-embeddings",
            "username": inputs.username,
            "batch_size": inputs.batch_size,
        },
    }
