from os import unlink
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

from task_manager.active_tigger_task import AbortableActiveTiggerTask, QueueName
from task_manager.celery import celery_app
from task_manager.task_input import TaskInputs
from task_manager.task_names import TaskNames
from task_manager.tasks.compute_multimodal import encode_multimodal


class ComputeMultimodalTaskInputs(TaskInputs):
    feature_name: str
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


class ComputeMultimodalTask(AbortableActiveTiggerTask):
    name: TaskNames = "compute_multimodal"
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
    self: ComputeMultimodalTask, inputs: ComputeMultimodalTaskInputs
) -> ComputeMultimodalTaskResult:
    paths = pd.read_pickle(inputs.paths_path)
    try:
        emb = encode_multimodal(
            paths,
            model_name=inputs.hf_name,
            update_progress=self.update_progress,
            batch_size=inputs.batch_size,
            is_aborted=self.is_aborted,
        )
    finally:
        unlink(inputs.paths_path)
    embeddings_path = inputs.path_process.joinpath(f"multimodal_{self.request.id}.parquet")
    emb.to_parquet(embeddings_path)
    return {
        "project_slug": inputs.project_slug,
        "username": inputs.user,
        "embeddings_path": str(embeddings_path),
        "parameters": {
            "model": inputs.model,
            "hf_name": inputs.hf_name,
            "name": inputs.feature_name,
            "kind": "multimodal-embeddings",
            "username": inputs.user,
            "batch_size": inputs.batch_size,
        },
    }
