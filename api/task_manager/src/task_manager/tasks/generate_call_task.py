from pathlib import Path
from typing import TypedDict

import pandas as pd
from activetigger.datamodels import GenerationModel
from pydantic import BaseModel

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.generate_call import generate_call


class GenerateCallTaskInput(BaseModel):
    project_slug: str
    username: str
    dataset: str
    prompt_name: str
    prompt: str
    cols_context: list[str]
    n_workers: int = 1
    # pickled {"data": DataFrame, "model": GenerationModel dump}; the model
    # carries decrypted credentials, so it never goes through the broker
    data_path: Path
    # progress file <task_id> and results gen_<task_id>.jsonl are written here
    path_process: Path


# Return type must be a dict, a BaseModel would not be serialized by Celery
class GenerateCallTaskResult(TypedDict):
    project_slug: str
    username: str
    batch: str
    jsonl_path: str
    n_results: int
    n_errors: int


class GenerateCallTask(AbortableAutoCallbackTask):
    name = "generation"
    queue = QueueName.CPU


@celery_app.task(
    bind=True,
    name=GenerateCallTask.name,
    queue=GenerateCallTask.queue,
    base=GenerateCallTask,
    pydantic=True,
)
def generate_call_task(
    self: GenerateCallTask, inputs: GenerateCallTaskInput
) -> GenerateCallTaskResult:
    task_id = self.request.id
    batch = f"{inputs.dataset}_{inputs.prompt_name}_{task_id}"
    path_jsonl = inputs.path_process.joinpath(f"gen_{task_id}.jsonl")
    try:
        payload = pd.read_pickle(inputs.data_path)
        n_results, n_errors = generate_call(
            payload["data"],
            GenerationModel(**payload["model"]),
            inputs.prompt,
            inputs.cols_context,
            inputs.username,
            inputs.project_slug,
            batch,
            path_progress=inputs.path_process.joinpath(task_id),
            path_jsonl=path_jsonl,
            n_workers=inputs.n_workers,
            is_aborted=self.is_aborted,
        )
    finally:
        inputs.data_path.unlink(missing_ok=True)
    return {
        "project_slug": inputs.project_slug,
        "username": inputs.username,
        "batch": batch,
        "jsonl_path": str(path_jsonl),
        "n_results": n_results,
        "n_errors": n_errors,
    }
