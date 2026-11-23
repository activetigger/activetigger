from pathlib import Path
from typing import TypedDict

import pandas as pd
from activetigger.datamodels import GenerationModel, TaskComputing

from task_manager.active_tigger_task import AbortableActiveTiggerTask, QueueName
from task_manager.celery import celery_app
from task_manager.task_input import TaskInputs
from task_manager.task_names import TaskNames
from task_manager.tasks.generate_call import generate_call


class GenerateCallTaskInputs(TaskInputs[str]):
    dataset: str
    prompt_name: str
    prompt: str
    cols_context: list[str]
    n_workers: int = 1
    model_id: int
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


class GenerateCallTask(AbortableActiveTiggerTask):
    name: TaskNames = "generation"
    queue = QueueName.CPU

    @classmethod
    def clean_temp_files(cls, computing_task: TaskComputing[GenerateCallTaskInputs]):
        computing_task.inputs.data_path.unlink(missing_ok=True)


@celery_app.task(
    bind=True,
    name=GenerateCallTask.name,
    queue=GenerateCallTask.queue,
    base=GenerateCallTask,
    pydantic=True,
)
def generate_call_task(
    self: GenerateCallTask, inputs: GenerateCallTaskInputs
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
            inputs.user,
            inputs.project_slug,
            batch,
            path_jsonl=path_jsonl,
            n_workers=inputs.n_workers,
            is_aborted=self.is_aborted,
            update_progress=self.update_progress,
        )
    finally:
        inputs.data_path.unlink(missing_ok=True)
    return {
        "project_slug": inputs.project_slug,
        "username": inputs.user,
        "batch": batch,
        "jsonl_path": str(path_jsonl),
        "n_results": n_results,
        "n_errors": n_errors,
    }
