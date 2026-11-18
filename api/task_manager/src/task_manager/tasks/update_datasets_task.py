from pathlib import Path
from typing import TypedDict

from activetigger.datamodels import ProjectUpdateModel, TaskComputing
from pydantic import BaseModel

from task_manager.active_tigger_task import AbortableActiveTiggerTask, QueueName
from task_manager.celery import celery_app
from task_manager.task_input import TaskInputs
from task_manager.task_names import TaskNames
from task_manager.tasks.update_datasets import update_datasets


class UpdateDatasetsTaskInputs(TaskInputs[str]):
    path_project: Path
    # current columns, to detect what actually changes
    cols_context: list[str]
    cols_text: list[str]
    update: ProjectUpdateModel


# Return type must be a dict, a BaseModel would not be serialized by Celery
class UpdateDatasetsTaskResult(TypedDict):
    project_slug: str
    cols_context: list[str]
    cols_text: list[str]
    n_train: int | None
    reset_features: bool
    train_path: str | None


class UpdateDatasetsTask(AbortableActiveTiggerTask):
    name: TaskNames = "update_datasets"
    queue = QueueName.CPU

    @classmethod
    def clean_temp_files(cls, computing_task: TaskComputing[UpdateDatasetsTaskInputs]):
        computing_task.inputs.path_project.joinpath(f"train_{computing_task.id}.parquet").unlink(
            missing_ok=True
        )


@celery_app.task(
    bind=True,
    name=UpdateDatasetsTask.name,
    queue=UpdateDatasetsTask.queue,
    base=UpdateDatasetsTask,
    pydantic=True,
)
def update_datasets_task(
    self: UpdateDatasetsTask, inputs: UpdateDatasetsTaskInputs
) -> UpdateDatasetsTaskResult:
    # the parquet rewrite cannot be interrupted midway, only skipped before it starts
    if self.is_aborted():
        raise Exception("Process interrupted by user")
    # provisional file, promoted to train.parquet by the API callback
    output_path = inputs.path_project.joinpath(f"train_{self.request.id}.parquet")
    try:
        fields = update_datasets(
            inputs.path_project, inputs.cols_context, inputs.cols_text, inputs.update, output_path
        )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return {
        "project_slug": inputs.project_slug,
        "cols_context": fields["cols_context"],
        "cols_text": fields["cols_text"],
        "n_train": fields["n_train"],
        "reset_features": fields["reset_features"],
        "train_path": fields["train_path"],
    }
