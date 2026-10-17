from pathlib import Path
from typing import TypedDict

from activetigger.datamodels import ProjectUpdateModel
from pydantic import BaseModel

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.update_datasets import update_datasets


class UpdateDatasetsTaskInput(BaseModel):
    project_slug: str
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


class UpdateDatasetsTask(AbortableAutoCallbackTask):
    name = "update datasets"
    queue = QueueName.CPU


@celery_app.task(
    bind=True,
    name=UpdateDatasetsTask.name,
    queue=UpdateDatasetsTask.queue,
    base=UpdateDatasetsTask,
    pydantic=True,
)
def update_datasets_task(
    self: UpdateDatasetsTask, inputs: UpdateDatasetsTaskInput
) -> UpdateDatasetsTaskResult:
    # the parquet rewrite cannot be interrupted midway, only skipped before it starts
    if self.is_aborted():
        raise Exception("Process interrupted by user")
    fields = update_datasets(
        inputs.path_project, inputs.cols_context, inputs.cols_text, inputs.update
    )
    return {
        "project_slug": inputs.project_slug,
        "cols_context": fields["cols_context"],
        "cols_text": fields["cols_text"],
        "n_train": fields["n_train"],
        "reset_features": fields["reset_features"],
    }
