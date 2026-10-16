from pathlib import Path
from typing import TypedDict

from activetigger.datamodels import PrepareSplitModel
from pydantic import BaseModel

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.prepare_dataset import prepare_dataset


class PrepareDatasetTaskInput(BaseModel):
    # toolbox tasks belong to a user session, not to a project
    username: str
    session_id: str
    path_session: Path
    params: PrepareSplitModel


# Return type must be a dict, a BaseModel would not be serialized by Celery
class PrepareDatasetTaskResult(TypedDict):
    username: str
    session_id: str
    result_path: str


class PrepareDatasetTask(AbortableAutoCallbackTask):
    name = "prepare dataset"
    # default queue; the dispatcher sends wtpsplit runs to the GPU queue
    queue = QueueName.CPU


@celery_app.task(
    bind=True,
    name=PrepareDatasetTask.name,
    queue=PrepareDatasetTask.queue,
    base=PrepareDatasetTask,
    pydantic=True,
)
def prepare_dataset_task(
    self: PrepareDatasetTask, inputs: PrepareDatasetTaskInput
) -> PrepareDatasetTaskResult:
    # progress lives at <session>/progress, read by Toolbox.status
    result_path = prepare_dataset(inputs.path_session, inputs.params, is_aborted=self.is_aborted)
    return {
        "username": inputs.username,
        "session_id": inputs.session_id,
        "result_path": str(result_path),
    }
