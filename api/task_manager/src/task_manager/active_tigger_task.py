from enum import StrEnum
from typing import Generic, Literal, Protocol, TypeVar

from activetigger.datamodels import TaskComputing
from celery import Task
from celery.contrib.abortable import ABORTED, AbortableTask
from celery.states import REVOKED

from task_manager.task_input import TaskInputs
from task_manager.task_names import TaskNames


# TODO: should we declare queue in celery_app directly?
class QueueName(StrEnum):
    CPU = "CPU"
    GPU = "GPU"


# typehint for UpdateProgressMethod methods
class UpdateProgressMethod(Protocol):
    def __call__(self, progress: float, label: str | None = None) -> None: ...


TInputs = TypeVar("TInputs", bound=TaskInputs)

TaskScope = Literal["project", "orchestrator"]


class ActiveTiggerTask(Task, Generic[TInputs]):
    """
    Base contract for tasks which need to callback the orchestrator.

    Subclasses must implement:
      - `run(**kwargs)` — the actual work (registered as the Celery task)
      - `on_success()` — send a callback request to orchestrator API
    """

    name: TaskNames
    queue: QueueName = QueueName.CPU
    scope: TaskScope = "project"

    # Progress state is used to monitor state in orchestrator
    def update_progress(self, progress: float, label: str | None = None):
        if not isinstance(self, AbortableActiveTiggerTask) or self.is_aborted() is False:
            self.update_state(
                state="PROGRESS", meta={"progress": progress, "progress_label": label}
            )

    # Clean temp files is used to remove temp files used to exchange data with orchestrator
    @classmethod
    def clean_temp_files(cls, computing_task: TaskComputing[TInputs]):
        pass


class AbortableActiveTiggerTask(ActiveTiggerTask, AbortableTask, Generic[TInputs]):
    def is_aborted(self, **kwargs) -> bool:
        task_id = kwargs.get("task_id", self.request.id)

        return self.AsyncResult(task_id).state in (ABORTED, REVOKED)
