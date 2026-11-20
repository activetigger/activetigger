from enum import StrEnum
from typing import Generic, Literal, Protocol, TypeVar

from activetigger.datamodels import TaskComputing
from celery import Task
from celery.contrib.abortable import ABORTED, AbortableTask
from celery.states import REVOKED

from task_manager.stop_signal import stop_requested
from task_manager.task_input import TaskInputs
from task_manager.task_names import TaskNames
from task_manager.utils import (
    TaskFailureReportForCallback,
    TaskResultForCallback,
    task_failure_callback,
    task_success_callback,
)


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
    scope: TaskScope = "project"

    # Callback to orchestrator on task success
    def on_success(self, retval, task_id, args, kwargs):

        results = TaskResultForCallback(
            results=retval,
            task_name=self.name,
            task_args=args,
            task_kwargs=kwargs,
        )
        task_success_callback(task_id, results)
        return super().on_success(retval, task_id, args, kwargs)

    # Callback to orchestrator on task failure
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        report = TaskFailureReportForCallback(
            exception=str(exc),
            task_args=args,
            task_kwargs=kwargs,
            task_name=self.name,
        )
        task_failure_callback(task_id, report)
        return super().on_failure(exc, task_id, args, kwargs, einfo)

    # Progress state is used to monitor state in orchestrator
    def update_progress(self, progress: float, label: str | None = None):
        self.update_state(state="PROGRESS", meta={"progress": progress, "progress_label": label})

    # Clean temp files is used to remove temp files used to exchange data with orchestrator
    @classmethod
    def clean_temp_files(cls, computing_task: TaskComputing[TInputs]):
        pass


class AbortableActiveTiggerTask(ActiveTiggerTask, AbortableTask, Generic[TInputs]):
    def is_aborted(self, **kwargs) -> bool:
        task_id = kwargs.get("task_id", self.request.id)
        # the stop key is the reference: the task state is overwritten by update_progress
        if stop_requested(task_id):
            return True
        # fallback on the state set by abort()/revoke() (the revoke overwrites ABORTED)
        return self.AsyncResult(task_id).state in (ABORTED, REVOKED)
