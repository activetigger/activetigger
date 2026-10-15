from enum import StrEnum

from celery import Task
from celery.contrib.abortable import ABORTED, AbortableTask
from celery.states import REVOKED

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


class AutoCallbackTask(Task):
    """
    Base contract for tasks which need to callback the orchestrator.

    Subclasses must implement:
      - `run(**kwargs)` — the actual work (registered as the Celery task)
      - `on_success()` — send a callback request to orchestrator API
    """

    # Callback to orchestrator on task success
    def on_success(self, retval, task_id, args, kwargs):
        results = TaskResultForCallback(
            results=retval,
            task_name=self.name,  # ty:ignore[invalid-argument-type]
        )
        task_success_callback(task_id, results)
        return super().on_success(retval, task_id, args, kwargs)

    # Callback to orchestrator on task failure
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        report = TaskFailureReportForCallback(
            exception=str(exc),
            task_args=args,
            task_kwargs=kwargs,
            task_name=self.name,  # ty:ignore[invalid-argument-type]
        )
        task_failure_callback(task_id, report)
        return super().on_failure(exc, task_id, args, kwargs, einfo)


class AbortableAutoCallbackTask(AutoCallbackTask, AbortableTask):
    def is_aborted(self, **kwargs) -> bool:
        # stop = abort() + revoke(); the revoke overwrites ABORTED with REVOKED in the backend
        task_id = kwargs.get("task_id", self.request.id)
        return self.AsyncResult(task_id).state in (ABORTED, REVOKED)
