import time

from celery import uuid
from celery.contrib.abortable import AbortableAsyncResult
from task_manager.celery import celery_app

# collections of utils methods to interact with celery task_manager


def enqueue_celery_task(task, inputs, queue: str | None = None) -> tuple[str, bool]:
    task_id = uuid()
    # queue overrides the task default (e.g. GPU for a wtpsplit preparation)
    options = {"queue": queue} if queue else {}
    task = task.s(inputs.model_dump(mode="json")).apply_async(task_id=task_id, **options)
    print(task, type(task).__name__)
    return task_id, isinstance(task, AbortableAsyncResult)


# ids stopped from the API, so a failure they trigger (e.g. an input pickle
# removed under a task that had just started) is reported as a cancellation
_stopped: dict[str, float] = {}
STOPPED_TTL = 3600


def was_stopped(task_id: str) -> bool:
    now = time.time()
    for k in [k for k, t in _stopped.items() if now - t > STOPPED_TTL]:
        del _stopped[k]
    return _stopped.pop(task_id, None) is not None


def stop_celery_task(task_id: str, abortable: bool | None):
    _stopped[task_id] = time.time()
    if abortable:
        task = AbortableAsyncResult(task_id, app=celery_app)
        print(f"Aborting {task_id} {type(task).__name__}")
        # abort() only flags a running task; revoke drops a task still in the queue.
        # revoke overwrites ABORTED with REVOKED, so is_aborted() accepts both states.
        task.abort()
        celery_app.control.revoke(task_id)
    else:
        print(f"Revoke task {task_id}")
        celery_app.control.revoke(task_id, terminate=True, signal="SIGKILL")
