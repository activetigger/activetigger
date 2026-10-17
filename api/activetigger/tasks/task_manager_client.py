import time
from pathlib import Path

from celery import uuid
from celery.contrib.abortable import AbortableAsyncResult, AbortableTask
from task_manager.celery import celery_app

from activetigger.datamodels import ProcessComputing

# collections of utils methods to interact with celery task_manager


def new_task_id(task) -> tuple[str, bool]:
    """
    Pre-allocate the id and the abortable flag so the process entry can exist before the task is sent
    """
    return uuid(), isinstance(task, AbortableTask)


def enqueue_celery_task(
    task,
    inputs,
    computing: list[ProcessComputing] | None = None,
    process: ProcessComputing | None = None,
    queue: str | None = None,
) -> tuple[str, bool]:
    """
    Send a task to the workers. With a process, register it in computing first:
    a task that fails or finishes at once calls back before the dispatcher returns.
    """
    if process is not None:
        if computing is None:
            raise ValueError("a process needs the computing list to be registered in")
        task_id, abortable = process.unique_id, bool(process.abortable_celery_task)
        computing.append(process)
    else:
        task_id, abortable = new_task_id(task)
    # queue overrides the task default (e.g. GPU for a wtpsplit preparation)
    options = {"queue": queue} if queue else {}
    try:
        task.s(inputs.model_dump(mode="json")).apply_async(task_id=task_id, **options)
    except Exception:
        if process is not None and computing is not None:
            computing.remove(process)
            for f in process.temp_files:
                Path(f).unlink(missing_ok=True)
        raise
    return task_id, abortable


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
