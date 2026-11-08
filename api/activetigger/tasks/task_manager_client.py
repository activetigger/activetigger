from pathlib import Path

from celery import uuid
from celery.contrib.abortable import AbortableAsyncResult, AbortableTask
from celery.result import AsyncResult
from task_manager.auto_callback_task import QueueName, TaskProgressState
from task_manager.celery import celery_app
from task_manager.stop_signal import request_stop, stop_requested

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
        if process is not None and computing is not None and process in computing:
            computing.remove(process)
            for f in process.temp_files:
                Path(f).unlink(missing_ok=True)
        raise
    return task_id, abortable


def was_stopped(task_id: str) -> bool:
    """
    A failure triggered by a stop (e.g. an input pickle removed under a task
    that had just started) is reported as a cancellation, not an error
    """
    return stop_requested(task_id)


def stop_celery_task(task_id: str, abortable: bool | None):
    # the stop key is what is_aborted() reads in the worker; it survives
    # progress updates, result expiry and API restarts
    request_stop(task_id)
    if abortable:
        task = AbortableAsyncResult(task_id, app=celery_app)
        print(f"Aborting {task_id} {type(task).__name__}")
        # abort() is a fallback flag; revoke drops a task still in the queue
        task.abort()
        celery_app.control.revoke(task_id)
    else:
        print(f"Revoke task {task_id}")
        celery_app.control.revoke(task_id, terminate=True, signal="SIGKILL")


def purge_celery_tasks() -> dict[str, int]:
    """
    Drop every queued task and stop every task the workers hold. 
    """
    if celery_app.conf.task_always_eager:
        return {"purged": 0, "stopped": 0}
    purged = 0
    with celery_app.connection_for_write() as conn:
        for queue in QueueName:
            purged += conn.default_channel.queue_purge(str(queue)) or 0
    # running (active) and prefetched (reserved) tasks are no longer in the queue
    stopped = 0
    inspect = celery_app.control.inspect(timeout=2)
    for report in (inspect.active() or {}, inspect.reserved() or {}):
        for tasks in report.values():
            for t in tasks:
                task = celery_app.tasks.get(t["name"])
                stop_celery_task(t["id"], isinstance(task, AbortableTask))
                stopped += 1
    return {"purged": purged, "stopped": stopped}


def get_celery_task_progress(task_id: str) -> TaskProgressState | None:
    r = AsyncResult(task_id, app=celery_app)
    if r.state == "PROGRESS":
        return r.info
    return None
