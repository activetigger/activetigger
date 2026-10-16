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


def stop_celery_task(task_id: str, abortable: bool | None):
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
