from celery import uuid
from celery.contrib.abortable import AbortableAsyncResult

from task_manager.celery import celery_app

# collections of utils methods to interact with celery task_manager


def enqueue_celery_task(task, inputs) -> tuple[str, bool]:
    task_id = uuid()
    task = task.s(inputs.model_dump(mode="json")).apply_async(task_id=task_id)
    print(task, type(task).__name__)
    return task_id, isinstance(task, AbortableAsyncResult)


def stop_celery_task(task_id: str, abortable: bool | None):
    if abortable:
        task = AbortableAsyncResult(task_id, app=celery_app)
        print(f"Aborting {task_id} {type(task).__name__}")
        task.abort()
    else:
        print(f"Revoke task {task_id}")
        celery_app.control.revoke(task_id, terminate=True, signal="SIGKILL")
