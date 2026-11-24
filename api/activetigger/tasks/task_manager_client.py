import json
from typing import cast

import redis
from celery import uuid
from celery.contrib.abortable import AbortableAsyncResult, AbortableTask
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from task_manager.active_tigger_task import QueueName
from task_manager.celery import celery_app

from task_manager import config

logger = get_task_logger("stop_signal")

_client: redis.Redis | None = None


def redis_client() -> redis.Redis:
    """
    Shared client (lazy: created after the prefork fork in a worker child)
    """
    global _client
    if _client is None:
        _client = redis.Redis.from_url(config.redis_url, socket_timeout=5, socket_connect_timeout=5)
    return _client


# collections of utils methods to interact with celery task_manager


def new_task_id(task) -> tuple[str, bool]:
    """
    Pre-allocate the id and the abortable flag so the process entry can exist before the task is sent
    """
    return uuid(), isinstance(task, AbortableTask)


def _stop_celery_task(task_id: str, abortable: bool | None):

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
                _stop_celery_task(t["id"], isinstance(task, AbortableTask))
                stopped += 1
    return {"purged": purged, "stopped": stopped}


def celery_task_state(task_id: str) -> str:
    """
    Backend state of a task (PENDING for an unknown or expired id)
    """
    return AsyncResult(task_id, app=celery_app).state


def celery_tasks_in_flight() -> set[str] | None:
    """
    Ids of the tasks still queued (broker lists) or held by a worker (active,
    reserved). None when no worker answered: nothing can be concluded then.
    """
    inspect = celery_app.control.inspect(timeout=2)
    held: set[str] = set()
    answered = False
    for report in (inspect.active(), inspect.reserved()):
        if report is None:
            continue
        answered = True
        for tasks in report.values():
            held.update(t["id"] for t in tasks)
    if not answered:
        return None
    # queued messages: kombu redis transport stores one JSON message per list entry
    r = redis_client()
    for queue in QueueName:
        for raw in cast(list, r.lrange(str(queue), 0, -1)):
            try:
                task_id = json.loads(raw)["headers"]["id"]
            except (ValueError, KeyError, TypeError):
                continue
            held.add(task_id)
    return held
