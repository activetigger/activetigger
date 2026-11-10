"""
Stop signal shared by the API and the workers.
The stop request therefore in its own redis key to avoid overlap.
"""

import redis
from celery.utils.log import get_task_logger

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


def stop_key(task_id: str) -> str:
    return f"{config.STOP_KEY_PREFIX}{task_id}"


def request_stop(task_id: str) -> bool:
    """
    Flag a task as stopped by the user (API side). Best effort: a redis failure
    is logged and reported as False so the caller can still revoke and clean up.
    """
    try:
        redis_client().set(stop_key(task_id), "1", ex=config.stop_signal_ttl)
        return True
    except redis.RedisError as e:
        logger.error(f"Cannot write stop signal for {task_id}: {e}")
        return False


def stop_requested(task_id: str) -> bool:
    """
    Whether a stop was requested for this task (both sides).
    A redis failure is logged and read as "no stop" so the caller can fall back.
    """
    try:
        return bool(redis_client().exists(stop_key(task_id)))
    except redis.RedisError as e:
        logger.error(f"Cannot read stop signal for {task_id}: {e}")
        return False
