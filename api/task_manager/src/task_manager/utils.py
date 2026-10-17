import time
from typing import Any

import requests
from celery.utils.log import get_task_logger
from pydantic import BaseModel

from task_manager import config


class TaskResultForCallback[R](BaseModel):
    results: R
    task_name: str


class TaskFailureReportForCallback(BaseModel):
    exception: str
    task_name: str
    task_args: Any
    task_kwargs: Any


class SuccessCallbackPayload(BaseModel):
    task_id: str
    task_name: str
    results: Any


class FailureCallbackPayload(BaseModel):
    task_id: str
    task_name: str
    report: TaskFailureReportForCallback


# backoff between attempts: long enough to ride out an API restart or reload
CALLBACK_RETRY_DELAYS = (2, 4, 8, 16, 32)
CALLBACK_TIMEOUT = (5, 60)


def _post_callback(url: str, payload: BaseModel) -> bool:
    """
    Deliver a callback to the API; retried on network errors and 5xx, never on 4xx.
    Returns False once abandoned (logged as an error, the worker log is then the only trace).
    """
    logger = get_task_logger("callback")
    logger.info(f"Sending callback {url} {payload}")
    body = payload.model_dump(mode="json")
    headers = {config.TASK_TOKEN_HEADER: config.task_callback_token}
    attempts = len(CALLBACK_RETRY_DELAYS) + 1
    error = ""
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(url, json=body, headers=headers, timeout=CALLBACK_TIMEOUT)
        except requests.RequestException as e:
            error = f"{type(e).__name__}: {e}"
        else:
            if response.ok:
                logger.info(f"Callback response: {response.status_code} - {response.text}")
                return True
            if response.status_code < 500:
                logger.error(f"Callback {url} rejected: {response.status_code} - {response.text}")
                return False
            error = f"HTTP {response.status_code}: {response.text[:300]}"
        if attempt < attempts:
            delay = CALLBACK_RETRY_DELAYS[attempt - 1]
            logger.warning(
                f"Callback {url} failed ({error}), retry {attempt}/{attempts - 1} in {delay}s"
            )
            time.sleep(delay)
    logger.error(f"Callback {url} abandoned after {attempts} attempts ({error}): {body}")
    return False


def task_success_callback(task_id: str, results: TaskResultForCallback[Any]):
    """
    Send to the FastAPI route the result of the task
    """
    payload = SuccessCallbackPayload(
        task_id=task_id, task_name=results.task_name, results=results.results
    )
    _post_callback(config.api_task_success_route, payload)


def task_failure_callback(task_id: str, report: TaskFailureReportForCallback):
    """
    Send to the FastAPI route the failure report of the task
    """
    payload = FailureCallbackPayload(task_id=task_id, task_name=report.task_name, report=report)
    _post_callback(config.api_task_failure_route, payload)
