from typing import Any

import requests
from celery import Task
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from openai import BaseModel

from task_manager import config
from task_manager.active_tigger_task import QueueName
from task_manager.celery import celery_app
from task_manager.task_names import TaskNames


class TaskFailureReportForCallback(BaseModel):
    exception: str
    exception_type: str
    task_name: TaskNames


class CallbackDeliveryError(Exception):
    pass


CALLBACK_TIMEOUT = (5, 60)


class SuccessCallbackPayload(BaseModel):
    task_id: str
    task_name: TaskNames
    results: Any


class FailureCallbackPayload(BaseModel):
    task_id: str
    task_name: TaskNames
    report: TaskFailureReportForCallback


class TaskResultForCallback[R](BaseModel):
    results: R
    task_name: TaskNames


def _post_callback(url: str, payload: BaseModel) -> None:
    """
    Deliver a callback to the API; retried on network errors and 5xx, never on 4xx.
    Returns False once abandoned (logged as an error, the worker log is then the only trace).
    """
    logger = get_task_logger("callback")
    logger.info(f"Sending callback {url} {payload}")
    body = payload.model_dump(mode="json")
    headers = {config.TASK_TOKEN_HEADER: config.task_callback_token}

    try:
        response = requests.post(url, json=body, headers=headers, timeout=CALLBACK_TIMEOUT)
    except requests.RequestException as e:
        logger.error(f"Callback {url} rejected: {type(e).__name__}:{e} ")
        raise CallbackDeliveryError(f"{type(e).__name__}: {e}") from e
    if not response.ok:
        logger.info(f"Callback response: {response.status_code} - {response.text}")
        raise CallbackDeliveryError(f"HTTP {response.status_code}: {response.text[:300]}")
    logger.info(f"Callback response: {response.status_code} - {response.text}")


class SuccessCallbackTask(Task):
    queue: QueueName = QueueName.CPU
    name = "success_callback"
    # retry settings
    autoretry_for = (CallbackDeliveryError,)
    retry_backoff = True  # 1s, 2s, 4s, 8s...
    retry_backoff_max = 32
    retry_jitter = True
    max_retries = 5

    def run(self, retval, task_name: TaskNames, main_task_id: str):

        results = TaskResultForCallback(
            results=retval[0],
            task_name=task_name,
        )
        try:
            payload = SuccessCallbackPayload(
                task_id=main_task_id, task_name=results.task_name, results=results.results
            )
            _post_callback(config.api_task_success_route, payload)
        except Exception:
            raise CallbackDeliveryError(f"callback for task {main_task_id} not accepted by API")


celery_app.register_task(SuccessCallbackTask())


class FailureCallbackTask(Task):
    name = "failure_callback"
    queue = QueueName.CPU
    autoretry_for = (CallbackDeliveryError,)
    retry_backoff = True
    retry_backoff_max = 32
    retry_jitter = True
    max_retries = 5

    def run(
        self,
        failed_task_id: str,
        task_name: TaskNames,
        main_task_id: str,
    ):
        # single-arg errback style: dispatched as a task, receives ONLY the
        # id of the failed task — the exception is not passed, fetch it
        exc = AsyncResult(failed_task_id, app=celery_app).result
        report = TaskFailureReportForCallback(
            exception=str(exc), exception_type=type(exc).__name__, task_name=task_name
        )
        try:
            payload = FailureCallbackPayload(
                task_id=failed_task_id, report=report, task_name=task_name
            )
            _post_callback(config.api_task_failure_route, payload)
        except Exception:
            raise CallbackDeliveryError(
                f"failure callback for task {main_task_id} not accepted by API"
            )


celery_app.register_task(FailureCallbackTask())
