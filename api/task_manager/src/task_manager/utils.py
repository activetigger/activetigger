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


def _post_callback(url: str, payload: BaseModel) -> None:
    logger = get_task_logger("callback")
    logger.info(f"Sending callback {url} {payload}")
    response = requests.post(url, json=payload.model_dump(mode="json"))
    logger.info(f"Callback response: {response.status_code} - {response.text}")


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
