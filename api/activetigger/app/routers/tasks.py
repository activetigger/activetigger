import logging

from fastapi import APIRouter, Body
from pydantic import BaseModel, ConfigDict

from activetigger.tasks import all_callbacks
from task_manager.celery import celery_app
from task_manager.utils import TaskFailureReportForCallback


class GenericCallbackModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    task_id: str
    task_name: str


class SuccessCallbackModel(GenericCallbackModel):
    results: any  # ty:ignore[invalid-type-form]


class FailureCallbackModel(GenericCallbackModel):
    report: TaskFailureReportForCallback


router = APIRouter(tags=["tasks"])
logger = logging.getLogger("activetigger.fastapi.tasks")


@router.post(
    "/tasks/done",
    # TODO: add a Task API key verification
    # dependencies=[Depends(verified_user)],
)
def task_success_callback(body: SuccessCallbackModel = Body(...)):
    """
    Task success callback
    """
    logger.info(f"task {body.task_id} succeed with result {body.results} {body.task_name}")

    # check if a completion callback is available
    callback = all_callbacks.task_callbacks[body.task_name]
    if callback:
        logger.info(f"execute callback for task {body.task_name} {body.task_id}")
        callback.on_complete(body.task_id, body.results)
    return True


@router.post(
    "/tasks/failed",
    # TODO: add a Task API key verification
    # dependencies=[Depends(verified_user)],
)
def task_failure_callback(body: FailureCallbackModel = Body(...)):
    """
    Task success callback
    """
    logger.info(f"task {body.task_id} failed with report {body.report} {body.task_name}")

    # check if a completion callback is available
    callback = all_callbacks.task_callbacks[body.task_name]
    if callback:
        logger.info(f"execute callback for task {body.task_name} {body.task_id}")
        callback.on_failure(body.task_id, body.report)
    return True


@router.get(
    "/tasks/monitor",
    # TODO: add a Task API key verification
    # dependencies=[Depends(verified_user)],
)
def task_monitor_callback():
    """
    Task success callback
    """
    logger.info("monitoring tasks")
    inspect = celery_app.control.inspect()
    return {"active": inspect.active(), "stats": inspect.stats()}
