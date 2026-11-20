import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException
from task_manager.celery import celery_app
from task_manager.utils import (
    FailureCallbackPayload,
    SuccessCallbackPayload,
)

from activetigger.app.dependencies import (
    ServerAction,
    test_rights,
    verified_task_worker,
    verified_user,
)
from activetigger.datamodels import UserInDBModel
from activetigger.tasks import all_callbacks
from activetigger.tasks.task_callback import ProjectNotTrackedError

router = APIRouter(tags=["tasks"])
logger = logging.getLogger("activetigger.fastapi.tasks")


@router.post("/tasks/done", dependencies=[Depends(verified_task_worker)])
def task_success_callback(body: SuccessCallbackPayload = Body(...)):
    """
    Task success callback
    """
    logger.info(f"task {body.task_id} succeed with result {body.results} {body.task_name}")

    callback = all_callbacks.task_callbacks.get(body.task_name)
    if callback is None:
        raise HTTPException(status_code=404, detail=f"No callback for task {body.task_name}")
    logger.info(f"execute callback for task {body.task_name} {body.task_id}")
    try:
        callback.on_complete(body)
    except ProjectNotTrackedError as e:
        logger.warning(f"task {body.task_id} result dropped: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from e
    return True


@router.post("/tasks/failed", dependencies=[Depends(verified_task_worker)])
def task_failure_callback(body: FailureCallbackPayload = Body(...)):
    """
    Task failure callback
    """
    logger.info(f"task {body.task_id} failed with report {body.report} {body.task_name}")

    callback = all_callbacks.task_callbacks.get(body.task_name)
    if callback is None:
        raise HTTPException(status_code=404, detail=f"No callback for task {body.task_name}")
    logger.info(f"execute callback for task {body.task_name} {body.task_id}")
    try:
        callback.on_failure(body)
    except ProjectNotTrackedError as e:
        logger.warning(f"task {body.task_id} failure dropped: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from e
    return True


@router.get("/tasks/monitor")
def task_monitor_callback(current_user: Annotated[UserInDBModel, Depends(verified_user)]):
    """
    Inspect the celery workers (server managers only)
    """
    test_rights(ServerAction.MANAGE_SERVER, current_user.username)
    logger.info("monitoring tasks")
    inspect = celery_app.control.inspect()
    return {"active": inspect.active(), "stats": inspect.stats()}
