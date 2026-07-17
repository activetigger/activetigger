from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    UploadFile,
)
from fastapi.responses import FileResponse

from activetigger.app.dependencies import ServerAction, test_rights, verified_user
from activetigger.datamodels import (
    PrepareSessionModel,
    PrepareSplitModel,
    PrepareStatusModel,
    PrepareTaskModel,
    UserInDBModel,
)
from activetigger.orchestrator import get_orchestrator

router = APIRouter(prefix="/toolbox", tags=["toolbox"])


@router.post("/upload")
def upload_prepare_file(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    file: UploadFile = File(...),
) -> PrepareSessionModel:
    """
    Upload a file (csv/xlsx/parquet or zip of txt/docx) for the dataset
    preparation tool. Normalize it as raw.parquet in a new session
    directory and return its columns and a preview.
    """
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)
    return get_orchestrator().toolbox.upload(current_user.username, file)


@router.post("/split")
def split_prepare_dataset(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    params: PrepareSplitModel,
) -> PrepareTaskModel:
    """
    Launch the split task on the queue (gpu queue for wtpsplit)
    """
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)
    return get_orchestrator().toolbox.split(current_user.username, params)


@router.post("/stop")
def stop_prepare_task(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    task_id: str,
) -> None:
    """
    Stop a running split task of the current user
    """
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)
    get_orchestrator().toolbox.stop(current_user.username, task_id)


@router.get("/status")
def get_prepare_status(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    session_id: str,
    task_id: str,
) -> PrepareStatusModel:
    """
    Status of a split task; when done, add the number of rows and a preview
    """
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)
    return get_orchestrator().toolbox.status(current_user.username, session_id, task_id)


@router.get("/export")
def export_prepared_dataset(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    session_id: str,
    format: str = "csv",
) -> FileResponse:
    """
    Download the prepared dataset in the requested format
    """
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)
    return get_orchestrator().toolbox.export(current_user.username, session_id, format)
