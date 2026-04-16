import os
import random
import shutil
import time
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
)

from activetigger.app.dependencies import (
    ProjectAction,
    ServerAction,
    get_project,
    test_rights,
    verified_user,
)
from activetigger.config import config
from activetigger.datamodels import (
    UserInDBModel,
)
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project

router = APIRouter(tags=["files"])


@router.post("/files/add/project")
def upload_file_project(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_name: str,
    file: UploadFile = File(...),
) -> None:
    """
    Upload a file on the server to create a new project
    use: type de file
    """
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)
    orchestrator = get_orchestrator()

    # add a delay if projects are already being created
    if len(orchestrator.project_creation_ongoing) >= 3:
        time.sleep(random.randint(1, 4))

    # check if the project does not already exist
    if orchestrator.exists(project_name):
        raise HTTPException(
            status_code=500, detail="Project already exists, please choose another name"
        )
    # test the incoming file
    if file.filename is None:
        raise HTTPException(status_code=500, detail="Problem with the file")
    if (
        not file.filename.endswith("csv")
        and not file.filename.endswith("parquet")
        and not file.filename.endswith("xlsx")
    ):
        raise HTTPException(status_code=500, detail="Only csv and parquet files are allowed")

    # try to upload the file
    try:
        # create a folder for the project to be created
        project_slug = orchestrator.check_project_name(project_name)
        project_path = Path(f"{config.data_path}/projects/{project_slug}")
        os.makedirs(project_path)

        # Read and write the file synchronously
        with open(project_path.joinpath(file.filename), "wb") as out_file:
            while chunk := file.file.read(1024 * 1024):
                out_file.write(chunk)
        print("File uploaded successfully")

    except Exception as e:
        # if failed, remove the project folder
        if project_path.exists():  # ty: ignore[possibly-unresolved-reference]
            shutil.rmtree(project_path)  # ty: ignore[possibly-unresolved-reference]
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/files/add/dataset")
def upload_file_dataset(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    file: UploadFile = File(...),
) -> None:
    """
    Upload a file on the server for a project in the data folder
    """
    test_rights(ProjectAction.MANAGE_FILES, current_user.username, project.params.project_slug)
    # test the incoming file
    if file.filename is None:
        raise HTTPException(status_code=500, detail="Problem with the file")
    if (
        not file.filename.endswith("csv")
        and not file.filename.endswith("parquet")
        and not file.filename.endswith("xlsx")
    ):
        raise HTTPException(status_code=500, detail="Only csv and parquet files are allowed")

    try:
        with open(project.data.path_datasets.joinpath(file.filename), "wb") as out_file:
            while chunk := file.file.read(1024 * 1024):
                out_file.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/files/copy/project")
def copy_existing_data(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_name: str,
    source_project: str,
    from_toy_dataset: bool = False,
) -> None:
    """
    Copy an existing project to create a new one
    if copy dataset from toy datasets: orchestrator.path_toy_datasets/NAME.parquet
    if copy from project: orchestrator.path/NAME/data_all.parquet
    """
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)

    orchestrator = get_orchestrator()

    # check if the project does not already exist
    if orchestrator.exists(project_name):
        raise HTTPException(
            status_code=500, detail="Project already exists, please choose another name"
        )
    # try to copy the project
    try:
        # create a folder for the project to be created
        project_slug = orchestrator.check_project_name(project_name)
        if from_toy_dataset:
            source_path = Path(f"{orchestrator.path_toy_datasets}/{source_project}.parquet")
        else:
            source_path = Path(f"{orchestrator.path}/{source_project}/{config.data_all}")
        project_path = Path(f"{orchestrator.path}/{project_slug}")
        os.makedirs(project_path)

        # copy the full dataset
        shutil.copyfile(
            source_path,
            project_path.joinpath(config.data_all),
        )

    except Exception as e:
        # if failed, remove the project folder
        if project_path.exists():  # ty: ignore[possibly-unresolved-reference]
            shutil.rmtree(project_path)  # ty: ignore[possibly-unresolved-reference]
        raise HTTPException(status_code=500, detail=str(e))
