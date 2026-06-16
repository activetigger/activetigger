import os
import random
import re
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

_ALLOWED_UPLOAD_EXTENSIONS = (".csv", ".parquet", ".xlsx")
_ALLOWED_IMAGE_UPLOAD_EXTENSIONS = (".zip",)


def _safe_upload_path(
    directory: Path,
    filename: str | None,
    allowed_extensions: tuple[str, ...] = _ALLOWED_UPLOAD_EXTENSIONS,
) -> Path:
    """
    Return a path inside `directory` that is safe to write to.

    Strips any directory components from `filename`, restricts the result to a
    conservative character set, and verifies the final path stays inside
    `directory` so a crafted filename cannot escape via traversal or absolute
    path tricks.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    base = Path(filename).name
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".")
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not safe_name.lower().endswith(allowed_extensions):
        allowed_str = ", ".join(ext.lstrip(".") for ext in allowed_extensions)
        raise HTTPException(
            status_code=400, detail=f"Only {allowed_str} files are allowed"
        )

    directory_resolved = directory.resolve()
    target = (directory_resolved / safe_name).resolve()
    if not target.is_relative_to(directory_resolved):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return target


@router.post("/files/add/project")
def upload_file_project(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_name: str,
    file: UploadFile = File(...),
    kind: str = "text",
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
    # Experimental image projects accept a .zip archive (see docs/image-projects-strategy.md)
    if kind == "image":
        if not file.filename.lower().endswith(".zip"):
            raise HTTPException(
                status_code=500,
                detail="Image projects require a .zip archive of images",
            )
    elif (
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

        allowed = (
            _ALLOWED_IMAGE_UPLOAD_EXTENSIONS if kind == "image" else _ALLOWED_UPLOAD_EXTENSIONS
        )
        target = _safe_upload_path(project_path, file.filename, allowed)

        # Read and write the file synchronously
        with open(target, "wb") as out_file:
            while chunk := file.file.read(1024 * 1024):
                out_file.write(chunk)
        print("File uploaded successfully")

    except HTTPException:
        if project_path.exists():  # ty: ignore[possibly-unresolved-reference]
            shutil.rmtree(project_path)  # ty: ignore[possibly-unresolved-reference]
        raise
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
    target = _safe_upload_path(project.data.path_datasets, file.filename)

    try:
        with open(target, "wb") as out_file:
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

    # Validate the source. `source_project` is user-supplied and used in a path,
    # so reduce it to a basename and require it to refer to a real, authorized
    # source before doing anything with the filesystem.
    source_name = Path(source_project).name
    if not source_name or source_name != source_project:
        raise HTTPException(status_code=400, detail="Invalid source_project")

    if from_toy_dataset:
        allowed = {d.project_slug for d in orchestrator.get_toy_datasets()}
        if source_name not in allowed:
            raise HTTPException(status_code=404, detail="Unknown toy dataset")
        source_path = orchestrator.path_toy_datasets / f"{source_name}.parquet"
    else:
        if source_name not in orchestrator.existing_projects():
            raise HTTPException(status_code=404, detail="Unknown source project")
        # Confirm the caller is actually authorized on the source project.
        test_rights(ProjectAction.GET, current_user.username, source_name)
        source_path = Path(orchestrator.path) / source_name / config.data_all

    # try to copy the project
    try:
        # create a folder for the project to be created
        project_slug = orchestrator.check_project_name(project_name)
        project_path = Path(f"{orchestrator.path}/{project_slug}")
        os.makedirs(project_path)

        # copy the full dataset
        shutil.copyfile(
            source_path,
            project_path.joinpath(config.data_all),
        )

    except HTTPException:
        if project_path.exists():  # ty: ignore[possibly-unresolved-reference]
            shutil.rmtree(project_path)  # ty: ignore[possibly-unresolved-reference]
        raise
    except Exception as e:
        # if failed, remove the project folder
        if project_path.exists():  # ty: ignore[possibly-unresolved-reference]
            shutil.rmtree(project_path)  # ty: ignore[possibly-unresolved-reference]
        raise HTTPException(status_code=500, detail=str(e))
