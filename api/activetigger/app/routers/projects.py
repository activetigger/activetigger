from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from activetigger.app.dependencies import (
    ProjectAction,
    ServerAction,
    check_auth_exists,
    get_project,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    AvailableProjectsModel,
    DatasetModel,
    EvalSetDataModel,
    ProjectAuthsModel,
    ProjectBaseModel,
    ProjectDescriptionModel,
    ProjectStateModel,
    ProjectUpdateModel,
    UserInDBModel,
)
from activetigger.functions import slugify
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project

router = APIRouter(tags=["projects"])


@router.post("/projects/close/{project_slug}", dependencies=[Depends(verified_user)])
def close_project(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_slug: str,
) -> None:
    """
    Close a project from memory
    """
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)
    try:
        get_orchestrator().stop_project(project_slug)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_slug}/statistics", dependencies=[Depends(verified_user)])
def get_project_statistics(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str | None = None,
) -> ProjectDescriptionModel:
    """
    Statistics for a scheme and a user
    """
    test_rights(ProjectAction.GET, current_user.username, project.project_slug)
    try:
        return project.get_statistics(scheme=scheme)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/auth", dependencies=[Depends(verified_user)])
def get_project_auth(
    current_user: Annotated[UserInDBModel, Depends(verified_user)], project_slug: str
) -> ProjectAuthsModel:
    """
    Users auth on a project
    """
    orchestrator = get_orchestrator()
    if not orchestrator.exists(project_slug):
        raise HTTPException(status_code=404, detail="Project doesn't exist")
    test_rights(ProjectAction.MONITOR, current_user.username, project_slug)
    try:
        return ProjectAuthsModel(auth=orchestrator.users.get_project_auth(project_slug))
    except Exception as e:
        raise HTTPException(status_code=500) from e


@router.post("/projects/new", dependencies=[Depends(verified_user)])
def new_project(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: ProjectBaseModel,
) -> str:
    """
    Start the creation of a new project
    """
    orchestrator = get_orchestrator()
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)
    try:
        project_slug = orchestrator.starting_project_creation(
            project=project,
            username=current_user.username,
        )
        orchestrator.log_action(
            current_user.username, f"START CREATING PROJECT: {project_slug}", project_slug
        )
        return project_slug
    except Exception as e:
        orchestrator.clean_unfinished_project(project_name=project.project_name)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/projects/update",
    dependencies=[Depends(verified_user), Depends(check_auth_exists)],
)
def update_project(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    update: ProjectUpdateModel,
) -> None:
    """
    Update a project
    - change the name
    - change the language
    - change context cols
    - change text cols
    - expand the number of elements in the trainset
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.project_slug)
    try:
        project.start_update_project(update, current_user.username)
        get_orchestrator().log_action(
            current_user.username,
            f"INFO UPDATE PROJECT: {project.project_slug}",
            project.project_slug,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/projects/delete",
    dependencies=[Depends(verified_user), Depends(check_auth_exists)],
)
def delete_project(
    project_slug: str,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> None:
    """
    Delete a project
    """
    test_rights(ServerAction.DELETE_PROJECT, current_user.username, project_slug)
    try:
        orchestrator = get_orchestrator()
        orchestrator.delete_project(project_slug)
        orchestrator.log_action(
            current_user.username, f"DELETE PROJECT: {project_slug}", project_slug
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/status", dependencies=[Depends(verified_user)])
def get_project_status(
    project_name: str,
) -> str:
    """
    Get the status of a project
    - not existing
    - creating
    - existing
    """
    try:
        orchestrator = get_orchestrator()
        slug = slugify(project_name)
        # if project is in creation
        if slug in orchestrator.project_creation_ongoing:
            return "creating"
        # if creation failed, return the error (consumed once)
        if slug in orchestrator.creation_errors:
            error_msg = orchestrator.creation_errors.pop(slug)
            return f"error: {error_msg}"
        elif orchestrator.exists(project_name):
            return "existing"
        else:
            return "not existing"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/evalset/delete", dependencies=[Depends(verified_user)])
def delete_evalset(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    dataset: str,
) -> None:
    """
    Delete an existing eval dataset
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.project_slug)
    try:
        project.drop_evalset(dataset=dataset)
        get_orchestrator().log_action(
            current_user.username, f"DELETE EVALSET {dataset}", project.project_slug
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/projects/evalset/add", dependencies=[Depends(verified_user)])
def add_testdata(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    dataset: str,
    evalset: EvalSetDataModel,
) -> str | None:
    """
    Delete existing eval/test dataset or
    Add a dataset for eval/test when there is none available
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.project_slug)
    try:
        if evalset is None:
            raise Exception("No evalset sent")
        id = project.add_evalset(dataset, evalset, current_user.username, project.project_slug)
        get_orchestrator().log_action(
            current_user.username, f"ADD EVALSET {dataset}", project.project_slug
        )
        return id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/projects")
def get_projects(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> AvailableProjectsModel:
    """
    Get general informations on the server
    depending of the status of connected user
    """
    try:
        orchestrator = get_orchestrator()
        return AvailableProjectsModel(
            projects=orchestrator.users.get_user_projects(current_user.username),
            storage_used=orchestrator.users.get_storage(current_user.username),
            storage_limit=orchestrator.users.get_storage_limit(current_user.username),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets", dependencies=[Depends(verified_user)])
def get_project_datasets(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    include_toy_datasets: bool = False,
) -> tuple[list[DatasetModel], list[DatasetModel] | None]:
    """
    Get all datasets already available for a specific user
    """
    try:
        orchestrator = get_orchestrator()
        toy_datasets = orchestrator.get_toy_datasets() if include_toy_datasets else []
        auth_datasets = orchestrator.users.get_auth_datasets(current_user.username)
        return auth_datasets, toy_datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/projects/{project_slug}",
    dependencies=[Depends(verified_user), Depends(check_auth_exists)],
)
def get_project_state(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
) -> ProjectStateModel:
    """
    Get the state of a specific project
    """
    test_rights(ProjectAction.GET, current_user.username, project.project_slug)
    try:
        return project.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
