from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from activetigger.app.dependencies import (
    ProjectAction,
    get_project,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    GenCredentialsInput,
    GenCredentialsOut,
    GenCredentialsTestOut,
    GenPipelineCreate,
    GenPipelineOut,
    GenRunOut,
    GenRunRequest,
    GenSandboxOut,
    GenSandboxRequest,
    PostprocessPreviewRequest,
    TableOutModel,
    UserInDBModel,
)
from activetigger.errors import APIError
from activetigger.generations import POSTPROCESS_STEPS, Generations
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project

router = APIRouter(tags=["generation"])

SANDBOX_MAX_ELEMENTS = 10


@router.get("/generate/credentials", dependencies=[Depends(verified_user)])
def list_credentials(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> list[GenCredentialsOut]:
    """
    List the credentials visible to the current user
    """
    try:
        return Generations.list_user_credentials(
            get_orchestrator().db_manager, current_user.username
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/credentials", dependencies=[Depends(verified_user)])
def add_credentials(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    credentials: GenCredentialsInput,
) -> GenCredentialsTestOut:
    """
    Save a user entry (an entry with the same name is replaced) and test it
    """
    try:
        return Generations.save_user_credentials(
            get_orchestrator().db_manager, current_user.username, credentials
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/credentials/test", dependencies=[Depends(verified_user)])
def test_credentials(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    credentials_id: int,
) -> GenCredentialsTestOut:
    """
    Re-test an entry against its endpoint
    """
    try:
        return Generations.test_user_credentials(
            get_orchestrator().db_manager, credentials_id, current_user.username
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/credentials/delete", dependencies=[Depends(verified_user)])
def delete_credentials(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    credentials_id: int,
) -> None:
    """
    Delete a user entry (instance entries are managed through generative.yaml).
    Refused if a pipeline still uses it.
    """
    try:
        Generations.delete_user_credentials(
            get_orchestrator().db_manager, credentials_id, current_user.username
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate/credentials/{credentials_id}/models", dependencies=[Depends(verified_user)])
def list_endpoint_models(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    credentials_id: int = Path(ge=0),
) -> list[str]:
    """
    List the models available with a credentials entry
    """
    try:
        return Generations.models_for_credentials(
            get_orchestrator().db_manager, credentials_id, current_user.username
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate/postprocess/steps")
def list_postprocess_steps() -> dict[str, dict]:
    """
    Available post-treatment steps available in the server
    """
    return {
        name: params_model.model_json_schema()
        for name, (function, params_model) in POSTPROCESS_STEPS.items()
    }


@router.get("/generate/pipelines", dependencies=[Depends(verified_user)])
def list_pipelines(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> list[GenPipelineOut]:
    """
    Pipelines of the project
    """
    test_rights(ProjectAction.GENERATE, current_user.username, project.name)
    try:
        return project.generations.available()
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/pipelines", dependencies=[Depends(verified_user)])
def add_pipeline(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    pipeline: GenPipelineCreate,
) -> int:
    """
    Create a new pipeline
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    try:
        schemes_kinds = {name: s.kind for name, s in project.schemes.available().items()}
        pipeline_id = project.generations.add_pipeline(
            pipeline, current_user.username, schemes_kinds, project.params.cols_context
        )
        get_orchestrator().log_action(
            current_user.username, "CREATE GENERATIVE PIPELINE", project.name
        )
        return pipeline_id
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/pipelines/delete", dependencies=[Depends(verified_user)])
def delete_pipeline(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    pipeline_id: int,
) -> None:
    """
    Delete a pipeline, its runs and their output files
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    try:
        project.generations.delete_pipeline(pipeline_id)
        get_orchestrator().log_action(
            current_user.username, "DELETE GENERATIVE PIPELINE", project.name
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def resolve_sampling(
    project: Project, scheme_name: str | None, request_scheme: str | None
) -> tuple[str, list[str] | None]:
    """
    The scheme used to sample elements, and the labels for post-treatment.
    """
    schemes = project.schemes.available()
    if scheme_name is not None:
        if scheme_name not in schemes:
            raise Exception(f"Scheme {scheme_name} does not exist anymore")
        return scheme_name, schemes[scheme_name].labels
    sampling_scheme = request_scheme or next(iter(schemes), None)
    if sampling_scheme is None or sampling_scheme not in schemes:
        raise Exception("No scheme available to sample elements")
    return sampling_scheme, None


@router.post("/generate/pipelines/{pipeline_id}/sandbox", dependencies=[Depends(verified_user)])
def sandbox_pipeline(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    request: GenSandboxRequest,
    pipeline_id: int = Path(ge=0),
) -> GenSandboxOut:
    """
    Test a pipeline synchronously on a few sampled elements.
    """
    test_rights(ProjectAction.GENERATE, current_user.username, project.name)
    try:
        pipeline = project.generations.get_pipeline(pipeline_id)
        sampling_scheme, labels = resolve_sampling(project, pipeline.scheme_name, request.scheme)
        df = project.schemes.get_sample(
            sampling_scheme,
            min(request.n_elements, SANDBOX_MAX_ELEMENTS),
            request.mode,
            dataset=request.dataset,
            random=True,
        )
        if len(df) == 0:
            raise Exception("No elements available for this selection")
        return project.generations.sandbox(pipeline_id, df, project.params.cols_context, labels)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/postprocess/preview", dependencies=[Depends(verified_user)])
def preview_postprocess(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    request: PostprocessPreviewRequest,
) -> GenSandboxOut:
    """
    Apply a candidate list of steps on raw outputs already generated
    """
    test_rights(ProjectAction.GENERATE, current_user.username, project.name)
    try:
        labels = None
        if request.scheme_name is not None:
            schemes = project.schemes.available()
            if request.scheme_name not in schemes:
                raise Exception(f"Scheme {request.scheme_name} does not exist")
            labels = schemes[request.scheme_name].labels
        return Generations.preview_postprocess(request.steps, request.raws, labels)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/pipelines/{pipeline_id}/start", dependencies=[Depends(verified_user)])
def start_run(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    request: GenRunRequest,
    pipeline_id: int = Path(ge=0),
) -> int:
    """
    Launch a pipeline on a dataset sample
    """
    test_rights(ProjectAction.GENERATE, current_user.username, project.name)
    try:
        pipeline = project.generations.get_pipeline(pipeline_id)
        sampling_scheme, labels = resolve_sampling(project, pipeline.scheme_name, request.scheme)
        run_id = project.start_generation(
            pipeline_id, request, sampling_scheme, labels, current_user.username
        )
        get_orchestrator().log_action(current_user.username, "START GENERATION RUN", project.name)
        return run_id
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate/runs", dependencies=[Depends(verified_user)])
def list_runs(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> list[GenRunOut]:
    """
    Runs of the project with their status and NA counts
    """
    test_rights(ProjectAction.GENERATE, current_user.username, project.name)
    try:
        return project.generations.runs()
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate/runs/{run_id}/elements", dependencies=[Depends(verified_user)])
def get_run_elements(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    run_id: int = Path(ge=0),
    limit: int = 100,
    offset: int = 0,
) -> TableOutModel:
    """
    Paginated outputs of a run
    """
    test_rights(ProjectAction.GENERATE, current_user.username, project.name)
    try:
        return project.generations.run_table(run_id, limit=limit, offset=offset)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/runs/delete", dependencies=[Depends(verified_user)])
def delete_run(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    run_id: int,
) -> None:
    """
    Delete a run and its output file
    """
    test_rights(ProjectAction.GENERATE, current_user.username, project.name)
    try:
        project.generations.delete_run(run_id)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
