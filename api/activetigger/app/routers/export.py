from io import StringIO
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
)
from fastapi.responses import FileResponse
from fastapi.responses import Response as FastAPIResponse

from activetigger.app.dependencies import ProjectAction, get_project, test_rights, verified_user
from activetigger.config import config
from activetigger.datamodels import (
    ExportGenerationsParams,
    ProjectStaticFiles,
    UserInDBModel,
)
from activetigger.project import Project

router = APIRouter(tags=["export"])


@router.get("/export/data", dependencies=[Depends(verified_user)])
def export_data(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
    format: str,
    dataset: str = "train",
) -> FileResponse:
    """
    Export labelled data
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.export_data(format=format, scheme=scheme, dataset=dataset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/features", dependencies=[Depends(verified_user)])
def export_features(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    features: list = Query(),
    format: str = Query(),
) -> FileResponse:
    """
    Export features
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.export_features(features=features, format=format)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/projection", dependencies=[Depends(verified_user)])
def export_projection(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    format: str = Query(),
) -> FileResponse:
    """
    Export features
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.projections.export(user_name=current_user.username, format=format)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/prediction", dependencies=[Depends(verified_user)])
def export_prediction(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    format: str = Query(),
    name: str = Query(),
    dataset: str = Query("all"),
) -> FileResponse:
    """
    Export annotations
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.languagemodels.export_prediction(
            name=name,
            file_name=f"predict_{dataset}.parquet",
            format=format,
            col_id=project.params.col_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/bert", dependencies=[Depends(verified_user)])
def export_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(),
) -> FastAPIResponse:
    """
    Export fine-tuned BERT model - file with redirect with nginx
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        file_path = project.languagemodels.export_bert(name=name)
        return FastAPIResponse(
            status_code=200,
            headers={
                "X-Accel-Redirect": f"/privatefiles/{file_path.path}",
                "Content-Disposition": f'attachment; filename="{name}.tar.gz"',
                "Content-Type": "application/octet-stream",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/raw", dependencies=[Depends(verified_user)])
def export_raw(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> FastAPIResponse:
    """
    Export raw data of the project
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        file_path = project.export_raw(project.name)
        return FastAPIResponse(
            status_code=200,
            headers={
                "X-Accel-Redirect": f"/privatefiles/{file_path.path}",
                "Content-Disposition": f'attachment; filename="{project.name}.parquet"',
                "Content-Type": "application/octet-stream",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# This is a temporary fix for the sqlite issue that will send static links only when sql database (without nginx redirection)
@router.get("/export/static", dependencies=[Depends(verified_user)])
def export_static(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    model: str | None = Query(default=None),
) -> ProjectStaticFiles | None:
    """
    Get static links of the project
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        # don't return nothing if not direct with sqlite
        if "sqlite" not in config.database_url:
            return None
        r = ProjectStaticFiles(dataset=project.export_raw(project.name))
        if model is not None:
            r.model = project.languagemodels.export_bert(name=model)
        return r
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export/generations", dependencies=[Depends(verified_user)])
def export_generations(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    params: ExportGenerationsParams,
) -> Response:
    """
    Export annotations
    """
    try:
        table = project.export_generations(
            project_slug=project.name,
            username=current_user.username,
            params=params,
        )

        # convert to payload
        output = StringIO()
        table.to_csv(output, index=True)
        csv_data = output.getvalue()
        output.close()
        headers = {
            "Content-Disposition": 'attachment; filename="data.csv"',
            "Content-Type": "text/csv",
        }

        return Response(content=csv_data, media_type="text/csv", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/bertopic/topics", dependencies=[Depends(verified_user)])
def export_bertopics_topics(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> FileResponse:
    """
    Export annotations
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.bertopic.export_topics(name=name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/bertopic/clusters", dependencies=[Depends(verified_user)])
def export_bertopics_clusters(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> FileResponse:
    """
    Export annotations
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.bertopic.export_clusters(name=name, col_id=project.params.col_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/bertopic/report", dependencies=[Depends(verified_user)])
def export_bertopics_report(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> FileResponse:
    """
    Export annotations
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.bertopic.export_report(name=name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/bertopic/embeddings", dependencies=[Depends(verified_user)])
def export_bertopics_embeddings(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> FileResponse:
    """
    Export annotations
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.bertopic.export_embeddings(name=name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @router.get("/export/prediction/quickmodel", dependencies=[Depends(verified_user)])
# def export_quickmodel_predictions(
#     project: Annotated[Project, Depends(get_project)],
#     current_user: Annotated[UserInDBModel, Depends(verified_user)],
#     name: str,
#     format: str = "csv",
# ) -> StreamingResponse:
#     """
#     Export prediction quickmodel for the project/user/scheme if any
#     """
#     test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
#     try:
#         output, headers = project.quickmodels.export_prediction(name, format)
#         return StreamingResponse(output, headers=headers)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
