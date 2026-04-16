from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from activetigger.app.dependencies import (
    ProjectAction,
    get_project,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    BertModelModel,
    ModelInformationsModel,
    QuickModelInModel,
    QuickModelOutModel,
    TextDatasetModel,
    UserInDBModel,
)
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project

router = APIRouter(tags=["models"])


@router.post("/models/quick/train", dependencies=[Depends(verified_user)])
def train_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    quickmodel: QuickModelInModel,
) -> None:
    """
    Compute quickmodel
    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    try:
        project.train_quickmodel(quickmodel, current_user.username)
        get_orchestrator().log_action(
            current_user.username, f"TRAIN SIMPLE MODEL {quickmodel.name}", project.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/quick/retrain", dependencies=[Depends(verified_user)])
def retrain_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
    name: str,
) -> None:
    """
    Retrain quickmodel
    """
    test_rights(ProjectAction.GET, current_user.username, project.name)
    try:
        project.retrain_quickmodel(name, scheme, current_user.username)
        get_orchestrator().log_action(
            current_user.username, f"RETRAIN SIMPLE MODEL {name}", project.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/quick/delete", dependencies=[Depends(verified_user)])
def delete_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str,
) -> None:
    """
    Delete quickmodel
    """
    test_rights(ProjectAction.DELETE, current_user.username, project.name)
    try:
        project.quickmodels.delete(name)
        get_orchestrator().log_action(
            current_user.username, f"DELETE SIMPLE MODEL + FEATURES: {name}", project.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/quick/rename", dependencies=[Depends(verified_user)])
def rename_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    former_name: str,
    new_name: str,
) -> None:
    """
    Rename quickmodel
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    try:
        project.quickmodels.rename(former_name, new_name)
        get_orchestrator().log_action(
            current_user.username,
            f"INFO RENAME QUICK MODEL: {former_name} -> {new_name}",
            project.name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/quick", dependencies=[Depends(verified_user)])
def get_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str,
) -> QuickModelOutModel | None:
    """
    Get available quickmodel by a name
    """
    test_rights(ProjectAction.GET, current_user.username, project.name)
    try:
        sm = project.quickmodels.get(name)
        return QuickModelOutModel(
            name=sm.name,
            model=sm.model_type,
            params=sm.model_params,
            features=sm.features,
            statistics_train=sm.statistics_train,
            statistics_test=sm.statistics_test,
            statistics_cv10=sm.statistics_cv10,
            balance_classes=sm.balance_classes,
            scheme=sm.scheme,
            username=sm.user,
            exclude_labels=sm.exclude_labels if hasattr(sm, "exclude_labels") else [],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/information", dependencies=[Depends(verified_user)])
def get_model_information(
    project: Annotated[Project, Depends(get_project)], name: str, kind: str
) -> ModelInformationsModel:
    """
    Get model information
    """
    try:
        if kind == "bert":
            return project.languagemodels.get_informations(name)
        elif kind == "quick":
            return project.quickmodels.get_informations(name)
        else:
            raise Exception(f"Model kind {kind} not recognized")
    except Exception as e:
        print(f"Erreur in /models/information:\n{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/predict", dependencies=[Depends(verified_user)])
def predict(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    model_name: str,
    scheme: str,
    kind: str,
    dataset_type: str = "annotable",
    batch_size: int = 32,
    external_dataset: TextDatasetModel | None = None,
) -> None:
    """
    Start prediction with a model
    - quick or bert model
    - types of dataset
    Manage specific cases for prediction

    TODO : optimize prediction on whole dataset
    TODO : manage prediction external/whole dataset for quick models

    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    try:
        # types of prediction
        if kind not in ["quick", "bert"]:
            raise Exception(f"Model kind {kind} not recognized")

        if dataset_type not in ["annotable", "external", "all"]:
            raise Exception(f"Dataset {dataset_type} not recognized")

        # managing the perimeter of the prediction
        datasets = None
        if dataset_type == "annotable":
            datasets = ["train"]
            if project.data.valid is not None:
                datasets.append("valid")
            if project.data.test is not None:
                datasets.append("test")
        if dataset_type == "external":
            if kind != "bert":
                raise Exception("External dataset prediction is only available for bert models")

        if kind == "bert":
            if dataset_type == "external" and external_dataset is None:
                raise Exception("External dataset must be provided for external prediction")
            if (
                dataset_type == "external"
                and external_dataset is not None
                and not project.data.get_path(external_dataset.filename).exists()
            ):
                raise HTTPException(
                    status_code=404,
                    detail=f"External dataset file {external_dataset.filename} not found",
                )
            project.start_language_model_prediction(
                username=current_user.username,
                dataset_type=dataset_type,
                datasets=datasets,
                scheme_name=scheme,
                model_name=model_name,
                external_dataset=external_dataset,
                batch_size=batch_size,
            )

        if kind == "quick":
            if datasets is None:
                raise Exception("Dataset parameter must be specified for quick model prediction")
            project.start_quick_model_prediction(
                username=current_user.username,
                dataset_type=dataset_type,
                datasets=datasets,
                scheme_name=scheme,
                model_name=model_name,
            )
        get_orchestrator().log_action(
            current_user.username,
            f"PREDICT MODEL: {model_name} - {kind} DATASET: {dataset_type}",
            project.name,
        )
    except Exception as e:
        print(f"ERROR in /models/predict\n{e}\n\n")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/bert/train", dependencies=[Depends(verified_user)])
def post_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bert: BertModelModel,
) -> None:
    """
    Compute bertmodel
    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    try:
        orchestrator = get_orchestrator()
        if not orchestrator.available_storage(current_user.username):
            raise HTTPException(
                status_code=403,
                detail="Storage limit exceeded. Please delete models or contact the administrator.",
            )
        project.start_languagemodel_training(
            bert=bert,
            username=current_user.username,
        )
        orchestrator.log_action(current_user.username, f"TRAIN MODEL: {bert.name}", project.name)
        return None

    except Exception as e:
        print(f"ERREUR : \n{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/bert/delete", dependencies=[Depends(verified_user)])
def delete_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bert_name: str,
) -> None:
    """
    Delete trained bert model
    # TODO : check the replace
    """
    test_rights(ProjectAction.DELETE, current_user.username, project.name)
    try:
        # delete the model
        project.languagemodels.delete(bert_name)
        # delete the features associated with the model
        for f in [i for i in project.features.map.keys() if bert_name.replace("__", "_") in i]:
            project.features.delete(f)
        get_orchestrator().log_action(
            current_user.username, f"DELETE MODEL + FEATURES: {bert_name}", project.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/bert/rename", dependencies=[Depends(verified_user)])
def rename_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    former_name: str,
    new_name: str,
) -> None:
    """
    Rename bertmodel
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    try:
        project.languagemodels.rename(former_name, new_name)
        get_orchestrator().log_action(
            current_user.username,
            f"INFO RENAME MODEL: {former_name} -> {new_name}",
            project.name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
