from typing import Annotated

from fastapi import APIRouter, Depends

from activetigger.app.dependencies import ServerAction, test_rights, verified_user
from activetigger.datamodels import MonitoringMetricsModel, UserInDBModel
from activetigger.orchestrator import get_orchestrator

router = APIRouter(tags=["monitoring"])


@router.get("/monitoring/metrics")
def get_monitoring_metrics(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> MonitoringMetricsModel:
    """
    Get monitoring metrics
    """
    test_rights(ServerAction.MANAGE_SERVER, current_user.username)
    metrics = get_orchestrator().monitoring.get_metrics()
    return metrics


@router.get("/monitoring/data")
def get_monitoring_data(
    current_user: Annotated[UserInDBModel, Depends(verified_user)], kind: str
) -> list:
    """
    Get monitoring data
    """
    test_rights(ServerAction.MANAGE_SERVER, current_user.username)
    data = get_orchestrator().monitoring.get_data(kind)
    return data.to_dict(orient="records")
