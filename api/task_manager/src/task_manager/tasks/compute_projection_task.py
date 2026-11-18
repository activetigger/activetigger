from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from activetigger.datamodels import ProjectionParametersModel, TaskComputing
from pydantic import BaseModel

from task_manager.active_tigger_task import AbortableActiveTiggerTask, QueueName
from task_manager.celery import celery_app
from task_manager.task_input import TaskInputs
from task_manager.task_names import TaskNames
from task_manager.tasks.compute_projection import compute_projection


class ComputeProjectionTaskInputs(TaskInputs[str]):
    project_slug: str
    name: str
    # pickled features DataFrame, removed by the worker
    features_path: Path
    # directory receiving the coordinates parquet
    path_process: Path
    normalize_features: bool = False
    projection: ProjectionParametersModel


# Return type must be a dict, a BaseModel would not be serialized by Celery
class ComputeProjectionTaskResult(TypedDict):
    project_slug: str
    name: str
    # coordinates parquet written by the worker, read then removed by the callback
    coordinates_path: str


class ComputeProjectionTask(AbortableActiveTiggerTask):
    name: TaskNames = "compute_projection"
    # umap/tsne on CPU (cuML picked up automatically when installed)
    queue = QueueName.CPU

    @classmethod
    def clean_temp_files(cls, computing_task: TaskComputing[ComputeProjectionTaskInputs]):
        computing_task.inputs.features_path.unlink(missing_ok=True)


@celery_app.task(
    bind=True,
    name=ComputeProjectionTask.name,
    queue=ComputeProjectionTask.queue,
    base=ComputeProjectionTask,
    pydantic=True,
)
def compute_projection_task(
    self: ComputeProjectionTask, inputs: ComputeProjectionTaskInputs
) -> ComputeProjectionTaskResult:
    try:
        features = pd.read_pickle(inputs.features_path)
        # one-shot fit: abort only before starting
        if self.is_aborted():
            raise Exception("Process interrupted by user")
        coordinates = compute_projection(
            features,
            kind=inputs.projection.method,
            params=inputs.projection.parameters,
            normalize_features=inputs.normalize_features,
        )
    finally:
        inputs.features_path.unlink(missing_ok=True)
    coordinates_path = inputs.path_process.joinpath(f"coordinates_{self.request.id}.parquet")
    # parquet requires string column names; axis columns are integers
    coordinates.columns = coordinates.columns.astype(str)
    coordinates.to_parquet(coordinates_path)
    return {
        "project_slug": inputs.project_slug,
        "name": inputs.name,
        "coordinates_path": str(coordinates_path),
    }
