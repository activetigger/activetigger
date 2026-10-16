from pathlib import Path

import pandas as pd

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.compute_projection_task import (
    ComputeProjectionTask,
    ComputeProjectionTaskResult,
)

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class ComputeProjectionCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = ComputeProjectionTask.name

    # load the coordinates written by the worker and register the projection
    @classmethod
    def on_complete_task_specific(
        cls, task_id: str, task_result: ComputeProjectionTaskResult, project_manager, process
    ):
        coordinates_path = Path(task_result["coordinates_path"])
        coordinates = pd.read_parquet(coordinates_path)
        # restore the integer axis column names expected by Projections.add
        coordinates.columns = [int(c) if c.isdigit() else c for c in coordinates.columns]
        project_manager.projections.add(process, coordinates)
        coordinates_path.unlink(missing_ok=True)
