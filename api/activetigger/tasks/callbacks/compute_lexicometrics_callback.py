import json
from pathlib import Path

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.compute_lexicometrics_task import (
    ComputeLexicometricsTask,
    ComputeLexicometricsTaskResult,
)

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class ComputeLexicometricsCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = ComputeLexicometricsTask.name

    # load the statistics written by the worker and store them on the project
    @classmethod
    def on_complete_task_specific(
        cls, task_id: str, task_result: ComputeLexicometricsTaskResult, project_manager, process
    ):
        statistics_path = Path(task_result["statistics_path"])
        with open(statistics_path) as f:
            results = json.load(f)
        project_manager.lexicometrics.add(process, results)
        statistics_path.unlink(missing_ok=True)
