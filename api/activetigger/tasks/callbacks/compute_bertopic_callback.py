import shutil
from typing import cast

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.compute_bertopic_task import (
    ComputeBertopicTask,
    ComputeBertopicTaskResult,
)

from activetigger.datamodels import BertopicComputing, EventsModel

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class ComputeBertopicCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = ComputeBertopicTask.name

    # the on_complete method will be executed when a task succeeds
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    def on_complete_task_specific(
        cls, task_id: str, task_result: ComputeBertopicTaskResult, project_manager, process
    ):
        bertopic_model = cast(BertopicComputing, process)
        events = EventsModel(events=task_result["events"])
        project_manager.bertopic.add(bertopic_model)
        project_manager.monitoring.close_process(bertopic_model.unique_id, events)

    # a run killed before its own cleanup must not leave a half-written run directory
    @classmethod
    def on_failure_task_specific(cls, task_id, task_report, project_manager):
        path_run = project_manager.bertopic.path.joinpath("runs", task_report.task_args[0]["name"])
        if path_run.exists():
            shutil.rmtree(path_run)
