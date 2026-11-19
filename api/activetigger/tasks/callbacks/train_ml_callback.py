from typing import cast

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.train_ml_task import TrainMLMultiClassInputs, TrainMLTask, TrainMLTaskResult

from activetigger.datamodels import EventsModel, QuickModelComputing, TaskComputing
from activetigger.project import Project

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class TrainMLCallback(TaskCallback):
    task_name = TrainMLTask.name

    @classmethod
    def on_complete_task_specific(
        cls,
        task_id: str,
        task_result: TrainMLTaskResult,
        project_manager: Project | None,
        computing_task: TaskComputing[TrainMLMultiClassInputs],
    ):
        # project scope task always receive a populated project_manager
        project_manager = cast(Project, project_manager)
        project_manager.quickmodels.add(computing_task)
