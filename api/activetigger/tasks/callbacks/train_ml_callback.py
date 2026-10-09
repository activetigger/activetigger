from typing import cast

from activetigger.datamodels import EventsModel, QuickModelComputing

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.train_ml_task import TrainMLTask, TrainMLTaskResult


class TrainMLCallback(TaskCallback):
    task_name = TrainMLTask.name

    @classmethod
    def on_complete_task_specific(
        cls, task_id: str, task_result: TrainMLTaskResult, project_manager, process
    ):
        model = cast(QuickModelComputing, process)
        events = EventsModel(events=task_result["events"])
        project_manager.quickmodels.add(model)
        project_manager.monitoring.close_process(model.unique_id, events)
