from typing import cast

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.train_ner_task import (
    TrainNerTask,
    TrainNerTaskResult,
)

from activetigger.datamodels import EventsModel, LMComputing

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class TrainNerCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = TrainNerTask.name

    # the on_complete method will be executed when a task succeeds
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    def on_complete_task_specific(
        cls, task_id: str, task_result: TrainNerTaskResult, project_manager, process
    ):
        if project_manager.nermodels is None:
            raise Exception("NER models are not available for this project")
        model = cast(LMComputing, process)
        events = EventsModel(events=task_result["events"])
        project_manager.nermodels.add(model)
        project_manager.monitoring.close_process(model.unique_id, events)

    # a failed training must not leave a half-registered model in the database
    @classmethod
    def on_failure_task_specific(cls, task_id, task_report, project_manager):
        model_name = task_report.task_args[0]["model_name"]
        project_manager.db_manager.language_models_service.delete_model(
            project_manager.name, model_name
        )
