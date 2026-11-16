from typing import cast

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.predict_bert_task import (
    PredictBertTask,
    PredictBertTaskInputs,
    PredictBertTaskResult,
)

from activetigger.datamodels import EventsModel, LMComputing, TaskComputing
from activetigger.project import Project

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class PredictBertCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = PredictBertTask.name

    # the on_complete method will be executed when a task succeeds
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    def on_complete_task_specific(
        cls,
        task_id: str,
        task_result: PredictBertTaskResult,
        project_manager: Project,
        computing_task: TaskComputing[PredictBertTaskInputs],
    ):
        # predictions on the annotable dataset become features
        if "predict_annotable.parquet" in task_result["path"]:
            errors = project_manager.features.add_predictions(
                {"predict_" + computing_task.inputs.model_name: task_result["path"]}
            )
            for err in errors:
                project_manager.errors.add(err)
        project_manager.languagemodels.add_predicting_result(computing_task)
        project_manager.monitoring.close_process(
            computing_task.id, EventsModel(events=task_result["events"])
        )
