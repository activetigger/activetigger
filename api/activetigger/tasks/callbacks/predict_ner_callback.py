from typing import cast

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.predict_ner_task import (
    PredictNerTask,
    PredictNerTaskInputs,
    PredictNerTaskResult,
)

from activetigger.datamodels import EventsModel, TaskComputing
from activetigger.project import Project

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class PredictNerCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = PredictNerTask.name

    # the on_complete method will be executed when a task succeeds
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    def on_complete_task_specific(
        cls,
        task_id: str,
        task_result: PredictNerTaskResult,
        project_manager: Project | None,
        computing_task: TaskComputing[PredictNerTaskInputs],
    ):
        # project scope task always receive a populated project_manager
        project_manager = cast(Project, project_manager)
        if project_manager.nermodels is None:
            raise Exception("NER models are not available for this project")

        # predictions on the annotable dataset become features
        if "predict_annotable.parquet" in task_result["path"]:
            errors = project_manager.features.add_predictions(
                {"predict_" + computing_task.inputs.model_name: task_result["path"]}
            )
            for err in errors:
                project_manager.errors.add(err)
        project_manager.nermodels.add_prediction(computing_task)
