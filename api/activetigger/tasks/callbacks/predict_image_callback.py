from typing import cast

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.predict_image_task import (
    PredictImageTask,
    PredictImageTaskResult,
)

from activetigger.datamodels import LMComputing

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class PredictImageCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = PredictImageTask.name

    # the on_complete method will be executed when a task succeeds
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    def on_complete_task_specific(
        cls, task_id: str, task_result: PredictImageTaskResult, project_manager, process
    ):
        if project_manager.imagemodels is None:
            raise Exception("Image models are not available for this project")
        prediction = cast(LMComputing, process)
        # predictions on the annotable dataset become features
        if "predict_annotable.parquet" in task_result["path"]:
            errors = project_manager.features.add_predictions(
                {"predict_" + prediction.model_name: task_result["path"]}
            )
            for err in errors:
                project_manager.errors.add(err)
        project_manager.imagemodels.add(prediction)
