# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.predict_with_features_task import (
    PredictWithFeaturesTask,
    PredictWithFeaturesTaskResult,
)

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class PredictWithFeaturesCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = PredictWithFeaturesTask.name

    # the proba parquet is already on disk; only flag the quickmodel row
    @classmethod
    def on_complete_task_specific(
        cls, task_id: str, task_result: PredictWithFeaturesTaskResult, project_manager, process
    ):
        project_manager.quickmodels.mark_prediction_done(
            task_result["name"], task_result["dataset"]
        )
