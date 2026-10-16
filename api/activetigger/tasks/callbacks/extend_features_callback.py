# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.extend_features_task import (
    ExtendFeaturesTask,
    ExtendFeaturesTaskResult,
)

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class ExtendFeaturesCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = ExtendFeaturesTask.name

    # the worker rewrote the features file: refresh the in-memory map
    @classmethod
    def on_complete_task_specific(
        cls, task_id: str, task_result: ExtendFeaturesTaskResult, project_manager, process
    ):
        features = project_manager.features
        features.map, features.n = features.get_map()
        if features.n != len(project_manager.data.index):
            project_manager._recover_features_after_failed_extension()

    # a failed extension leaves a features file with the pre-eval-set shape
    @classmethod
    def on_failure_task_specific(cls, task_id, task_report, project_manager):
        project_manager._recover_features_after_failed_extension()
