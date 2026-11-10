# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.compute_similarity_with_features_task import (
    ComputeSimilarityWithFeaturesTask,
    ComputeSimilarityWithFeaturesTaskResult,
)

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class ComputeSimilarityWithFeaturesCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = ComputeSimilarityWithFeaturesTask.name

    # the similarity parquet is on disk; availability is derived from the file
    @classmethod
    def on_complete_task_specific(
        cls,
        task_id: str,
        task_result: ComputeSimilarityWithFeaturesTaskResult,
        project_manager,
        process,
    ):
        pass
