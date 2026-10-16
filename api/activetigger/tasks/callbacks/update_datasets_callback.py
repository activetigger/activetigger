from fastapi.encoders import jsonable_encoder

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.update_datasets_task import (
    UpdateDatasetsTask,
    UpdateDatasetsTaskResult,
)

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class UpdateDatasetsCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = UpdateDatasetsTask.name

    # the worker rewrote train.parquet: save the new params and reload the data
    @classmethod
    def on_complete_task_specific(
        cls, task_id: str, task_result: UpdateDatasetsTaskResult, project_manager, process
    ):
        params = project_manager.params
        params.cols_context = task_result["cols_context"]
        params.cols_text = task_result["cols_text"]
        if task_result["n_train"] is not None:
            params.n_train = task_result["n_train"]
        project_manager.db_manager.projects_service.update_project(
            params.project_slug, jsonable_encoder(params)
        )
        # reload the data in memory
        project_manager.data.load_dataset("all")
        # reset the features file and load the dataset again
        if task_result["reset_features"]:
            project_manager.features.reset_features_file()
            project_manager.bertopic.clear_bertopic()
            project_manager.projections.clear_projections()

    @classmethod
    def on_failure_task_specific(cls, task_id, task_report, project_manager):
        return None
