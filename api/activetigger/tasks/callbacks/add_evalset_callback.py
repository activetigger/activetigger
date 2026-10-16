import json
from pathlib import Path

from fastapi.encoders import jsonable_encoder

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.add_evalset_task import AddEvalsetTask, AddEvalsetTaskResult

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class AddEvalsetCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = AddEvalsetTask.name

    # the worker wrote test/valid.parquet: insert the labels, update params, reload
    @classmethod
    def on_complete_task_specific(
        cls, task_id: str, task_result: AddEvalsetTaskResult, project_manager, process
    ):
        eval_dataset = task_result["dataset"]
        user_name = task_result["username"]
        annotations_path = Path(task_result["annotations_path"])
        try:
            with open(annotations_path) as f:
                schemes_elements = json.load(f)
        finally:
            annotations_path.unlink(missing_ok=True)
        for scheme_name, elements in schemes_elements:
            if elements:
                project_manager.db_manager.projects_service.add_annotations(
                    eval_dataset,
                    user_name,
                    task_result["project_slug"],
                    scheme_name,
                    elements,
                )
        # update params with the new evalset
        setattr(project_manager.params, eval_dataset, True)
        setattr(project_manager.params, f"n_{eval_dataset}", task_result["n_rows"])
        project_manager.db_manager.projects_service.update_project(
            project_manager.params.project_slug, jsonable_encoder(project_manager.params)
        )
        # load the new eval set before refreshing features so the
        # features file covers the full train+valid+test index
        project_manager.data.load_dataset(eval_dataset)
        project_manager._refresh_features_after_evalset(eval_dataset, user_name)

    # the frontend matches the error on "add_evalset_<dataset>" (old in-process message)
    @classmethod
    def on_failure_task_specific(cls, task_id, task_report, project_manager):
        dataset = task_report.task_args[0]["dataset"]
        return f"Error in add_evalset_{dataset} : {task_report.exception}"
