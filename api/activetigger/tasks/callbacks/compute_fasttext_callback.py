from os import unlink

import pandas as pd

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.compute_fasttext_task import (
    ComputeFasttextTask,
    ComputeFasttextTaskResult,
)

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class ComputeFasttextCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = ComputeFasttextTask.name

    # the on_complete method will be executed when a task succeeds
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    def on_complete_task_specific(
        cls, task_id: str, task_result: ComputeFasttextTaskResult, project_manager, process
    ):
        # load the embeddings written by the worker and register them as a feature
        results = pd.read_parquet(task_result["features_path"])
        project_manager.features.add(
            name=task_result["parameters"]["name"],
            kind="fasttext",
            username=task_result["username"],
            parameters=task_result["parameters"],
            new_content=results,
        )
        unlink(task_result["features_path"])
