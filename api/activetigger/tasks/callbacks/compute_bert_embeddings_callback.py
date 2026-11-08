from pathlib import Path
from typing import cast

import pandas as pd

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.compute_bert_embeddings_task import (
    ComputeBertEmbeddingsTask,
    ComputeBertEmbeddingsTaskResult,
)

from activetigger.project import Project

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class ComputeBertEmbeddingsCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = ComputeBertEmbeddingsTask.name

    # the on_complete method will be executed when a task succeeds
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    def on_complete_task_specific(
        cls,
        task_id: str,
        task_result: ComputeBertEmbeddingsTaskResult,
        project_manager: Project,
        process,
    ):
        embeddings_path = Path(task_result["embeddings_path"])
        try:
            results = pd.read_parquet(embeddings_path)
            project_manager.features.add(
                name=task_result["parameters"]["name"],
                kind="bert-embeddings",
                username=task_result["username"],
                parameters=cast(dict, task_result["parameters"]),
                new_content=results,
            )
        finally:
            embeddings_path.unlink(missing_ok=True)
