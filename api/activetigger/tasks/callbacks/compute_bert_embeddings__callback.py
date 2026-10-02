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
    def on_complete_task_specific(cls, task_id:str, task_result:ComputeBertEmbeddingsTaskResult, project_manager: Project, process):
        # load DataFrame from filesystem
        results = pd.read_parquet(task_result['embeddings_path'])
        # add Feature to project manager
        project_manager.features.add(
                name=task_result['parameters']['name'],
                kind="compute_bert_features",
                username= task_result['username'],
                parameters=cast(dict, task_result['parameters']),
                new_content=results,
        )
    
