import numpy as np

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.compute_prompt_embedding_task import (
    ComputePromptEmbeddingTask,
    ComputePromptEmbeddingTaskResult,
)

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class ComputePromptEmbeddingCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = ComputePromptEmbeddingTask.name

    # append the prompt row (text, user, feature from the process entry) to prompts.parquet
    @classmethod
    def on_complete_task_specific(
        cls, task_id: str, task_result: ComputePromptEmbeddingTaskResult, project_manager, process
    ):
        if project_manager.prompts is None:
            return
        project_manager.prompts.receive_result(
            process, np.asarray(task_result["embedding"], dtype=float)
        )
