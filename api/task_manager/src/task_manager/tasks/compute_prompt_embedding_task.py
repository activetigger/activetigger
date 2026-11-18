from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel

from task_manager.active_tigger_task import AbortableActiveTiggerTask, QueueName
from task_manager.celery import celery_app
from task_manager.task_input import TaskInputs
from task_manager.task_names import TaskNames
from task_manager.tasks.compute_prompt_embedding import encode_prompt


class ComputePromptEmbeddingTaskInputs(TaskInputs[str]):
    prompt_id: str
    text: str
    model_name: str
    # feature kind: "sentence-embeddings" or "multimodal-embeddings"
    kind: str
    path_process: Path
    feature_name: str
    hf_name: str


# Return type must be a dict, a BaseModel would not be serialized by Celery
class ComputePromptEmbeddingTaskResult(TypedDict):
    project_slug: str
    prompt_id: str
    embedding: list[float]


class ComputePromptEmbeddingTask(AbortableActiveTiggerTask):
    name: TaskNames = "compute_prompt_embedding"
    # loads the feature's sentence-transformer, possibly a large VL model
    queue = QueueName.GPU


@celery_app.task(
    bind=True,
    name=ComputePromptEmbeddingTask.name,
    queue=ComputePromptEmbeddingTask.queue,
    base=ComputePromptEmbeddingTask,
    pydantic=True,
)
def compute_prompt_embedding(
    self: ComputePromptEmbeddingTask, inputs: ComputePromptEmbeddingTaskInputs
) -> ComputePromptEmbeddingTaskResult:

    vec = encode_prompt(
        inputs.text,
        model_name=inputs.model_name,
        kind=inputs.kind,
        is_aborted=self.is_aborted,
        update_progress=self.update_progress,
    )

    return {
        "project_slug": inputs.project_slug,
        "prompt_id": inputs.prompt_id,
        "embedding": [float(x) for x in vec],
    }
