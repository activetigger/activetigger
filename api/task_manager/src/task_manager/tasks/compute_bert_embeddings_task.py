from pathlib import Path
from typing import Literal, TypedDict

import pandas as pd
from pydantic import BaseModel

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.compute_bert_embeddings import encode_bert_embeddings


class ComputeBertEmbeddingsTaskParameters(TypedDict):
    model: str
    pooling: Literal["mean", "cls"]
    name: str
    kind: Literal["bert-embeddings"]
    username: str
    max_length_tokens: int
    batch_size: int


# Return type must be a dict, a BaseModel would not be serialized by Celery
class ComputeBertEmbeddingsTaskResult(TypedDict):
    username: str
    project_slug: str
    embeddings_path: str
    parameters: ComputeBertEmbeddingsTaskParameters


class ComputeBertEmbeddingsTaskInput(BaseModel):
    feature_name: str
    username: str
    project_slug: str
    texts_path: Path
    path_process: Path
    model_name: str
    model_dir: Path
    pooling: Literal["mean", "cls"] = "mean"
    batch_size: int = 32
    max_tokens: int = 512
    min_gpu: int = 6
    path_progress: Path | None = None


class ComputeBertEmbeddingsTask(AbortableAutoCallbackTask):
    name = "compute bert embeddings"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU


@celery_app.task(
    bind=True,
    name=ComputeBertEmbeddingsTask.name,
    queue=ComputeBertEmbeddingsTask.queue,
    base=ComputeBertEmbeddingsTask,
    pydantic=True,
)
def compute_bert_embeddings(
    self: ComputeBertEmbeddingsTask, props: ComputeBertEmbeddingsTaskInput
) -> ComputeBertEmbeddingsTaskResult:
    # the API reads the progress file named after the task id
    path_progress = props.path_progress or props.path_process.joinpath(self.request.id)
    try:
        texts = pd.read_pickle(props.texts_path)
        emb = encode_bert_embeddings(
            texts,
            model_path=props.model_dir.joinpath(props.model_name),
            path_progress=path_progress,
            pooling=props.pooling,
            batch_size=props.batch_size,
            max_tokens=props.max_tokens,
            min_gpu=props.min_gpu,
            is_aborted=self.is_aborted,
        )
    finally:
        path_progress.unlink(missing_ok=True)
        props.texts_path.unlink(missing_ok=True)
    # one file per task: several bert-embeddings features can run concurrently
    embeddings_path = props.path_process.joinpath(f"bert_{self.request.id}.parquet")
    emb.to_parquet(embeddings_path)
    return {
        "project_slug": props.project_slug,
        "username": props.username,
        "embeddings_path": str(embeddings_path),
        "parameters": {
            "model": props.model_name,
            "pooling": props.pooling,
            "name": props.feature_name,
            "kind": "bert-embeddings",
            "username": props.username,
            "max_length_tokens": props.max_tokens,
            "batch_size": props.batch_size,
        },
    }
