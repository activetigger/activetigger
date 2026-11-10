import json
from pathlib import Path
from typing import Any, TypedDict

from pydantic import BaseModel

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.compute_lexicometrics import compute_lexicometrics


class ComputeLexicometricsTaskInput(BaseModel):
    project_slug: str
    username: str
    # project directory holding the train file
    path_data: Path
    train_file: str = "train.parquet"
    col_text: str = "text"
    language: str = "en"
    tokenizer_name: str = "bert-base-multilingual-cased"
    n_most_frequent: int = 100
    tfidf_n_words: int = 300
    tfidf_n_docs_per_word: int = 25
    tfidf_n_words_per_doc: int = 10
    tfidf_min_term_freq: int = 5
    tfidf_max_documents: int = 10000


# Return type must be a dict, a BaseModel would not be serialized by Celery
class ComputeLexicometricsTaskResult(TypedDict):
    project_slug: str
    username: str
    # JSON statistics written by the worker, read then removed by the callback
    statistics_path: str


class ComputeLexicometricsTask(AbortableAutoCallbackTask):
    name = "compute lexicometrics"
    # sklearn + tokenizer counting, never uses the GPU
    queue = QueueName.CPU


@celery_app.task(
    bind=True,
    name=ComputeLexicometricsTask.name,
    queue=ComputeLexicometricsTask.queue,
    base=ComputeLexicometricsTask,
    pydantic=True,
)
def compute_lexicometrics_task(
    self: ComputeLexicometricsTask, inputs: ComputeLexicometricsTaskInput
) -> ComputeLexicometricsTaskResult:
    statistics: dict[str, Any] = compute_lexicometrics(
        inputs.path_data,
        train_file=inputs.train_file,
        col_text=inputs.col_text,
        language=inputs.language,
        tokenizer_name=inputs.tokenizer_name,
        n_most_frequent=inputs.n_most_frequent,
        tfidf_n_words=inputs.tfidf_n_words,
        tfidf_n_docs_per_word=inputs.tfidf_n_docs_per_word,
        tfidf_n_words_per_doc=inputs.tfidf_n_words_per_doc,
        tfidf_min_term_freq=inputs.tfidf_min_term_freq,
        tfidf_max_documents=inputs.tfidf_max_documents,
        is_aborted=self.is_aborted,
    )
    # kept out of the result payload: tfidf slices can weigh a few MB
    statistics_path = inputs.path_data.joinpath(f"lexicometrics_{self.request.id}.json")
    with open(statistics_path, "w") as f:
        json.dump(statistics, f)
    return {
        "project_slug": inputs.project_slug,
        "username": inputs.username,
        "statistics_path": str(statistics_path),
    }
