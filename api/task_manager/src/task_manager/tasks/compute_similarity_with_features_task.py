from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.compute_features_from_specs import (
    FeatureSpec,
    compute_features_from_specs,
    write_progress,
)


class ComputeSimilarityWithFeaturesTaskInput(BaseModel):
    project_slug: str
    prompt_id: str
    dataset: str
    specs: list[FeatureSpec]
    # pickled Series shared by every spec (texts or image paths)
    data_path: Path
    path_process: Path
    path_models: Path
    language: str
    prompt_vector: list[float]
    path_output: Path
    file_name: str


# Return type must be a dict, a BaseModel would not be serialized by Celery
class ComputeSimilarityWithFeaturesTaskResult(TypedDict):
    project_slug: str
    prompt_id: str
    dataset: str
    output_path: str


class ComputeSimilarityWithFeaturesTask(AbortableAutoCallbackTask):
    name = "compute similarity with features"
    # recomputes an embedding feature on the complete dataset
    queue = QueueName.GPU


def cosine_similarity_ranking(features: DataFrame, prompt_vector: np.ndarray) -> DataFrame:
    """
    Cosine similarity between the prompt vector and each row of the embedding
    matrix (same math as Prompts.get_ranking), with a dense rank.
    """
    mat = features.to_numpy(dtype=float)
    if mat.shape[1] != prompt_vector.shape[0]:
        raise ValueError(
            f"Dimension mismatch between prompt ({prompt_vector.shape[0]}) "
            f"and computed feature ({mat.shape[1]}). "
            "The feature may have been recomputed with a different model."
        )
    row_norms = np.linalg.norm(mat, axis=1)
    prompt_norm = float(np.linalg.norm(prompt_vector))
    sims = (mat @ prompt_vector) / (row_norms * prompt_norm + 1e-12)
    out = pd.DataFrame({"similarity": sims}, index=features.index)
    out["rank"] = out["similarity"].rank(ascending=False, method="first").astype(int)
    return out


@celery_app.task(
    bind=True,
    name=ComputeSimilarityWithFeaturesTask.name,
    queue=ComputeSimilarityWithFeaturesTask.queue,
    base=ComputeSimilarityWithFeaturesTask,
    pydantic=True,
)
def compute_similarity_with_features(
    self: ComputeSimilarityWithFeaturesTask, inputs: ComputeSimilarityWithFeaturesTaskInput
) -> ComputeSimilarityWithFeaturesTaskResult:
    # the API reads the progress file named after the task id
    path_progress = inputs.path_process.joinpath(self.request.id)
    try:
        data = pd.read_pickle(inputs.data_path)
        # one extra progress unit for the final similarity step
        features = compute_features_from_specs(
            inputs.specs,
            data,
            path_process=inputs.path_process,
            path_models=inputs.path_models,
            language=inputs.language,
            path_progress=path_progress,
            is_aborted=self.is_aborted,
            total=len(inputs.specs) + 1,
        )
        prompt_vector = np.asarray(inputs.prompt_vector, dtype=float).reshape(-1)
        similarity = cosine_similarity_ranking(features, prompt_vector)
        inputs.path_output.mkdir(parents=True, exist_ok=True)
        output_path = inputs.path_output.joinpath(inputs.file_name)
        similarity.to_parquet(output_path, index=True)
        write_progress(path_progress, 100.0)
    finally:
        path_progress.unlink(missing_ok=True)
        inputs.data_path.unlink(missing_ok=True)
    return {
        "project_slug": inputs.project_slug,
        "prompt_id": inputs.prompt_id,
        "dataset": inputs.dataset,
        "output_path": str(output_path),
    }
