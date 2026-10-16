import pickle
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel
from scipy.stats import entropy

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.compute_features_from_specs import (
    FeatureSpec,
    compute_features_from_specs,
    write_progress,
)


class PredictWithFeaturesTaskInput(BaseModel):
    project_slug: str
    # quickmodel name and dataset predicted, echoed to the callback
    name: str
    dataset: str
    specs: list[FeatureSpec]
    # pickled Series shared by every spec (texts or image paths)
    data_path: Path
    path_process: Path
    path_models: Path
    language: str
    # directory holding model.pkl; the proba parquet is written there too
    path_quickmodel: Path
    file_name: str


# Return type must be a dict, a BaseModel would not be serialized by Celery
class PredictWithFeaturesTaskResult(TypedDict):
    project_slug: str
    name: str
    dataset: str
    output_path: str


class PredictWithFeaturesTask(AbortableAutoCallbackTask):
    name = "predict with features"
    # the embeddings may be recomputed here, but sklearn prediction is CPU
    queue = QueueName.CPU


def predict_proba(model, features: DataFrame) -> DataFrame:
    """
    Apply a trained quickmodel to the computed feature matrix: label columns
    + `prediction` + `entropy` (+ per-label entropy), like the training-time proba.
    """
    expected = list(model.feature_names_in_)
    missing = [c for c in expected if c not in features.columns]
    if missing:
        raise ValueError(
            f"Feature columns required by quickmodel are missing: {missing[:5]}"
            + ("..." if len(missing) > 5 else "")
        )
    X = features.reindex(columns=expected)

    proba_values = model.predict_proba(X)
    labels = list(model.classes_)
    proba = pd.DataFrame(proba_values, columns=labels, index=X.index)
    proba["prediction"] = proba[labels].idxmax(axis=1)
    proba["entropy"] = entropy(proba_values, axis=1)
    for label in labels:
        try:
            prob_A_not_A = np.column_stack([proba[label], 1 - proba[label]])
            proba[f"entropy-{label}"] = entropy(prob_A_not_A, axis=1)
        except Exception:
            pass
    return proba


@celery_app.task(
    bind=True,
    name=PredictWithFeaturesTask.name,
    queue=PredictWithFeaturesTask.queue,
    base=PredictWithFeaturesTask,
    pydantic=True,
)
def predict_with_features(
    self: PredictWithFeaturesTask, inputs: PredictWithFeaturesTaskInput
) -> PredictWithFeaturesTaskResult:
    # the API reads the progress file named after the task id
    path_progress = inputs.path_process.joinpath(self.request.id)
    try:
        data = pd.read_pickle(inputs.data_path)
        with open(inputs.path_quickmodel.joinpath("model.pkl"), "rb") as f:
            model = pickle.load(f).model
        # one extra progress unit for the final prediction step
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
        proba = predict_proba(model, features)
        inputs.path_quickmodel.mkdir(parents=True, exist_ok=True)
        output_path = inputs.path_quickmodel.joinpath(inputs.file_name)
        proba.to_parquet(output_path, index=True)
        write_progress(path_progress, 100.0)
    finally:
        path_progress.unlink(missing_ok=True)
        inputs.data_path.unlink(missing_ok=True)
    return {
        "project_slug": inputs.project_slug,
        "name": inputs.name,
        "dataset": inputs.dataset,
        "output_path": str(output_path),
    }
