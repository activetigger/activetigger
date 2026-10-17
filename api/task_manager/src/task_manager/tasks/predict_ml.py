import json
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from activetigger.datamodels import MLStatisticsModel, QuickModelComputed
from activetigger.functions import get_metrics_multiclass
from activetigger.monitoring import TaskTimer
from pydantic import BaseModel


class PredictMLTaskInput(BaseModel):
    path: Path  # model directory (model.pkl, training_data.parquet)
    path_df: Path
    project_slug: str
    model_name: str
    col_dataset: str
    col_features: list[str]
    file_name: str
    col_label: str | None = None
    col_text: str | None = None
    statistics: list[str] | None = None
    exclude_labels: list[str] = []


# Return type must be a dict, a BaseModel would not be serialized by Celery
class PredictMLResult(TypedDict):
    path: str
    metrics: dict[str, Any] | None
    events: dict[str, dict[str, str | None]]


class PredictMLMultiClass:
    """
    Predict with a sklearn model
    """

    kind = "predict_model"

    def __init__(self, unique_id: str, inputs: PredictMLTaskInput, is_aborted: Callable[[], bool]):
        self.path = inputs.path
        self.path_df = inputs.path_df
        self.file_name = inputs.file_name
        self.unique_id = unique_id
        self.statistics = inputs.statistics
        self.col_label = inputs.col_label
        self.col_text = inputs.col_text
        self.col_dataset = inputs.col_dataset
        self.col_features = inputs.col_features
        self.exclude_labels = inputs.exclude_labels
        self.is_aborted = is_aborted
        if self.statistics is not None and self.col_label is None:
            raise ValueError("Labels must be provided to compute statistics")

    def __load_model(self):
        with open(self.path / "model.pkl", "rb") as file:
            sm: QuickModelComputed = pickle.load(file)
        return sm.model

    def __check_data_and_features(self, df: pd.DataFrame, expected_cols) -> None:
        """Check that the dataframe contains the dataset column and the features
        the model was trained on"""
        if self.col_dataset not in df.columns:
            raise ValueError(f"Dataset column {self.col_dataset} not in dataframe")

        if not all(col in df.columns for col in self.col_features):
            raise ValueError("Some feature columns are not in dataframe")

        missing = set(expected_cols) - set(self.col_features)
        extra = set(self.col_features) - set(expected_cols)
        if len(missing) > 0 or len(extra) > 0:
            raise ValueError(f"Feature mismatch. Missing: {missing}, Extra: {extra}")

    def __compute_metrics(
        self, Y_pred_full: pd.DataFrame, statistics: list[str]
    ) -> dict[str, MLStatisticsModel]:
        """Compute the metrics for the whole train dataset as well as the
        train_valid dataset"""
        metrics: dict[str, MLStatisticsModel] = {}

        # Compute the statistics for each dataset
        for dataset in statistics:
            filter_dataset = Y_pred_full["dataset"] == dataset
            if filter_dataset.sum() < 5:
                # TODO: Warn user that there are not enough elements to compute statistics
                continue
            sub_Y_pred_full = Y_pred_full[filter_dataset]
            metrics[dataset] = get_metrics_multiclass(
                sub_Y_pred_full["predicted_label"],
                sub_Y_pred_full["prediction"],
                texts=sub_Y_pred_full["text"] if self.col_text else None,
            )

        # Compute the statistics on elements of the trainset not used during training ("out of sample")
        index_of_texts_used_during_training = pd.read_parquet(
            self.path.joinpath("training_data.parquet"), columns=[]
        ).index

        not_in_training = ~Y_pred_full.index.isin(index_of_texts_used_during_training)
        filter_dataset = Y_pred_full["dataset"] == "train"
        filter_oos = not_in_training & filter_dataset

        if filter_oos.sum() > 10:
            metrics["outofsample"] = get_metrics_multiclass(
                Y_pred_full[filter_oos]["predicted_label"],
                Y_pred_full[filter_oos]["prediction"],
                texts=Y_pred_full[filter_oos]["text"] if self.col_text else None,
            )
        return metrics

    def __create_saving_files(
        self, Y_pred: pd.DataFrame, metrics: dict[str, MLStatisticsModel]
    ) -> None:
        """Save the predictions and metrics"""
        Y_pred.to_parquet(self.path.joinpath(self.file_name), index=True)
        with open(str(self.path.joinpath(f"metrics_predict_{time.time()}.json")), "w") as f:
            json.dump({k: v.model_dump(mode="json") for k, v in metrics.items()}, f)

    def run(self) -> PredictMLResult:
        """
        Predict with quickmodel and calculate statistics
        """
        task_timer = TaskTimer(compulsory_steps=["setup", "predict", "save_files"])
        task_timer.start("setup")
        metrics: dict[str, MLStatisticsModel] | None = None
        try:
            model = self.__load_model()
            df = pd.read_parquet(self.path_df)
            expected_cols = model.feature_names_in_
            self.__check_data_and_features(df, expected_cols)

            X: pd.DataFrame = df.reindex(columns=list(expected_cols) + [self.col_dataset])
            Y: pd.Series = df[self.col_label]

            # Remove rows for which the label is to be excluded
            rows_to_keep = np.logical_and(Y.notna(), np.isin(Y, self.exclude_labels, invert=True))
            X = X.loc[rows_to_keep, :]
            task_timer.stop("setup")

            task_timer.start("predict")
            if self.is_aborted():
                raise Exception("Process interrupted by user")
            Y_pred = model.predict(X.drop(columns=["dataset"]))
            Y_proba = model.predict_proba(X.drop(columns=["dataset"]))
            labels_list = [label for label in model.classes_]

            # Aggregate data into one DataFrame
            Y_pred_full = pd.DataFrame(
                {
                    **{label: Y_proba[:, ilabel] for ilabel, label in enumerate(labels_list)},
                    "prediction": Y_pred,
                    "dataset": X["dataset"],
                    "predicted_label": df[self.col_label],
                },
                index=X.index,
            )
            if self.col_text:
                Y_pred_full["text"] = df[self.col_text]
            task_timer.stop("predict")

            task_timer.start("save_files")
            if self.statistics is not None:
                metrics = self.__compute_metrics(Y_pred_full, self.statistics)
                self.__create_saving_files(Y_pred_full, metrics)
            task_timer.stop("save_files")
        finally:
            Path(self.path_df).unlink(missing_ok=True)

        return PredictMLResult(
            path=str(self.path.joinpath(self.file_name)),
            metrics={k: v.model_dump(mode="json") for k, v in metrics.items()}
            if metrics is not None
            else None,
            events=task_timer.get_events(),
        )
