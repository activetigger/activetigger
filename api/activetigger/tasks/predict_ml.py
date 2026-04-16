import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from activetigger.datamodels import MLStatisticsModel
from activetigger.functions import get_metrics_multiclass
from activetigger.tasks.base_task import BaseTask


class PredictMLMultiClass(BaseTask):
    """
    Predict with a sklearn model
    """

    kind = "predict_model"

    def __init__(
        self,
        model: BaseEstimator,
        df: pd.DataFrame,
        col_dataset: str,
        col_features: list[str],
        file_name: str,
        path: Path,
        col_label: str | None = None,
        col_text: str | None = None,
        unique_id: Optional[str] = None,
        statistics: list[str] | None = None,
        exclude_labels: list[str] = [],
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.df = df
        self.file_name = file_name
        self.path = path
        self.unique_id = unique_id
        self.statistics = statistics
        self.col_label = col_label
        self.col_text = col_text

        expected_cols = self.model.feature_names_in_  # ty: ignore[unresolved-attribute]
        self.__check_data_and_features(col_dataset, expected_cols, col_features)

        self.X: pd.DataFrame = self.df.reindex(columns=list(expected_cols) + [col_dataset])
        self.Y: pd.Series = self.df[col_label]

        # Remove rows for which the label is to be excluded
        rows_to_keep = np.logical_and(self.Y.notna(), np.isin(self.Y, exclude_labels, invert=True))
        self.X = self.X.loc[rows_to_keep, :]
        self.Y = self.Y.loc[rows_to_keep]

    def __check_data_and_features(self, col_dataset, expected_cols, col_features):
        """Check that the dafatrame profided contains the required columns (text
        and features) and check that the labels are provided"""

        if col_dataset not in self.df.columns:
            raise ValueError(f"Dataset column {col_dataset} not in dataframe")

        if not all(col in self.df.columns for col in col_features):
            raise ValueError("Some feature columns are not in dataframe")

        if self.statistics is not None and self.col_label is None:
            raise ValueError("Labels must be provided to compute statistics")

        missing = set(expected_cols) - set(col_features)
        extra = set(col_features) - set(expected_cols)
        if len(missing) > 0 or len(extra) > 0:
            raise ValueError(f"Feature mismatch. Missing: {missing}, Extra: {extra}")

    def __compute_metrics(
        self, Y_pred_full: pd.DataFrame, statistics: list[str]
    ) -> dict[str, MLStatisticsModel]:
        """Compute the metrics for the whole train dataset as well as the
        train_valid dataset"""
        # Compute statistics if labels are provided
        metrics: dict[str, MLStatisticsModel] = {}

        # Compute the statistics for each dataset
        for dataset in statistics:
            # Select the correct rows
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
        # write the metrics in a json file
        with open(str(self.path.joinpath(f"metrics_predict_{time.time()}.json")), "w") as f:
            json.dump({k: v.model_dump(mode="json") for k, v in metrics.items()}, f)

    def __call__(self):
        """
        Predict with simplemodel and calculate statistics
        """

        # Predict
        Y_pred = self.model.predict(self.X.drop(columns=["dataset"]))  # ty: ignore[unresolved-attribute]
        Y_proba = self.model.predict_proba(self.X.drop(columns=["dataset"]))  # ty: ignore[unresolved-attribute]
        labels_list = [label for label in self.model.classes_]  # ty: ignore[unresolved-attribute]

        # Aggregate data into one DataFrame
        Y_pred_full = pd.DataFrame(
            {
                **{label: Y_proba[:, ilabel] for ilabel, label in enumerate(labels_list)},
                "prediction": Y_pred,
                "dataset": self.X["dataset"],
                "predicted_label": self.df[self.col_label],
            },
            index=self.X.index,
        )

        if self.col_text:
            Y_pred_full["text"] = self.df[self.col_text]

        if self.statistics is not None:
            metrics = self.__compute_metrics(Y_pred_full, self.statistics)
            self.__create_saving_files(Y_pred_full, metrics)
