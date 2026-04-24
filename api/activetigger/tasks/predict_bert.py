import gc
import json
import multiprocessing
import multiprocessing.synchronize
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from scipy.stats import entropy
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,  # ty: ignore[possibly-missing-import]
)

from activetigger.data import Data
from activetigger.datamodels import MLStatisticsModel, ReturnTaskPredictModel, TextDatasetModel
from activetigger.functions import (
    activate_probs,
    annotations_to_matrix,
    concat_text_columns,
    dichotomize,
    get_device,
    get_metrics_multiclass,
    get_metrics_multilabel,
    logits_to_probs,
)
from activetigger.tasks.base_task import BaseTask


class PredictBertMultiClass(BaseTask):
    """
    Class to predict with a bert model
    """

    kind = "predict_bert"
    default_max_length = 512

    def __init__(
        self,
        dataset: str,
        path: Path,
        df: DataFrame | None,
        col_text: str,
        training_kind: str,
        scheme_labels: list[str],
        col_label: str | None = None,
        path_data: Path | None = None,
        external_dataset: TextDatasetModel | None = None,
        col_id_external: str | None = None,
        col_datasets: str | None = None,
        file_name: str = "predict.parquet",
        batch: int = 32,
        statistics: list | None = None,
        event: Optional[multiprocessing.synchronize.Event] = None,
        unique_id: Optional[str] = None,
        path_train: Path | None = None,
        path_valid: Path | None = None,
        path_test: Path | None = None,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.df = df
        self.dataset = dataset
        self.col_text = col_text
        self.col_label = col_label
        self.col_id_external = col_id_external
        self.col_datasets = col_datasets
        self.event = event
        self.unique_id = unique_id
        self.file_name = file_name
        self.batch = batch
        self.statistics = statistics
        self.path_train = path_train
        self.path_valid = path_valid
        self.path_test = path_test
        self.progress_path = self.path / "progress_predict"

        if self.df is None and path_data is not None:
            self.df = self.__load_external_file(path_data, external_dataset)

        if self.df is None:
            raise ValueError("Dataframe must be provided for prediction")

        if col_text not in self.df.columns:
            raise ValueError(f"Column text {col_text} not in dataframe")

        if col_label is not None and col_label not in self.df.columns:
            raise ValueError(f"Column label {col_label} not in dataframe")

        if col_datasets is not None and col_datasets not in self.df.columns:
            raise ValueError(f"Column datasets {col_datasets} not in dataframe")

        if col_id_external is not None and col_id_external not in self.df.columns:
            raise ValueError(f"Column id {col_id_external} not in dataframe")

        if statistics is not None and col_label is None:
            raise ValueError("Column label must be provided to compute statistics")

        self.training_kind = training_kind
        self.scheme_labels = scheme_labels
        self.threshold = None

        # read the config file
        with open(self.path / "parameters.json", "r") as jsonfile:
            self.model_config = json.load(jsonfile)
            self.max_length = int(self.model_config.get("max_length", self.default_max_length))
            if "base_model" in self.model_config:
                self.modeltype = self.model_config["base_model"]
            else:
                raise ValueError("No model type found in config.json. Please check the file.")
            if "threshold" in self.model_config and self.training_kind == "multilabel":
                self.threshold = float(self.model_config["threshold"])
            elif self.training_kind == "multiclass":
                pass  # we don't need it
            else:
                raise ValueError("Threshold not found in config.json while required for multilabel")

    def __load_external_file(
        self, path_data: Path, external_dataset: TextDatasetModel | None
    ) -> DataFrame:
        """
        Load file for prediction with specific rules to match the expected format
        """
        df = Data.read_dataset(path_data)

        if self.dataset == "external" and external_dataset is not None:
            df["text"] = concat_text_columns(df, external_dataset.cols_text)
            df["index"] = df[external_dataset.id].apply(str)
            df["id_external"] = df["index"]
            df["dataset"] = "external"
            df.set_index("index", inplace=True)
            df = df[["id_external", "dataset", "text"]].dropna()

        if self.dataset == "all":
            df["id_external"] = df[self.col_id_external].astype(str)
            df["dataset"] = "all"
            existing_ids = set(df["id_external"])
            # add also all elements that may have been imported
            subsets = {
                "train": self.path_train,
                "valid": self.path_valid,
                "test": self.path_test,
            }
            extra_frames = []
            for name, subset_path in subsets.items():
                if subset_path is None or not subset_path.exists():
                    continue
                subset = Data.read_dataset(subset_path)
                if "id_external" not in subset.columns:
                    continue
                subset["id_external"] = subset["id_external"].astype(str)
                # label rows from the main corpus that belong to this split
                in_main = subset["id_external"].isin(existing_ids)
                if in_main.any():
                    main_mask = df["id_external"].isin(set(subset.loc[in_main, "id_external"]))
                    df.loc[main_mask, "dataset"] = name
                # collect imported rows (present only in the split parquet)
                imported = subset.loc[~in_main]
                if not imported.empty and "text" in imported.columns:
                    imported = imported[["id_external", "text"]].copy()
                    imported["dataset"] = name
                    extra_frames.append(imported)

            if extra_frames:
                df = pd.concat([df, *extra_frames])

        return df

    def __write_progress(self, progress: float) -> None:
        """
        Write progress to the progress file
        """
        with open(self.progress_path, "w") as f:
            f.write(str(progress))

    def __load_model(self) -> tuple:
        """
        Load the model and tokenizer from the path
        """
        # load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.modeltype, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.path, trust_remote_code=True
        )

        return tokenizer, model, self.max_length

    def __listen_stop_event(self):
        """
        Raise exception if stop event is set
        """
        if self.event is not None:
            if self.event.is_set():
                raise Exception("Process interrupted by user")

    def __transform_to_dataframe(
        self, prob_predictions: np.ndarray, id2label: dict[int, str]
    ) -> DataFrame:
        """
        Transform the prob_predictions into a dataframe
        """
        if self.df is None:
            raise ValueError("Dataframe is required to transform to predictions")

        id2label = dict(sorted(id2label.items(), key=lambda u: u[0]))  # sort by index 0,1,2 ...

        if list(id2label.keys()) != [i for i in range(len(id2label))]:
            raise ValueError(f"Warning, something is off with the id2label: {id2label}")

        pred = pd.DataFrame(
            prob_predictions,
            columns=list(id2label.values()),
            index=self.df.index,
        )

        pred["entropy"] = entropy(prob_predictions, axis=1)

        # Add entropy-LABEL defined as the entropy of p(A) / 1-p(A)
        for label in list(id2label.values()):
            prob_A_not_A = np.column_stack([pred[label], 1 - pred[label]])
            pred[f"entropy-{label}"] = entropy(prob_A_not_A, axis=1)

        if self.training_kind == "multiclass":
            y_pred = activate_probs(
                probs=prob_predictions, strategy="max", force_max_1_per_row=True
            )  # shape: n_rows x n_labels
            pred["prediction"] = np.argmax(y_pred, axis=1)  # 0,1,2,0
            pred["prediction"] = pred["prediction"].replace(id2label)  # label1, label2 ...

        elif self.training_kind == "multilabel":
            # Save labels as index + pipe -> [1,0,0,1] -> 0|3
            y_pred = activate_probs(
                probs=prob_predictions,
                strategy="threshold",
                threshold=self.threshold if self.threshold is not None else 0.5,
            )  # shape: n_rows x n_labels
            pred["prediction"] = [
                "|".join(
                    [id2label[index] for index, activation in enumerate(row) if activation == 1]
                )
                for row in y_pred
            ]  # label1|label2, label1, label2|label3

        # add text in the dataframe to be able to get mismatch
        pred["text"] = self.df[self.col_text]

        if self.col_datasets:
            pred[self.col_datasets] = self.df[self.col_datasets]
        if self.col_id_external:
            pred[self.col_id_external] = self.df[self.col_id_external].astype(str)
        if self.col_label:
            pred["GS-label"] = self.df[self.col_label]
            if self.model_config.get("use_dichotomization", False):
                pred["GS-label-non-dichotomized"] = pred["GS-label"].copy()
                label_for_dichotomization = self.model_config["label_for_dichotomization"]
                pred, _ = dichotomize(pred, "GS-label", label_for_dichotomization)
        return pred

    def __compute_statistics(
        self, pred: DataFrame, id2label: dict[int, str]
    ) -> dict[str, MLStatisticsModel]:
        """
        Compute statistics for the predictions
        """
        if self.df is None:
            raise ValueError("Dataframe is required to compute statistics")
        if self.statistics is None:
            raise ValueError("Statistics list is required to compute statistics")

        # compute statistics
        metrics: dict[str, MLStatisticsModel] = {}

        filter_label = pred["GS-label"].notna()  # only non null values

        # compute the statistics per dataset
        for dataset in self.statistics:
            filter_dataset = pred[self.col_datasets] == dataset
            filter = filter_label & filter_dataset
            if filter.sum() < 5:
                continue
            if self.training_kind == "multiclass":
                metrics[dataset] = get_metrics_multiclass(
                    Y_true=pred.loc[filter, "GS-label"],
                    Y_pred=pred.loc[filter, "prediction"],
                    texts=pred[filter]["text"],
                    id2label=id2label,
                )
            elif self.training_kind == "multilabel":
                labels = list(id2label.values())
                y_true = annotations_to_matrix(pred.loc[filter, "GS-label"], labels)
                y_pred = annotations_to_matrix(pred.loc[filter, "prediction"], labels)
                metrics[dataset] = get_metrics_multilabel(
                    Y_true=y_true,
                    Y_pred=y_pred,
                    id2label=id2label,
                    texts=pred[filter]["text"],
                )

        # add out of sample (labelled data not in training data)
        index_training_data = pd.read_parquet(
            self.path.joinpath("training_data.parquet"), columns=[]
        ).index
        filter_oos = (
            ~pred.index.isin(index_training_data) & filter_label & pred[self.col_datasets]
            == "train"
        )
        if filter_oos.sum() > 10:
            if self.training_kind == "multiclass":
                metrics["outofsample"] = get_metrics_multiclass(
                    Y_true=pred.loc[filter_oos, "GS-label"],
                    Y_pred=pred.loc[filter_oos, "prediction"],
                    texts=pred[filter_oos]["text"],
                    id2label=id2label,
                )
            elif self.training_kind == "multilabel":
                labels = list(id2label.values())
                y_true = annotations_to_matrix(pred.loc[filter_oos, "GS-label"], labels)
                y_pred = annotations_to_matrix(pred.loc[filter_oos, "prediction"], labels)
                metrics["outofsample"] = get_metrics_multilabel(
                    Y_true=y_true,
                    Y_pred=y_pred,
                    id2label=id2label,
                    texts=pred[filter_oos]["text"],
                )

        # write the metrics in a json file
        with open(str(self.path.joinpath(f"metrics_predict_{time.time()}.json")), "w") as f:
            json.dump({k: v.model_dump(mode="json") for k, v in metrics.items()}, f)

        return metrics

    def __call__(self) -> ReturnTaskPredictModel:
        """
        Main process to predict
        """
        print(f"start predicting ({self.training_kind})")

        if self.df is None:
            raise ValueError("Dataframe is required for prediction")

        # load the model
        tokenizer, model, max_length = self.__load_model()
        id2label = model.config.id2label

        # select device
        device = get_device()
        print(f"Using {device} for prediction")
        model.to(device)
        try:
            models_id2label = model.config.id2label
            num_labels = len(models_id2label)
        except Exception as e:
            raise ValueError("Model is wrong, id2label missing from the model's config." + str(e))

        if not np.isin(self.scheme_labels, list(models_id2label.values())).all():
            # WARNING
            print(
                f"The scheme labels ({self.scheme_labels}) are different "
                f"from the labels used during training. Will use {list(models_id2label.values())} "
                f"instead."
            )

        try:
            # prediction by batches
            proba_predictions = np.zeros((0, num_labels))
            for i in range(0, self.df.shape[0], self.batch):
                self.__listen_stop_event()
                chunk = tokenizer(
                    list(self.df[self.col_text][i : i + self.batch]),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length,
                )
                chunk = chunk.to(device)
                with torch.no_grad():
                    outputs = model(**chunk)
                logits = outputs[0].detach().cpu().numpy()
                proba = logits_to_probs(logits, kind=self.training_kind)
                proba_predictions = np.append(proba_predictions, proba, axis=0)
                self.__write_progress(100 * (i + self.batch) / self.df.shape[0])

            # transform predictions to clean dataframe
            pred = self.__transform_to_dataframe(proba_predictions, id2label=models_id2label)
            # save the prediction to file
            pred.to_parquet(self.path.joinpath(self.file_name))

            # compute statistics if required
            if self.statistics:
                metrics = self.__compute_statistics(pred, id2label)
            else:
                metrics = None

        except Exception as e:
            print("Error in prediction", e)
            raise e
        finally:
            # delete the temporary logs
            if self.progress_path.exists():
                os.remove(self.progress_path)
            # clean memory
            self.df = None
            self.event = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        return ReturnTaskPredictModel(path=str(self.path.joinpath(self.file_name)), metrics=metrics)
