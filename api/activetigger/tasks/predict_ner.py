import gc
import json
import multiprocessing
import multiprocessing.synchronize
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,  # ty: ignore[possibly-missing-import]
)

from activetigger.data import Data
from activetigger.datamodels import (
    EventsModel,
    MLStatisticsModel,
    ReturnTaskPredictModel,
    TextDatasetModel,
)
from activetigger.functions import (
    concat_text_columns,
    get_device,
)
from activetigger.monitoring import TaskTimer
from activetigger.ner_metrics import compute_ner_metrics
from activetigger.tasks.base_task import BaseTask
from activetigger.tasks.train_ner import decode_spans, parse_spans


class PredictNer(BaseTask):
    """
    Predict spans on a dataset with a trained NER (token-classification) model.

    Mirrors PredictBertMultiClass's I/O contract (paths, dataset modes,
    progress file, parquet output) so it slots into the same orchestration
    layer, but writes a `prediction` column containing JSON-encoded span
    lists instead of single labels.
    """

    kind = "predict_ner"
    default_max_length = 512

    def __init__(
        self,
        dataset: str,
        path: Path,
        df: DataFrame | None,
        col_text: str,
        scheme_labels: list[str],
        col_label: str | None = None,
        path_data: Path | None = None,
        external_dataset: TextDatasetModel | None = None,
        col_id_external: str | None = None,
        col_datasets: str | None = None,
        file_name: str = "predict.parquet",
        batch: int = 16,
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
        self.external_id_col = (
            external_dataset.id if dataset == "external" and external_dataset else None
        )
        self.path_data = path_data
        self.external_dataset = external_dataset

        if self.df is None and path_data is None:
            raise ValueError("A dataframe or a data file path must be provided for prediction")
        if statistics is not None and col_label is None:
            raise ValueError("Column label must be provided to compute statistics")
        if self.df is not None:
            self.__validate_dataframe(self.df)

        self.scheme_labels = scheme_labels
        with open(self.path / "parameters.json", "r") as f:
            self.model_config = json.load(f)
        self.max_length = int(self.model_config.get("max_length", self.default_max_length))
        self.modeltype = self.model_config.get("base_model")
        if self.modeltype is None:
            raise ValueError("No base_model found in parameters.json")

    def __validate_dataframe(self, df: DataFrame) -> None:
        if self.col_text not in df.columns:
            raise ValueError(f"Column text {self.col_text} not in dataframe")
        if self.col_label is not None and self.col_label not in df.columns:
            raise ValueError(f"Column label {self.col_label} not in dataframe")
        if self.col_datasets is not None and self.col_datasets not in df.columns:
            raise ValueError(f"Column datasets {self.col_datasets} not in dataframe")
        if self.col_id_external is not None and self.col_id_external not in df.columns:
            raise ValueError(f"Column id {self.col_id_external} not in dataframe")

    def __load_external_file(
        self, path_data: Path, external_dataset: TextDatasetModel | None
    ) -> DataFrame:
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
                in_main = subset["id_external"].isin(existing_ids)
                if in_main.any():
                    main_mask = df["id_external"].isin(set(subset.loc[in_main, "id_external"]))
                    df.loc[main_mask, "dataset"] = name
                imported = subset.loc[~in_main]
                if not imported.empty and "text" in imported.columns:
                    imported = imported[["id_external", "text"]].copy()
                    imported["dataset"] = name
                    extra_frames.append(imported)
            if extra_frames:
                df = pd.concat([df, *extra_frames])
        return df

    def __write_progress(self, progress: float) -> None:
        with open(self.progress_path, "w") as f:
            f.write(str(progress))

    def __listen_stop_event(self):
        if self.event is not None and self.event.is_set():
            raise Exception("Process interrupted by user")

    def __compute_statistics(
        self, pred: DataFrame, id2label: dict[int, str]
    ) -> dict[str, MLStatisticsModel]:
        """Compute exact/partial/type metrics on rows with gold annotations,
        broken down by split (train/valid/test) and out-of-sample.

        Writes the full three-flavor breakdown to ``metrics_predict_*.json``
        for the frontend, and returns the "exact" flavor per dataset so the
        orchestrator's ``ReturnTaskPredictModel`` exposes headline NER
        metrics in the same shape as classification metrics.
        """
        if self.df is None or self.statistics is None:
            return {}
        metrics: dict[str, MLStatisticsModel] = {}
        metrics_file: dict[str, dict[str, Any]] = {}
        filter_label = pred["GS-label"].notna()
        for dataset in self.statistics:
            filter_dataset = pred[self.col_datasets] == dataset
            mask = filter_label & filter_dataset
            if mask.sum() < 5:
                continue
            gold = pred.loc[mask, "GS-label"].apply(parse_spans).tolist()
            preds = pred.loc[mask, "prediction"].apply(parse_spans).tolist()
            texts = pred.loc[mask, "text"].astype(str).tolist()
            ids = [str(i) for i in pred.loc[mask].index]
            flavors = compute_ner_metrics(gold, preds, self.scheme_labels, texts=texts, ids=ids)
            metrics[dataset] = flavors["exact"]
            metrics_file[dataset] = {
                "training_kind": "ner",
                "exact": flavors["exact"].model_dump(mode="json"),
                "partial": flavors["partial"].model_dump(mode="json"),
                "type": flavors["type"].model_dump(mode="json"),
            }
        with open(str(self.path.joinpath(f"metrics_predict_{time.time()}.json")), "w") as f:
            json.dump(metrics_file, f)
        return metrics

    def __call__(self) -> ReturnTaskPredictModel:
        print("start predicting (ner)", flush=True)
        task_timer = TaskTimer(compulsory_steps=["setup", "predict", "save_files"])
        task_timer.start("setup")

        if self.df is None:
            if self.path_data is None:
                raise ValueError("Dataframe is required for prediction")
            self.df = self.__load_external_file(self.path_data, self.external_dataset)
            self.__validate_dataframe(self.df)

        # Load the tokenizer that was saved alongside the trained model
        # instead of pulling from the upstream base model. This guarantees
        # we use the exact same vocab / special tokens / offset behavior
        # that produced the model's predictions during training.
        tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True, use_fast=True)
        if not tokenizer or not tokenizer.is_fast:
            raise Exception(
                "NER prediction requires a fast tokenizer for offset_mapping "
                "and word_ids(); the saved tokenizer at "
                f"{self.path} is slow."
            )
        model = AutoModelForTokenClassification.from_pretrained(self.path, trust_remote_code=True)
        id2label = {int(k): v for k, v in model.config.id2label.items()}

        device = get_device()
        model.to(device)
        task_timer.stop("setup")

        n_rows = self.df.shape[0]
        n_batches = (n_rows + self.batch - 1) // self.batch
        predictions_json: list[str] = []
        metrics: dict[str, MLStatisticsModel] | None = None

        try:
            task_timer.start("predict")
            texts = self.df[self.col_text].astype(str).tolist()
            for i in tqdm(
                range(0, n_rows, self.batch),
                total=n_batches,
                desc="Predicting",
                unit="batch",
            ):
                self.__listen_stop_event()
                batch_texts = texts[i : i + self.batch]
                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    max_length=self.max_length,
                )
                # offset_mapping and word_ids are not tensors / not on device
                offset_mappings = enc.pop("offset_mapping").tolist()
                word_ids_per_row = [enc.word_ids(j) for j in range(len(batch_texts))]
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    outputs = model(**enc)
                logits = outputs.logits.detach().cpu().numpy()
                pred_ids = np.argmax(logits, axis=-1)
                for j in range(len(batch_texts)):
                    spans = decode_spans(
                        pred_ids[j].tolist(),
                        offset_mappings[j],
                        word_ids_per_row[j],
                        id2label,
                        text=batch_texts[j],
                    )
                    predictions_json.append(json.dumps(spans))
                self.__write_progress(100 * (i + self.batch) / n_rows)
            task_timer.stop("predict")

            task_timer.start("save_files")
            pred = pd.DataFrame(
                {"prediction": predictions_json, "text": texts},
                index=self.df.index,
            )
            if self.col_datasets:
                pred[self.col_datasets] = self.df[self.col_datasets]
            if self.col_id_external and self.col_id_external in self.df.columns:
                pred[self.col_id_external] = self.df[self.col_id_external].astype(str)
            if self.col_label:
                pred["GS-label"] = self.df[self.col_label]

            pred.to_parquet(self.path.joinpath(self.file_name))
            if self.external_id_col is not None:
                with open(self.path.joinpath(f"{self.file_name}.meta.json"), "w") as f:
                    json.dump({"col_id": self.external_id_col}, f)

            if self.statistics:
                metrics = self.__compute_statistics(pred, id2label)

            task_timer.stop("save_files")
        except Exception as e:
            print("Error in NER prediction", e)
            raise
        finally:
            if self.progress_path.exists():
                os.remove(self.progress_path)
            self.df = None
            self.event = None
            try:
                del model, tokenizer
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        return ReturnTaskPredictModel(
            path=str(self.path.joinpath(self.file_name)),
            # Headline (exact-match) metrics per dataset. The full
            # exact/partial/type breakdown is written to
            # ``metrics_predict_*.json`` by ``__compute_statistics``.
            metrics=metrics,
            events=EventsModel(events=task_timer.get_events()),
        )
