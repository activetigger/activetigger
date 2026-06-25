import gc
import json
import logging
import multiprocessing
import multiprocessing.synchronize
import os
import shutil
from logging import Logger
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import datasets
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from PIL import Image, ImageOps
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,  # ty: ignore[possibly-missing-import]
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

from activetigger.config import config
from activetigger.datamodels import EventsModel, LMParametersModel, MLStatisticsModel
from activetigger.functions import (
    activate_probs,
    get_device,
    get_metrics_multiclass,
    get_metrics_multilabel,
    logits_to_probs,
    matrix_to_label,
    split_annotation,
)
from activetigger.functions_image import filter_readable_images
from activetigger.monitoring import TaskTimer
from activetigger.tasks.base_task import BaseTask
from activetigger.tasks.predict_bert import annotations_to_matrix
from activetigger.tasks.train_bert import (
    CustomTrainer,
    compute_class_weights,
)

# Cap PIL's maximum decoded pixel count to ~64 megapixels. The default of
# ~89 MP only raises DecompressionBombError at 2× the limit, which is high
# enough that a small attacker-supplied JPEG can decompress to several GB
# and OOM the GPU worker mid-epoch. 64 MP covers any realistic 8K×8K input;
# anything bigger is almost certainly a bomb.
Image.MAX_IMAGE_PIXELS = 64_000_000


class ImageLoggingCallback(TrainerCallback):
    """
    Image-classification training callback.

    Mirrors the BERT CustomLoggingCallback but **does not** divide the logged
    training loss by gradient_accumulation_steps. In modern HF Transformers
    (5.x), `fixed_cross_entropy` uses `reduction="sum"` and divides by
    `num_items_in_batch` (the total items across all sub-batches in the
    optimizer step) when the model accepts loss kwargs — so the reported
    train_loss is already at per-item scale, matching eval_loss. Applying
    the old gradacc correction under-scales train_loss by gradacc and
    breaks comparability with eval_loss on the loss chart.
    """

    event: Optional[multiprocessing.synchronize.Event]
    current_path: Path
    logger: Logger

    def __init__(self, event, logger, current_path):
        super().__init__()
        self.event = event
        self.current_path = current_path
        self.logger = logger

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.logger.info(f"Step {state.global_step}")
        if state.max_steps:
            progress_percentage = (state.global_step / state.max_steps) * 100
        else:
            progress_percentage = 0.0
        with open(self.current_path.joinpath("progress_train"), "w") as f:
            f.write(str(progress_percentage))
        with open(self.current_path.joinpath("log_history.txt"), "w") as f:
            json.dump(state.log_history, f)
        if self.event is not None and self.event.is_set():
            self.logger.info("Event set, stopping training.")
            control.should_training_stop = True
            raise Exception("Process interrupted by user")


def _open_image_rgb(path: str) -> Image.Image | None:
    """
    Open an image and normalize it: EXIF rotation + RGB conversion.

    Used by the dataset transform during training and prediction; callers
    are expected to have already pre-flighted the path so a raise here is
    a true error (the path passed verify() once but failed on second open).
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img is not None and img.mode != "RGB":
        img = img.convert("RGB")
    return img


class TrainImage(BaseTask):
    """
    Fine-tune an image-classification model on a labelled image set.

    Works with any HuggingFace AutoModelForImageClassification (ViT,
    ConvNeXt, EfficientNet, Swin, BEiT, etc.). Mirrors TrainBert: same
    disk layout under <project>/image/<name>/, same progress and log
    files, same archive output. Images are loaded lazily through
    Dataset.with_transform so we never materialize a multi-GB tensor
    parquet on disk.
    """

    kind = "train_image"

    def __init__(
        self,
        path: Path,
        project_slug: str,
        model_name: str,
        df: DataFrame,
        training_kind: str,
        scheme_labels: list[str],
        col_text: str,
        col_label: str,
        base_model: str,
        params: LMParametersModel,
        test_size: float,
        event: Optional[multiprocessing.synchronize.Event] = None,
        unique_id: Optional[str] = None,
        loss: Optional[str] = "cross_entropy",
        class_balance: bool = False,
        class_min_freq: int = 1,
        fp16: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.project_slug = project_slug
        self.name = model_name
        df.index.name = "id"
        self.df = df
        if training_kind not in ["multiclass", "multilabel"]:
            raise ValueError(
                f"TrainImage only supports multiclass and multilabel; got {training_kind}"
            )
        self.training_kind = training_kind
        if len(scheme_labels) != len(set(scheme_labels)):
            raise ValueError(f"Labels in your scheme are not unique: {scheme_labels}")
        self.scheme_labels = scheme_labels
        self.col_text = col_text  # holds the image path
        self.col_label = col_label
        self.base_model = base_model
        self.params = params
        self.test_size = test_size
        self.event = event
        self.unique_id = unique_id
        if loss == "weighted_cross_entropy" and training_kind == "multilabel":
            raise ValueError("weighted_cross_entropy is not supported for multilabel.")
        self.loss = loss
        self.class_balance = class_balance
        self.class_min_freq = class_min_freq
        self.fp16 = fp16

    # --- setup helpers ---------------------------------------------------

    def __init_paths(self) -> Tuple[Path, Path]:
        # Absolute path: Loky workers use "spawn" and don't always inherit the
        # parent's CWD, so a relative self.path (e.g. "projects/foo/image") could
        # resolve to a different filesystem location in the worker and the
        # progress-file writes from the training callback would fail.
        current_path = (self.path / self.name).absolute()
        current_path.mkdir(parents=True, exist_ok=True)
        log_path = current_path / "status.log"
        return current_path, log_path

    def __init_logger(self, log_path) -> Logger:
        logger = logging.getLogger("train_vit_model")
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Start {self.base_model}")
        return logger

    def __listen_stop_event(self) -> None:
        """
        Raise if the user requested cancellation. Called in the long phases
        (pre-flight scan, evaluation, saving) that the training callback
        does not cover — without these checks a kill is only honored
        during training steps and the GPU slot stays occupied.
        """
        if self.event is not None and self.event.is_set():
            raise Exception("Process interrupted by user")

    def __check_data(self, df: pd.DataFrame, col_label: str, col_text: str) -> pd.DataFrame:
        """
        Drop rows with missing labels, missing paths, unrecognised labels,
        or images that PIL cannot open. Pre-flighting paths here is the
        only way to keep training collators well-shaped when the dataset
        contains the occasional corrupt / non-image file.
        """
        df = df.copy()
        if df[col_label].isnull().sum() > 0:
            df = df[df[col_label].notnull()]
            self.logger.info(f"Missing labels - reducing training data to {len(df)}")

        if df[col_text].isnull().sum() > 0:
            df = df[df[col_text].notnull()]
            self.logger.info(f"Missing image paths - reducing training data to {len(df)}")

        scheme_set = set(self.scheme_labels)

        def _check_labels(annotation: object) -> bool:
            if not isinstance(annotation, str):
                return False
            parts = split_annotation(annotation)
            if not isinstance(parts, list):
                return False
            return all(part in scheme_set for part in parts)

        condition = df[col_label].apply(_check_labels)
        if (~condition).sum() > 0:
            df = df[condition]
            self.logger.info(f"Labels unrecognised - reducing training data to {len(df)}")

        # Pre-flight: drop unreadable / corrupt images so the lazy transform
        # never raises mid-batch and breaks the collator.
        readable_flags = filter_readable_images(
            df[col_text].tolist(),
            stop_check=self.__listen_stop_event,
        )
        readable = pd.Series(readable_flags, index=df.index)
        n_dropped = int((~readable).sum())
        if n_dropped > 0:
            df = df[readable]
            self.logger.info(f"Dropped {n_dropped} unreadable/corrupt images")
            print(f"Dropped {n_dropped} unreadable/corrupt images", flush=True)

        if len(df) < 2:
            raise ValueError("Too few usable images after pre-flight; aborting.")
        return df

    def __retrieve_labels(self, scheme_labels):
        if len(scheme_labels) < 2:
            raise ValueError(
                "Not enough classes. Either you excluded classes or "
                "there are not enough annotations."
            )
        label2id = {j: i for i, j in enumerate(scheme_labels)}
        id2label = {i: j for i, j in enumerate(scheme_labels)}
        return scheme_labels, label2id, id2label

    def __transform_to_dataset(
        self,
        training_kind: str,
        df: pd.DataFrame,
        col_label: str,
        col_text: str,
        label2id: dict[str, int],
    ) -> datasets.Dataset:
        """
        Build a Dataset of (id, path, labels). The image processor will be
        applied lazily by with_transform — see __make_transform.
        """
        ids = df.reset_index()["id"].copy().to_list()
        paths = df[col_text].copy().to_list()
        one_hot = "weight" in self.loss.lower() if self.loss is not None else False
        if training_kind == "multiclass":
            labels_as_list = df[col_label].copy().replace(label2id).tolist()
            if one_hot:
                labels = torch.tensor(
                    [[int(i == j) for j in range(len(label2id))] for i in labels_as_list],
                    dtype=torch.float32,
                )
            else:
                labels = torch.tensor(labels_as_list, dtype=torch.long)
        elif training_kind == "multilabel":
            labels = torch.tensor(
                annotations_to_matrix(df[col_label], list(label2id.keys())).tolist(),
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unsupported training_kind: {training_kind}")

        return datasets.Dataset.from_dict({"id": ids, "path": paths, "labels": labels})

    def __make_transform(self, image_processor):
        """
        Build a transform suitable for Dataset.with_transform.

        Returns dicts with pixel_values + labels. Must be applied to every
        split after train_test_split — DatasetDict does not propagate
        with_transform settings.
        """

        def _transform(batch: dict) -> dict:
            images = [_open_image_rgb(p) for p in batch["path"]]
            processed = image_processor(images=images, return_tensors="pt")
            # stack labels back into a tensor (with_transform receives lists)
            labels = batch["labels"]
            if isinstance(labels, list):
                labels_tensor = torch.stack([torch.as_tensor(label) for label in labels])
            else:
                labels_tensor = labels
            return {"pixel_values": processed["pixel_values"], "labels": labels_tensor}

        return _transform

    def __load_trainer(
        self,
        current_path: Path,
        ds: datasets.DatasetDict,
        model,
        params: LMParametersModel,
        loss: str,
    ) -> Trainer:
        has_test = "test" in ds

        total_steps = (float(params.epochs) * len(ds["train"])) // (
            int(params.batchsize) * float(params.gradacc)
        )
        warmup_steps = int(total_steps // 10)
        eval_steps = (total_steps - warmup_steps) // params.eval
        eval_steps = max(int(eval_steps), 1)

        use_cpu = config.cpu_only or not bool(params.gpu)
        # Only enable fp16 when actually on CUDA; HF rejects it otherwise.
        fp16 = bool(self.fp16) and not use_cpu and torch.cuda.is_available()
        seed = int(config.random_seed)

        training_args = TrainingArguments(
            output_dir=str(current_path.joinpath("train")),
            logging_dir=str(current_path.joinpath("logs")),
            learning_rate=float(params.lrate),
            weight_decay=float(params.wdecay),
            num_train_epochs=float(params.epochs),
            warmup_steps=int(warmup_steps),
            gradient_accumulation_steps=int(params.gradacc),
            per_device_train_batch_size=int(params.batchsize),
            per_device_eval_batch_size=int(params.batchsize),
            eval_strategy="steps" if has_test else "no",
            eval_steps=eval_steps if has_test else None,
            save_strategy="best" if has_test else "epoch",
            metric_for_best_model="eval_loss" if has_test else None,
            save_steps=float(eval_steps) if has_test else 500,
            logging_steps=int(eval_steps),
            do_eval=has_test,
            greater_is_better=False if has_test else None,
            load_best_model_at_end=params.best if has_test else False,
            use_cpu=use_cpu,
            fp16=fp16,
            # ViT-Large checkpoints are ~1.2 GB each; keep only one.
            save_total_limit=1,
            # Image classification: with_transform produces pixel_values + labels.
            # remove_unused_columns=True would strip pixel_values before it
            # reaches the model — see HF transformers image-classification guide.
            remove_unused_columns=False,
            label_names=["labels"],
            # Reproducibility: same run inputs → same model weights and metrics.
            seed=seed,
            data_seed=seed,
        )

        callback = ImageLoggingCallback(self.event, current_path=current_path, logger=self.logger)
        eval_dataset = ds["test"] if has_test else None
        if loss == "cross_entropy":
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=eval_dataset,
                callbacks=[callback],
            )
        elif loss == "weighted_cross_entropy":
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=eval_dataset,
                callbacks=[callback],
                class_weights=compute_class_weights(ds["train"], label_key="labels"),
                training_kind=self.training_kind,
            )
        else:
            raise ValueError(f"Loss function {loss} not recognized.")
        return trainer

    def __create_save_files(
        self,
        current_path: Path,
        log_path: Path,
        df_train_results: pd.DataFrame,
        df_test_results: pd.DataFrame | None,
        training_data: pd.DataFrame,
        model,
        image_processor,
        params_to_save: dict[str, Any],
        metrics_train: MLStatisticsModel,
        metrics_test: MLStatisticsModel | None,
    ) -> None:
        df_train_results[[c for c in df_train_results.columns if c not in ["pixel_values"]]].to_csv(
            current_path.joinpath("train_dataset_eval.csv")
        )
        if df_test_results is not None:
            df_test_results[
                [c for c in df_test_results.columns if c not in ["pixel_values"]]
            ].to_csv(current_path.joinpath("test_dataset_eval.csv"))
        training_data.to_parquet(current_path.joinpath("training_data.parquet"))

        model.save_pretrained(current_path)
        image_processor.save_pretrained(current_path)

        with open(current_path.joinpath("parameters.json"), "w") as f:
            json.dump(params_to_save, f)

        # remove intermediate checkpoint dir
        train_dir = current_path.joinpath("train")
        if train_dir.exists():
            shutil.rmtree(train_dir)
        os.rename(log_path, current_path.joinpath("finished"))

        path_static = f"{config.data_path}/projects/static/{self.project_slug}"
        os.makedirs(path_static, exist_ok=True)
        shutil.make_archive(
            f"{path_static}/{self.name}",
            "gztar",
            str(self.path.joinpath(self.name)),
        )

        metrics_data: dict[str, Any] = {"train": metrics_train.model_dump(mode="json")}
        if metrics_test is not None:
            metrics_data["trainvalid"] = metrics_test.model_dump(mode="json")
        with open(str(current_path.joinpath("metrics_training.json")), "w") as f:
            json.dump(metrics_data, f)

    # --- main entry -------------------------------------------------------

    def __call__(self) -> EventsModel:
        task_timer = TaskTimer(compulsory_steps=["setup", "train", "evaluate", "save_files"])
        task_timer.start("setup")

        # Seed everything (python random, numpy, torch, torch.cuda) so that
        # successive runs of the same training produce the same model.
        # HF Trainer also gets the seed via TrainingArguments below; both
        # layers are needed because operations before Trainer.__init__
        # (train_test_split, dataset shuffling) don't use HF's seed.
        seed = int(config.random_seed)
        set_seed(seed)

        current_path, log_path = self.__init_paths()
        self.logger = self.__init_logger(log_path)
        device = get_device()

        # bound before the try so the finally can clean them up even when
        # setup fails partway (model download, OOM on .to(device), ...)
        image_processor = None
        model = None
        trainer = None

        try:
            assert self.df is not None
            self.df = self.__check_data(self.df, self.col_label, self.col_text)
            labels, label2id, id2label = self.__retrieve_labels(self.scheme_labels)

            ds_base = self.__transform_to_dataset(
                self.training_kind, self.df, self.col_label, self.col_text, label2id
            )

            # Build a fresh image processor for this base model and reuse it both
            # for transforms and for save_pretrained / inference.
            image_processor = AutoImageProcessor.from_pretrained(self.base_model)
            transform_fn = self.__make_transform(image_processor)

            if self.test_size > 0:
                split = ds_base.train_test_split(test_size=self.test_size, seed=seed)
                # Capture id / path lists BEFORE with_transform — once the transform
                # is set, column-name access through ds[col] is replaced by the
                # transform output (pixel_values, labels) and the raw columns become
                # unreachable, which breaks post-train evaluation.
                train_ids = list(split["train"]["id"])
                train_paths = list(split["train"]["path"])
                test_ids = list(split["test"]["id"])
                test_paths = list(split["test"]["path"])
                split["train"] = split["train"].with_transform(transform_fn)
                split["test"] = split["test"].with_transform(transform_fn)
                self.ds = split
            else:
                train_ids = list(ds_base["id"])
                train_paths = list(ds_base["path"])
                test_ids = None
                test_paths = None
                self.ds = datasets.DatasetDict({"train": ds_base.with_transform(transform_fn)})
            self.logger.info("Train/test dataset created")

            self.__listen_stop_event()
            model = AutoModelForImageClassification.from_pretrained(
                self.base_model,
                num_labels=len(labels),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
                problem_type="multi_label_classification"
                if self.training_kind == "multilabel"
                else "single_label_classification",
            ).to(device=device)
            model.config.use_cache = False
            self.logger.info(f"Model loaded on {model.device}")

            trainer = self.__load_trainer(
                current_path, self.ds, model, self.params, self.loss or "cross_entropy"
            )
            task_timer.stop("setup")

            task_timer.start("train")
            trainer.train()
            self.logger.info(f"Model trained {current_path}")
            task_timer.stop("train")

            # ----- post-training evaluation -----
            task_timer.start("evaluate")
            self.__listen_stop_event()
            train_ds = cast(datasets.Dataset, self.ds["train"])
            predictions_train = trainer.predict(cast(TorchDataset, train_ds))
            train_label_ids = cast(np.ndarray, predictions_train.label_ids)
            train_logits = cast(np.ndarray, predictions_train.predictions)

            df_train_results = pd.DataFrame({"id": train_ids}).set_index("id")
            df_train_results["path"] = train_paths
            df_train_results["true_label-matrix"] = train_label_ids.tolist()
            df_train_results["true_label"] = [
                "|".join(matrix_to_label(row, id2label)) for row in train_label_ids
            ]

            y_prob_pred = logits_to_probs(train_logits, self.training_kind)

            # Hoisted so it stays defined for the test-set branch below;
            # only meaningful when training_kind != "multiclass".
            threshold: float = 0.5
            if self.training_kind == "multiclass":
                labels_predicted = activate_probs(
                    probs=y_prob_pred, strategy="max", force_max_1_per_row=True
                )
            else:
                labels_predicted = activate_probs(
                    probs=y_prob_pred,
                    strategy="threshold",
                    threshold=threshold,
                    force_max_1_per_row=False,
                )

            df_train_results["predicted_label-matrix"] = y_prob_pred.tolist()
            df_train_results["predicted_label"] = [
                "|".join(matrix_to_label(row, id2label)) for row in labels_predicted
            ]

            if self.training_kind == "multiclass":
                metrics_train = get_metrics_multiclass(
                    Y_true=df_train_results["true_label"],
                    Y_pred=df_train_results["predicted_label"],
                    texts=df_train_results["path"],
                    id2label=id2label,
                )
            else:
                metrics_train = get_metrics_multilabel(
                    Y_true=train_label_ids,
                    Y_pred=labels_predicted,
                    texts=df_train_results["path"],
                    id2label=id2label,
                )

            if "test" in self.ds:
                self.__listen_stop_event()
                test_ds = cast(datasets.Dataset, self.ds["test"])
                predictions_test = trainer.predict(cast(TorchDataset, test_ds))
                test_label_ids = cast(np.ndarray, predictions_test.label_ids)
                test_logits = cast(np.ndarray, predictions_test.predictions)
                df_test_results = pd.DataFrame({"id": test_ids}).set_index("id")
                df_test_results["path"] = test_paths
                df_test_results["true_label-matrix"] = test_label_ids.tolist()
                df_test_results["true_label"] = [
                    "|".join(matrix_to_label(row, id2label)) for row in test_label_ids
                ]

                y_prob_pred_test = logits_to_probs(test_logits, kind=self.training_kind)
                if self.training_kind == "multiclass":
                    y_label_pred = activate_probs(
                        y_prob_pred_test, strategy="max", force_max_1_per_row=True
                    )
                else:
                    y_label_pred = activate_probs(y_prob_pred_test, threshold, strategy="threshold")
                df_test_results["predicted_label-matrix"] = y_prob_pred_test.tolist()
                df_test_results["predicted_label"] = [
                    "|".join(matrix_to_label(row, id2label)) for row in y_label_pred
                ]

                if self.training_kind == "multiclass":
                    metrics_test = get_metrics_multiclass(
                        Y_true=df_test_results["true_label"],
                        Y_pred=df_test_results["predicted_label"],
                        texts=df_test_results["path"],
                        id2label=id2label,
                    )
                else:
                    metrics_test = get_metrics_multilabel(
                        Y_true=test_label_ids,
                        Y_pred=y_label_pred,
                        texts=df_test_results["path"],
                        id2label=id2label,
                    )
            else:
                df_test_results = None
                metrics_test = None
            task_timer.stop("evaluate")

            task_timer.start("save_files")
            # last cancellation point before saving + archiving the model dir
            self.__listen_stop_event()
            params_to_save = self.params.model_dump()
            params_to_save.update(
                {
                    "training_kind": self.training_kind,
                    "test_size": self.test_size,
                    "threshold": 0.5 if self.training_kind == "multilabel" else None,
                    "base_model": self.base_model,
                    "n_train": len(self.ds["train"]),
                    "device": str(device),
                    "loss": self.loss,
                    "balance classes": self.class_balance,
                    "class_min_freq": self.class_min_freq,
                    "fp16": self.fp16,
                    "kind": "image",
                }
            )
            self.__create_save_files(
                current_path=current_path,
                log_path=log_path,
                df_train_results=df_train_results,
                df_test_results=df_test_results,
                training_data=self.df[[self.col_text, self.col_label]],
                model=model,
                image_processor=image_processor,
                params_to_save=params_to_save,
                metrics_train=metrics_train,
                metrics_test=metrics_test,
            )
            task_timer.stop("save_files")

        except Exception as e:
            print("Error in image-classification training", e)
            # Mirrors TrainBert: scrub the partial model dir on failure /
            # cancellation so it doesn't masquerade as a trained model.
            if current_path.exists():
                shutil.rmtree(current_path)
            raise e
        finally:
            print("Cleaning memory")
            # release references one by one: a single failed del must not
            # skip the CUDA cleanup (a polluted reused worker breaks the
            # next GPU task in the queue)
            trainer = None
            model = None
            image_processor = None
            self.df = None
            self.ds = None
            self.event = None
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception as e:
                print("Error in cleaning GPU memory", e)

        return EventsModel(events=task_timer.get_events())
