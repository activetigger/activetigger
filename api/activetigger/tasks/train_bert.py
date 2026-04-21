import gc
import json
import logging
import multiprocessing
import multiprocessing.synchronize
import os
import shutil
from collections import Counter
from logging import Logger
from pathlib import Path
from typing import Any, Optional, Tuple

import datasets
import pandas as pd
import torch
from pandas import DataFrame
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,  # ty: ignore[possibly-missing-import]
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
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
from activetigger.monitoring import TaskTimer
from activetigger.tasks.base_task import BaseTask
from activetigger.tasks.predict_bert import annotations_to_matrix
from activetigger.tasks.utils import length_after_tokenizing, retrieve_model_max_length

pd.set_option("future.no_silent_downcasting", True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomLoggingCallback(TrainerCallback):
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
        progress_percentage = (state.global_step / state.max_steps) * 100
        with open(self.current_path.joinpath("progress_train"), "w") as f:
            f.write(str(progress_percentage))
        # Normalize training loss: HuggingFace Trainer accumulates raw losses
        # across forward passes but divides only by optimizer steps, inflating
        # the logged training loss by gradient_accumulation_steps.
        gradacc = args.gradient_accumulation_steps
        adjusted_history = []
        for entry in state.log_history:
            if "loss" in entry and "eval_loss" not in entry and gradacc > 1:
                entry = dict(entry)
                entry["loss"] = entry["loss"] / gradacc
            adjusted_history.append(entry)
        with open(self.current_path.joinpath("log_history.txt"), "w") as f:
            json.dump(adjusted_history, f)
        # end if event set
        if self.event is not None:
            if self.event.is_set():
                self.logger.info("Event set, stopping training.")
                control.should_training_stop = True
                raise Exception("Process interrupted by user")


# Function for the weighted loss computation


# Rescaling the weights
def compute_class_weights(dataset, label_key="labels"):
    # Labels are stored as one-hot vectors; convert to class indices
    labels = [example[label_key].argmax().item() for example in dataset]
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    num_classes = len(label_counts)

    # Inverse frequency weight, ordered by label index
    weights = [total / (num_classes * label_counts[k]) for k in sorted(label_counts.keys())]
    return torch.tensor(weights, dtype=torch.float)


# CustomTrainer is a subclass of Trainer that allows for custom loss computation.
# https://stackoverflow.com/questions/70979844/using-weights-with-transformers-huggingface
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        print("CustomTrainer initialized with class weights:", self.class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # ty: ignore[invalid-method-override]
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Convert one-hot labels to class indices for CrossEntropyLoss
        label_indices = labels.argmax(dim=-1)
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),  # ty: ignore[unresolved-attribute]
        )
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),  # ty: ignore[unresolved-attribute]
            label_indices.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


class TrainBert(BaseTask):
    """
    Class to train a bert model

    Parameters:
    ----------
    path (Path): path to save the files
    name (str): name of the model
    df (DataFrame): labelled data
    col_text (str): text column
    col_label (str): label column
    base_model (str): model to use
    params (dict) : training parameters
    test_size (dict): train/test distribution
    event : possibility to interrupt
    unique_id : unique id for the current task
    loss : loss function to use (cross_entropy, weighted_cross_entropy)

    TODO : test more weighted loss entropy
    """

    kind = "train_bert"

    def __init__(
        self,
        path: Path,
        project_slug: str,
        model_name: str,
        df: DataFrame | datasets.Dataset,
        training_kind: str,
        scheme_labels: list[str],
        use_dichotomization: bool,
        col_text: str,
        col_label: str,
        base_model: str,
        params: LMParametersModel,
        test_size: float,
        label_for_dichotomization: str | None = None,
        event: Optional[multiprocessing.synchronize.Event] = None,
        unique_id: Optional[str] = None,
        loss: Optional[str] = "cross_entropy",
        max_length: int = 512,
        auto_max_length: bool = False,
        class_balance: bool = False,
        class_min_freq: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.project_slug = project_slug
        self.name = model_name
        df.index.name = "id"  # ty: ignore[unresolved-attribute]
        self.df = df
        if training_kind not in ["multiclass", "multilabel"]:
            raise ValueError(
                (
                    f"TrainBERT only works for multiclass and "
                    f"multilabel but you set training_kind = {training_kind}"
                )
            )
        self.training_kind = training_kind
        if len(scheme_labels) != len(set(scheme_labels)):
            raise ValueError(
                (f"Labels in your scheme are not unique.\nLabels provided : {scheme_labels}")
            )
        if use_dichotomization:
            raise ValueError("Dichotomization not supported in multilabel.")
        self.use_dichotomization = use_dichotomization
        self.label_for_dichotomization = label_for_dichotomization
        self.scheme_labels = scheme_labels
        self.col_text = col_text
        self.col_label = col_label
        self.base_model = base_model
        self.params = params
        self.test_size = test_size
        self.event = event
        self.unique_id = unique_id
        if loss == "weighted_cross_entropy" and training_kind == "multilabel":
            raise ValueError(
                "weighted_cross_entropy loss is not supported for multilabel classification."
            )
        self.loss = loss
        self.max_length = max_length
        self.auto_max_length = auto_max_length
        self.class_balance = class_balance
        self.class_min_freq = class_min_freq

    def __init_paths(self) -> Tuple[Path, Path]:
        """Initiate the current path (directory for the model) and for the logger"""
        #  create repertory for the specific model
        current_path = self.path.joinpath(self.name)
        if not current_path.exists():
            os.makedirs(current_path)
        # logging the process
        log_path = current_path.joinpath("status.log")
        return current_path, log_path

    def __init_logger(self, log_path) -> Logger:
        """Load the logger and set it up"""
        logger = logging.getLogger("train_bert_model")
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Start {self.base_model}")
        return logger

    def __check_data(self, df: pd.DataFrame, col_label: str, col_text: str) -> pd.DataFrame:
        """Remove rows missing labels or text"""
        df = df.copy()
        # test labels missing values and remove them
        if df[col_label].isnull().sum() > 0:
            df = df[df[col_label].notnull()]
            self.logger.info(f"Missing labels - reducing training data to {len(df)}")
            print(f"Missing labels - reducing training data to {len(df)}")

        # test empty texts and remove them
        if df[col_text].isnull().sum() > 0:
            df = df[df[col_text].notnull()]
            self.logger.info(f"Missing texts - reducing training data to {len(df)}")
            print(f"Missing texts - reducing training data to {len(df)}")

        # Test that all labels in the label column appear in the scheme labels
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
            print(f"Labels not recognised - reducing training data to {len(df)}")

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
        """Transform the dataframe into a dataset with the right format for
        training"""
        ids = df.reset_index()["id"].copy().to_list()
        texts = df[col_text].copy().to_list()
        if training_kind == "multiclass":
            print("Preprocess multiclass")
            labels_as_list = df[col_label].copy().replace(label2id)
            labels_as_matrix = [
                [int(id_label == i_column) for i_column in range(len(label2id))]
                for id_label in labels_as_list
            ]
        elif training_kind == "multilabel":
            print("Preprocess multilabel")
            labels_as_matrix = annotations_to_matrix(df[col_label], list(label2id.keys())).tolist()

        return datasets.Dataset.from_dict(
            {"id": ids, "text": texts, "labels": torch.Tensor(labels_as_matrix)}  # ty: ignore[possibly-unresolved-reference]
        ).with_format("torch")

    def __load_tokenizer(self, base_model: str):
        """Load the tokenize"""
        return AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    def __cap_tokenizer_max_length(
        self,
        texts: pd.Series,
        tokenizer,
        auto_max_length: bool,
        original_max_length: int,
        base_model_max_length: int,
        adapt: bool,
    ) -> Tuple[Any, int]:
        """Cap the tokenizer max length and create a tokenizing function"""

        # if auto_max_length set max_length to the maximum length of tokenized sentences
        # Tokenize the text column
        def get_n_tokens(txt):
            return length_after_tokenizing(txt, tokenizer)

        if auto_max_length:
            max_length = int(texts.apply(get_n_tokens).dropna().max())

        # cap max_length
        max_length = min(original_max_length, base_model_max_length)
        # evaluate the proportion of elements truncated
        percentage_truncated = int(100 * (texts.apply(get_n_tokens).dropna() > max_length).mean())

        if adapt:

            def tokenizing_function(e):
                return tokenizer(
                    e["text"],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=int(max_length),
                )
        else:

            def tokenizing_function(e):
                return tokenizer(
                    e["text"],
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                    max_length=max_length,
                )

        return tokenizing_function, percentage_truncated

    def __load_trainer(
        self,
        current_path: Path,
        ds: datasets.DatasetDict,
        bert_model,
        params: LMParametersModel,
        loss: str,
    ) -> Trainer:
        """Load the training arguments and update the configuration"""

        # Calculate the number of steps (total, warmup and eval)
        has_test = "test" in ds

        total_steps = (float(params.epochs) * len(ds["train"])) // (
            int(params.batchsize) * float(params.gradacc)
        )
        warmup_steps = int((total_steps) // 10)
        eval_steps = (total_steps - warmup_steps) // params.eval
        eval_steps = max(eval_steps, 1)

        # Load the training arguments
        training_args = TrainingArguments(
            # Directories
            output_dir=str(current_path.joinpath("train")),
            logging_dir=str(current_path.joinpath("logs")),
            # Hyperparameters
            learning_rate=float(params.lrate),
            weight_decay=float(params.wdecay),
            num_train_epochs=float(params.epochs),
            warmup_steps=int(warmup_steps),
            # Batch sizes
            gradient_accumulation_steps=int(params.gradacc),
            per_device_train_batch_size=int(params.batchsize),
            per_device_eval_batch_size=int(params.batchsize),
            # Logging and saving parameters
            eval_strategy="steps" if has_test else "no",
            eval_steps=eval_steps if has_test else None,
            save_strategy="best" if has_test else "epoch",
            metric_for_best_model="eval_loss" if has_test else None,
            save_steps=float(eval_steps) if has_test else 500,
            logging_steps=int(eval_steps),
            do_eval=has_test,
            greater_is_better=False if has_test else None,
            load_best_model_at_end=params.best if has_test else False,
            use_cpu=config.cpu_only or not bool(params.gpu),  # deactivate gpu
        )

        callback = CustomLoggingCallback(self.event, current_path=current_path, logger=self.logger)
        eval_dataset = ds["test"] if has_test else None
        if loss == "cross_entropy":
            trainer = Trainer(
                model=bert_model,
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=eval_dataset,
                callbacks=[callback],
            )
        elif loss == "weighted_cross_entropy":
            print("Using weighted cross entropy loss - EXPERIMENTAL")
            trainer = CustomTrainer(
                model=bert_model,
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=eval_dataset,
                callbacks=[callback],
                class_weights=compute_class_weights(ds["train"], label_key="labels"),
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
        bert_model,
        params_to_save: dict[str, Any],
        metrics_train: MLStatisticsModel,
        metrics_test: MLStatisticsModel | None,
    ) -> None:
        """Save the model and parameters
        Save the following objects:
        - predictions of the train set (csv)
        - predictions of the test set  (csv)
        - data used during the training (parquet)
        - the trained model
        - the parameters used during the training (json)
        - metrics (json)

        Also delete intermediate files
        """

        # Save results for the train and test set
        (
            df_train_results[
                [c for c in df_train_results.columns if c not in ["input_ids", "attention_mask"]]
            ].to_csv(current_path.joinpath("train_dataset_eval.csv"))
        )
        if df_test_results is not None:
            (
                df_test_results[
                    [c for c in df_test_results.columns if c not in ["input_ids", "attention_mask"]]
                ].to_csv(current_path.joinpath("test_dataset_eval.csv"))
            )
        training_data.to_parquet(current_path.joinpath("training_data.parquet"))

        # save the trained bert model
        bert_model.save_pretrained(current_path)

        # Save parameters
        with open(current_path.joinpath("parameters.json"), "w") as f:
            json.dump(params_to_save, f)

        # remove intermediate steps and logs if succeed
        shutil.rmtree(current_path.joinpath("train"))
        os.rename(log_path, current_path.joinpath("finished"))

        # make archive (create dir if needed)
        path_static = f"{config.data_path}/projects/static/{self.project_slug}"
        os.makedirs(path_static, exist_ok=True)
        shutil.make_archive(
            f"{path_static}/{self.name}",
            "gztar",
            str(self.path.joinpath(self.name)),
        )

        metrics_data: dict[str, Any] = {
            "train": metrics_train.model_dump(mode="json"),
        }
        if metrics_test is not None:
            metrics_data["trainvalid"] = metrics_test.model_dump(mode="json")
        with open(str(current_path.joinpath("metrics_training.json")), "w") as f:
            json.dump(metrics_data, f)

    def __call__(self) -> EventsModel:
        """
        Main process to the task
        """
        task_timer = TaskTimer(compulsory_steps=["setup", "train", "evaluate", "save_files"])
        task_timer.start("setup")

        current_path, log_path = self.__init_paths()
        self.logger = self.__init_logger(log_path)
        device = get_device()

        self.df = self.__check_data(
            self.df,  # ty: ignore[invalid-argument-type]
            self.col_label,
            self.col_text,
        )
        labels, label2id, id2label = self.__retrieve_labels(self.scheme_labels)
        self.ds = self.__transform_to_dataset(
            self.training_kind, self.df, self.col_label, self.col_text, label2id
        )

        tokenizer = self.__load_tokenizer(self.base_model)
        tokenizing_function, percentage_truncated = self.__cap_tokenizer_max_length(
            texts=self.df[self.col_text],
            tokenizer=tokenizer,
            auto_max_length=self.auto_max_length,
            original_max_length=self.max_length,
            base_model_max_length=retrieve_model_max_length(self.base_model),
            adapt=self.params.adapt,
        )
        self.ds = self.ds.map(tokenizing_function, batched=True)

        # Build train/test dataset for dev eval
        if self.test_size > 0:
            self.ds = self.ds.train_test_split(test_size=self.test_size)
        else:
            self.ds = datasets.DatasetDict({"train": self.ds})
        self.logger.info("Train/test dataset created")

        # Model
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=True,
            problem_type="multi_label_classification"
            if self.training_kind == "multilabel"
            else None,
        ).to(device=device)
        bert_model.config.use_cache = False
        self.logger.info(f"Model loaded on {bert_model.device}")
        print(f"Model loaded on {bert_model.device}")

        try:
            trainer = self.__load_trainer(
                current_path, self.ds, bert_model, self.params, self.loss or "cross_entropy"
            )
            task_timer.stop("setup")

            task_timer.start("train")
            trainer.train()
            self.logger.info(f"Model trained {current_path}")
            task_timer.stop("train")

            # predict on the data (separation validation set and training set)
            task_timer.start("evaluate")
            predictions_train = trainer.predict(self.ds["train"])  # ty: ignore[invalid-argument-type]

            # Compute the metrics
            df_train_results = self.ds["train"].to_pandas().set_index("id")  # ty: ignore[unresolved-attribute]

            df_train_results["true_label-matrix"] = predictions_train.label_ids.tolist()  # ty: ignore[unresolved-attribute]
            df_train_results["true_label"] = [
                "|".join(matrix_to_label(row, id2label))  # ty: ignore[invalid-argument-type]
                for row in predictions_train.label_ids  # ty: ignore[not-iterable]
            ]

            y_prob_pred = logits_to_probs(
                predictions_train.predictions,  # ty: ignore[invalid-argument-type]
                self.training_kind,
            )

            if self.training_kind == "multiclass":
                labels_predicted = activate_probs(
                    probs=y_prob_pred, strategy="max", force_max_1_per_row=True
                )
            elif self.training_kind == "multilabel":
                # threshold = find_best_threshold(
                #     y_true = predictions_train.label_ids,
                #     y_prob_pred = y_prob_pred,
                # )
                threshold = 0.5  # Force threshold = 0.5
                labels_predicted = activate_probs(
                    probs=y_prob_pred,
                    strategy="threshold",
                    threshold=threshold,
                    force_max_1_per_row=False,
                )

            df_train_results["predicted_label-matrix"] = y_prob_pred.tolist()
            df_train_results["predicted_label"] = [
                "|".join(matrix_to_label(row, id2label))
                for row in labels_predicted  # ty: ignore[possibly-unresolved-reference]
            ]

            if self.training_kind == "multiclass":
                metrics_train = get_metrics_multiclass(
                    Y_true=df_train_results["true_label"],
                    Y_pred=df_train_results["predicted_label"],
                    texts=df_train_results["text"],
                    id2label=id2label,
                )
            elif self.training_kind == "multilabel":
                metrics_train = get_metrics_multilabel(
                    Y_true=predictions_train.label_ids,  # ty: ignore[invalid-argument-type]
                    Y_pred=labels_predicted,  # ty: ignore[possibly-unresolved-reference]
                    texts=df_train_results["text"],
                    id2label=id2label,
                )

            if "test" in self.ds:
                predictions_test = trainer.predict(self.ds["test"])  # ty: ignore[invalid-argument-type]
                df_test_results = self.ds["test"].to_pandas().set_index("id")  # ty: ignore[unresolved-attribute]

                df_test_results["true_label-matrix"] = predictions_test.label_ids.tolist()  # ty: ignore[unresolved-attribute]
                df_test_results["true_label"] = [
                    "|".join(matrix_to_label(row, id2label))  # ty: ignore[invalid-argument-type]
                    for row in predictions_test.label_ids  # ty: ignore[not-iterable]
                ]

                y_prob_pred = logits_to_probs(
                    predictions_test.predictions,  # ty: ignore[invalid-argument-type]
                    kind=self.training_kind,
                )
                if self.training_kind == "multiclass":
                    y_label_pred = activate_probs(
                        y_prob_pred, strategy="max", force_max_1_per_row=True
                    )
                else:
                    y_label_pred = activate_probs(y_prob_pred, threshold, strategy="threshold")  # ty: ignore[possibly-unresolved-reference]
                df_test_results["predicted_label-matrix"] = y_prob_pred.tolist()
                df_test_results["predicted_label"] = [
                    "|".join(matrix_to_label(row, id2label)) for row in y_label_pred
                ]

                if self.training_kind == "multiclass":
                    metrics_test = get_metrics_multiclass(
                        Y_true=df_test_results["true_label"],
                        Y_pred=df_test_results["predicted_label"],
                        texts=df_test_results["text"],
                        id2label=id2label,
                    )
                elif self.training_kind == "multilabel":
                    metrics_test = get_metrics_multilabel(
                        Y_true=predictions_test.label_ids,  # ty: ignore[invalid-argument-type]
                        Y_pred=y_label_pred,
                        texts=df_test_results["text"],
                        id2label=id2label,
                    )

            else:
                df_test_results = None
                metrics_test = None
            task_timer.stop("evaluate")

            task_timer.start("save_files")
            params_to_save = self.params.model_dump()
            params_to_save.update(
                {
                    "training_kind": self.training_kind,
                    "test_size": self.test_size,
                    "threshold": threshold if self.training_kind == "multilabel" else None,  # ty: ignore[possibly-unresolved-reference]
                    "use_dichotomization": self.use_dichotomization,
                    "label_for_dichotomization": self.label_for_dichotomization,
                    "base_model": self.base_model,
                    "n_train": len(self.ds["train"]),
                    "max_length": self.max_length,
                    "device": str(device),
                    "Proportion of elements truncated (%)": percentage_truncated,
                    "loss": self.loss,
                    "auto context length": self.params.adapt,
                    "balance classes": self.class_balance,
                    "class_min_freq": self.class_min_freq,
                }
            )
            self.__create_save_files(
                current_path=current_path,
                log_path=log_path,
                df_train_results=df_train_results,
                df_test_results=df_test_results,
                training_data=self.df[[self.col_text, self.col_label]],
                bert_model=bert_model,
                params_to_save=params_to_save,
                metrics_train=metrics_train,  # ty: ignore[possibly-unresolved-reference]
                metrics_test=metrics_test,  # ty: ignore[possibly-unresolved-reference]
            )
            task_timer.stop("save_files")

        except Exception as e:
            print("Error in training", e)
            shutil.rmtree(current_path)
            raise e
        finally:
            print("Cleaning memory")
            try:
                del (
                    trainer,
                    bert_model,
                    self.df,
                    self.ds,
                    device,
                    self.event,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()

            except Exception as e:
                print("Error in cleaning memory", e)

        return EventsModel(events=task_timer.get_events())
