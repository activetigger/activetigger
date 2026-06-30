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
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,  # ty: ignore[possibly-missing-import]
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from activetigger.config import config
from activetigger.datamodels import EventsModel, LMParametersModel
from activetigger.functions import get_device
from activetigger.monitoring import TaskTimer
from activetigger.ner_metrics import compute_ner_metrics
from activetigger.tasks.base_task import BaseTask
from activetigger.tasks.train_bert import CustomLoggingCallback
from activetigger.tasks.utils import retrieve_model_max_length

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Sentinel for tokens that should not contribute to the loss (special tokens,
# subword continuations). Matches HuggingFace's convention.
IGNORE_LABEL = -100


def parse_spans(raw: Any) -> list[dict]:
    """Parse a single annotation cell into a list of {start, end, tag} dicts.

    Empty / non-string values yield an empty list, which means "no entities"
    (a valid label, not missing data — caller filters NaN before reaching here).
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, dict)]
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [s for s in parsed if isinstance(s, dict)]
    return []


def build_bio_labels(scheme_labels: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Build BIO label vocabulary from the user-defined scheme labels.

    "O" gets id 0 so we can default to it cheaply when initializing label tensors.
    """
    labels = ["O"]
    for tag in scheme_labels:
        labels.append(f"B-{tag}")
        labels.append(f"I-{tag}")
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return labels, label2id, id2label


def align_labels(
    spans: list[dict],
    offset_mapping: list[tuple[int, int]],
    word_ids: list[int | None],
    label2id: dict[str, int],
) -> list[int]:
    """Token-level BIO labels aligned to offset_mapping.

    Create a mask that indicate the token that starts each entity based on the word splitting

    First subword of a word gets the BIO tag, continuation subwords and
    special tokens get IGNORE_LABEL.
    """
    n = len(offset_mapping)
    out = [IGNORE_LABEL] * n
    o_id = label2id["O"]

    seen_word: set[int] = set()
    first_indices: list[int] = []
    for i, w in enumerate(word_ids):
        if w is None:
            continue
        if w in seen_word:
            continue
        seen_word.add(w)
        first_indices.append(i)
        out[i] = o_id

    # For each gold span, mark the first first-subword it touches as B-tag,
    # the subsequent first-subwords as I-tag. Spans dropped if no overlap
    # (e.g. truncated past max_length).
    for span in spans:
        gold_start = span.get("start")
        gold_end = span.get("end")
        tag = span.get("tag")
        if gold_start is None or gold_end is None or tag is None:
            continue
        b_key = f"B-{tag}"
        i_key = f"I-{tag}"
        if b_key not in label2id:
            continue
        first = True
        for i in first_indices:
            tok_start, tok_end = offset_mapping[i]
            if tok_end <= gold_start or tok_start >= gold_end:
                continue
            out[i] = label2id[b_key] if first else label2id[i_key]
            first = False
    return out


def token_labels_to_entities(
    label_ids: list[int], id2label: dict[int, str]
) -> set[tuple[int, int, str]]:
    """Collapse a token-level BIO id sequence into a set of entity tuples
    (start_token_idx, end_token_idx_inclusive, tag).

    Used by ``compute_entity_f1`` to score predictions during training.
    Tokens labeled ``IGNORE_LABEL`` (-100, e.g. subword continuations and
    special tokens) are skipped — they neither extend nor break a current
    entity, mirroring how HuggingFace handles the loss mask. A bare ``I-X``
    without a preceding ``B-X`` is treated as ``B-X`` (permissive decode).
    """
    entities: set[tuple[int, int, str]] = set()
    current_tag: str | None = None
    current_start: int | None = None
    last_idx: int | None = None
    for i, lab_id in enumerate(label_ids):
        if lab_id == IGNORE_LABEL:
            continue
        label = id2label.get(int(lab_id), "O")
        if label == "O":
            if current_tag is not None and current_start is not None and last_idx is not None:
                entities.add((current_start, last_idx, current_tag))
            current_tag = None
            current_start = None
        elif label.startswith("B-") or label.startswith("I-"):
            tag = label[2:]
            new_entity = label.startswith("B-") or current_tag != tag
            if new_entity:
                if current_tag is not None and current_start is not None and last_idx is not None:
                    entities.add((current_start, last_idx, current_tag))
                current_tag = tag
                current_start = i
            last_idx = i
    if current_tag is not None and current_start is not None and last_idx is not None:
        entities.add((current_start, last_idx, current_tag))
    return entities


def compute_entity_f1(eval_pred, id2label: dict[int, str]) -> dict[str, float]:
    """Trainer ``compute_metrics`` callback: returns entity-level
    precision / recall / F1, computed by extracting (token_start, token_end,
    tag) tuples from gold and predicted BIO sequences and intersecting.

    Token-position F1 (rather than character-position) is intentional here:
    during training we don't have offset_mapping in the Trainer's reach,
    and token-level entity F1 already correlates strongly with the
    character-level metric. Final evaluation still runs the full
    character-level pipeline in ``__evaluate_split``.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    total_tp = total_fp = total_fn = 0
    for pred_seq, gold_seq in zip(predictions, labels):
        gold_entities = token_labels_to_entities(list(gold_seq), id2label)
        pred_entities = token_labels_to_entities(list(pred_seq), id2label)
        tp = len(gold_entities & pred_entities)
        total_tp += tp
        total_fp += len(pred_entities) - tp
        total_fn += len(gold_entities) - tp
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall}


def decode_spans(
    pred_label_ids: list[int],
    offset_mapping: list[tuple[int, int]],
    word_ids: list[int | None],
    id2label: dict[int, str],
    text: str | None = None,
) -> list[dict]:
    """Decode token-level BIO predictions into a list of {start, end, tag}.

    Predictions are read only at the first subword of each word; continuation
    subwords share the predicted entity but extend the span's character range.
    Contiguous B-X/I-X (or I-X without a preceding B, which we still accept)
    of the same tag are merged into a single span.

    If ``text`` is provided, leading/trailing whitespace inside each decoded
    span's character range is trimmed. SentencePiece tokenizers (CamemBERT,
    XLM-R, …) often include the leading space in a token's offset, so the
    raw span would otherwise span one extra character on each side.
    """
    if not offset_mapping:
        return []

    word_to_subwords: dict[int, list[int]] = {}
    word_order: list[int] = []
    for i, w in enumerate(word_ids):
        if w is None:
            continue
        if w not in word_to_subwords:
            word_to_subwords[w] = []
            word_order.append(w)
        word_to_subwords[w].append(i)

    spans: list[dict] = []
    current: dict | None = None
    for w in word_order:
        subs = word_to_subwords[w]
        first = subs[0]
        last = subs[-1]
        label_id = pred_label_ids[first]
        label = id2label.get(label_id, "O")
        char_start = offset_mapping[first][0]
        char_end = offset_mapping[last][1]
        if label == "O":
            if current is not None:
                spans.append(current)
                current = None
            continue
        if label.startswith("B-") or label.startswith("I-"):
            tag = label[2:]
        else:
            tag = label
        if current is not None and current["tag"] == tag and not label.startswith("B-"):
            current["end"] = char_end
        else:
            if current is not None:
                spans.append(current)
            current = {"start": char_start, "end": char_end, "tag": tag}
    if current is not None:
        spans.append(current)

    if text is not None:
        for s in spans:
            start, end = s["start"], s["end"]
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            s["start"] = start
            s["end"] = end
        spans = [s for s in spans if s["end"] > s["start"]]

    # Merge consecutive same-tag spans whose gap is empty or whitespace-only.
    # BIO decoding fragments an entity into adjacent spans when the model
    # emits `B-X` instead of `I-X` on a continuation (e.g. "John Smith"
    # predicted as `B-PER B-PER`). After whitespace trimming we can safely
    # stitch them back into a single span.
    if spans:
        spans.sort(key=lambda s: s["start"])
        merged: list[dict] = [dict(spans[0])]
        for span in spans[1:]:
            prev = merged[-1]
            if prev["tag"] == span["tag"] and prev["end"] <= span["start"]:
                gap_clean = text is None or text[prev["end"] : span["start"]].strip() == ""
                if gap_clean:
                    prev["end"] = span["end"]
                    continue
            merged.append(dict(span))
        spans = merged
    return spans


class TrainNer(BaseTask):
    """
    Fine-tune a token-classification model on span-scheme annotations.

    Mirrors TrainBert's lifecycle (progress / loss / log files, save layout)
    so the existing language-model orchestration code can drive it without
    branching, but operates on BIO token tags instead of sequence labels.
    """

    kind = "train_ner"

    def __init__(
        self,
        path: Path,
        project_slug: str,
        model_name: str,
        df: DataFrame,
        scheme_labels: list[str],
        col_text: str,
        col_label: str,
        base_model: str,
        params: LMParametersModel,
        test_size: float,
        event: Optional[multiprocessing.synchronize.Event] = None,
        unique_id: Optional[str] = None,
        max_length: int = 512,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.project_slug = project_slug
        self.name = model_name
        df.index.name = "id"
        self.df = df
        if len(scheme_labels) == 0:
            raise ValueError("Scheme has no labels — define labels before training NER.")
        if len(scheme_labels) != len(set(scheme_labels)):
            raise ValueError(f"Labels not unique: {scheme_labels}")
        self.scheme_labels = scheme_labels
        self.col_text = col_text
        self.col_label = col_label
        self.base_model = base_model
        self.params = params
        self.test_size = test_size
        self.event = event
        self.unique_id = unique_id
        self.max_length = max_length

    def __init_paths(self) -> Tuple[Path, Path]:
        current_path = self.path.joinpath(self.name)
        if not current_path.exists():
            os.makedirs(current_path)
        log_path = current_path.joinpath("status.log")
        return current_path, log_path

    def __init_logger(self, log_path: Path) -> Logger:
        logger = logging.getLogger("train_ner_model")
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Start NER training on {self.base_model}")
        return logger

    def __prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[df[self.col_text].notnull()]
        df = df[df[self.col_label].notnull()]
        df["__spans__"] = df[self.col_label].apply(parse_spans)
        # filter spans referencing unknown tags so they don't get silently
        # mapped to O — keep their text examples, just drop the bad spans
        scheme_set = set(self.scheme_labels)

        def _clean(spans: list[dict]) -> list[dict]:
            cleaned = []
            for s in spans:
                if not isinstance(s, dict):
                    continue
                tag = s.get("tag")
                if tag in scheme_set:
                    cleaned.append(s)
            return cleaned

        df["__spans__"] = df["__spans__"].apply(_clean)
        return df

    def __tokenize_and_align(
        self,
        df: pd.DataFrame,
        tokenizer,
        label2id: dict[str, int],
        effective_max_length: int,
    ) -> datasets.Dataset:
        texts = df[self.col_text].astype(str).tolist()
        ids = df.reset_index()["id"].tolist()
        spans_per_doc = df["__spans__"].tolist()

        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=effective_max_length,
            return_offsets_mapping=True,
            padding=False,
        )

        all_labels: list[list[int]] = []
        for idx, spans in enumerate(spans_per_doc):
            offsets = encoded["offset_mapping"][idx]
            word_ids = encoded.word_ids(batch_index=idx)
            all_labels.append(align_labels(spans, offsets, word_ids, label2id))

        ds = datasets.Dataset.from_dict(
            {
                "id": [str(i) for i in ids],
                "text": texts,
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "offset_mapping": encoded["offset_mapping"],
                # word_ids needs to be stored per row; recompute later for eval
                "labels": all_labels,
            }
        )
        return ds

    def __load_trainer(
        self,
        current_path: Path,
        ds: datasets.DatasetDict,
        model,
        tokenizer,
        params: LMParametersModel,
        id2label: dict[int, str],
    ) -> Trainer:
        has_test = "test" in ds
        total_steps = (float(params.epochs) * len(ds["train"])) // (
            int(params.batchsize) * float(params.gradacc)
        )
        warmup_steps = int(total_steps // 10)
        eval_steps = max(int((total_steps - warmup_steps) // params.eval), 1)
        seed = int(config.random_seed)

        # The Trainer sees only input_ids / attention_mask / labels — the
        # offset_mapping and text columns are kept on the dataset for our
        # own decoding step after training finishes.
        train_for_trainer = ds["train"].remove_columns(
            [
                c
                for c in ds["train"].column_names
                if c not in ("input_ids", "attention_mask", "labels")
            ]
        )
        eval_for_trainer = None
        if has_test:
            eval_for_trainer = ds["test"].remove_columns(
                [
                    c
                    for c in ds["test"].column_names
                    if c not in ("input_ids", "attention_mask", "labels")
                ]
            )

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
            # NER is selected on entity F1 rather than eval_loss because a
            # token-classification model can hit lower loss by over-predicting
            # the dominant `O` class without improving entity-level quality.
            metric_for_best_model="eval_f1" if has_test else None,
            save_steps=float(eval_steps) if has_test else 500,
            logging_steps=int(eval_steps),
            do_eval=has_test,
            greater_is_better=True if has_test else None,
            load_best_model_at_end=params.best if has_test else False,
            use_cpu=config.cpu_only or not bool(params.gpu),
            seed=seed,
            data_seed=seed,
        )

        collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        callback = CustomLoggingCallback(self.event, current_path=current_path, logger=self.logger)
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_for_trainer,
            eval_dataset=eval_for_trainer,
            data_collator=collator,
            callbacks=[callback],
            # Only attach compute_metrics when there's an eval split; without
            # one HF skips evaluation entirely and the callback never runs.
            compute_metrics=(
                (lambda eval_pred: compute_entity_f1(eval_pred, id2label)) if has_test else None
            ),
        )

    def __evaluate_split(
        self,
        trainer: Trainer,
        ds: datasets.Dataset,
        tokenizer,
        id2label: dict[int, str],
    ) -> Tuple[pd.DataFrame, dict[str, Any]]:
        eval_ds = ds.remove_columns(
            [c for c in ds.column_names if c not in ("input_ids", "attention_mask", "labels")]
        )
        predictions = trainer.predict(cast(Any, eval_ds))
        logits = cast(np.ndarray, predictions.predictions)
        pred_ids = np.argmax(logits, axis=-1)

        gold_per_doc: list[list[dict]] = []
        pred_per_doc: list[list[dict]] = []
        texts: list[str] = []
        ids: list[str] = []
        records: list[dict[str, Any]] = []
        for row_idx in range(len(ds)):
            text = ds[row_idx]["text"]
            texts.append(text)
            ids.append(str(ds[row_idx]["id"]))
            offsets = ds[row_idx]["offset_mapping"]
            # Rebuild word_ids on the fly — encode the text again to recover
            # word boundaries. Cheaper than persisting word_ids in the dataset.
            enc = tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=False,
                padding=False,
            )
            word_ids = enc.word_ids()
            # offsets coming back from the dataset may have been padded by the
            # collator; trim to word_ids length so they align.
            trimmed_offsets = offsets[: len(word_ids)]
            trimmed_pred = pred_ids[row_idx][: len(word_ids)].tolist()
            pred_spans = decode_spans(trimmed_pred, trimmed_offsets, word_ids, id2label, text=text)
            gold_spans = self.gold_by_id.get(str(ds[row_idx]["id"]), [])
            gold_per_doc.append(list(gold_spans))
            pred_per_doc.append(pred_spans)
            records.append(
                {
                    "id": ds[row_idx]["id"],
                    "text": text,
                    "gold_spans": json.dumps(gold_per_doc[-1]),
                    "predicted_spans": json.dumps(pred_spans),
                }
            )

        flavors = compute_ner_metrics(
            gold_per_doc, pred_per_doc, self.scheme_labels, texts=texts, ids=ids
        )
        metrics_block: dict[str, Any] = {
            "training_kind": "ner",
            "exact": flavors["exact"].model_dump(mode="json"),
            "partial": flavors["partial"].model_dump(mode="json"),
            "type": flavors["type"].model_dump(mode="json"),
        }
        return pd.DataFrame.from_records(records), metrics_block

    def __save_files(
        self,
        current_path: Path,
        log_path: Path,
        df_train_results: pd.DataFrame,
        df_test_results: pd.DataFrame | None,
        training_data: pd.DataFrame,
        model,
        tokenizer,
        params_to_save: dict[str, Any],
        metrics_train: dict[str, Any],
        metrics_test: dict[str, Any] | None,
    ) -> None:
        df_train_results.to_csv(current_path.joinpath("train_dataset_eval.csv"), index=False)
        if df_test_results is not None:
            df_test_results.to_csv(current_path.joinpath("test_dataset_eval.csv"), index=False)
        training_data.to_parquet(current_path.joinpath("training_data.parquet"))
        model.save_pretrained(current_path)
        tokenizer.save_pretrained(current_path)
        with open(current_path.joinpath("parameters.json"), "w") as f:
            json.dump(params_to_save, f)
        shutil.rmtree(current_path.joinpath("train"), ignore_errors=True)
        os.rename(log_path, current_path.joinpath("finished"))
        path_static = f"{config.data_path}/projects/static/{self.project_slug}"
        os.makedirs(path_static, exist_ok=True)
        shutil.make_archive(
            f"{path_static}/{self.name}",
            "gztar",
            str(self.path.joinpath(self.name)),
        )
        metrics_data: dict[str, Any] = {"train": metrics_train}
        if metrics_test is not None:
            metrics_data["trainvalid"] = metrics_test
        with open(str(current_path.joinpath("metrics_training.json")), "w") as f:
            json.dump(metrics_data, f)

    def __call__(self) -> EventsModel:
        task_timer = TaskTimer(compulsory_steps=["setup", "train", "evaluate", "save_files"])
        task_timer.start("setup")
        set_seed(int(config.random_seed))

        current_path, log_path = self.__init_paths()
        self.logger = self.__init_logger(log_path)
        device = get_device()

        self.df = self.__prepare_dataframe(self.df)
        if len(self.df) < 2:
            raise Exception("Not enough annotated documents to train (need at least 2).")

        # Stringify ids for the gold-span lookup used during evaluation. The
        # dataset rows carry `str(id)` (see __tokenize_and_align), while
        # self.df.index keeps the caller's dtype — typically int — so a
        # `self.df.loc[str_id]` lookup would silently miss every row.
        self.gold_by_id: dict[str, list[dict]] = {
            str(idx): spans for idx, spans in self.df["__spans__"].items()
        }

        labels, label2id, id2label = build_bio_labels(self.scheme_labels)
        # Fast tokenizer is required: word_ids() and return_offsets_mapping
        # are not available on slow (Python) tokenizers, so failing loudly
        # here beats a confusing AttributeError mid-tokenization.
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True, use_fast=True
        )
        if not tokenizer or not tokenizer.is_fast:
            raise Exception(
                f"{self.base_model} does not provide a fast tokenizer; "
                f"NER training requires it for word_ids() and offset_mapping. "
                f"Choose a model whose tokenizer is Rust-backed."
            )
        effective_max_length = min(self.max_length, retrieve_model_max_length(self.base_model))
        self.max_length = effective_max_length

        full_ds = self.__tokenize_and_align(self.df, tokenizer, label2id, effective_max_length)
        if self.test_size > 0:
            split = full_ds.train_test_split(test_size=self.test_size, seed=int(config.random_seed))
            ds_dict = datasets.DatasetDict({"train": split["train"], "test": split["test"]})
        else:
            ds_dict = datasets.DatasetDict({"train": full_ds})

        model = AutoModelForTokenClassification.from_pretrained(
            self.base_model,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=True,
        ).to(device=device)
        model.config.use_cache = False

        try:
            trainer = self.__load_trainer(
                current_path, ds_dict, model, tokenizer, self.params, id2label
            )
            task_timer.stop("setup")

            task_timer.start("train")
            trainer.train()
            task_timer.stop("train")

            task_timer.start("evaluate")
            df_train_results, metrics_train = self.__evaluate_split(
                trainer, ds_dict["train"], tokenizer, id2label
            )
            df_test_results = None
            metrics_test = None
            if "test" in ds_dict:
                df_test_results, metrics_test = self.__evaluate_split(
                    trainer, ds_dict["test"], tokenizer, id2label
                )
            task_timer.stop("evaluate")

            task_timer.start("save_files")
            params_to_save = self.params.model_dump()
            params_to_save.update(
                {
                    "training_kind": "ner",
                    "tagging_scheme": "BIO",
                    "test_size": self.test_size,
                    "base_model": self.base_model,
                    "n_train": len(ds_dict["train"]),
                    "max_length": self.max_length,
                    "device": str(device),
                    "scheme_labels": self.scheme_labels,
                }
            )
            self.__save_files(
                current_path=current_path,
                log_path=log_path,
                df_train_results=df_train_results,
                df_test_results=df_test_results,
                training_data=self.df[[self.col_text, self.col_label]],
                model=model,
                tokenizer=tokenizer,
                params_to_save=params_to_save,
                metrics_train=metrics_train,
                metrics_test=metrics_test,
            )
            task_timer.stop("save_files")

        except Exception as e:
            print("Error in NER training", e)
            shutil.rmtree(current_path, ignore_errors=True)
            if isinstance(e, torch.cuda.OutOfMemoryError) or "NVML_SUCCESS" in str(e):
                raise Exception(
                    "GPU ran out of memory during training. "
                    "Reduce the batch size or increase gradient accumulation."
                ) from e
            raise e
        finally:
            try:
                del trainer, model, tokenizer, self.df, device, self.event
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()
            except Exception as e:
                print("Error cleaning memory", e)

        return EventsModel(events=task_timer.get_events())
