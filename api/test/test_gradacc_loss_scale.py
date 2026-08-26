"""
Regression test for the training-loss plot scale with gradient accumulation (#1109).

With gradacc > 1, whether HuggingFace's raw logged train loss is comparable to
eval_loss depends on the transformers version and model class: when the model's
forward accepts **kwargs, the Trainer skips its own normalization and the raw
logged loss is inflated by gradacc (because encoder classification heads ignore
num_items_in_batch). CustomLoggingCallback corrects this, gated on
model_accepts_loss_kwargs.

These tests train a tiny random BERT with learning_rate=0 (every forward yields
the same loss scale) so train and eval losses must match once corrected. If a
transformers upgrade changes the logging semantics (e.g. encoder heads become
num_items_in_batch-aware), the raw-scale test fails and the correction in
train_bert.CustomLoggingCallback must be revisited.
"""

import json
import logging

import pytest
import torch
from torch.utils.data import Dataset
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainerControl,
    TrainingArguments,
)

from activetigger.tasks.train_bert import CustomLoggingCallback

GRADACC = 4
NUM_LABELS = 3


class _TinyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, size: int = 16, seq_len: int = 12, vocab_size: int = 100):
        g = torch.Generator().manual_seed(0)
        self.input_ids = torch.randint(1, vocab_size, (size, seq_len), generator=g)
        self.labels = torch.tensor([i % NUM_LABELS for i in range(size)], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": torch.ones_like(self.input_ids[index]),
            "labels": self.labels[index],
        }


def _train_tiny_bert(tmp_path, gradacc: int) -> Trainer:
    torch.manual_seed(42)
    config = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        num_labels=NUM_LABELS,
    )
    model = BertForSequenceClassification(config)
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=gradacc,
        num_train_epochs=1,
        learning_rate=0.0,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        seed=42,
        use_cpu=True,
        disable_tqdm=True,
    )
    callback = CustomLoggingCallback(
        event=None, logger=logging.getLogger("test"), current_path=tmp_path
    )
    ds = _TinyDataset()
    trainer = Trainer(
        model=model, args=args, train_dataset=ds, eval_dataset=ds, callbacks=[callback]
    )
    # mirror the wiring done in TrainBert.__load_trainer
    callback.needs_gradacc_correction = getattr(trainer, "model_accepts_loss_kwargs", False)
    trainer.train()
    # on_step_end fires before the step's log entries are appended, so the file
    # written during training lags by one step; flush the complete history the
    # same way the callback does during training
    callback.on_step_end(args=trainer.args, state=trainer.state, control=TrainerControl())
    return trainer


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("gradacc")
    trainer = _train_tiny_bert(tmp_path, GRADACC)
    return tmp_path, trainer


def _split_losses(log_history):
    train = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
    evals = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    return train, evals


def test_raw_logged_loss_is_inflated_by_gradacc(trained):
    """Pin the transformers logging semantics the correction relies on.

    If this fails after a dependency upgrade, HF changed how the train loss is
    aggregated with gradient accumulation: re-audit the gradacc correction in
    CustomLoggingCallback before trusting the loss plot.
    """
    _, trainer = trained
    assert getattr(trainer, "model_accepts_loss_kwargs", False), (
        "BertForSequenceClassification.forward no longer accepts **kwargs: "
        "the Trainer now normalizes the logged loss itself and the gradacc "
        "correction must stay disabled (needs_gradacc_correction False)."
    )
    raw_train, evals = _split_losses(trainer.state.log_history)
    assert raw_train and evals
    ratio = (sum(raw_train) / len(raw_train)) / evals[-1]
    assert ratio == pytest.approx(GRADACC, rel=0.15)


def test_corrected_log_history_matches_eval_scale(trained):
    """The log_history.txt used by the loss chart has train ≈ eval scale."""
    tmp_path, _ = trained
    with open(tmp_path / "log_history.txt") as f:
        log = json.load(f)
    train, evals = _split_losses(log)
    assert train and evals
    for value in train:
        assert value == pytest.approx(evals[-1], rel=0.15)
