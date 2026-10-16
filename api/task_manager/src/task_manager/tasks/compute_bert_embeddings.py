import gc
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from activetigger.functions import get_device, release_device_memory
from pandas import DataFrame, Series
from transformers import (
    AutoModel,  # ty: ignore[possibly-missing-import]
    AutoTokenizer,  # ty: ignore[possibly-missing-import]
)


def _load_base_model_name(model_path: Path) -> str:
    params_file = model_path.joinpath("parameters.json")
    with open(params_file, "r") as f:
        params = json.load(f)
    base = params.get("base_model")
    if not base:
        raise ValueError(f"base_model missing from {params_file}")
    return str(base)


def encode_bert_embeddings(
    texts: Series,
    model_path: Path,
    path_progress: Path,
    pooling: str = "mean",
    batch_size: int = 32,
    max_tokens: int = 512,
    min_gpu: int = 6,
    is_aborted: Callable[[], bool] | None = None,
) -> DataFrame:
    """
    Embeddings from a fine-tuned BERT saved by train_bert: load the encoder
    (classification head dropped), forward pass, pool token states into one
    normalised vector per text. Shared by the celery task and the in-process
    callers. Progress is written as a percentage to path_progress; the caller
    owns that file.
    """
    model_path = Path(model_path)
    if pooling not in {"cls", "mean"}:
        raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
    batch_size = int(batch_size)
    max_tokens = int(max_tokens)

    if texts.isnull().sum() > 0:
        raise ValueError("There are missing values in the input data, so we can't proceed")

    device = get_device()
    if device.type == "cuda":
        if torch.cuda.get_device_properties(0).total_memory / (1024**3) <= min_gpu:
            print("Not enough GPU memory, fallback to CPU")
            device = torch.device("cpu")

    # prefer the tokenizer saved with the fine-tuned checkpoint
    if model_path.joinpath("tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            _load_base_model_name(model_path), trust_remote_code=True
        )
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
    model.to(device)
    model.eval()

    # clamp request to what the model actually supports; HF tokenizers may
    # report 1e30 when unset, so we also fall back to the encoder config
    model_max = getattr(tokenizer, "model_max_length", None)
    if not model_max or model_max > 100000:
        model_max = getattr(model.config, "max_position_embeddings", max_tokens)
    max_length = int(min(max_tokens, model_max))

    try:
        print("start computation")
        with open(path_progress, "w") as f:
            f.write("0")
        embeddings: list[np.ndarray] = []
        total_batches = math.ceil(len(texts) / batch_size)
        for i, start in enumerate(range(0, len(texts), batch_size), 1):
            if is_aborted is not None and is_aborted():
                print("Process interrupted by user")
                raise Exception("Process interrupted by user")

            batch_texts = list(texts.iloc[start : start + batch_size])
            encoded = tokenizer(  # ty: ignore[call-non-callable]
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = model(**encoded)

            last_hidden = outputs.last_hidden_state  # [B, T, H]
            if pooling == "cls":
                pooled = last_hidden[:, 0, :]
            else:
                # mean pool over real tokens (ignore padding)
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts

            pooled = F.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())

            progress_percent = (i / total_batches) * 100
            with open(path_progress, "w") as f:
                f.write(str(round(progress_percent, 1)))
            print(progress_percent)

        stacked = np.vstack(embeddings)
        return DataFrame(
            stacked,
            index=texts.index,
            columns=["be%03d" % (x + 1) for x in range(stacked.shape[1])],
        )
    finally:
        del model, tokenizer
        gc.collect()
        release_device_memory()
