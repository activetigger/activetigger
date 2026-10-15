import gc
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from activetigger.config import config
from activetigger.functions import get_device, release_device_memory
from pandas import DataFrame, Series
from sentence_transformers import SentenceTransformer


def encode_sbert(
    texts: Series,
    model: str,
    path_progress: Path,
    batch_size: int = 32,
    min_gpu: int = 6,
    max_tokens: int = 1024,
    is_aborted: Callable[[], bool] | None = None,
) -> DataFrame:
    """
    Compute normalised sentence-transformers embeddings by batches.

    Shared by the celery task and the in-process callers (bertopic, feature
    recomputation). Progress is written as a percentage to path_progress;
    the caller owns that file.
    """
    model_args = config.models_embeddings.get(model, {})
    max_tokens = int(max_tokens)

    if texts.isnull().sum() > 0:
        raise ValueError("There are missing values in the input data, so we can't proceed")

    device = get_device()
    if device.type == "cuda":
        if torch.cuda.get_device_properties(0).total_memory / (1024**3) <= min_gpu:
            print("Not enough GPU memory, fallback to CPU")
            device = torch.device("cpu")

    sbert = None
    try:
        # transformers>=5 loads bf16 checkpoints as-is; on CPU that hits a very slow matmul path
        model_kwargs = {"dtype": torch.float32} if device.type == "cpu" else {}
        sbert = SentenceTransformer(
            model, device=str(device), trust_remote_code=True, model_kwargs=model_kwargs
        )
        max_seq_length = sbert.max_seq_length
        if max_seq_length is None:
            sbert.max_seq_length = max_tokens
        else:
            sbert.max_seq_length = int(min(max_tokens, max_seq_length))

        print("start computation")
        embeddings = []
        total_batches = math.ceil(len(texts) / batch_size)
        for i, start in enumerate(range(0, len(texts), batch_size), 1):
            # check if the user want to stop the process
            if is_aborted is not None and is_aborted():
                print("Process interrupted by user")
                raise Exception("Process interrupted by user")
            batch_texts = list(texts.iloc[start : start + batch_size])
            embeddings.append(
                sbert.encode(
                    batch_texts,
                    device=str(device),
                    normalize_embeddings=True,
                    **model_args,
                )
            )

            progress_percent = (i / total_batches) * 100
            with open(path_progress, "w") as f:
                f.write(str(round(progress_percent, 1)))
            print(progress_percent)

        return DataFrame(
            np.vstack(embeddings),
            index=texts.index,
            columns=["sb%03d" % (x + 1) for x in range(len(embeddings[0][0]))],
        )
    finally:
        if sbert is not None:
            del sbert
        gc.collect()
        release_device_memory()
