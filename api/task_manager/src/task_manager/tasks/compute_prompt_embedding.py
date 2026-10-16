import gc
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from activetigger.config import config
from activetigger.functions import get_device, release_device_memory


def encode_prompt(
    text: str,
    model_name: str,
    kind: str,
    path_progress: Path,
    is_aborted: Callable[[], bool] | None = None,
) -> np.ndarray:
    """
    Encode one text prompt into the vector space of an embedding feature.

    `kind` is the feature kind: "sentence-embeddings" mirrors encode_sbert
    (normalised, model-specific encode args from config.models_embeddings),
    "multimodal-embeddings" mirrors encode_multimodal (plain encode, the
    query lands in the image space). Returns a 1-D array of shape (dim,).
    Progress is written to path_progress; the caller owns that file.
    """
    if kind not in {"sentence-embeddings", "multimodal-embeddings"}:
        raise ValueError(f"No prompt encoder for feature kind '{kind}'")

    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    device = get_device()
    model = None
    try:
        with open(path_progress, "w") as f:
            f.write("10.0")

        if kind == "sentence-embeddings":
            # same loading as encode_sbert: bf16 checkpoints are slow on CPU
            model_kwargs = {"dtype": torch.float32} if device.type == "cpu" else {}
            model = SentenceTransformer(
                model_name, device=str(device), trust_remote_code=True, model_kwargs=model_kwargs
            )
        else:
            # trust_remote_code enables BGE-VL / Qwen-VL custom modeling code
            model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

        if is_aborted is not None and is_aborted():
            raise Exception("Process interrupted by user")

        if kind == "sentence-embeddings":
            model_args = config.models_embeddings.get(model_name, {}) or {}
            emb = model.encode(
                [text],
                device=str(device),
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                **model_args,
            )
        else:
            emb = model.encode([text], show_progress_bar=False, convert_to_numpy=True)
        vec = np.asarray(emb).reshape(-1)

        with open(path_progress, "w") as f:
            f.write("100.0")
        return vec
    finally:
        if model is not None:
            del model
        gc.collect()
        release_device_memory()
