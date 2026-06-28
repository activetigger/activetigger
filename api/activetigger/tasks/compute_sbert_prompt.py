"""
Queueable task that encodes a single text prompt into the same vector space
as a previously-computed `sentence-embeddings` feature. Goes through the GPU
queue so loading the sentence-transformer model doesn't block the request
thread. Mirrors `ComputeMultimodalPrompt`, but uses the same encode call shape
as `ComputeSbert` (`normalize_embeddings=True`, model-specific args from
`config.models_embeddings`) so the query lives in the same normalized space as
the stored document vectors.
"""

import gc
import multiprocessing
import multiprocessing.synchronize
from pathlib import Path
from typing import Optional

import numpy as np

from activetigger.config import config
from activetigger.functions import get_device
from activetigger.tasks.base_task import BaseTask


class ComputeSbertPrompt(BaseTask):
    """
    Encode a prompt text with the same sentence-transformer model used to
    compute a `sentence-embeddings` feature.

    Returns a 1-D numpy array of shape (dim,).
    """

    kind = "compute_prompt_embedding"

    def __init__(
        self,
        text: str,
        model_name: str,
        path_process: Path,
        path_progress: Path | None = None,
        event: Optional[multiprocessing.synchronize.Event] = None,
    ):
        super().__init__()
        self.text = text
        self.model_name = model_name
        self.model_args = config.models_embeddings.get(self.model_name, {}) or {}
        self.path_process = path_process
        self.event = event
        if path_progress:
            self.progress_file_temporary = False
            self.path_progress = path_progress
        else:
            self.path_progress = self.path_process.joinpath(self.unique_id)
            self.progress_file_temporary = True

    def __call__(self) -> np.ndarray:
        if self.progress_file_temporary:
            self.path_progress = self.path_process.joinpath(self.unique_id)

        try:
            import torch  # type: ignore[import]
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "Prompt embedding requires `sentence-transformers`. "
                "Install with: `pip install -U sentence-transformers`."
            ) from e

        device = get_device()
        model = None
        try:
            with open(self.path_progress, "w") as f:
                f.write("10.0")

            model = SentenceTransformer(self.model_name, device=str(device), trust_remote_code=True)

            if self.event is not None and self.event.is_set():
                raise Exception("Process interrupted by user")

            emb = model.encode(
                [self.text],
                device=str(device),
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                **self.model_args,
            )
            vec = np.asarray(emb).reshape(-1)

            with open(self.path_progress, "w") as f:
                f.write("100.0")

            if self.progress_file_temporary:
                try:
                    self.path_progress.unlink()
                except OSError:
                    pass
            return vec
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
