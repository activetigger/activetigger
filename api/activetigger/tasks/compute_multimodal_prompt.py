# Experimental image projects — see docs/multimodal-prompt-selection.md
"""
Queueable task that encodes a single text prompt into the same vector space
as a previously-computed `multimodal-embeddings` feature. Goes through the
GPU queue so big models (Qwen3-VL 2B/8B) don't block the request thread.
"""

import gc
import multiprocessing
import multiprocessing.synchronize
from pathlib import Path
from typing import Optional

import numpy as np

from activetigger.functions import get_device
from activetigger.tasks.base_task import BaseTask


class ComputeMultimodalPrompt(BaseTask):
    """
    Encode a prompt text with the same sentence-transformer model used to
    compute the image embeddings it will be compared against.

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
                "Install with: `pip install -U 'sentence-transformers[image]'`."
            ) from e

        device = get_device()
        model = None
        try:
            with open(self.path_progress, "w") as f:
                f.write("10.0")

            # trust_remote_code enables BGE-VL / Qwen-VL custom modeling code;
            # native CLIP ignores it.
            model = SentenceTransformer(self.model_name, device=device, trust_remote_code=True)

            if self.event is not None and self.event.is_set():
                raise Exception("Process interrupted by user")

            emb = model.encode(
                [self.text],
                show_progress_bar=False,
                convert_to_numpy=True,
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
