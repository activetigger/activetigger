# Experimental image projects — see docs/multimodal-prompt-selection.md
"""
Queueable task that computes multimodal image embeddings using
sentence-transformers (Qwen3-VL / BGE-VL / ...). Output lives in the same
shared vector space as prompt embeddings produced by the same model, which
is what enables the `prompt` selection mode in get_next.

Mirrors ComputeClipImagexp so it plugs into the existing Features/queue
machinery (same kind="feature", same FeatureComputing entry, same
path_progress contract).
"""

import gc
import math
import multiprocessing
import multiprocessing.synchronize
from pathlib import Path
from typing import Optional

import numpy as np
from pandas import DataFrame, Series

from activetigger.functions import get_device
from activetigger.tasks.base_task import BaseTask


class ComputeMultimodal(BaseTask):
    """
    Compute multimodal image embeddings with a sentence-transformers model.
    `paths` is a Series indexed by id_internal whose values are absolute
    paths to image files (typically the `path` column of data_all.parquet,
    restricted to the annotable rows).

    `model_name` is the full HF name (e.g. "BAAI/BGE-VL-base"), stored so
    the Prompts layer can reload the exact same encoder for prompt text.
    """

    kind = "compute_feature_multimodal_embeddings"

    def __init__(
        self,
        paths: Series,
        path_process: Path,
        model_name: str,
        batch_size: int = 8,
        path_progress: Path | None = None,
        event: Optional[multiprocessing.synchronize.Event] = None,
    ):
        super().__init__()
        self.paths = paths
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.path_process = path_process
        self.event = event
        if path_progress:
            self.progress_file_temporary = False
            self.path_progress = path_progress
        else:
            self.path_progress = self.path_process.joinpath(self.unique_id)
            self.progress_file_temporary = True

    def __call__(self) -> DataFrame:
        if self.progress_file_temporary:
            self.path_progress = self.path_process.joinpath(self.unique_id)

        try:
            import torch  # type: ignore[import]
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "Multimodal embedding features require `sentence-transformers` "
                "with image support. Install with: "
                "`pip install -U 'sentence-transformers[image]'`."
            ) from e

        try:
            from PIL import Image  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "Multimodal embeddings require Pillow. Install with: `pip install Pillow`."
            ) from e

        device = get_device()
        model = None
        try:
            # trust_remote_code is required for BGE-VL / Qwen-VL models that
            # ship custom modeling code; the native CLIP models ignore it.
            model = SentenceTransformer(
                self.model_name, device=device, trust_remote_code=True
            )

            ids = list(self.paths.index)
            paths = [str(p) for p in self.paths.values]
            n = len(paths)
            total_batches = max(1, math.ceil(n / self.batch_size))

            kept_ids: list = []
            vectors: list = []

            for i, start in enumerate(range(0, n, self.batch_size), 1):
                if self.event is not None and self.event.is_set():
                    raise Exception("Process interrupted by user")

                batch_ids = ids[start : start + self.batch_size]
                batch_paths = paths[start : start + self.batch_size]

                # Load PIL Images per-batch; skip unreadable files rather than
                # fail the whole job.
                valid_ids: list = []
                images: list = []
                for id_, p in zip(batch_ids, batch_paths):
                    try:
                        images.append(Image.open(Path(p)).convert("RGB"))
                        valid_ids.append(id_)
                    except Exception as ex:
                        print(f"Skip image {p}: {ex}")

                if images:
                    try:
                        emb = model.encode(
                            images,
                            batch_size=self.batch_size,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                        )
                        vectors.append(np.asarray(emb))
                        kept_ids.extend(valid_ids)
                    except Exception as ex:
                        print(f"Skip batch at {start}: {ex}")
                    finally:
                        for img in images:
                            try:
                                img.close()
                            except Exception:
                                pass

                progress_percent = (i / total_batches) * 100
                with open(self.path_progress, "w") as f:
                    f.write(str(round(progress_percent, 1)))

            if not vectors:
                raise RuntimeError("No multimodal embeddings could be computed")

            stacked = np.vstack(vectors)
            df = DataFrame(
                stacked,
                index=kept_ids,
                columns=["mm%03d" % (x + 1) for x in range(stacked.shape[1])],
            )
            if self.progress_file_temporary:
                try:
                    self.path_progress.unlink()
                except OSError:
                    pass
            return df
        finally:
            if model is not None:
                del model
            del self.paths
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
