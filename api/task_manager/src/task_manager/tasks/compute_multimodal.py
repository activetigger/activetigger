# Experimental image projects — see docs/multimodal-prompt-selection.md
"""
Multimodal image embeddings with sentence-transformers (CLIP / BGE-VL / Qwen3-VL).
Output shares the vector space of prompt embeddings from the same model, which
enables the `prompt` selection mode in get_next.

`encode_multimodal` is shared by the celery task and the in-process callers
(feature recomputation).
"""

import gc
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from activetigger.functions import get_device, release_device_memory
from pandas import DataFrame, Series


def encode_multimodal(
    paths: Series,
    model_name: str,
    path_progress: Path,
    batch_size: int = 8,
    is_aborted: Callable[[], bool] | None = None,
) -> DataFrame:
    """
    `paths` is indexed by id_internal, values are absolute image paths; unreadable
    images and failing batches are skipped. `model_name` is the full HF name so the
    Prompts layer can reload the same encoder. Progress is written as a percentage
    to path_progress; the caller owns that file.
    """
    try:
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

    batch_size = int(batch_size)
    device = get_device()
    model = None
    try:
        # trust_remote_code is required for BGE-VL / Qwen-VL models that
        # ship custom modeling code; the native CLIP models ignore it.
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

        ids = list(paths.index)
        files = [str(p) for p in paths.values]
        n = len(files)
        total_batches = max(1, math.ceil(n / batch_size))

        kept_ids: list[Any] = []
        vectors: list[np.ndarray] = []

        for i, start in enumerate(range(0, n, batch_size), 1):
            if is_aborted is not None and is_aborted():
                raise Exception("Process interrupted by user")

            batch_ids = ids[start : start + batch_size]
            batch_paths = files[start : start + batch_size]

            # Load PIL Images per-batch; skip unreadable files rather than
            # fail the whole job.
            valid_ids: list[Any] = []
            images: list[Any] = []
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
                        batch_size=batch_size,
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
            with open(path_progress, "w") as f:
                f.write(str(round(progress_percent, 1)))

        if not vectors:
            raise RuntimeError("No multimodal embeddings could be computed")

        stacked = np.vstack(vectors)
        return DataFrame(
            stacked,
            index=kept_ids,
            columns=["mm%03d" % (x + 1) for x in range(stacked.shape[1])],
        )
    finally:
        if model is not None:
            del model
        gc.collect()
        release_device_memory()
