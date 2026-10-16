# Experimental image projects — see docs/image-projects-strategy.md
"""
CLIP-style image embeddings for an image project.

`encode_clip` is shared by the celery task and the in-process callers
(feature recomputation). Model specs live in features.IMAGE_EMBEDDING_MODELS_IMAGEXP
as (model_name, pretrained_tag) consumable by `open_clip.create_model_and_transforms`.
"""

import gc
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from activetigger.functions import get_device, release_device_memory
from pandas import DataFrame, Series


def encode_clip(
    paths: Series,
    model: str,
    pretrained: str,
    path_progress: Path,
    batch_size: int = 16,
    is_aborted: Callable[[], bool] | None = None,
) -> DataFrame:
    """
    Normalised CLIP image embeddings by batches. `paths` is indexed by id_internal,
    values are absolute image paths; unreadable images are skipped. Progress is
    written as a percentage to path_progress; the caller owns that file.
    """
    # lazy import — open_clip / torch / PIL are optional deps for image projects
    try:
        import open_clip  # type: ignore[import]
        import torch  # type: ignore[import]
        from PIL import Image  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "Image embedding features require the optional dependency "
            "`open_clip_torch`. Install it with: `pip install open_clip_torch` "
            "(see docs/image-projects-strategy.md)."
        ) from e

    batch_size = int(batch_size)
    device = get_device()
    clip_model = None
    try:
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained
        )
        clip_model = clip_model.to(device)
        clip_model.eval()

        ids = list(paths.index)
        files = list(paths.values)
        n = len(files)
        total_batches = max(1, math.ceil(n / batch_size))

        kept_ids: list[Any] = []
        vectors: list[np.ndarray] = []

        with torch.no_grad():
            for i, start in enumerate(range(0, n, batch_size), 1):
                if is_aborted is not None and is_aborted():
                    raise Exception("Process interrupted by user")

                batch_ids = ids[start : start + batch_size]
                batch_paths = files[start : start + batch_size]

                tensors = []
                valid_ids = []
                for id_, p in zip(batch_ids, batch_paths):
                    try:
                        img = Image.open(Path(str(p))).convert("RGB")
                        tensors.append(preprocess(img))
                        valid_ids.append(id_)
                    except Exception as ex:
                        print(f"Skip image {p}: {ex}")

                if tensors:
                    batch = torch.stack(tensors).to(device)
                    feats = clip_model.encode_image(batch)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    vectors.append(feats.cpu().numpy())
                    kept_ids.extend(valid_ids)

                progress_percent = (i / total_batches) * 100
                with open(path_progress, "w") as f:
                    f.write(str(round(progress_percent, 1)))

        if not vectors:
            raise RuntimeError("No image embeddings could be computed")

        stacked = np.vstack(vectors)
        return DataFrame(
            stacked,
            index=kept_ids,
            columns=["im%03d" % (x + 1) for x in range(stacked.shape[1])],
        )
    finally:
        if clip_model is not None:
            del clip_model
        gc.collect()
        release_device_memory()
