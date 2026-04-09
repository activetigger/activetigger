# Experimental image projects — see docs/image-projects-strategy.md
"""
Queueable task that computes CLIP-style image embeddings for an image project.

Mirrors the shape of `ComputeSbert` so it plugs into the existing
Features/queue machinery (same kind="feature", same FeatureComputing entry,
same path_progress contract).

The model list lives in features.IMAGE_EMBEDDING_MODELS — each entry is a
(model_name, pretrained_tag) tuple consumable by `open_clip.create_model_and_transforms`.
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


class ComputeClipImagexp(BaseTask):
    """
    Compute image embeddings for the train/valid/test elements of an image
    project. `paths` is a Series indexed by id_internal whose values are
    absolute paths to image files (typically the `path` column of
    data_all.parquet, restricted to the annotable rows).
    """

    kind = "compute_feature_image_embeddings"

    def __init__(
        self,
        paths: Series,
        path_process: Path,
        model: str,
        pretrained: str = "openai",
        batch_size: int = 16,
        path_progress: Path | None = None,
        event: Optional[multiprocessing.synchronize.Event] = None,
    ):
        super().__init__()
        self.paths = paths
        self.model = model
        self.pretrained = pretrained
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
        # update temporary file with the current ID
        if self.progress_file_temporary:
            self.path_progress = self.path_process.joinpath(self.unique_id)

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

        device = get_device()
        clip_model = None
        try:
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                self.model, pretrained=self.pretrained
            )
            clip_model = clip_model.to(device)
            clip_model.eval()

            ids = list(self.paths.index)
            paths = list(self.paths.values)
            n = len(paths)
            total_batches = max(1, math.ceil(n / self.batch_size))

            kept_ids: list = []
            vectors: list = []

            with torch.no_grad():
                for i, start in enumerate(range(0, n, self.batch_size), 1):
                    if self.event is not None and self.event.is_set():
                        raise Exception("Process interrupted by user")

                    batch_ids = ids[start : start + self.batch_size]
                    batch_paths = paths[start : start + self.batch_size]

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
                    with open(self.path_progress, "w") as f:
                        f.write(str(round(progress_percent, 1)))

            if not vectors:
                raise RuntimeError("No image embeddings could be computed")

            stacked = np.vstack(vectors)
            df = DataFrame(
                stacked,
                index=kept_ids,
                columns=["im%03d" % (x + 1) for x in range(stacked.shape[1])],
            )
            if self.progress_file_temporary:
                try:
                    self.path_progress.unlink()
                except OSError:
                    pass
            return df
        finally:
            if clip_model is not None:
                del clip_model
            del self.paths
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
