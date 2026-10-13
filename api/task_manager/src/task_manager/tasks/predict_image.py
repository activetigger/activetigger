import contextlib
import gc
import json
import multiprocessing
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import torch
from activetigger.data import Data
from activetigger.datamodels import MLStatisticsModel
from activetigger.functions import (
    activate_probs,
    annotations_to_matrix,
    get_device,
    get_metrics_multiclass,
    get_metrics_multilabel,
    logits_to_probs,
    release_device_memory,
)
from pandas import DataFrame
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from scipy.stats import entropy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,  # ty: ignore[possibly-missing-import]
)

# Cap PIL's maximum decoded pixel count to ~64 megapixels. Without this a
# small attacker-supplied JPEG can decompress to several GB and OOM the
# prediction worker. Matches the cap in train_image.py for consistency.
Image.MAX_IMAGE_PIXELS = 64_000_000


def _open_image_rgb_safe(path: str, fallback_size: tuple[int, int]) -> Image.Image | None:
    """
    Open an image with EXIF rotation + RGB conversion. On any error return
    a uniform-gray placeholder of the target size. Prediction tolerates
    bad rows (the output will just be ~uniform probabilities for that
    element) — unlike training, which pre-flights and drops them.
    """
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        if img is not None and img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except (
        FileNotFoundError,
        UnidentifiedImageError,
        OSError,
        ValueError,
        # Decompression-bomb error subclasses Exception, not OSError — must
        # be caught explicitly or the predict loop crashes on the first
        # oversized image.
        Image.DecompressionBombError,
    ):
        return Image.new("RGB", fallback_size, color=(128, 128, 128))


@contextlib.contextmanager
def _force_spawn_start_method() -> Iterator[None]:
    """
    Temporarily switch multiprocessing's global default start method to
    "spawn" for the duration of the with-block.

    DataLoader sub-workers spawned with multiprocessing_context="spawn"
    inherit the global default start method in their bootstrap data (see
    CPython's multiprocessing.spawn.get_preparation_data) and fail if that
    method is not resolvable in the child. Forcing "spawn" while the loader
    is alive sidesteps this; the previous method is restored on exit.
    """
    prev = multiprocessing.get_start_method(allow_none=True)
    try:
        multiprocessing.set_start_method("spawn", force=True)
        yield
    finally:
        if prev is not None and prev != "spawn":
            try:
                multiprocessing.set_start_method(prev, force=True)
            except (ValueError, RuntimeError):
                pass


class _ImagePredictDataset(Dataset[torch.Tensor]):
    """
    Loads + preprocesses one image per __getitem__ so a DataLoader with
    num_workers>0 can overlap decode/resize with GPU inference. Returns the
    (C, H, W) pixel_values tensor — default collate stacks to (B, C, H, W).
    """

    def __init__(self, paths: list[str], image_processor, fallback_size: tuple[int, int]):
        self.paths = paths
        self.image_processor = image_processor
        self.fallback_size = fallback_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index) -> torch.Tensor:
        img = _open_image_rgb_safe(self.paths[index], self.fallback_size)
        out = self.image_processor(images=img, return_tensors="pt")
        return out["pixel_values"].squeeze(0)


class PredictImageTaskInput(BaseModel):
    path: Path
    project_slug: str
    model_name: str
    dataset: str
    col_text: str  # holds the image path
    training_kind: str
    scheme_labels: list[str]
    # annotable dataset written to disk by the API; None for the all dataset
    path_df: Path | None = None
    col_label: str | None = None
    path_data: Path | None = None
    col_id_external: str | None = None
    col_datasets: str | None = None
    file_name: str = "predict.parquet"
    batch: int = 32
    statistics: list[str] | None = None
    path_train: Path | None = None
    path_valid: Path | None = None
    path_test: Path | None = None
    basemodel: str | None = None


# Return type must be a dict, a BaseModel would not be serialized by Celery
class PredictImageResult(TypedDict):
    path: str
    metrics: dict[str, Any] | None


class PredictImage:
    """
    Predict with a fine-tuned image-classification model.

    Output schema matches PredictBertMultiClass so that downstream
    Features.add_predictions can register predict_annotable.parquet as
    a feature without changes.
    """

    kind = "predict_image"

    def __init__(
        self, unique_id: str, inputs: PredictImageTaskInput, is_aborted: Callable[[], bool]
    ):
        self.path = inputs.path
        self.path_df = inputs.path_df
        self.path_data = inputs.path_data
        self.df: DataFrame | None = None
        self.dataset = inputs.dataset
        self.col_text = inputs.col_text
        self.col_label = inputs.col_label
        self.col_id_external = inputs.col_id_external
        self.col_datasets = inputs.col_datasets
        self.is_aborted = is_aborted
        self.unique_id = unique_id
        self.file_name = inputs.file_name
        self.batch = inputs.batch
        self.statistics = inputs.statistics
        self.path_train = inputs.path_train
        self.path_valid = inputs.path_valid
        self.path_test = inputs.path_test
        self.progress_path = self.path / "progress_predict"

        if inputs.dataset == "external":
            raise ValueError("External-dataset prediction is not supported for image models yet.")

        if self.path_df is None and self.path_data is None:
            raise ValueError("Dataframe must be provided for prediction")

        if self.statistics is not None and self.col_label is None:
            raise ValueError("Column label must be provided to compute statistics")

        self.training_kind = inputs.training_kind
        self.scheme_labels = inputs.scheme_labels
        self.threshold: float | None = None

        with open(self.path / "parameters.json", "r") as f:
            self.model_config = json.load(f)
            if "base_model" in self.model_config:
                self.basemodel = self.model_config["base_model"]
            elif inputs.basemodel is not None:
                self.basemodel = inputs.basemodel
            else:
                raise ValueError("No base_model in parameters.json.")
            if "threshold" in self.model_config and self.training_kind == "multilabel":
                t = self.model_config["threshold"]
                if t is not None:
                    self.threshold = float(t)
            if self.training_kind == "multilabel" and self.threshold is None:
                self.threshold = 0.5

    def __validate_dataframe(self, df: DataFrame) -> None:
        if self.col_text not in df.columns:
            raise ValueError(f"Column text {self.col_text} not in dataframe")
        if self.col_label is not None and self.col_label not in df.columns:
            raise ValueError(f"Column label {self.col_label} not in dataframe")
        if self.col_datasets is not None and self.col_datasets not in df.columns:
            raise ValueError(f"Column datasets {self.col_datasets} not in dataframe")
        if self.col_id_external is not None and self.col_id_external not in df.columns:
            raise ValueError(f"Column id {self.col_id_external} not in dataframe")

    def __load_all_file(self, path_data: Path) -> DataFrame:
        """
        Load data_all.parquet for whole-dataset prediction.
        """
        df = Data.read_dataset(path_data)

        if self.dataset == "all":
            df["id_external"] = (
                df[self.col_id_external].astype(str)
                if self.col_id_external and self.col_id_external in df.columns
                else df.index.astype(str)
            )
            df["dataset"] = "all"
            existing_ids = set(df["id_external"])
            subsets = {
                "train": self.path_train,
                "valid": self.path_valid,
                "test": self.path_test,
            }
            extra_frames = []
            for name, subset_path in subsets.items():
                if subset_path is None or not subset_path.exists():
                    continue
                subset = Data.read_dataset(subset_path)
                if "id_external" not in subset.columns:
                    continue
                subset["id_external"] = subset["id_external"].astype(str)
                in_main = subset["id_external"].isin(existing_ids)
                if in_main.any():
                    main_mask = df["id_external"].isin(set(subset.loc[in_main, "id_external"]))
                    df.loc[main_mask, "dataset"] = name
                imported = subset.loc[~in_main]
                if not imported.empty and "text" in imported.columns:
                    imported = imported[["id_external", "text"]].copy()
                    imported["dataset"] = name
                    extra_frames.append(imported)
            if extra_frames:
                df = pd.concat([df, *extra_frames])
        return df

    def __write_progress(self, progress: float) -> None:
        with open(self.progress_path, "w") as f:
            f.write(str(progress))

    def __load_model(self):
        # backend="torchvision" picks the fast (tensor-native) image processor
        # when available; transformers falls back to the numpy/PIL processor
        # for model families that don't have one. Several × faster on the
        # decode/resize/normalize path that runs per image in __getitem__.
        image_processor = AutoImageProcessor.from_pretrained(self.path, backend="torchvision")
        model = AutoModelForImageClassification.from_pretrained(self.path)
        return image_processor, model

    def __listen_stop_event(self):
        if self.is_aborted():
            raise Exception("Process interrupted by user")

    def __transform_to_dataframe(
        self, prob_predictions: np.ndarray, id2label: dict[int, str]
    ) -> DataFrame:
        if self.df is None:
            raise ValueError("Dataframe is required to transform to predictions")
        id2label = dict(sorted(id2label.items(), key=lambda u: u[0]))
        if list(id2label.keys()) != [i for i in range(len(id2label))]:
            raise ValueError(f"Bad id2label mapping: {id2label}")

        pred = pd.DataFrame(
            prob_predictions,
            columns=list(id2label.values()),
            index=self.df.index,
        )
        pred["entropy"] = entropy(prob_predictions, axis=1)
        for label in list(id2label.values()):
            prob_A_not_A = np.column_stack([pred[label], 1 - pred[label]])
            pred[f"entropy-{label}"] = entropy(prob_A_not_A, axis=1)

        if self.training_kind == "multiclass":
            y_pred = activate_probs(
                probs=prob_predictions, strategy="max", force_max_1_per_row=True
            )
            pred["prediction"] = np.argmax(y_pred, axis=1)
            pred["prediction"] = pred["prediction"].replace(id2label)
        else:
            y_pred = activate_probs(
                probs=prob_predictions,
                strategy="threshold",
                threshold=self.threshold if self.threshold is not None else 0.5,
            )
            pred["prediction"] = [
                "|".join([id2label[i] for i, activation in enumerate(row) if activation == 1])
                for row in y_pred
            ]

        # carry over the image path under the "text" column name to match
        # the BERT output schema (features layer drops it on registration).
        pred["text"] = self.df[self.col_text]
        if self.col_datasets:
            pred[self.col_datasets] = self.df[self.col_datasets]
        if self.col_id_external:
            pred[self.col_id_external] = self.df[self.col_id_external].astype(str)
        if self.col_label:
            pred["GS-label"] = self.df[self.col_label]
        return pred

    def __compute_statistics(
        self, pred: DataFrame, id2label: dict[int, str]
    ) -> dict[str, MLStatisticsModel]:
        if self.df is None or self.statistics is None:
            raise ValueError("df + statistics required")
        metrics: dict[str, MLStatisticsModel] = {}
        filter_label = pred["GS-label"].notna()
        for dataset in self.statistics:
            filter_dataset = pred[self.col_datasets] == dataset
            f = filter_label & filter_dataset
            if f.sum() < 5:
                continue
            if self.training_kind == "multiclass":
                metrics[dataset] = get_metrics_multiclass(
                    Y_true=pred.loc[f, "GS-label"],
                    Y_pred=pred.loc[f, "prediction"],
                    texts=pred[f]["text"],
                    id2label=id2label,
                )
            else:
                labels = list(id2label.values())
                y_true = annotations_to_matrix(pred.loc[f, "GS-label"], labels)
                y_pred = annotations_to_matrix(pred.loc[f, "prediction"], labels)
                metrics[dataset] = get_metrics_multilabel(
                    Y_true=y_true, Y_pred=y_pred, id2label=id2label, texts=pred[f]["text"]
                )
        with open(str(self.path.joinpath(f"metrics_predict_{time.time()}.json")), "w") as f_out:
            json.dump({k: v.model_dump(mode="json") for k, v in metrics.items()}, f_out)
        return metrics

    def run(self) -> PredictImageResult:
        print(f"start predicting image ({self.training_kind})", flush=True)
        if self.path_df is not None:
            self.df = pd.read_parquet(self.path_df)
        elif self.path_data is not None:
            self.df = self.__load_all_file(self.path_data)
        else:
            raise ValueError("Dataframe is required for prediction")
        self.__validate_dataframe(self.df)

        image_processor, model = self.__load_model()
        id2label = model.config.id2label
        try:
            num_labels = len(id2label)
        except Exception as e:
            raise ValueError("Model missing id2label: " + str(e))

        device = get_device()
        print(f"Using {device} for prediction", flush=True)
        model.to(device)
        print(f"Model moved to {device}", flush=True)
        model.eval()

        # Image size for the placeholder fallback; preprocessing rescales
        # so any reasonable size works, but match the processor's expected
        # size when possible.
        size_cfg = getattr(image_processor, "size", None)
        if isinstance(size_cfg, dict):
            h = size_cfg.get("height") or size_cfg.get("shortest_edge") or 224
            w = size_cfg.get("width") or size_cfg.get("shortest_edge") or 224
            fallback_size = (int(w), int(h))
        else:
            fallback_size = (224, 224)

        try:
            n = self.df.shape[0]
            n_batches = (n + self.batch - 1) // self.batch
            print(f"Predicting {n} images in {n_batches} batches", flush=True)
            paths = self.df[self.col_text].tolist()

            use_cuda = device.type == "cuda"
            # Decode/resize/normalize is CPU-bound and was the dominant cost
            # with num_workers=0 — the GPU sat idle between batches. Only
            # spin up workers when there are enough batches to amortize
            # their start-up cost. A celery prefork child is daemonic and
            # cannot spawn sub-processes at all (the GPU pool is solo, so
            # it can): fall back to in-process loading there.
            num_workers = min(4, n_batches) if n_batches >= 4 else 0
            if multiprocessing.current_process().daemon:
                num_workers = 0
            loader_kwargs: dict = {
                "batch_size": self.batch,
                "shuffle": False,
                "num_workers": num_workers,
                "pin_memory": use_cuda,
            }
            if num_workers > 0:
                loader_kwargs["multiprocessing_context"] = "spawn"
                loader_kwargs["persistent_workers"] = True
                loader_kwargs["prefetch_factor"] = 4

            proba_chunks: list[np.ndarray] = []
            processed = 0
            with _force_spawn_start_method() if num_workers > 0 else contextlib.nullcontext():
                loader = DataLoader(
                    _ImagePredictDataset(paths, image_processor, fallback_size),
                    **loader_kwargs,
                )
                for batch_tensor in tqdm(loader, total=n_batches, desc="Predicting", unit="batch"):
                    self.__listen_stop_event()
                    batch_tensor = batch_tensor.to(device, non_blocking=use_cuda)
                    with torch.no_grad():
                        if use_cuda:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                outputs = model(pixel_values=batch_tensor)
                        else:
                            outputs = model(pixel_values=batch_tensor)
                    logits = outputs.logits.float().detach().cpu().numpy()
                    proba_chunks.append(logits_to_probs(logits, kind=self.training_kind))
                    processed += batch_tensor.shape[0]
                    self.__write_progress(100 * processed / n)
                # ensure persistent worker processes are torn down (and the
                # restored start method actually takes effect on next use)
                # before we exit the context manager.
                del loader

            proba_predictions = (
                np.concatenate(proba_chunks, axis=0) if proba_chunks else np.zeros((0, num_labels))
            )

            pred = self.__transform_to_dataframe(proba_predictions, id2label=id2label)
            pred.to_parquet(self.path.joinpath(self.file_name))

            metrics = self.__compute_statistics(pred, id2label) if self.statistics else None

        except Exception as e:
            print("Error in image-classification prediction", e)
            raise e
        finally:
            if self.progress_path.exists():
                os.remove(self.progress_path)
            self.df = None
            try:
                del model, image_processor
            except Exception:
                pass
            gc.collect()
            # guard the CUDA release: on a corrupted context synchronize()
            # raises, and an exception here would mask the real error from
            # the prediction loop
            try:
                release_device_memory()
            except Exception as e:
                print("Error in cleaning GPU memory", e)
            if self.path_df is not None:
                try:
                    os.unlink(self.path_df)
                except Exception as e:
                    print("Error removing temporary prediction data", e)

        return PredictImageResult(
            path=str(self.path.joinpath(self.file_name)),
            metrics={k: v.model_dump(mode="json") for k, v in metrics.items()}
            if metrics is not None
            else None,
        )
