import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from turtle import up
from typing import Any

import pandas as pd
import regex
from pandas import DataFrame, Series
from pydantic import BaseModel

from task_manager.auto_callback_task import UpdateProgressMethod
from task_manager.tasks.compute_bert_embeddings import encode_bert_embeddings
from task_manager.tasks.compute_clip_imagexp import encode_clip
from task_manager.tasks.compute_dfm import compute_dfm
from task_manager.tasks.compute_fasttext import encode_fasttext
from task_manager.tasks.compute_multimodal import encode_multimodal
from task_manager.tasks.compute_sbert import encode_sbert


class FeatureSpec(BaseModel):
    """
    One feature to (re)compute, as built by Features.build_compute_specs.
    `parameters` are the ones stored in the DB by Features.add; `model_path`
    is required for bert-embeddings (resolved by the caller).
    """

    name: str
    kind: str
    parameters: dict[str, Any]
    model_path: Path | None = None


def compute_features_from_specs(
    specs: list[FeatureSpec],
    data: Series,
    path_process: Path,
    path_models: Path,
    language: str,
    update_progress: UpdateProgressMethod,
    is_aborted: Callable[[], bool] | None = None,
    total: int | None = None,
) -> DataFrame:
    """
    Run every spec on `data` (texts for text kinds, image paths for image
    kinds) and return the concatenated matrix with `<name>__<subcol>` columns,
    the scheme of the project features parquet.

    Overall progress is update throug the given update_progress method (using task state).
    `total` is the number of progress units; pass `len(specs) + k`
    to reserve k units for follow-up steps. Sub-tasks that report progress
    get a custom update_sub_progress method which uses the given update_progress method but rescale subtask progression to its unit bounds.
    """
    if total is None:
        total = max(len(specs), 1)

    def check_aborted() -> None:
        if is_aborted is not None and is_aborted():
            raise Exception("Process interrupted by user")

    frames: list[DataFrame] = []

    update_progress(0.0)
    check_aborted()
    for i, spec in enumerate(specs):
        check_aborted()
        update_sub_progress = _make_update_sub_progress(
            update_progress, i, total, f"{spec.name} ({spec.kind})"
        )
        sub_callable = _prepare(spec, data, path_models, language, update_sub_progress, is_aborted)

        result = sub_callable()

        if isinstance(result, Series):
            result = result.to_frame()
        result.columns = [f"{spec.name}__{c}" for c in result.columns]
        frames.append(result)

        update_sub_progress(100)

    check_aborted()
    return pd.concat(frames, axis=1) if frames else DataFrame()


# factory for the sub_progress monitoring
def _make_update_sub_progress(
    update_progress: UpdateProgressMethod,
    sub_task_index: int,
    nb_tasks: int,
    task_label: str | None = None,
) -> UpdateProgressMethod:
    def update_sub_progress(progress: float, label: str | None = None):
        # rescale sub_progress [0, 100] to the portion of its group progress
        update_progress(((sub_task_index + 1 + progress) / nb_tasks), label or task_label or None)

    return update_sub_progress


def _prepare(
    spec: FeatureSpec,
    data: Series,
    path_models: Path,
    language: str,
    update_progress: UpdateProgressMethod,
    is_aborted: Callable[[], bool] | None,
) -> Callable[[], Any]:
    """
    Build the callable computing one spec. Returns `(callable, reports_progress)`;
    when true the callable writes 0..100 into `sub_progress`.
    """
    kind = spec.kind
    parameters = spec.parameters

    if kind == "regex":
        pattern = regex.compile(parameters["regex"])
        count_mode = parameters.get("mode") == "count"

        def run_regex() -> Series:
            if count_mode:
                return data.apply(lambda x: len(pattern.findall(x)))
            update_progress(100)
            return data.apply(lambda x: bool(pattern.search(x)))

        return run_regex

    if kind == "sentence-embeddings":
        return lambda: encode_sbert(
            data,
            model=parameters["model"],
            update_progress=update_progress,
            batch_size=int(parameters.get("batch_size", 32)),
            max_tokens=int(parameters.get("max_length_tokens", 1024)),
            is_aborted=is_aborted,
        )

    if kind == "bert-embeddings":
        if spec.model_path is None:
            raise ValueError(
                "bert-embeddings spec is missing 'model_path' "
                "(caller must resolve it before invoking the task)"
            )
        model_path = spec.model_path
        return lambda: encode_bert_embeddings(
            data,
            model_path=model_path,
            update_progress=update_progress,
            pooling=parameters.get("pooling", "mean"),
            batch_size=int(parameters.get("batch_size", 32)),
            max_tokens=int(parameters.get("max_length_tokens", 512)),
            is_aborted=is_aborted,
        )

    if kind == "fasttext":
        return lambda: encode_fasttext(
            data,
            language=language,
            path_models=path_models,
            update_progress=update_progress,
            model=parameters["model"],
            is_aborted=is_aborted,
        )
    if kind == "dfm":
        # one-shot vectoriser; no incremental progress
        def dfm():
            compute_dfm(
                data,
                tfidf=parameters.get("tfidf", False),
                ngrams=parameters.get("dfm_ngrams", 1),
                min_term_freq=parameters.get("dfm_min_term_freq", 5),
                max_term_freq=parameters.get("dfm_max_term_freq", 100),
                log=parameters.get("dfm_log", False),
                language=language,
                norm=parameters.get("dfm_norm", None),
            )
            update_progress(100)

        return dfm
    if kind == "image-embeddings":
        from activetigger.features import IMAGE_EMBEDDING_MODELS_IMAGEXP

        model_spec = IMAGE_EMBEDDING_MODELS_IMAGEXP.get(parameters["model"])
        if model_spec is None:
            raise ValueError(f"Unknown image embedding model: {parameters['model']}")
        return lambda: encode_clip(
            data,
            model=model_spec["model"],
            pretrained=model_spec["pretrained"],
            update_progress=update_progress,
            batch_size=int(parameters.get("batch_size", 16)),
            is_aborted=is_aborted,
        )

    if kind == "multimodal-embeddings":
        hf_name = parameters.get("hf_name")
        if not hf_name:
            from activetigger.features import MULTIMODAL_EMBEDDING_MODELS

            hf_name = MULTIMODAL_EMBEDDING_MODELS.get(parameters["model"])
            if hf_name is None:
                raise ValueError(f"Unknown multimodal embedding model: {parameters['model']}")
        return lambda: encode_multimodal(
            data,
            model_name=hf_name,
            update_progress=update_progress,
            batch_size=int(parameters.get("batch_size", 8)),
            is_aborted=is_aborted,
        )

    raise ValueError(f"Unsupported feature kind: {kind}")
