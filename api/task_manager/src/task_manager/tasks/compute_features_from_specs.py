import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import regex
from pandas import DataFrame, Series
from pydantic import BaseModel

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


def write_progress(path_progress: Path, pct: float) -> None:
    pct = max(0.0, min(100.0, pct))
    try:
        with open(path_progress, "w") as f:
            f.write(str(round(pct, 1)))
    except OSError:
        pass


def compute_features_from_specs(
    specs: list[FeatureSpec],
    data: Series,
    path_process: Path,
    path_models: Path,
    language: str,
    path_progress: Path,
    is_aborted: Callable[[], bool] | None = None,
    total: int | None = None,
) -> DataFrame:
    """
    Run every spec on `data` (texts for text kinds, image paths for image
    kinds) and return the concatenated matrix with `<name>__<subcol>` columns,
    the scheme of the project features parquet.

    Overall progress is written to `path_progress` (the caller owns that
    file). `total` is the number of progress units; pass `len(specs) + k`
    to reserve k units for follow-up steps. Sub-tasks that report progress
    get a temporary file in `path_process`, watched by a thread so the bar
    moves within a step.
    """
    if total is None:
        total = max(len(specs), 1)

    def check_aborted() -> None:
        if is_aborted is not None and is_aborted():
            raise Exception("Process interrupted by user")

    frames: list[DataFrame] = []
    write_progress(path_progress, 0.0)
    check_aborted()
    for i, spec in enumerate(specs):
        check_aborted()
        sub_progress = path_process.joinpath(f"sub_{uuid.uuid4().hex}")
        sub_callable, watched = _prepare(
            spec, data, path_models, language, sub_progress, is_aborted
        )

        stop: threading.Event | None = None
        watcher: threading.Thread | None = None
        if watched:
            stop = threading.Event()
            watcher = threading.Thread(
                target=_watch,
                args=(sub_progress, path_progress, i, total, stop),
                daemon=True,
            )
            watcher.start()

        try:
            result = sub_callable()
        finally:
            if stop is not None:
                stop.set()
            if watcher is not None:
                watcher.join(timeout=2)
            sub_progress.unlink(missing_ok=True)

        if isinstance(result, Series):
            result = result.to_frame()
        result.columns = [f"{spec.name}__{c}" for c in result.columns]
        frames.append(result)

        write_progress(path_progress, ((i + 1) / total) * 100)

    check_aborted()
    return pd.concat(frames, axis=1) if frames else DataFrame()


def _prepare(
    spec: FeatureSpec,
    data: Series,
    path_models: Path,
    language: str,
    sub_progress: Path,
    is_aborted: Callable[[], bool] | None,
) -> tuple[Callable[[], Any], bool]:
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
            return data.apply(lambda x: bool(pattern.search(x)))

        return run_regex, False

    if kind == "sentence-embeddings":
        return (
            lambda: encode_sbert(
                data,
                model=parameters["model"],
                path_progress=sub_progress,
                batch_size=int(parameters.get("batch_size", 32)),
                max_tokens=int(parameters.get("max_length_tokens", 1024)),
                is_aborted=is_aborted,
            ),
            True,
        )

    if kind == "bert-embeddings":
        if spec.model_path is None:
            raise ValueError(
                "bert-embeddings spec is missing 'model_path' "
                "(caller must resolve it before invoking the task)"
            )
        model_path = spec.model_path
        return (
            lambda: encode_bert_embeddings(
                data,
                model_path=model_path,
                path_progress=sub_progress,
                pooling=parameters.get("pooling", "mean"),
                batch_size=int(parameters.get("batch_size", 32)),
                max_tokens=int(parameters.get("max_length_tokens", 512)),
                is_aborted=is_aborted,
            ),
            True,
        )

    if kind == "fasttext":
        return (
            lambda: encode_fasttext(
                data,
                language=language,
                path_models=path_models,
                path_progress=sub_progress,
                model=parameters["model"],
                is_aborted=is_aborted,
            ),
            True,
        )

    if kind == "dfm":
        # one-shot vectoriser; no incremental progress
        return (
            lambda: compute_dfm(
                data,
                tfidf=parameters.get("tfidf", False),
                ngrams=parameters.get("dfm_ngrams", 1),
                min_term_freq=parameters.get("dfm_min_term_freq", 5),
                max_term_freq=parameters.get("dfm_max_term_freq", 100),
                log=parameters.get("dfm_log", False),
                language=language,
                norm=parameters.get("dfm_norm", None),
            ),
            False,
        )

    if kind == "image-embeddings":
        from activetigger.features import IMAGE_EMBEDDING_MODELS_IMAGEXP

        model_spec = IMAGE_EMBEDDING_MODELS_IMAGEXP.get(parameters["model"])
        if model_spec is None:
            raise ValueError(f"Unknown image embedding model: {parameters['model']}")
        return (
            lambda: encode_clip(
                data,
                model=model_spec["model"],
                pretrained=model_spec["pretrained"],
                path_progress=sub_progress,
                batch_size=int(parameters.get("batch_size", 16)),
                is_aborted=is_aborted,
            ),
            True,
        )

    if kind == "multimodal-embeddings":
        hf_name = parameters.get("hf_name")
        if not hf_name:
            from activetigger.features import MULTIMODAL_EMBEDDING_MODELS

            hf_name = MULTIMODAL_EMBEDDING_MODELS.get(parameters["model"])
            if hf_name is None:
                raise ValueError(f"Unknown multimodal embedding model: {parameters['model']}")
        return (
            lambda: encode_multimodal(
                data,
                model_name=hf_name,
                path_progress=sub_progress,
                batch_size=int(parameters.get("batch_size", 8)),
                is_aborted=is_aborted,
            ),
            True,
        )

    raise ValueError(f"Unsupported feature kind: {kind}")


def _watch(
    sub_progress: Path,
    path_progress: Path,
    step_index: int,
    total: int,
    stop: threading.Event,
) -> None:
    """
    Poll the sub-task progress file (0..100) and rewrite the overall progress
    as `(step_index + sub_pct/100) / total * 100` until `stop` is set.
    Read/parse errors are ignored: the file may not exist yet or be mid-write.
    """
    while not stop.is_set():
        try:
            with open(sub_progress) as f:
                raw = f.read().strip()
            if raw:
                sub_pct = float(raw)
                write_progress(path_progress, ((step_index + sub_pct / 100.0) / total) * 100.0)
        except (FileNotFoundError, ValueError, OSError):
            pass
        stop.wait(0.5)
