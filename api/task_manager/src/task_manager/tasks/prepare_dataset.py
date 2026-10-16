import gc
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from activetigger.datamodels import PrepareSplitModel
from activetigger.functions import concat_text_columns, get_device
from wtpsplit import SaT

WTPSPLIT_MODEL = "sat-3l-sm"
WTPSPLIT_BATCH_CHARS = 500_000
WTPSPLIT_BATCH_SIZE = 8


def split_by_chars(text: str, chunk_size: int) -> list[str]:
    """
    Split a text into chunks of about chunk_size characters without
    breaking words (a chunk can exceed chunk_size only for a single
    word longer than the limit).
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in text.split():
        candidate = current_len + len(word) + (1 if current else 0)
        if current and candidate > chunk_size:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = candidate
    if current:
        chunks.append(" ".join(current))
    return chunks


def split_by_regex(text: str, pattern: str) -> list[str]:
    """
    Split a text on a regex pattern, dropping empty segments
    """
    return [s.strip() for s in re.split(pattern, text) if s and s.strip()]


def _write_progress(path_progress: Path, done: int, total: int) -> None:
    try:
        with open(path_progress, "w") as f:
            f.write(str(round((done / max(total, 1)) * 100, 1)))
    except OSError:
        pass


def _check_interrupt(is_aborted: Callable[[], bool] | None) -> None:
    if is_aborted is not None and is_aborted():
        raise Exception("Process interrupted by user")


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def split_wtpsplit(
    texts: list[str],
    granularity: str | None,
    path_progress: Path,
    is_aborted: Callable[[], bool] | None = None,
) -> list[list[str]]:
    device = get_device()
    sat = SaT(WTPSPLIT_MODEL)
    # half precision divides the activation memory by two
    if device.type in ("cuda", "mps"):
        sat.half().to(device)

    paragraphs = granularity == "paragraph"
    all_chunks: list[list[str]] = []
    total = len(texts)
    try:
        start = 0
        while start < total:
            _check_interrupt(is_aborted)
            # take documents until the character budget is reached (at least one)
            end = start + 1
            batch_chars = len(texts[start])
            while end < total and batch_chars + len(texts[end]) <= WTPSPLIT_BATCH_CHARS:
                batch_chars += len(texts[end])
                end += 1
            segmented = sat.split(
                texts[start:end],
                do_paragraph_segmentation=paragraphs,
                batch_size=WTPSPLIT_BATCH_SIZE,
            )
            for element in segmented:
                if paragraphs:
                    chunks = ["".join(sentences).strip() for sentences in element]
                else:
                    chunks = [s.strip() for s in element]
                all_chunks.append([c for c in chunks if c])
            start = end
            _write_progress(path_progress, start, total)
            _empty_device_cache(device)
    finally:
        del sat
        gc.collect()
        _empty_device_cache(device)
    return all_chunks


def prepare_dataset(
    path_session: Path,
    params: PrepareSplitModel,
    path_progress: Path | None = None,
    is_aborted: Callable[[], bool] | None = None,
) -> Path:
    """
    Split the texts of an uploaded dataset (raw.parquet in the session
    directory) into chunks and write result.parquet, one row per chunk
    with id = old_id + chunk number and the kept columns duplicated.
    wtpsplit segmentation can use the GPU. Returns the result path.
    """
    path_raw = path_session.joinpath("raw.parquet")
    path_result = path_session.joinpath("result.parquet")
    if path_progress is None:
        path_progress = path_session.joinpath("progress")
    if not path_raw.exists():
        raise Exception("Uploaded data not found, please upload the file again")

    try:
        df = pd.read_parquet(path_raw)
        texts = list(concat_text_columns(df, params.cols_text))

        if params.col_id in df.columns:
            ids = [str(i) for i in df[params.col_id]]
        else:
            ids = [str(i) for i in range(len(df))]

        # suffix duplicated ids so every source row gets a distinct id
        if params.force_unique_id:
            seen_ids: set[str] = set()
            unique_ids = []
            for i in ids:
                candidate = i
                n = 1
                while candidate in seen_ids:
                    n += 1
                    candidate = f"{i}-{n}"
                seen_ids.add(candidate)
                unique_ids.append(candidate)
            ids = unique_ids

        cols_keep = [c for c in params.cols_keep if c in df.columns]
        keep_records = (
            df[cols_keep].to_dict("records") if cols_keep else [{} for _ in range(len(df))]
        )
        # everything needed is extracted, free the dataframe before splitting
        del df
        gc.collect()

        # optional cleaning before splitting
        if params.remove_html:
            texts = [re.sub(r"</?[a-zA-Z][^>]*>|<!--.*?-->", "", t, flags=re.S) for t in texts]
        if params.remove_urls:
            texts = [re.sub(r"https?://\S+|www\.\S+", "", t) for t in texts]
        if params.remove_html or params.remove_urls:
            # collapse spaces left behind by the removals (newlines kept)
            texts = [re.sub(r"[ \t]{2,}", " ", t) for t in texts]

        if params.method == "chunk":
            if not params.chunk_size or params.chunk_size <= 0:
                raise Exception("A positive chunk size is required")
            all_chunks = []
            for i, text in enumerate(texts, 1):
                _check_interrupt(is_aborted)
                all_chunks.append(split_by_chars(text, params.chunk_size))
                if i % 1000 == 0:
                    _write_progress(path_progress, i, len(texts))
        elif params.method == "regex":
            if not params.regex_pattern:
                raise Exception("A regex pattern is required")
            all_chunks = []
            for i, text in enumerate(texts, 1):
                _check_interrupt(is_aborted)
                all_chunks.append(split_by_regex(text, params.regex_pattern))
                if i % 1000 == 0:
                    _write_progress(path_progress, i, len(texts))
        elif params.method == "wtpsplit":
            all_chunks = split_wtpsplit(texts, params.granularity, path_progress, is_aborted)
        elif params.method == "none":
            all_chunks = [[c for c in [text.strip()] if c] for text in texts]
        else:
            raise Exception(f"Unknown split method {params.method}")

        # drop the units that are too short
        min_chars = max(params.min_chars, 0)
        if min_chars > 0:
            all_chunks = [[c for c in chunks if len(c) >= min_chars] for chunks in all_chunks]

        # drop units whose trimmed text already appeared (first occurrence kept)
        if params.drop_duplicates:
            seen: set[str] = set()
            deduped = []
            for chunks in all_chunks:
                kept = []
                for c in chunks:
                    key = c.strip()
                    if key in seen:
                        continue
                    seen.add(key)
                    kept.append(c)
                deduped.append(kept)
            all_chunks = deduped

        del texts
        rows = []
        for old_id, chunks, keep in zip(ids, all_chunks, keep_records):
            for num, chunk in enumerate(chunks, 1):
                rows.append({**keep, "id": f"{old_id}_{num}", "text": chunk})
        ordered = ["id", "text"] + [c for c in cols_keep if c not in ("id", "text")]
        result = pd.DataFrame(rows, columns=ordered)
        result.to_parquet(path_result, index=False)
        return path_result
    finally:
        path_progress.unlink(missing_ok=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
