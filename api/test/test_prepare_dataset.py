from pathlib import Path

import pandas as pd

from activetigger.datamodels import PrepareSplitModel
from activetigger.tasks.prepare_dataset import (
    PrepareDataset,
    split_by_chars,
    split_by_regex,
)


def test_split_by_chars_respects_words():
    text = "the quick brown fox jumps over the lazy dog"
    chunks = split_by_chars(text, 15)
    assert all(len(c) <= 15 for c in chunks)
    assert " ".join(chunks) == text
    # no word is broken
    assert all(w in text.split() for c in chunks for w in c.split())


def test_split_by_chars_long_word_kept_whole():
    chunks = split_by_chars("hi supercalifragilistic yes", 5)
    assert "supercalifragilistic" in chunks


def test_split_by_chars_empty_text():
    assert split_by_chars("", 100) == []
    assert split_by_chars("   ", 100) == []


def test_split_by_regex_drops_empty_segments():
    assert split_by_regex("a. b.  . c", r"\.") == ["a", "b", "c"]


def _run_task(tmp_path: Path, df: pd.DataFrame, params: PrepareSplitModel) -> pd.DataFrame:
    df.to_parquet(tmp_path.joinpath("raw.parquet"), index=False)
    PrepareDataset(tmp_path, params)()
    return pd.read_parquet(tmp_path.joinpath("result.parquet"))


def test_prepare_dataset_chunk_method(tmp_path: Path):
    df = pd.DataFrame(
        {
            "doc": ["doc1", "doc2"],
            "content": ["one two three four", "five"],
            "source": ["a", "b"],
        }
    )
    params = PrepareSplitModel(
        session_id="x",
        cols_text=["content"],
        col_id="doc",
        cols_keep=["source"],
        method="chunk",
        chunk_size=10,
        min_chars=0,
    )
    result = _run_task(tmp_path, df, params)
    assert list(result.columns) == ["id", "text", "source"]
    assert list(result["id"]) == ["doc1_1", "doc1_2", "doc2_1"]
    assert list(result["text"]) == ["one two", "three four", "five"]
    # kept column duplicated on every chunk of the same document
    assert list(result["source"]) == ["a", "a", "b"]


def test_prepare_dataset_regex_method_row_number_id(tmp_path: Path):
    df = pd.DataFrame({"content": ["a---b", "c"]})
    params = PrepareSplitModel(
        session_id="x",
        cols_text=["content"],
        col_id="row_number",
        cols_keep=[],
        method="regex",
        regex_pattern="---",
        min_chars=0,
    )
    result = _run_task(tmp_path, df, params)
    assert list(result["id"]) == ["0_1", "0_2", "1_1"]
    assert list(result["text"]) == ["a", "b", "c"]


def test_prepare_dataset_min_chars_drops_short_units(tmp_path: Path):
    df = pd.DataFrame({"content": ["a long enough segment---no---another long segment"]})
    params = PrepareSplitModel(
        session_id="x",
        cols_text=["content"],
        col_id="row_number",
        cols_keep=[],
        method="regex",
        regex_pattern="---",
        # default min_chars=10 drops the "no" segment and renumbers the chunks
    )
    result = _run_task(tmp_path, df, params)
    assert list(result["text"]) == ["a long enough segment", "another long segment"]
    assert list(result["id"]) == ["0_1", "0_2"]


def test_prepare_dataset_concatenates_text_columns(tmp_path: Path):
    df = pd.DataFrame({"t1": ["hello"], "t2": ["world"]})
    params = PrepareSplitModel(
        session_id="x",
        cols_text=["t1", "t2"],
        col_id="row_number",
        cols_keep=[],
        method="chunk",
        chunk_size=1000,
    )
    result = _run_task(tmp_path, df, params)
    assert list(result["text"]) == ["hello world"]
