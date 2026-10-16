from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import Series
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.fr import French
from spacy.lang.nb import Norwegian
from transformers import AutoTokenizer  # ty: ignore[possibly-missing-import]

from task_manager.tasks.compute_dfm import compute_dfm


def distribution_statistics(values: Series, bins: int = 30) -> dict:
    """
    Summary + binned histogram of a numeric per-document series.

    Binned (not per-document) so the stored JSON stays small on large corpora.
    Matches DistributionModel in datamodels.
    """

    def _round(value) -> float | None:
        return None if np.isnan(value) else round(float(value), 2)

    summary = {
        "count": int(values.count()),
        "mean": _round(values.mean()),
        "std": _round(values.std()),
        "min": _round(values.min()),
        "q25": _round(values.quantile(0.25)),
        "median": _round(values.median()),
        "q75": _round(values.quantile(0.75)),
        "max": _round(values.max()),
    }
    counts, bin_edges = np.histogram(values, bins=bins)
    return {
        "summary": summary,
        "histogram": {
            "bin_edges": [float(edge) for edge in bin_edges],
            "counts": [int(count) for count in counts],
        },
    }


def _check_aborted(is_aborted: Callable[[], bool] | None) -> None:
    if is_aborted is not None and is_aborted():
        raise Exception("Process interrupted by user")


def compute_lexicometrics(
    path_data: Path,
    train_file: str = "train.parquet",
    col_text: str = "text",
    language: str = "en",
    tokenizer_name: str = "bert-base-multilingual-cased",
    n_most_frequent: int = 100,
    tfidf_n_words: int = 300,
    tfidf_n_docs_per_word: int = 25,
    tfidf_n_words_per_doc: int = 10,
    tfidf_min_term_freq: int = 5,
    tfidf_max_documents: int = 10000,
    is_aborted: Callable[[], bool] | None = None,
) -> dict:
    """
    Lexicometry statistics of the train dataset read from the project directory.

    Returns a flat dict keyed by statistic name so new statistics can be
    added later without changing the storage format.
    """
    texts = load_texts(path_data, train_file, col_text)

    words_per_doc = texts.str.split().str.len()
    _check_aborted(is_aborted)

    tokens_per_doc = count_tokens(texts, tokenizer_name, is_aborted=is_aborted)
    _check_aborted(is_aborted)

    frequent_words = most_frequent_words(texts, language, n_most_frequent)
    _check_aborted(is_aborted)

    tfidf_words, tfidf_documents = tfidf_statistics(
        texts,
        language=language,
        tfidf_n_words=tfidf_n_words,
        tfidf_n_docs_per_word=tfidf_n_docs_per_word,
        tfidf_n_words_per_doc=tfidf_n_words_per_doc,
        tfidf_min_term_freq=tfidf_min_term_freq,
        tfidf_max_documents=tfidf_max_documents,
        is_aborted=is_aborted,
    )

    return {
        "words_per_doc": distribution_statistics(words_per_doc),
        "tokens_per_doc": (
            distribution_statistics(tokens_per_doc) if tokens_per_doc is not None else None
        ),
        "most_frequent_words": frequent_words,
        "tfidf_words": tfidf_words,
        "tfidf_documents": tfidf_documents,
    }


def load_texts(path_data: Path, train_file: str, col_text: str) -> Series:
    df = pd.read_parquet(path_data.joinpath(train_file), columns=[col_text])
    return df[col_text].fillna("").astype(str)


def count_tokens(
    texts: Series,
    tokenizer_name: str,
    batch_size: int = 1000,
    is_aborted: Callable[[], bool] | None = None,
) -> Series | None:
    """
    Number of subword tokens per document (relevant to model context limits).
    """
    try:
        try:
            # prefer the local cache to avoid any network dependency
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Could not load tokenizer {tokenizer_name}, skipping token counts: {e}")
        return None
    if tokenizer is None:
        print(f"Could not load tokenizer {tokenizer_name}, skipping token counts")
        return None
    counts: list[int] = []
    values = texts.tolist()
    for i in range(0, len(values), batch_size):
        batch = tokenizer(values[i : i + batch_size], truncation=False)
        counts.extend(len(input_ids) for input_ids in batch.input_ids)
        _check_aborted(is_aborted)
    return Series(counts, index=texts.index)


def tfidf_statistics(
    texts: Series,
    language: str,
    tfidf_n_words: int,
    tfidf_n_docs_per_word: int,
    tfidf_n_words_per_doc: int,
    tfidf_min_term_freq: int,
    tfidf_max_documents: int,
    is_aborted: Callable[[], bool] | None = None,
) -> tuple[list[dict] | None, list[dict] | None]:
    """
    TF-IDF slices, kept small on purpose (the full matrix is n_docs x vocab):
    - per word (vocabulary capped to tfidf_n_words), the documents with the
      highest scores
    - per document, its most distinctive words — only stored when the train
      set has at most tfidf_max_documents documents
    """
    try:
        dtm = compute_dfm(
            texts,
            tfidf=True,
            norm="l2",
            min_term_freq=tfidf_min_term_freq,
            max_features=tfidf_n_words,
            language=language,
        )
    except ValueError as e:
        # corpus too small for the vocabulary constraints
        print("TF-IDF statistics skipped:", e)
        return None, None
    _check_aborted(is_aborted)

    tfidf_words = []
    for word in dtm.columns:
        col = dtm[word]
        top = col.nlargest(tfidf_n_docs_per_word)
        top = top[top > 0]
        tfidf_words.append(
            {
                "word": str(word),
                "n_documents": int((col > 0).sum()),
                "top_documents": [
                    {"element_id": str(element_id), "score": round(float(score), 3)}
                    for element_id, score in top.items()
                ],
            }
        )
    tfidf_words.sort(key=lambda w: w["n_documents"], reverse=True)
    _check_aborted(is_aborted)

    if len(dtm) > tfidf_max_documents:
        return tfidf_words, None

    tfidf_documents = []
    words = dtm.columns.to_numpy()
    matrix = dtm.to_numpy()
    order = np.argsort(matrix, axis=1)[:, ::-1][:, :tfidf_n_words_per_doc]
    for row, element_id in enumerate(dtm.index):
        top_words = [
            {"word": str(words[j]), "score": round(float(matrix[row, j]), 3)}
            for j in order[row]
            if matrix[row, j] > 0
        ]
        tfidf_documents.append({"element_id": str(element_id), "top_words": top_words})
    return tfidf_words, tfidf_documents


def most_frequent_words(texts: Series, language: str, n_most_frequent: int) -> list[dict]:
    """
    Top words by corpus frequency, stopwords excluded.
    Based on Countvectorizer
    """
    # load stopwords
    if language == "fr":
        stop_words = list(French.Defaults.stop_words)
    elif language == "es":
        stop_words = list(Spanish.Defaults.stop_words)
    elif language == "de":
        stop_words = list(German.Defaults.stop_words)
    elif language == "en":
        stop_words = list(English.Defaults.stop_words)
    elif language == "nb":
        stop_words = list(Norwegian.Defaults.stop_words)
    else:
        stop_words = list(English.Defaults.stop_words)
        print(f"Language {language} not supported, using English stop words.")

    vectorizer = CountVectorizer(stop_words=stop_words)
    dtm = vectorizer.fit_transform(texts)
    frequencies = np.asarray(dtm.sum(axis=0)).ravel()
    words = vectorizer.get_feature_names_out()
    top = np.argsort(frequencies)[::-1][:n_most_frequent]
    return [{"word": str(words[i]), "count": int(frequencies[i])} for i in top]
