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

from activetigger.config import config
from activetigger.tasks.base_task import BaseTask


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


class ComputeTextometrics(BaseTask):
    """
    Compute textometry statistics on a corpus.

    Returns a flat dict keyed by statistic name so new statistics can be
    added later without changing the storage format.
    """

    kind = "textometrics"

    def __init__(
        self,
        path_data: Path,
        col_text: str = "text",
        language: str = "en",
        tokenizer_name: str = "bert-base-multilingual-cased",
        n_most_frequent: int = 100,
        **kwargs,
    ):
        super().__init__()
        self.path_data = path_data
        self.col_text = col_text
        self.language = language
        self.tokenizer_name = tokenizer_name
        self.n_most_frequent = n_most_frequent

    def __stop_process_opportunity(self):
        if self.event is not None and self.event.is_set():
            raise Exception("Textometrics computation interrupted by user")

    def __call__(self) -> dict:
        texts = self.load_texts()

        words_per_doc = texts.str.split().str.len()
        self.__stop_process_opportunity()

        tokens_per_doc = self.count_tokens(texts)
        self.__stop_process_opportunity()

        most_frequent_words = self.most_frequent_words(texts)

        return {
            "words_per_doc": distribution_statistics(words_per_doc),
            "tokens_per_doc": distribution_statistics(tokens_per_doc),
            "most_frequent_words": most_frequent_words,
        }

    def load_texts(self) -> Series:
        """
        Read the train dataset from the project directory.
        """
        df = pd.read_parquet(self.path_data.joinpath(config.train_file), columns=[self.col_text])
        return df[self.col_text].fillna("").astype(str)

    def count_tokens(self, texts: Series, batch_size: int = 1000) -> Series:
        """
        Number of subword tokens per document (relevant to model context limits).
        """
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if tokenizer is None:
            raise Exception(f"Could not load tokenizer {self.tokenizer_name}")
        counts: list[int] = []
        values = texts.tolist()
        for i in range(0, len(values), batch_size):
            batch = tokenizer(values[i : i + batch_size], truncation=False)
            counts.extend(len(input_ids) for input_ids in batch.input_ids)
            self.__stop_process_opportunity()
        return Series(counts, index=texts.index)

    def most_frequent_words(self, texts: Series) -> list[dict]:
        """
        Top words by corpus frequency, stopwords excluded.
        """
        # load stopwords
        if self.language == "fr":
            stop_words = list(French.Defaults.stop_words)
        elif self.language == "es":
            stop_words = list(Spanish.Defaults.stop_words)
        elif self.language == "de":
            stop_words = list(German.Defaults.stop_words)
        elif self.language == "en":
            stop_words = list(English.Defaults.stop_words)
        elif self.language == "nb":
            stop_words = list(Norwegian.Defaults.stop_words)
        else:
            stop_words = list(English.Defaults.stop_words)
            print(f"Language {self.language} not supported, using English stop words.")

        vectorizer = CountVectorizer(stop_words=stop_words)
        dtm = vectorizer.fit_transform(texts)
        frequencies = np.asarray(dtm.sum(axis=0)).ravel()
        words = vectorizer.get_feature_names_out()
        top = np.argsort(frequencies)[::-1][: self.n_most_frequent]
        return [{"word": str(words[i]), "count": int(frequencies[i])} for i in top]
