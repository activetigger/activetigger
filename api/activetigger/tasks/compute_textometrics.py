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
from activetigger.tasks.compute_dfm import ComputeDfm


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
        tfidf_n_words: int = 300,
        tfidf_n_docs_per_word: int = 25,
        tfidf_n_words_per_doc: int = 10,
        tfidf_min_term_freq: int = 5,
        tfidf_max_documents: int = 10000,
        **kwargs,
    ):
        super().__init__()
        self.path_data = path_data
        self.col_text = col_text
        self.language = language
        self.tokenizer_name = tokenizer_name
        self.n_most_frequent = n_most_frequent
        self.tfidf_n_words = tfidf_n_words
        self.tfidf_n_docs_per_word = tfidf_n_docs_per_word
        self.tfidf_n_words_per_doc = tfidf_n_words_per_doc
        self.tfidf_min_term_freq = tfidf_min_term_freq
        self.tfidf_max_documents = tfidf_max_documents

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
        self.__stop_process_opportunity()

        tfidf_words, tfidf_documents = self.tfidf_statistics(texts)

        return {
            "words_per_doc": distribution_statistics(words_per_doc),
            "tokens_per_doc": distribution_statistics(tokens_per_doc),
            "most_frequent_words": most_frequent_words,
            "tfidf_words": tfidf_words,
            "tfidf_documents": tfidf_documents,
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

    def tfidf_statistics(self, texts: Series) -> tuple[list[dict] | None, list[dict] | None]:
        """
        TF-IDF slices, kept small on purpose (the full matrix is n_docs x vocab):
        - per word (vocabulary capped to tfidf_n_words), the documents with the
          highest scores
        - per document, its most distinctive words — only stored when the train
          set has at most tfidf_max_documents documents
        """
        try:
            dtm = ComputeDfm(
                texts=texts,
                tfidf=True,
                norm="l2",
                min_term_freq=self.tfidf_min_term_freq,
                max_features=self.tfidf_n_words,
                language=self.language,
            )()
        except ValueError as e:
            # corpus too small for the vocabulary constraints
            print("TF-IDF statistics skipped:", e)
            return None, None
        self.__stop_process_opportunity()

        tfidf_words = []
        for word in dtm.columns:
            col = dtm[word]
            top = col.nlargest(self.tfidf_n_docs_per_word)
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
        self.__stop_process_opportunity()

        if len(dtm) > self.tfidf_max_documents:
            return tfidf_words, None

        tfidf_documents = []
        words = dtm.columns.to_numpy()
        matrix = dtm.to_numpy()
        order = np.argsort(matrix, axis=1)[:, ::-1][:, : self.tfidf_n_words_per_doc]
        for row, element_id in enumerate(dtm.index):
            top_words = [
                {"word": str(words[j]), "score": round(float(matrix[row, j]), 3)}
                for j in order[row]
                if matrix[row, j] > 0
            ]
            tfidf_documents.append({"element_id": str(element_id), "top_words": top_words})
        return tfidf_words, tfidf_documents

    def most_frequent_words(self, texts: Series) -> list[dict]:
        """
        Top words by corpus frequency, stopwords excluded.
        Based on Countvectorizer
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
