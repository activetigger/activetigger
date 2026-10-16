from typing import Any

import pandas as pd
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.fr import French
from spacy.lang.nb import Norwegian


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _as_norm(value: Any) -> str | None:
    # stored parameters may hold "True"/"False" (UI select) or an sklearn norm name
    if value is None or value is False:
        return None
    if value is True:
        return "l2"
    v = str(value).strip().lower()
    if v in {"", "none", "false", "null"}:
        return None
    if v == "true":
        return "l2"
    return v


def compute_dfm(
    texts: Series,
    tfidf: Any = False,
    ngrams: Any = 1,
    min_term_freq: Any = 5,
    max_term_freq: Any = 1.0,
    max_features: Any = None,
    log: Any = False,
    language: str = "en",
    norm: Any = None,
) -> DataFrame:
    """
    Document Term Matrix, shared by the celery task and the in-process callers
    (lexicometrics, feature recomputation).

    Parameters are normalised because stored feature parameters come from the UI
    as floats and "True"/"False" strings. max_term_freq above 1 is an absolute
    document count, at most 1 a proportion.

    Norm :  None, l1, l2
    sublinear_tf : log
    Pas pris en compte : DFM : Min Docfreq
    https://quanteda.io/reference/dfm_tfidf.html
    """
    tfidf = _as_bool(tfidf)
    log = _as_bool(log)
    ngrams = int(float(ngrams))
    min_term_freq = int(float(min_term_freq))
    max_term_freq = float(max_term_freq)
    if max_term_freq > 1:
        max_term_freq = int(max_term_freq)
    max_features = None if max_features in (None, "", "None") else int(float(max_features))
    norm = _as_norm(norm)

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

    # compute matrix
    if tfidf:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, ngrams),
            min_df=min_term_freq,
            sublinear_tf=log,
            norm=norm,
            max_df=max_term_freq,
            max_features=max_features,
            stop_words=stop_words,
        )
    else:
        vectorizer = CountVectorizer(
            ngram_range=(1, ngrams),
            min_df=min_term_freq,
            max_df=max_term_freq,
            max_features=max_features,
            stop_words=stop_words,
        )

    dtm = vectorizer.fit_transform(texts)
    names = vectorizer.get_feature_names_out()
    return pd.DataFrame(dtm.toarray(), columns=names, index=texts.index)
