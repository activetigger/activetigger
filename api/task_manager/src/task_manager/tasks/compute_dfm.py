import pandas as pd
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.fr import French
from spacy.lang.nb import Norwegian


def compute_dfm(
    texts: Series,
    tfidf: bool = False,
    ngrams: int = 1,
    min_term_freq: int = 5,
    max_term_freq: int | float = 1.0,
    max_features: int | None = None,
    log: bool = False,
    language: str = "en",
    norm: str | None = None,
) -> DataFrame:
    """
    Document Term Matrix, shared by the celery task and the in-process callers
    (lexicometrics, feature recomputation).

    Norm :  None, l1, l2
    sublinear_tf : log
    Pas pris en compte : DFM : Min Docfreq
    https://quanteda.io/reference/dfm_tfidf.html
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
