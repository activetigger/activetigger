import pandas as pd
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.fr import French

from activetigger.tasks.base_task import BaseTask


class ComputeDfm(BaseTask):
    """
    Compute DFM feature
    """

    kind = "compute_feature_dfm"

    def __init__(
        self,
        texts: Series,
        tfidf: bool = False,
        ngrams: int = 1,
        min_term_freq: int = 5,
        max_term_freq: int | float = 1.0,
        log: bool = False,
        language: str = "en",
        norm=None,
        **kwargs,
    ):
        super().__init__()
        self.texts = texts
        self.tfidf = tfidf
        self.ngrams = ngrams
        self.min_term_freq = min_term_freq
        self.max_term_freq = max_term_freq
        self.log = log
        self.language = language
        self.norm = norm

    def __call__(self) -> DataFrame:
        """
        Compute Document Term Matrix

        Norm :  None, l1, l2
        sublinear_tf : log
        Pas pris en compte : DFM : Min Docfreq
        https://quanteda.io/reference/dfm_tfidf.html
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
        else:
            stop_words = list(English.Defaults.stop_words)
            print(f"Language {self.language} not supported, using English stop words.")

        # compute matrix
        if self.tfidf:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, self.ngrams),
                min_df=self.min_term_freq,
                sublinear_tf=self.log,
                norm=self.norm,
                max_df=self.max_term_freq,
                stop_words=stop_words,
            )
        else:
            vectorizer = CountVectorizer(
                ngram_range=(1, self.ngrams),
                min_df=self.min_term_freq,
                max_df=self.max_term_freq,
                stop_words=stop_words,
            )

        dtm = vectorizer.fit_transform(self.texts)
        names = vectorizer.get_feature_names_out()
        dtm = pd.DataFrame(dtm.toarray(), columns=names, index=self.texts.index)
        return dtm
