from pandas import DataFrame, Series
from task_manager.tasks.compute_dfm import compute_dfm

from activetigger.tasks.base_task import BaseTask


class ComputeDfm(BaseTask):
    """
    In-process DFM for composite tasks (lexicometrics, feature recomputation).
    Standalone feature computation goes through the celery task; both share compute_dfm.
    """

    kind = "compute_feature_dfm"

    def __init__(
        self,
        texts: Series,
        tfidf: bool = False,
        ngrams: int = 1,
        min_term_freq: int = 5,
        max_term_freq: int | float = 1.0,
        max_features: int | None = None,
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
        self.max_features = max_features
        self.log = log
        self.language = language
        self.norm = norm

    def __call__(self) -> DataFrame:
        return compute_dfm(
            self.texts,
            tfidf=self.tfidf,
            ngrams=self.ngrams,
            min_term_freq=self.min_term_freq,
            max_term_freq=self.max_term_freq,
            max_features=self.max_features,
            log=self.log,
            language=self.language,
            norm=self.norm,
        )
