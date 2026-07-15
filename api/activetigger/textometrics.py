import json
from datetime import datetime, timezone
from pathlib import Path

from activetigger.datamodels import (
    TextometricsComputing,
    TextometricsModel,
    TextometricsParametersModel,
    TextometricsProjectStateModel,
    TextometricsStatisticsModel,
)
from activetigger.queue_manager import Queue
from activetigger.tasks.compute_textometrics import ComputeTextometrics


class Textometrics:
    """
    Manage textometry statistics of the annotable dataset.

    A single result per project, computed on demand and persisted as a JSON
    file so new statistics can be added later without a schema migration.
    """

    path: Path
    computing: list
    queue: Queue
    available: TextometricsModel | None

    def __init__(self, path: Path, computing: list, queue: Queue) -> None:
        self.path = path
        self.computing = computing
        self.queue = queue
        self.available = None
        self.load()

    @property
    def file_path(self) -> Path:
        return self.path.joinpath("textometrics.json")

    def load(self) -> None:
        if not self.file_path.exists():
            return
        try:
            with open(self.file_path, "r") as f:
                self.available = TextometricsModel(**json.load(f))
        except Exception as e:
            print("Error in loading textometrics", e)

    def _save(self) -> None:
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.available.model_dump() if self.available else None, f)
        except Exception as e:
            print("Error in saving textometrics", e)

    def training(self) -> dict[str, str]:
        """
        Currently computing, keyed by username.
        """
        return {e.user: "computing" for e in self.computing if e.kind == "textometrics"}

    def compute(self, project_slug: str, username: str, language: str) -> None:
        """
        Launch the textometrics computation in the queue.

        The task reads the train dataset from the project directory itself.
        """
        if len(self.training()) > 0:
            raise Exception("Textometrics are already being computed")

        parameters = TextometricsParametersModel(language=language)
        unique_id = self.queue.add_task(
            "textometrics",
            project_slug,
            ComputeTextometrics(
                path_data=self.path,
                language=parameters.language,
                tokenizer_name=parameters.tokenizer,
                n_most_frequent=parameters.n_most_frequent,
                tfidf_n_words=parameters.tfidf_n_words,
                tfidf_n_docs_per_word=parameters.tfidf_n_docs_per_word,
                tfidf_n_words_per_doc=parameters.tfidf_n_words_per_doc,
                tfidf_min_term_freq=parameters.tfidf_min_term_freq,
                tfidf_max_documents=parameters.tfidf_max_documents,
            ),
        )
        self.computing.append(
            TextometricsComputing(
                unique_id=unique_id,
                user=username,
                time=datetime.now(timezone.utc),
                kind="textometrics",
                parameters=parameters,
            )
        )

    def add(self, element: TextometricsComputing, results: dict) -> None:
        """
        Store statistics after computation.
        """
        self.available = TextometricsModel(
            computed_at=datetime.now(timezone.utc).isoformat(),
            user=element.user,
            parameters=element.parameters,
            statistics=TextometricsStatisticsModel(**results),
        )
        self._save()

    def get(self) -> TextometricsModel | None:
        return self.available

    def state(self) -> TextometricsProjectStateModel:
        return TextometricsProjectStateModel(
            available=self.available is not None,
            training=self.training(),
        )
