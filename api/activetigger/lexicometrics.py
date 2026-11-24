import json
from datetime import datetime, timezone
from pathlib import Path

from task_manager.tasks.compute_lexicometrics_task import (
    ComputeLexicometricsTask,
    ComputeLexicometricsTaskInputs,
)

from activetigger.config import config
from activetigger.datamodels import (
    LexicometricsModel,
    LexicometricsParametersModel,
    LexicometricsProjectStateModel,
    LexicometricsStatisticsModel,
    TaskComputing,
)
from activetigger.errors import AlreadyExistsError
from activetigger.task_registry import TaskRegistry


class Lexicometrics:
    """
    Manage lexicometry statistics of the annotable dataset.

    A single result per project, computed on demand and persisted as a JSON
    file so new statistics can be added later without a schema migration.
    """

    project_slug: str
    path: Path
    task_registry: TaskRegistry
    available: LexicometricsModel | None

    def __init__(self, project_slug: str, path: Path, task_registry: TaskRegistry) -> None:
        self.project_slug = project_slug
        self.path = path
        self.task_registry = task_registry
        self.available = None
        self.load()

    @property
    def file_path(self) -> Path:
        return self.path.joinpath("lexicometrics.json")

    @property
    def _legacy_file_path(self) -> Path:
        # results computed before the textometrics -> lexicometrics rename
        return self.path.joinpath("textometrics.json")

    def load(self) -> None:
        if not self.file_path.exists() and self._legacy_file_path.exists():
            try:
                self._legacy_file_path.rename(self.file_path)
            except Exception as e:
                print("Error migrating legacy textometrics file", e)
        if not self.file_path.exists():
            return
        try:
            with open(self.file_path, "r") as f:
                self.available = LexicometricsModel(**json.load(f))
        except Exception as e:
            print("Error in loading lexicometrics", e)

    def _save(self) -> None:
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.available.model_dump() if self.available else None, f)
        except Exception as e:
            print("Error in saving lexicometrics", e)

    def training(self) -> dict[str, str]:
        """
        Currently computing, keyed by username.
        """
        return {
            e.user: "computing"
            for e in self.task_registry.project_computing_tasks(
                self.project_slug, kinds=["compute_lexicometrics"]
            )
        }

    def start_compute_task(self, project_slug: str, username: str, language: str) -> str:
        """
        Launch the lexicometrics computation as a celery task.

        The worker reads the train dataset from the project directory itself.
        """
        if len(self.training()) > 0:
            raise AlreadyExistsError("Lexicometrics are already being computed")

        parameters = LexicometricsParametersModel(language=language)

        task_inputs = ComputeLexicometricsTaskInputs(
            project_slug=project_slug,
            user=username,
            path_data=self.path,
            train_file=config.train_file,
            language=parameters.language,
            tokenizer_name=parameters.tokenizer,
            n_most_frequent=parameters.n_most_frequent,
            tfidf_n_words=parameters.tfidf_n_words,
            tfidf_n_docs_per_word=parameters.tfidf_n_docs_per_word,
            tfidf_n_words_per_doc=parameters.tfidf_n_words_per_doc,
            tfidf_min_term_freq=parameters.tfidf_min_term_freq,
            tfidf_max_documents=parameters.tfidf_max_documents,
            parameters=parameters,
        )
        return self.task_registry.start_task(ComputeLexicometricsTask.name, task_inputs)

    def add(
        self, task_computing: TaskComputing[ComputeLexicometricsTaskInputs], results: dict
    ) -> None:
        """
        Store statistics after computation.
        """
        self.available = LexicometricsModel(
            computed_at=datetime.now(timezone.utc).isoformat(),
            user=task_computing.user,
            parameters=task_computing.inputs.parameters,
            statistics=LexicometricsStatisticsModel(**results),
        )
        self._save()

    def get(self) -> LexicometricsModel | None:
        return self.available

    def state(self) -> LexicometricsProjectStateModel:
        return LexicometricsProjectStateModel(
            available=self.available is not None,
            training=self.training(),
        )
