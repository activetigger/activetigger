import datetime
import json
import math
import os
import pickle
import shutil
from collections.abc import Callable
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
from activetigger.config import config
from activetigger.datamodels import (
    EventsDict,
    KnnParams,
    LogisticL1Params,
    LogisticL2Params,
    MLStatisticsModel,
    Multi_naivebayesParams,
    QuickModelComputed,
    RandomforestParams,
)
from activetigger.functions import get_metrics_multiclass
from activetigger.monitoring import TaskTimer
from pydantic import BaseModel
from scipy.stats import entropy
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


class TrainMLMultiClassInput(BaseModel):
    project_slug: str
    path: Path
    path_df: Path
    col_Y: str
    cols_X: list[str]
    col_text: str | None = None
    name: str
    user: str
    model_type: str
    model_params: dict
    scheme: str
    features: list
    labels: list[str]
    standardize: bool = False
    cv10: bool = False
    balance_classes: bool = False
    exclude_labels: list[str] = []
    test_size: float = 0.2
    retrain: bool = False
    random_seed: int = 42


# models for which balance_classes is not supported
MODELS_WITHOUT_BALANCE = ("knn", "multi_naivebayes")


def build_model(
    model_type: str, model_params: dict, balance_classes: bool
) -> tuple[BaseEstimator, dict]:
    """
    Instantiate the sklearn estimator from its type and params.
    Returns the estimator and the validated params.
    """
    class_weight = "balanced" if balance_classes else None
    seed = config.random_seed
    if model_type == "knn":
        p_knn = KnnParams(**model_params)
        return (
            KNeighborsClassifier(n_neighbors=int(p_knn.n_neighbors), n_jobs=-1),
            p_knn.model_dump(),
        )
    if model_type == "logistic-l1":
        p_l1 = LogisticL1Params(**model_params)
        return (
            LogisticRegression(
                solver="saga",
                l1_ratio=1.0,
                C=p_l1.costLogL1,
                class_weight=class_weight,
                random_state=seed,
            ),
            p_l1.model_dump(),
        )
    if model_type == "logistic-l2":
        p_l2 = LogisticL2Params(**model_params)
        return (
            LogisticRegression(
                solver="lbfgs",
                C=p_l2.costLogL2,
                class_weight=class_weight,
                random_state=seed,
            ),
            p_l2.model_dump(),
        )
    if model_type == "randomforest":
        # mtry in R is max_features in sklearn
        p_rf = RandomforestParams(**model_params)
        return (
            RandomForestClassifier(
                n_estimators=int(p_rf.n_estimators),
                max_features=int(p_rf.max_features) if p_rf.max_features is not None else None,
                class_weight=class_weight,
                n_jobs=-1,
                random_state=seed,
            ),
            p_rf.model_dump(),
        )
    if model_type == "multi_naivebayes":
        # TODO: calculate class prior for docfreq & termfreq
        p_nb = Multi_naivebayesParams(**model_params)
        return (
            MultinomialNB(alpha=p_nb.alpha, fit_prior=p_nb.fit_prior, class_prior=p_nb.class_prior),
            p_nb.model_dump(),
        )
    raise ValueError(f"Unknown model type: {model_type}")


class TrainMLMultiClass:
    """
    Fit a sklearn model
    """

    kind = "train_ml"

    def __init__(
        self, unique_id: str, inputs: TrainMLMultiClassInput, is_aborted: Callable[[], bool]
    ):
        self.unique_id = unique_id
        self.random_seed = inputs.random_seed
        self.is_aborted = is_aborted
        self.name = inputs.name
        self.user = inputs.user
        self.cv10 = inputs.cv10
        self.balance_classes = (
            inputs.balance_classes and inputs.model_type not in MODELS_WITHOUT_BALANCE
        )
        self.model, self.model_params = build_model(
            inputs.model_type, inputs.model_params, self.balance_classes
        )
        self.exclude_labels = inputs.exclude_labels  # labels are excluded earlier on in the pipeline, but we must save this information somewhere
        self.test_size = inputs.test_size
        self.path = inputs.path
        self.path_df = inputs.path_df
        self.model_path = inputs.path.joinpath(inputs.name)
        self.work_path = inputs.path.joinpath(f"{inputs.name}.tmp")
        self.retrain = inputs.retrain
        self.scheme = inputs.scheme
        self.features = inputs.features
        self.labels = inputs.labels
        self.model_type = inputs.model_type
        self.standardize = inputs.standardize
        self.X, self.X_f, self.Y, self.Y_f, self.texts = self.__check_data(
            inputs.cols_X, inputs.col_Y, inputs.col_text, self.exclude_labels
        )

    def __check_data(
        self, cols_X: list[str], col_Y: str, col_text: str | None, exclude_labels: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series | None]:
        """
        Load the data from file : X, Y and texts if available
        """
        df = pd.read_parquet(self.path_df)
        X = df[cols_X]
        Y = df[col_Y]
        texts = None
        if col_text is not None:
            texts = df[col_text]
        rows_to_exclude = np.logical_or(np.isin(Y, exclude_labels), Y.isna())
        rows_to_keep = np.invert(rows_to_exclude)
        return X, X.loc[rows_to_keep, :], Y, Y[rows_to_keep], texts

    def __init_paths(self, retrain: bool) -> None:
        """
        Create the staging directory for the files to be saved. The
        existing model directory (retrain case) is left untouched until
        the new training has fully succeeded.
        """
        if not retrain and self.model_path.exists():
            raise Exception("The model already exists")
        if self.work_path.exists():
            shutil.rmtree(self.work_path)
        os.mkdir(self.work_path)

    def __promote_staging(self) -> None:
        """
        Atomically replace the model directory with the staged files.
        """
        if self.model_path.exists():
            shutil.rmtree(self.model_path)
        os.rename(self.work_path, self.model_path)

    def __split_set(
        self, X, Y, test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Remove null elements and return train/test splits
        (equivalent of train_test_split from sklearn)
        """
        index = X.copy().index.to_series()
        index = index.sample(frac=1.0, random_state=self.random_seed)
        n_total = len(index)
        n_test = math.ceil(n_total * test_size)
        n_train = n_total - n_test
        index_train = index.head(n_train)
        index_test = index.tail(n_test)
        X_train = X.loc[index_train.index, :]
        Y_train = Y.loc[index_train.index]
        X_test = X.loc[index_test.index, :]
        Y_test = Y.loc[index_test.index]
        return X_train, X_test, Y_train, Y_test

    def __compute_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> MLStatisticsModel:
        """
        Compute metrics
        """
        texts = self.texts.loc[y_true.index] if self.texts is not None else None
        metrics = get_metrics_multiclass(
            y_true,
            y_pred,
            texts=texts,
        )
        return metrics

    def __compute_cv10(self) -> MLStatisticsModel:
        """
        Compute cv (predict and compute metrics)
        """
        num_folds = 10
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=self.random_seed)
        Y_pred_10cv = pd.Series(
            cross_val_predict(self.model, self.X_f, self.Y_f, cv=kf), index=self.Y_f.index
        )

        statistics_cv10 = get_metrics_multiclass(
            self.Y_f,
            Y_pred_10cv,
        )
        # overwrite false_predictions
        statistics_cv10.false_predictions = None

        return statistics_cv10

    def __create_saving_files(
        self,
        proba: pd.DataFrame,
        X_train: pd.DataFrame,
        Y_train: pd.Series,
        metrics_train: MLStatisticsModel,
        metrics_test: MLStatisticsModel,
        statistics_cv10: MLStatisticsModel | None,
    ) -> None:
        """Save the following files in the staging directory:
        - proba.csv with the probabilities
        - data using during training (training_data.parquet)
        - a pickle version of the database entry (#NOTE: AM: Artefact ?)
        - metrics for the training (train, trainvalid and cv10)
        """
        # Write the proba
        proba.to_csv(self.work_path / "proba.csv")

        # Write the training data
        X_train["label"] = Y_train
        X_train.to_parquet(self.work_path / "training_data.parquet")

        # Dump it in the folder
        element = QuickModelComputed(
            time=datetime.datetime.now(timezone.utc),
            model=self.model,
            user=self.user,
            name=self.name,
            scheme=self.scheme,
            features=self.features,
            labels=self.labels,
            model_type=self.model_type,
            model_params=self.model_params,
            standardize=self.standardize,
            cv10=self.cv10,
            balance_classes=self.balance_classes,
            exclude_labels=self.exclude_labels,
            test_size=self.test_size,
            retrain=self.retrain,
            proba=proba,
            statistics_train=metrics_train,
            statistics_test=metrics_test,
            statistics_cv10=statistics_cv10,
        )

        with open(self.work_path / "model.pkl", "wb") as file:
            pickle.dump(element, file)

        # Write the statistics
        with open(self.work_path / "metrics_training.json", "w") as file:
            json.dump(
                {
                    "train": metrics_train.model_dump(mode="json"),
                    "trainvalid": metrics_test.model_dump(mode="json"),
                    "cv10": statistics_cv10.model_dump(mode="json") if statistics_cv10 else None,
                },
                file,
            )

    def _check_cancelled(self) -> None:
        """Raise if the user requested cancellation."""
        if self.is_aborted():
            raise Exception("Process interrupted by user")

    def run(self) -> EventsDict:
        """
        Fit quickmodel and calculate statistics.
        On failure the staged files are removed and the previous model
        directory (retrain case) is left untouched.
        """
        try:
            return self.__run()
        except Exception:
            shutil.rmtree(self.work_path, ignore_errors=True)
            raise
        finally:
            self.path_df.unlink(missing_ok=True)

    def __run(self) -> EventsDict:
        task_timer = TaskTimer(
            compulsory_steps=["setup", "train", "evaluate", "save_files"], optional_steps=["cv10"]
        )

        task_timer.start("setup")
        self.__init_paths(self.retrain)

        X_train, X_test, Y_train, Y_test = self.__split_set(self.X_f, self.Y_f, self.test_size)
        task_timer.stop("setup")

        self._check_cancelled()

        # Fit model --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        try:
            task_timer.start("train")
            self.model.fit(X_train, Y_train)  # ty: ignore[unresolved-attribute]
            task_timer.stop("train")
        except Exception as e:
            raise Exception((f"Problem fitting the model (TrainMLMultiClass.__call__)\nError: {e}"))

        self._check_cancelled()

        # predict on test data --- --- --- --- --- --- --- --- --- --- --- --- -
        try:
            Y_pred_train = pd.Series(
                self.model.predict(X_train),  # ty: ignore[unresolved-attribute]
                index=X_train.index,
            )
            Y_pred_test = pd.Series(
                self.model.predict(X_test),  # ty: ignore[unresolved-attribute]
                index=X_test.index,
            )
        except Exception as e:
            raise Exception(
                (
                    f"Problem computing predictions after fitting (TrainMLMultiClass.__call__)\nError: {e}"
                )
            )

        # compute probabilities for all data
        try:
            task_timer.start("evaluate")
            proba_values = self.model.predict_proba(self.X)  # ty: ignore[unresolved-attribute]
            proba = pd.DataFrame(
                proba_values,
                columns=self.model.classes_,  # ty: ignore[unresolved-attribute]
                index=self.X.index,
            )
            proba["prediction"] = proba.idxmax(axis=1)
            proba["entropy"] = entropy(proba_values, axis=1)
            # Add entropy-LABEL defined as the entropy of p(A) / 1-p(A)
            for label in self.model.classes_:  # ty: ignore[unresolved-attribute]
                prob_A_not_A = np.column_stack([proba[label], 1 - proba[label]])
                proba[f"entropy-{label}"] = entropy(prob_A_not_A, axis=1)
        except Exception as e:
            raise Exception(
                (f"Problem calculating the entropy (TrainMLMultiClass.__call__)\nError: {e}")
            )

        self._check_cancelled()

        # Compute metrics --- --- --- --- --- --- --- --- --- --- --- --- --- --
        try:
            metrics_train = self.__compute_metrics(y_true=Y_train, y_pred=Y_pred_train)
            metrics_test = self.__compute_metrics(y_true=Y_test, y_pred=Y_pred_test)
            task_timer.stop("evaluate")
        except Exception as e:
            raise Exception(
                (f"Problem computing the metrics (TrainMLMultiClass.__call__)\nError: {e}")
            )

        self._check_cancelled()

        if self.cv10:
            try:
                task_timer.start("cv10")
                statistics_cv10 = self.__compute_cv10()
                task_timer.stop("cv10")
            except Exception as e:
                raise Exception(
                    (
                        f"Problem computing the cross valisation (TrainMLMultiClass.__compute_cv10)\nError: {e}"
                    )
                )
        else:
            statistics_cv10 = None

        task_timer.start("save_files")
        self.__create_saving_files(
            proba, X_train, Y_train, metrics_train, metrics_test, statistics_cv10
        )
        self.__promote_staging()
        task_timer.stop("save_files")

        return EventsDict({"events": task_timer.get_events()})
