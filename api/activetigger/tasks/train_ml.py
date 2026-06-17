import datetime
import json
import math
import os
import pickle
import shutil
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from activetigger.config import config
from activetigger.datamodels import (
    EventsModel,
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
from activetigger.tasks.base_task import BaseTask

PARAM_GRIDS = {
    "logistic-l1": {"C": [0.01, 0.1, 1, 10, 100]},
    "logistic-l2": {"C": [0.01, 0.1, 1, 10, 100]},
    "knn": {"n_neighbors": [3, 5, 7, 9]},
    "multi_naivebayes": {"alpha": [0.1, 0.5, 1.0, 2.0]},
}


SKLEARN_TO_PYDANTIC = {
    "logistic-l1": {"C": "costLogL1"},
    "logistic-l2": {"C": "costLogL2"},
    "knn": {},
    "randomforest": {},
    "multi_naivebayes": {},
}


def build_model(
    model_type: str, model_params: dict, balance_classes: bool
) -> tuple[BaseEstimator, dict, bool]:

    if model_type == "knn":
        params_knn = KnnParams(**model_params)
        model = KNeighborsClassifier(n_neighbors=int(params_knn.n_neighbors), n_jobs=-1)
        balance_classes = False
        model_params = params_knn.model_dump()
        return model, model_params, balance_classes

    if model_type == "logistic-l1":
        params_libL1 = LogisticL1Params(**model_params)
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=params_libL1.costLogL1,
            class_weight="balanced" if balance_classes else None,
            n_jobs=-1,
            random_state=config.random_seed,
        )
        model_params = params_libL1.model_dump()
        return model, model_params, balance_classes

    if model_type == "logistic-l2":
        params_libL2 = LogisticL2Params(**model_params)
        model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=params_libL2.costLogL2,
            class_weight="balanced" if balance_classes else None,
            n_jobs=-1,
            random_state=config.random_seed,
        )
        model_params = params_libL2.model_dump()
        return model, model_params, balance_classes

    if model_type == "randomforest":
        # params  Num. trees mtry  Sample fraction
        # Number of variables randomly sampled as candidates at each split:
        # it is “mtry” in R and it is “max_features” Python
        #  The sample.fraction parameter specifies the fraction of observations to be used in each tree
        params_rf = RandomforestParams(**model_params)
        model = RandomForestClassifier(
            n_estimators=int(params_rf.n_estimators),
            max_features=(
                int(params_rf.max_features) if params_rf.max_features is not None else None
            ),
            class_weight="balanced"
            if balance_classes
            else None,  # AM: Need to choose between balanced and balanced_subsample
            n_jobs=-1,
            random_state=config.random_seed,
        )
        model_params = params_rf.model_dump()
        return model, model_params, balance_classes

    if model_type == "multi_naivebayes":
        # small workaround for parameters
        params_nb = Multi_naivebayesParams(**model_params)
        if params_nb.class_prior is not None:
            class_prior = params_nb.class_prior
        else:
            class_prior = None
        # Only with dtf or tfidf for features
        # TODO: calculate class prior for docfreq & termfreq
        model = MultinomialNB(
            alpha=params_nb.alpha,
            fit_prior=params_nb.fit_prior,
            class_prior=class_prior,
        )
        balance_classes = False  # Force the parameter to be set as False
        model_params = params_nb.model_dump()
        return model, model_params, balance_classes


def _randomforest_grid(n_features: int) -> dict[str, list]:
    """ """
    candidates = [max(1, int(n_features**0.5)), max(1, int(n_features / 3)), n_features]
    values = sorted({v for v in candidates if 1 <= v <= n_features})
    return {"max_features": values}


def optimize_hyperparameters(
    model_type: str,
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    balance_classes: bool,
    cv: int = 10,
    n_jobs: int = -1,
    model_params: dict = {},
) -> dict:
    """ """
    if model_type == "randomforest":
        param_grid = _randomforest_grid(X_train.shape[1])
    elif model_type in PARAM_GRIDS:
        param_grid = PARAM_GRIDS[model_type]

    base_estimator, _, _ = build_model(model_type, model_params, balance_classes)

    search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
    )
    search.fit(X_train, Y_train)

    rename = SKLEARN_TO_PYDANTIC[model_type]
    best = {rename.get(k, k): v for k, v in search.best_params_.items()}
    return best


def check_data(
    X: pd.DataFrame, Y: pd.Series, exclude_labels: list[str]
) -> tuple[pd.DataFrame, pd.Series]:

    rows_to_exclude = np.logical_or(np.isin(Y, exclude_labels), Y.isna())
    rows_to_keep = np.invert(rows_to_exclude)
    return X.loc[rows_to_keep, :], Y[rows_to_keep]


def cv_score(model, X, Y, num_folds, random_seed) -> float:
    """ """
    num_folds = 10
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

    f = Y.notnull()

    Y_pred_10cv = pd.Series(cross_val_predict(model, X[f], Y[f], cv=kf), index=Y[f].index)

    statistics_cv10 = get_metrics_multiclass(
        Y[f],
        Y_pred_10cv,
    )

    statistics_cv10.false_predictions = None

    return statistics_cv10


def split_test(
    X, Y, random_seed, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Remove null elements and return train/test splits
    (equivalent of train_test_split from sklearn)
    """
    index = X.copy().index.to_series()
    index = index.sample(frac=1.0, random_state=random_seed)
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


class TrainMLMultiClass(BaseTask):
    """
    Fit a sklearn model
    """

    kind = "train_ml"

    def __init__(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        Y: pd.Series,
        path: Path,
        name: str,
        user: str,
        model_params: dict,
        scheme: str,
        features: list,
        labels: list[str],
        model_type: str,
        standardize: bool = False,
        cv10: bool = False,
        balance_classes: bool = False,
        exclude_labels: list[str] = [],
        test_size: float = 0.2,
        retrain: bool = False,
        texts: pd.Series | None = None,
        random_seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.random_seed = random_seed
        self.model = model
        self.name = name
        self.X = X
        self.Y = Y
        self.user = user
        self.cv10 = cv10
        self.balance_classes = balance_classes
        self.exclude_labels = exclude_labels  # labels are excluded earlier on in the pipeline, but we must save this information somewhere
        self.test_size = test_size
        self.path = path
        self.model_path = path.joinpath(name)
        self.retrain = retrain
        self.model_params = model_params
        self.scheme = scheme
        self.features = features
        self.labels = labels
        self.model_type = model_type
        self.standardize = standardize
        self.texts = texts

    def __init_paths(self, retrain: bool) -> None:
        """
        Create a directory for the files to be saved
        """
        # if retrain, clear the folder
        if retrain:
            shutil.rmtree(self.model_path)
            os.mkdir(self.model_path)
        else:
            if self.model_path.exists():
                raise Exception("The model already exists")
            os.mkdir(self.model_path)

    def __check_data(
        self, X: pd.DataFrame, Y: pd.Series, exclude_labels: list[str]
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Remove labels to exclude and nan values"""
        return check_data(X, Y, exclude_labels)

    def __split_set(
        self, X, Y, test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """ """
        return split_test(X, Y, self.random_seed, test_size)

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

        statistics_cv10 = cv_score(self.model, self.X, self.Y, num_folds, self.random_seed)

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
        """Add an entry in the data base and save the following files:
        - proba.csv with the probabilities
        - data using during training (training_data.parquet)
        - a pickle version of the database entry (#NOTE: AM: Artefact ?)
        - metrics for the training (train, trainvalid and cv10)
        """
        # Write the proba
        proba.to_csv(self.model_path / "proba.csv")

        # Write the training data
        X_train["label"] = Y_train
        X_train.to_parquet(self.model_path / "training_data.parquet")

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

        path_to_model_tmp = self.model_path / f"model{str(self.unique_id)}.pkl"
        path_to_model = self.model_path / "model.pkl"
        with open(path_to_model_tmp, "wb") as file:
            pickle.dump(element, file)
        os.replace(path_to_model_tmp, path_to_model)

        # Write the statistics
        path_to_metrics_json_tmp = str(
            self.path.joinpath(self.name).joinpath(f"metrics_training_{str(self.unique_id)}.json")
        )
        path_to_metrics_json = str(self.path.joinpath(self.name).joinpath("metrics_training.json"))

        with open(path_to_metrics_json_tmp, "w") as file:
            json.dump(
                {
                    "train": metrics_train.model_dump(mode="json"),
                    "trainvalid": metrics_test.model_dump(mode="json"),
                    "cv10": statistics_cv10.model_dump(mode="json") if statistics_cv10 else None,
                },
                file,
            )
        os.replace(path_to_metrics_json_tmp, path_to_metrics_json)

    def _check_cancelled(self) -> None:
        """Raise if the user requested cancellation."""
        if self.event is not None and self.event.is_set():
            raise Exception("Process interrupted by user")

    def __call__(self) -> EventsModel:
        """
        Fit quickmodel and calculate statistics
        """
        task_timer = TaskTimer(
            compulsory_steps=["setup", "train", "evaluate", "save_files"], optional_steps=["cv10"]
        )

        task_timer.start("setup")
        self.__init_paths(self.retrain)

        X_for_training, Y_for_training = self.__check_data(self.X, self.Y, self.exclude_labels)

        X_train, X_test, Y_train, Y_test = self.__split_set(
            X_for_training, Y_for_training, self.test_size
        )
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
            Y_pred_train = pd.Series(self.model.predict(X_train), index=X_train.index)  # ty: ignore[unresolved-attribute]
            Y_pred_test = pd.Series(self.model.predict(X_test), index=X_test.index)  # ty: ignore[unresolved-attribute]
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
            proba = pd.DataFrame(proba_values, columns=self.model.classes_, index=self.X.index)  # ty: ignore[unresolved-attribute]
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
        task_timer.stop("save_files")

        return EventsModel(events=task_timer.get_events())
