import os
import pickle
import shutil
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from activetigger.config import config
from activetigger.datamodels import (
    KnnParams,
    LogisticL1Params,
    LogisticL2Params,
    ModelDescriptionModel,
    ModelInformationsModel,
    ModelScoresModel,
    Multi_naivebayesParams,
    PredictedLabel,
    QuickModelComputed,
    QuickModelComputing,
    QuickModelsProjectStateModel,
    RandomforestParams,
)
from activetigger.db.languagemodels import ModelsService
from activetigger.db.manager import DatabaseManager
from activetigger.functions import get_model_metrics
from activetigger.queue_manager import Queue
from activetigger.tasks.predict_ml import PredictMLMultiClass
from activetigger.tasks.train_ml import TrainMLMultiClass


class QuickModels:
    """
    Module to manage quickmodels
    - define available models
    - save a quickmodel/user
    - train quickmodels
    """

    path: Path
    queue: Queue
    available_models: dict[str, Any]
    computing: list
    loaded: dict
    language_models_service: ModelsService

    def __init__(
        self,
        project_slug: str,
        path: Path,
        queue: Queue,
        computing: list,
        db_manager: DatabaseManager,
    ) -> None:
        """
        Init Quickmodels class
        """
        self.path: Path = path.joinpath("quickmodels")
        if not self.path.exists():
            os.mkdir(self.path)
        self.project_slug = project_slug
        self.language_models_service = db_manager.language_models_service
        self.computing = computing
        self.queue = queue

        # Models and default parameters
        self.available_models = {
            "logistic-l1": LogisticL1Params(costLogL1=1),
            "logistic-l2": LogisticL2Params(costLogL2=1),
            "knn": KnnParams(n_neighbors=3),
            "randomforest": RandomforestParams(n_estimators=500, max_features=None),
            "multi_naivebayes": Multi_naivebayesParams(alpha=1, fit_prior=True, class_prior=None),
        }

        self.loaded = {}

    def compute_quickmodel(
        self,
        project_slug: str,
        user: str,
        scheme: str,
        features: list,
        name: str,
        model_type: str,
        df: DataFrame,
        col_labels: str,
        col_features: list,
        standardize: bool = True,
        model_params: dict | None = None,
        cv10: bool = False,
        balance_classes: bool = False,
        exclude_labels: list[str] = [],
        test_size: float = 0.2,
        retrain: bool = False,
        texts: pd.Series | None = None,
    ) -> str:
        """
        Add a new quickmodel for a user and a scheme
        """
        X, Y, labels = self.transform_data(df, col_labels, col_features, standardize)

        # default parameters
        if model_params is None:
            model_params = self.available_models[model_type].dict()

        # Select model
        if model_type == "knn":
            params_knn = KnnParams(**model_params)
            model = KNeighborsClassifier(n_neighbors=int(params_knn.n_neighbors), n_jobs=-1)
            balance_classes = False  # Force the parameter to be set as False
            model_params = params_knn.model_dump()

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

        # launch the compuation (model + statistics) as a future process
        args = {
            "model": model,  # ty: ignore[possibly-unresolved-reference]
            "X": X,
            "Y": Y,
            "labels": labels,
            "cv10": cv10,
            "balance_classes": balance_classes,
            "path": self.path,
            "name": name,
            "retrain": retrain,
            "scheme": scheme,
            "model_type": model_type,
            "user": user,
            "standardize": standardize,
            "features": features,
            "model_params": model_params,
            "texts": texts,
            "random_seed": config.random_seed,
            "exclude_labels": exclude_labels,
            "test_size": test_size,
        }
        unique_id = self.queue.add_task("train_quickmodel", project_slug, TrainMLMultiClass(**args))  # ty: ignore[invalid-argument-type]
        del args

        req = QuickModelComputing(
            status="training",
            user=user,
            unique_id=unique_id,
            time=datetime.now(timezone.utc),
            kind="train_quickmodel",
            scheme=scheme,
            model_type=model_type,
            name=name,
            features=features,
            labels=labels,
            model_params=model_params,
            standardize=standardize,
            cv10=cv10,
            balance_classes=balance_classes,
            exclude_labels=exclude_labels,
            test_size=test_size,
            retrain=retrain,
            dataset="train",
        )
        self.computing.append(req)
        return unique_id

    def add(self, element: QuickModelComputing) -> None:
        """
        Add computed model in the database
        """
        model_path = self.path.joinpath(element.name)
        self.language_models_service.add_model(
            kind="quickmodel",
            name=element.name,
            user=element.user,
            project=self.project_slug,
            scheme=element.scheme,
            params={**element.model_params, "exclude_labels": element.exclude_labels},
            path=str(model_path),
            status="trained",
            retrain=element.retrain,
        )

    def available(self) -> dict[str, list[ModelDescriptionModel]]:
        """
        Return available models per scheme
        """
        existing = self.language_models_service.available_models(self.project_slug, "quickmodel")
        r: dict[str, list[ModelDescriptionModel]] = {}
        for m in existing:
            if m.scheme not in r:
                r[m.scheme] = []
            r[m.scheme].append(m)
        return r

    def get(self, name: str) -> QuickModelComputed:
        """
        Load the content of a specific model
        (cache in memory)
        """
        if not self.exists(name):
            raise Exception("The model does not exist in database")
        if name in self.loaded:
            return self.loaded[name]
        else:
            path = self.path.joinpath(name)
            if not path.exists():
                raise Exception("The model path does not exist")
            if not (path / "model.pkl").exists():
                # wait until the model is available if still writing
                print("Waiting for model file to be available...")
                time.sleep(1)
            with open(path / "model.pkl", "rb") as file:
                sm: QuickModelComputed = pickle.load(file)
            return sm

    def get_prediction(self, name: str) -> DataFrame:
        """
        Get a specific quickmodel
        """
        sm = self.get(name)
        if sm.proba is None:
            raise ValueError("No probability available for this model")
        return sm.proba

    def get_prediction_element(self, model_name: str, element_id: Any) -> PredictedLabel:
        """
        Get prediction for a specific element
        """
        sm = self.get(model_name)
        if sm.proba is None:
            raise ValueError("No probability available for this model")
        if element_id not in sm.proba.index:
            raise ValueError("Element ID not found in the predictions")
        predicted_label = sm.proba.loc[element_id, "prediction"]
        predicted_proba = round(sm.proba.loc[element_id, predicted_label], 2)
        predicted_entropy = round(sm.proba.loc[element_id, "entropy"], 2)
        return PredictedLabel(
            label=predicted_label,
            proba=None if np.isnan(predicted_proba) else predicted_proba,
            entropy=None if np.isnan(predicted_entropy) else predicted_entropy,
        )

    def training(self) -> dict[str, list[str]]:
        """
        Currently under training
        """
        return {e.user: [e.scheme] for e in self.computing if e.kind == "train_quickmodel"}

    def exists(self, name: str) -> bool:
        """
        Test if a quickmodel exists for a user/scheme
        """
        existing = self.language_models_service.available_models(self.project_slug, "quickmodel")
        return name in [m.name for m in existing]

    def transform_data(
        self, data, col_label, col_predictors, standardize
    ) -> tuple[DataFrame, DataFrame, list]:
        """
        Load data
        """
        f_na = data[col_predictors].isna().sum(axis=1) > 0
        if f_na.sum() > 0:
            print(f"There is {f_na.sum()} predictor rows with missing values")

        # normalize X data
        if standardize:
            scaler = StandardScaler()
            df = data[col_predictors]
            df_stand = scaler.fit_transform(df)
            df_pred = pd.DataFrame(df_stand, columns=df.columns, index=df.index)
        else:
            df_pred = data[col_predictors]

        # create global dataframe with no missing predictor
        df = pd.concat([data[~f_na][col_label], df_pred], axis=1)

        # data for training
        Y = df[col_label]
        X = df[col_predictors]
        labels = [str(i) for i in Y.unique() if pd.notna(i)]

        return X, Y, labels

    def export_prediction(self, name: str, format: str = "csv") -> tuple[BytesIO, dict[str, str]]:
        """
        Function to export the prediction of a quickmodel
        """
        # get data
        table = self.get_prediction(name)
        # convert to payload
        if format == "csv":
            output = BytesIO()
            pd.DataFrame(table).to_csv(output)
            output.seek(0)
            headers = {
                "Content-Disposition": 'attachment; filename="data.csv"',
                "Content-Type": "text/csv",
            }
            return output, headers
        elif format == "xlsx":
            output = BytesIO()
            pd.DataFrame(table).to_excel(output)
            output.seek(0)
            headers = {
                "Content-Disposition": 'attachment; filename="data.xlsx"',
                "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            }
            return output, headers
        elif format == "parquet":
            output = BytesIO()
            pd.DataFrame(table).to_parquet(output)
            output.seek(0)
            headers = {
                "Content-Disposition": 'attachment; filename="data.parquet"',
                "Content-Type": "application/octet-stream",
            }
            return output, headers
        else:
            raise ValueError("Format not supported")

    def state(self) -> QuickModelsProjectStateModel:
        return QuickModelsProjectStateModel(
            options=self.available_models,
            available=self.available(),
            training=self.training(),
        )

    def delete(self, name: str) -> None:
        """
        Delete a specific quickmodel
        """
        if not self.exists(name):
            raise Exception("The model does not exist")

        # delete from the database
        self.language_models_service.delete_model(self.project_slug, name)

        # delete from the filesystem
        model_path = self.path.joinpath(name)
        if model_path.exists():
            shutil.rmtree(model_path)

        # delete from the loaded cache
        if name in self.loaded:
            del self.loaded[name]

    def start_predicting_process(
        self,
        name: str,
        username: str,
        df: DataFrame,
        dataset: str,
        col_dataset: str,
        cols_features: list,
        col_label: str | None = None,
        col_text: str | None = None,
        statistics: list[str] | None = None,
    ) -> None:
        """
        Start the predicting process for a specific model
        """
        if not self.exists(name):
            raise Exception("The model does not exist")
        sm = self.get(name)
        file_name = f"predict_{dataset}.parquet"
        unique_id = self.queue.add_task(
            "prediction",
            self.project_slug,
            PredictMLMultiClass(
                model=sm.model,
                df=df,
                col_dataset=col_dataset,
                col_features=cols_features,
                col_label=col_label,
                path=self.path.joinpath(name),
                file_name=file_name,
                statistics=statistics,
                col_text=col_text,
                exclude_labels=sm.exclude_labels,
            ),
            queue="cpu",
        )
        self.computing.append(
            QuickModelComputing(
                user=username,
                unique_id=unique_id,
                time=datetime.now(timezone.utc),
                kind="predict_quickmodel",
                status="predicting",
                name=name,
                dataset=dataset,
                features=sm.features,
                scheme=sm.scheme,
                model_type=sm.model_type,
                model_params=sm.model_params,
                labels=sm.labels,
            )
        )
        print("Predicting process started")

    def get_informations(self, model_name) -> ModelInformationsModel:
        """
        Informations on the bert model from the files
        """

        if not self.exists(model_name):
            raise Exception(f"The model {model_name} does not exist")
        sm = self.get(model_name)
        # params = self.get_parameters(model_name)
        metrics = get_model_metrics(self.path.joinpath(model_name))
        if metrics is None:
            metrics = {}

        return ModelInformationsModel(
            params={"exclude_labels": sm.exclude_labels},
            scores=ModelScoresModel(
                internalvalid_scores=metrics.get("trainvalid", None),
                valid_scores=metrics.get("valid", None),
                test_scores=metrics.get("test", None),
                outofsample_scores=metrics.get("outofsample", None),
                train_scores=metrics.get("train", None),
            ),
        )

    def rename(self, former_name: str, new_name: str) -> None:
        """
        Rename a model (copy it)
        """
        # get model
        model = self.language_models_service.get_model(self.project_slug, former_name)
        if model is None:
            raise Exception("Model does not exist")
        if (Path(model.path) / "status.log").exists():
            raise Exception("Model is currently computing")
        self.language_models_service.rename_model(self.project_slug, former_name, new_name)
        model_path = Path(model.path)
        new_path = model_path.parent / model_path.name.replace(former_name, new_name)
        os.rename(model_path, new_path)

    def drop_models(self, which: str = "all") -> None:
        """
        Drop all quickmodels of a specific kind
        """
        existing = self.language_models_service.available_models(self.project_slug, "quickmodel")
        if which == "all":
            for m in existing:
                self.delete(m.name)
        else:
            raise ValueError("Kind not recognized")
