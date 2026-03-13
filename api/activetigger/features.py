import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import]
import pyarrow.parquet as pq  # type: ignore[import]
from pandas import DataFrame, Series

from activetigger.config import config
from activetigger.data import Data
from activetigger.datamodels import (
    FeatureComputing,
    FeatureDescriptionModelOut,
    FeaturesProjectStateModel,
)
from activetigger.db.projects import ProjectsService
from activetigger.queue_manager import Queue
from activetigger.tasks.compute_dfm import ComputeDfm
from activetigger.tasks.compute_fasttext import ComputeFasttext
from activetigger.tasks.compute_sbert import ComputeSbert


class Features:
    """
    Manage project features
    Comment :
    - for the moment as a file
    - database for informations
    - use "__" as separator
    - add a column dataset to separate train, valid, test
    """

    project_slug: str
    data: Data
    path_model: Path
    queue: Queue
    computing: list
    informations: dict
    content: DataFrame
    map: dict
    options: dict
    lang: str
    projects_service: ProjectsService
    n: int

    def __init__(
        self,
        project_slug: str,
        data: Data,
        models_path: Path,
        queue: Any,
        computing: list,
        db_manager,
        lang: str,
    ) -> None:
        """
        Initit features
        """
        self.project_slug = project_slug
        self.projects_service = db_manager.projects_service
        self.data = data
        self.path_features = data.path_features
        self.path_all = data.path_data_all
        self.path_models = models_path
        self.queue = queue
        self.informations = {}

        self.lang = lang
        self.computing = computing

        # load possible embeddings models
        fasttext_models = [f for f in os.listdir(self.path_models) if f.endswith(".bin")]
        # possibility to create a embeddings.yaml file to add models

        # options
        self.options: dict = {
            "sentence-embeddings": {
                "models": config.models_embeddings,
                "default": "jinaai/jina-embeddings-v5-text-small",
            },
            "fasttext": {"models": fasttext_models},
            "dfm": {
                "tfidf": False,
                "ngrams": 1,
                "min_term_freq": 5,
                "max_term_freq": 100,
                "norm": None,
                "log": None,
            },
            "regex": {"formula": None},
            "dataset": {},
        }

        # create the features file if not exist
        if not self.path_features.exists():
            self.create_features_file()
        self.map, self.n = self.get_map()

    def create_features_file(self):
        """
        Create the features file with the dataset information
        based on train, valid, test files
        """

        # clear database
        self.projects_service.delete_project_features(self.project_slug)

        # read file to create the structure
        train = pd.DataFrame(index=self.data.train.index)
        train["dataset"] = "train"

        to_concat = [train]

        if self.data.valid is not None:
            valid = pd.DataFrame(index=self.data.valid.index)
            valid["dataset"] = "valid"
            to_concat.append(valid)
        if self.data.test is not None:
            test = pd.DataFrame(index=self.data.test.index)
            test["dataset"] = "test"
            to_concat.append(test)

        df = pd.DataFrame(pd.concat(to_concat))
        df.to_parquet(self.path_features, index=True)
        del df

    def reset_features_file(self):
        """
        Reset the features file with only the dataset information
        based on train, valid, test files
        """
        if self.path_features.exists():
            self.path_features.unlink()
        self.projects_service.delete_project_features(self.project_slug)
        self.create_features_file()
        self.map, self.n = self.get_map()

    def get_map(self) -> tuple[dict, int]:
        """
        Get the structure of features from the parquet file
        """
        parquet_file = pq.ParquetFile(self.path_features)
        column_names = [i for i in parquet_file.schema.names if i != "dataset"]

        def find_strings_with_pattern(strings, pattern):
            matching_strings = [s for s in strings if re.match(pattern, s)]
            return matching_strings

        var = set(
            [
                i.split("__")[0]
                for i in column_names
                if "__index" not in i and i not in ["id_internal", "id_external"]
            ]
        )
        dic = {i: find_strings_with_pattern(column_names, i) for i in var}
        num_rows = parquet_file.metadata.num_rows
        return dic, num_rows

    def exists(self, name: str) -> bool:
        """
        Check if a feature exists
        """
        return name in self.map

    def add_predictions(self, predictions: dict) -> list:
        """
        Add BERT predictions to features
        """
        errors = []
        for f in predictions:
            try:
                # load the prediction probabilities minus one
                df = pd.read_parquet(predictions[f])
                df = df.drop(columns=["entropy", "prediction", "dataset", "label"])
                df = df[df.columns[0:-1]]
                name = f.replace("__", "_")  # avoid __ in the name for features
                # if the feature already exists, delete it first
                if self.exists(name):
                    self.delete(name)
                # add it
                self.add(
                    name=name,
                    kind="prediction",
                    parameters={},
                    username="system",
                    new_content=df,
                )
            except Exception as ex:
                errors.append(f"Error in adding prediction : {str(ex)}")
        return errors

    def add(
        self,
        name: str,
        kind: str,
        username: str,
        parameters: dict[str, Any],
        new_content: DataFrame | Series,
    ):
        """
        Add feature(s) after computing
        """
        # test name
        if name in self.map:
            raise Exception("Feature already exists")

        # test length
        if len(new_content) != self.n:
            raise ValueError("Features don't have the right shape")

        # change type for series
        if type(new_content) is Series:
            new_content = pd.DataFrame(new_content)

        # change column name with a prefix
        new_content.columns = [f"{name}__{i}" for i in new_content.columns]

        # read data, add the feature to the dataset and save
        content = pd.read_parquet(self.path_features)
        content = pd.concat(
            [
                content[[i for i in content.columns if i not in new_content.columns]],
                new_content,
            ],
            axis=1,
        )
        content.to_parquet(self.path_features)
        del content

        # add informations to database
        self.projects_service.add_feature(
            project_slug=self.project_slug,
            kind=kind,
            name=name,
            parameters=parameters,
            user_name=username,
            data=list(new_content.columns),
        )

        # refresh the map
        self.map = self.get_map()[0]

    def delete(self, name: str):
        """
        Delete feature
        """
        if not self.exists(name):
            raise Exception("Feature doesn't exist")

        col = self.map[name]
        # read data, delete columns and save
        content = pd.read_parquet(self.path_features)

        content[[i for i in content.columns if i not in col]].to_parquet(self.path_features)
        del content

        # delete from database
        self.projects_service.delete_feature(self.project_slug, name)

        # refresh the map
        self.map = self.get_map()[0]

    def get(
        self, features: list | str, dataset: list | str = "all", keep_dataset_column: bool = False
    ) -> DataFrame:
        """
        Get content for specific features with a filter on dataset
        """
        features = [i for i in features if i is not None]
        if features == "all":
            features = list(self.map.keys())
        if type(features) is str:
            features = [features]

        # get needed columns
        cols = ["dataset"]
        missing = []
        for i in features:
            if i in self.map:
                cols += self.map[i]
            else:
                missing.append(i)
        if len(missing) > 0:
            print("Missing features:", missing)

        # load only needed data from file
        data = pd.read_parquet(self.path_features, columns=cols)

        # filter on dataset
        if dataset == "annotable":
            if not keep_dataset_column:
                return data.drop(columns=["dataset"])
            return data

        if type(dataset) is str:
            raise Exception("Dataset should be a list of strings")

        data = data.loc[data["dataset"].isin(dataset)]
        if not keep_dataset_column:
            return data.drop(columns=["dataset"])
        return data

    def info(self, name: str):
        feature = self.projects_service.get_feature(self.project_slug, name)
        if feature is None:
            raise Exception("Feature doesn't exist in database")
        return {
            "time": feature.time,
            "name": name,
            "kind": feature.kind,
            "username": feature.user,
            "parameters": feature.parameters,
            "columns": json.loads(feature.data),
        }

    def get_available(self) -> dict[str, FeatureDescriptionModelOut]:
        """
        Informations on features + update
        """
        return self.projects_service.get_project_features(self.project_slug)

    def get_column_raw(
        self, column_name: str, index: str = "annotable", add_real_index: bool = True
    ) -> Series:
        """
        Get column raw dataset
        """
        parquet_file = pq.ParquetFile(self.path_all)
        column_names = parquet_file.schema.names
        if column_name not in list(column_names):
            raise Exception("Column doesn't exist")
        df = pd.read_parquet(self.path_all, columns=[column_name])
        if index == "annotable":  # filter only train id
            df_annotable = pd.read_parquet(self.path_features, columns=[])  # only the index
            return df.loc[df_annotable.index][column_name]
        elif index == "all":
            return df[column_name]
        else:
            raise Exception("Index not recognized")

    def current_user_processes(self, user: str):
        return [e for e in self.computing if e.user == user]

    def current_computing(self) -> dict[str, dict[str, str | None]]:
        return {
            e.name: {"progress": self.computing_progress(e.unique_id), "name": e.name}
            for e in self.computing
            if e.kind == "feature"
        }

    def compute(self, df: pd.Series, name: str, kind: str, parameters: dict, username: str):
        """
        Compute new feature
        TODO : the parameters management is really bad
        """
        if len(self.current_user_processes(username)) > 0:
            raise ValueError("A process is already running")

        if kind not in {"sentence-embeddings", "fasttext", "dfm", "regex", "dataset"}:
            raise ValueError("Kind not recognized")

        name = f"{kind}_{name}"
        if self.exists(name):
            raise ValueError("This regex already exists")

        if kind == "regex":
            if "value" not in parameters:
                raise ValueError("No value for regex")

            pattern = re.compile(parameters["value"])
            if parameters.get("regex_count", False):
                f = df.apply(lambda x: len(pattern.findall(x)))
            else:
                f = df.apply(lambda x: bool(pattern.search(x)))
            parameters = {
                "name": name,
                "kind": kind,
                "regex": parameters["value"],
                "count": int(f.sum()),
                "username": username,
            }
            self.add(name, kind, username, parameters, f)
            return None

        if kind == "dataset":
            # force the name
            name = f"{parameters['dataset_col']}_{parameters['dataset_type'].lower()}_{name.replace('Feature-', '')}"
            # get the raw column for the train set
            column = self.get_column_raw(parameters["dataset_col"])

            # convert the column to a specific format
            if len(column.dropna()) != len(column):
                raise ValueError("Column contains null values")
            if parameters["dataset_type"] == "Numeric":
                try:
                    column = column.apply(float)
                except Exception:
                    raise Exception("The column can't be transform into numerical feature")
            else:
                column = column.apply(str)
                column_encoded = pd.get_dummies(
                    column, prefix=parameters["dataset_col"], drop_first=True
                )  # type: ignore[misc]

            # add the feature to the project
            parameters = {
                "name": name,
                "kind": kind,
                "dataset_col": parameters["dataset_col"],
                "dataset_type": parameters["dataset_type"],
                "username": username,
            }
            self.add(name, kind, username, parameters, column_encoded)
            return None

        # features with queue
        unique_id = None

        if kind == "sentence-embeddings":
            if (
                "model" not in parameters
                or parameters["model"] is None
                or parameters["model"] == "generic"
            ):
                model = self.options["sentence-embeddings"]["default"]
            else:
                model = parameters["model"]
            if "max_length_tokens" not in parameters:
                parameters["max_length_tokens"] = 1024
            unique_id = self.queue.add_task(
                "feature",
                self.project_slug,
                ComputeSbert(
                    texts=df,
                    path_process=self.path_all.parent,
                    model=model,
                    max_tokens=parameters["max_length_tokens"],
                ),
                queue="gpu",
            )
            # append short model name (without provider prefix)
            short_model = model.split("/")[-1] if "/" in model else model
            name = f"{name}_{short_model}"
            parameters = {
                "model": model,
                "name": name,
                "kind": kind,
                "username": username,
                "max_length_tokens": parameters["max_length_tokens"],
            }

        if kind == "fasttext":
            unique_id = self.queue.add_task(
                "feature",
                self.project_slug,
                ComputeFasttext(
                    texts=df,
                    language=self.lang,
                    path_process=self.path_all.parent,
                    path_models=self.path_models,
                    model=parameters["model"],
                ),
            )
            if parameters["model"] is not None and parameters["model"] != "":
                short_model = (
                    parameters["model"].split("/")[-1]
                    if "/" in parameters["model"]
                    else parameters["model"]
                )
                name = f"{name}_{short_model}"
            parameters = {
                "model": parameters["model"],
                "name": name,
                "kind": kind,
                "username": username,
            }

        if kind == "dfm":
            args = parameters.copy()
            args["texts"] = df
            args["language"] = self.lang
            unique_id = self.queue.add_task("feature", self.project_slug, ComputeDfm(**args))
            del args
            parameters = {
                "name": name,
                "kind": kind,
                "username": username,
                "tfidf": parameters.get("dfm_tfidf", False),
                "dfm_ngrams": parameters.get("dfm_ngrams", 1),
                "dfm_max_term_freq": parameters.get("dfm_max_term_freq", 100),
                "dfm_min_term_freq": parameters.get("dfm_min_term_freq", 5),
                "dfm_norm": parameters.get("dfm_norm", None),
                "dfm_log": parameters.get("dfm_log", None),
            }

        if unique_id:
            self.computing.append(
                FeatureComputing(
                    unique_id=unique_id,
                    kind="feature",
                    parameters=parameters,
                    type=kind,
                    user=username,
                    name=name,
                    time=datetime.now(),
                )
            )
            return None
        raise ValueError("Error in the process")

    def computing_progress(self, unique_id: str) -> str | None:
        """
        Get the progress of a computing feature
        """
        try:
            with open(self.path_all.parent.joinpath(unique_id), "r") as f:
                r = f.read()
            return r
        except Exception:
            return None

    def state(self) -> FeaturesProjectStateModel:
        return FeaturesProjectStateModel(
            options=self.options,
            available=list(self.map.keys()),
            training=self.current_computing(),
        )
