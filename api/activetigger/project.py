import io
import json
import math
import os
import shutil
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import]
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from pandas import DataFrame

from activetigger.bertopic_manager import Bertopic
from activetigger.config import config
from activetigger.data import Data
from activetigger.datamodels import (
    ActiveModel,
    AuthUserModel,
    BertModelModel,
    BertopicComputing,
    ElementInModel,
    ElementOutModel,
    EvalSetDataModel,
    EventsModel,
    ExportGenerationsParams,
    FeatureComputing,
    GenerationComputing,
    GenerationModel,
    GenerationRequest,
    GenerationResult,
    LMComputing,
    NextInModel,
    NextProjectStateModel,
    PredictedLabel,
    ProcessComputing,
    ProjectBaseModel,
    ProjectCreatingModel,
    ProjectDescriptionModel,
    ProjectionComputing,
    ProjectionOutModel,
    ProjectionOutModelNode,
    ProjectModel,
    ProjectStateModel,
    ProjectUpdateModel,
    QuickModelComputing,
    QuickModelInModel,
    StaticFileModel,
    UpdateComputing,
)
from activetigger.db.manager import DatabaseManager
from activetigger.features import Features
from activetigger.functions import clean_regex, get_dir_size, slugify
from activetigger.generation.generations import Generations
from activetigger.languagemodels import LanguageModels
from activetigger.messages import Messages
from activetigger.monitoring import Monitoring
from activetigger.projections import Projections
from activetigger.queue_manager import Queue
from activetigger.quickmodels import QuickModels
from activetigger.schemes import Schemes
from activetigger.tasks.create_project import CreateProject
from activetigger.tasks.generate_call import GenerateCall
from activetigger.tasks.update_datasets import UpdateDatasets
from activetigger.users import Users


class Errors:
    """
    Runtime error object
    """

    def __init__(self, timeout: int = 15) -> None:
        """
        Initialize the error stack
        """
        self.timeout = timeout
        self.__stack: list[list] = []

    def add(self, message: str) -> None:
        """
        Add an error to the stack
        """
        self.__stack.append([message, datetime.now(config.timezone)])

    def clean(self) -> None:
        """
        Clean old errors
        """
        now = datetime.now(config.timezone)
        self.__stack = [e for e in self.__stack if e[1] >= now - timedelta(minutes=self.timeout)]

    def state(self) -> list[list]:
        """
        Get the current stack
        """
        self.clean()
        return self.__stack


class Project:
    """
    Project object
    """

    status: str
    starting_time: float
    name: str
    queue: Queue
    computing: list
    path_models: Path
    data: Data
    db_manager: DatabaseManager
    params: ProjectModel
    schemes: Schemes
    features: Features
    languagemodels: LanguageModels
    quickmodels: QuickModels
    generations: Generations
    projections: Projections
    messages: Messages
    errors: Errors
    monitoring: Monitoring

    def __init__(
        self,
        project_slug: str,
        queue: Queue,
        db_manager: DatabaseManager,
        path_models: Path,
        users: Users,
        messages: Messages,
    ) -> None:
        """
        Initialize the project
        - load if it exits in database
        """
        self.status = "initialize"
        self.starting_time = time.time()
        self.queue = queue
        self.computing = []
        self.db_manager = db_manager
        self.path_models = path_models
        self.name = project_slug
        self.project_slug = project_slug
        self.errors = Errors()
        self.users = users
        self.messages = messages
        self.monitoring = Monitoring(db_manager, project_slug)

        # cached project directory size (refreshed every _memory_cache_interval seconds)
        self._memory_cache: float = 0.0
        self._memory_cache_time: float = 0.0
        self._memory_cache_interval: float = 60

        # cached project state (avoids recomputing for every polling user)
        self._state_cache: ProjectStateModel | None = None
        self._state_cache_time: float = 0.0
        self._state_cache_interval: float = 2  # seconds

        # load the project if exist
        if self.exists():
            self.status = "created"
            self.load_project(project_slug)

    def exists(self) -> bool:
        """
        Check if the project exists
        """
        if self.db_manager.projects_service.get_project(self.project_slug):
            return True
        return False

    def load_project(self, project_slug: str) -> None:
        """
        Load existing project
        """
        # get projet parameters
        existing_project = self.db_manager.projects_service.get_project(project_slug)

        if not existing_project:
            raise ValueError("This project does not exist")

        self.params = ProjectModel(**existing_project["parameters"])

        # check if directory exists
        if self.params.dir is None:
            raise ValueError("No directory exists for this project")

        # Tabular data management
        self.data = Data(
            self.params.dir,
            self.params.dir.joinpath(config.data_all),
            self.params.dir.joinpath(config.features_file),
            self.params.dir.joinpath(config.train_file),
            self.params.dir.joinpath(config.valid_file),
            self.params.dir.joinpath(config.test_file),
        )

        # create specific management objets
        self.schemes = Schemes(
            project_slug,
            self.db_manager,
            self.data,
        )
        self.features = Features(
            project_slug,
            self.data,
            self.path_models,
            self.queue,
            cast(list[FeatureComputing], self.computing),
            self.db_manager,
            self.params.language,
        )
        self.languagemodels = LanguageModels(
            project_slug,
            self.params.dir,
            self.queue,
            self.computing,
            self.db_manager,
            config.file_bert_models,
        )
        self.quickmodels = QuickModels(
            project_slug, self.params.dir, self.queue, self.computing, self.db_manager
        )
        self.generations = Generations(
            self.db_manager, cast(list[GenerationComputing], self.computing)
        )
        self.projections = Projections(self.params.dir, self.computing, self.queue)
        self.bertopic = Bertopic(
            project_slug,
            self.params.dir,
            self.queue,
            self.computing,
            self.features,
            self.db_manager,
        )

    def start_project_creation(self, params: ProjectBaseModel, username: str, path: Path) -> None:
        """
        Manage process creation, sending the heavy process to the queue
        """
        self.status = "creating"

        # test if the name of the column is specified
        if params.col_id is None or params.col_id == "":
            raise Exception("No column selected for the id")
        if params.cols_text is None or len(params.cols_text) == 0:
            raise Exception("No column selected for the text")

        # add the dedicated directory
        params.dir = path.joinpath(self.project_slug)

        # send the process to the queue
        unique_id = self.queue.add_task(
            "create_project",
            self.project_slug,
            CreateProject(
                self.project_slug,
                params,
                username,
                data_all=config.data_all,
                train_file=config.train_file,
                valid_file=config.valid_file,
                test_file=config.test_file,
                features_file=config.features_file,
                random_seed=config.random_seed,
            ),
            queue="cpu",
        )

        # Update the register
        self.computing.append(
            ProjectCreatingModel(
                username=username,
                project_slug=self.project_slug,
                unique_id=unique_id,
                time=datetime.now(),
                kind="create_project",
                status="training",
            )
        )

    def finish_project_creation(
        self,
        username: str,
        project: ProjectModel,
        import_trainset_labels: pd.DataFrame | None = None,
        import_validset_labels: pd.DataFrame | None = None,
        import_testset_labels: pd.DataFrame | None = None,
    ) -> None:
        """
        Get the result of the queue and finish the creation process
        """
        # add the project to the database
        self.db_manager.projects_service.add_project(
            project.project_slug, jsonable_encoder(project), username
        )

        # add the default scheme if needed
        if import_trainset_labels is None or len(import_trainset_labels.columns) == 0:
            self.db_manager.projects_service.add_scheme(
                self.project_slug, config.default_scheme, [], "multiclass", "system"
            )

        # if labels/schemes to import, add them to the database
        else:
            for col in import_trainset_labels.columns:
                scheme_name = col.replace("dataset_", "")
                delimiters = import_trainset_labels[col].str.contains("|", regex=False).sum()
                if delimiters < 5:
                    scheme_type = "multiclass"
                    scheme_labels = list(import_trainset_labels[col].dropna().unique())
                else:
                    scheme_type = "multilabel"
                    scheme_labels = list(
                        import_trainset_labels[col].dropna().str.split("|").explode().unique()
                    )
                self.db_manager.projects_service.add_scheme(
                    self.project_slug,
                    scheme_name,
                    scheme_labels,
                    scheme_type,
                    "system",
                )
                elements = [
                    {"element_id": element_id, "annotation": label, "comment": ""}
                    for element_id, label in import_trainset_labels[col].dropna().items()
                ]
                self.db_manager.projects_service.add_annotations(
                    dataset="train",
                    user_name=username,
                    project_slug=self.project_slug,
                    scheme=scheme_name,
                    elements=elements,
                )
                if import_validset_labels is not None and col in import_validset_labels.columns:
                    elements = [
                        {"element_id": element_id, "annotation": label, "comment": ""}
                        for element_id, label in import_validset_labels[col].dropna().items()
                    ]
                    self.db_manager.projects_service.add_annotations(
                        dataset="valid",
                        user_name=username,
                        project_slug=self.project_slug,
                        scheme=scheme_name,
                        elements=elements,
                    )
                if import_testset_labels is not None and col in import_testset_labels.columns:
                    elements = [
                        {"element_id": element_id, "annotation": label, "comment": ""}
                        for element_id, label in import_testset_labels[col].dropna().items()
                    ]
                    self.db_manager.projects_service.add_annotations(
                        dataset="test",
                        user_name=username,
                        project_slug=self.project_slug,
                        scheme=scheme_name,
                        elements=elements,
                    )

        # add user authorizations
        self.users.set_auth(
            AuthUserModel(username=username, project_slug=project.project_slug, status="manager")
        )
        self.users.set_auth(
            AuthUserModel(username="root", project_slug=project.project_slug, status="manager")
        )
        self.status = "created"

    def delete(self) -> None:
        """
        Delete completely a project
        """

        if self.params.dir is None:
            raise ValueError("No directory for this project")

        # remove from database
        try:
            print("Deleting project from the database")
            self.db_manager.projects_service.delete_project(self.params.project_slug)
        except Exception as e:
            print(f"Problem with the database: {e}")
            raise ValueError("Problem with the database " + str(e))

        # remove folder of the project
        try:
            shutil.rmtree(self.params.dir)
        except Exception as e:
            raise ValueError("No directory to delete " + str(e))

        # remove static files
        if Path(f"{config.data_path}/projects/static/{self.name}").exists():
            shutil.rmtree(f"{config.data_path}/projects/static/{self.name}")

    def drop_evalset(self, dataset: str) -> None:
        """
        Clean all the test data of the project
        - remove the file
        - remove all the annotations in the database
        - set the flag to False
        """
        if not self.params.dir:
            raise Exception("No directory for project")
        path = self.params.dir.joinpath(f"{dataset}.parquet")
        if not path.exists():
            raise Exception("No eval data available")
        os.remove(path)
        self.db_manager.projects_service.delete_annotations_evalset(
            self.params.project_slug, dataset
        )
        if dataset == "test":
            self.data.test = None
            self.params.test = False
        if dataset == "valid":
            self.data.valid = None
            self.params.valid = False
        self.db_manager.projects_service.update_project(
            self.params.project_slug, jsonable_encoder(self.params)
        )

        # reset the features file
        self.features.reset_features_file()
        self.quickmodels.drop_models(which="all")

    def add_evalset(
        self, dataset, evalset: EvalSetDataModel, username: str, project_slug: str
    ) -> None:
        """
        Add a eval dataset (test or valid)

        The eval dataset should :
        - not contains NA
        - have a unique id different from the complete dataset

        The id will be modified to indicate imported

        TODO : put this task in the queue

        """
        if len(evalset.cols_text) == 0:
            raise Exception("No text column selected for the evalset")

        if self.params.dir is None:
            raise Exception("Cannot add eval data without a valid dir")

        if evalset.col_label == "":
            evalset.col_label = None

        if dataset not in ["test", "valid"]:
            raise Exception("Dataset should be test or valid")

        if dataset == "test" and self.params.test:
            raise Exception("There is already a test dataset")

        if dataset == "valid" and self.params.valid:
            raise Exception("There is already a valid dataset")

        csv_buffer = io.StringIO(evalset.csv)
        df = pd.read_csv(
            csv_buffer,
            dtype={evalset.col_id: str, **{col: str for col in evalset.cols_text}},
            nrows=evalset.n_eval,
        )

        if len(df) > 10000:
            raise Exception("You valid set is too large")

        # create text column
        df["text"] = df[evalset.cols_text].apply(
            lambda x: "\n\n".join([str(i) for i in x if pd.notnull(i)]), axis=1
        )

        # change names
        if not evalset.col_label:
            df = df.rename(columns={evalset.col_id: "id"})
        else:
            df = df.rename(
                columns={
                    evalset.col_id: "id",
                    evalset.col_label: "label",
                }
            )
            df["label"] = df["label"].apply(lambda x: str(x) if pd.notna(x) else None)

        # deal with non-unique id
        # TODO : compare with the general dataset
        df["id_external"] = df["id"].apply(str)
        if not ((df["id"].astype(str).apply(slugify)).nunique() == len(df)):
            df["id"] = [str(i) for i in range(len(df))]
            print("ID not unique, changed to default id")

        # identify the dataset as imported and set the id
        df["id"] = df["id"].apply(lambda x: f"imported-{str(x)}")
        df = df.set_index("id")

        # import labels if specified + scheme // check if the labels are in the scheme
        if evalset.col_label and evalset.scheme:
            # Check the label columns if they match the scheme or raise error
            scheme = self.schemes.available()[evalset.scheme].labels
            for label in df["label"].dropna().unique():
                if label not in scheme:
                    raise Exception(f"Label {label} not in the scheme {evalset.scheme}")

            elements = [
                {"element_id": element_id, "annotation": label, "comment": ""}
                for element_id, label in df["label"].dropna().items()
            ]
            self.db_manager.projects_service.add_annotations(
                dataset=dataset,
                user_name=username,
                project_slug=project_slug,
                scheme=evalset.scheme,
                elements=elements,
            )
            print("Valid labels imported")

        # write the dataset
        if dataset == "test":
            df[["id_external", "text"]].to_parquet(self.params.dir.joinpath(config.test_file))
            self.params.test = True
            self.data.load_dataset("test")
        elif dataset == "valid":
            df[["id_external", "text"]].to_parquet(self.params.dir.joinpath(config.valid_file))
            self.params.valid = True
            self.data.load_dataset("valid")
        else:
            raise Exception("Dataset should be test or valid")

        # update the database
        self.db_manager.projects_service.update_project(
            self.params.project_slug, jsonable_encoder(self.params)
        )

        # reset the features file
        self.features.reset_features_file()
        self.quickmodels.drop_models(which="all")

        # reload the data
        self.data.load_dataset(dataset)

    def train_quickmodel(
        self,
        quickmodel: QuickModelInModel,
        username: str,
        n_min_annotated: int = 3,
        retrain: bool = False,
    ) -> str:
        """
        Build all the information before calling the quickmodel computation
        retrain : if True, will delete the previous model with the same name
        """
        # Tests
        availabe_schemes = self.schemes.available()
        quickmodel.features = [i for i in quickmodel.features if i is not None]
        if quickmodel.features is None or len(quickmodel.features) == 0:
            raise Exception("No features selected")
        if quickmodel.model not in list(self.quickmodels.available_models.keys()):
            raise Exception("Model not available")
        if quickmodel.scheme not in availabe_schemes:
            raise Exception("Scheme not available")
        if len(availabe_schemes[quickmodel.scheme].labels) < 2:
            raise Exception("Not enough labels in the scheme")
        exist = self.quickmodels.exists(quickmodel.name)
        if exist and not retrain:
            raise Exception("A quickmodel with this name already exists")
        if not exist and retrain:
            raise Exception("No quickmodel with this name to retrain")

        # only dfm feature for multi_naivebayes (FORCE IT if available else error)
        if quickmodel.model == "multi_naivebayes":
            dfm_features = [f for f in self.features.map if f.startswith("dfm")]
            if not dfm_features:
                raise Exception("No dfm feature available")
            quickmodel.features = dfm_features
            quickmodel.standardize = False

        if quickmodel.params is None:
            params = None
        else:
            params = dict(quickmodel.params)
        # add information on the target of the model
        if quickmodel.dichotomize is not None and params is not None:
            params["dichotomize"] = quickmodel.dichotomize

        # get data
        df_features = self.features.get(quickmodel.features, dataset=["train"])
        df_scheme = self.schemes.get_scheme(scheme=quickmodel.scheme)

        # management for multilabels / dichotomize
        if quickmodel.dichotomize is not None:
            df_scheme["labels"] = df_scheme["labels"].apply(
                lambda x: self.schemes.dichotomize(x, quickmodel.dichotomize)
            )

        # test for a minimum of annotated elements
        counts = df_scheme["labels"].value_counts()
        valid_categories = counts[counts >= n_min_annotated]
        if len(valid_categories) < 2:
            raise Exception(
                f"Not enough annotated elements (should be more than {n_min_annotated})"
            )

        col_features = list(df_features.columns)
        data = pd.concat([df_scheme, df_features], axis=1)
        process_id = self.quickmodels.compute_quickmodel(
            project_slug=self.params.project_slug,
            user=username,
            scheme=quickmodel.scheme,
            features=quickmodel.features,
            name=quickmodel.name,
            model_type=quickmodel.model,
            df=data,
            col_labels="labels",
            col_features=col_features,
            model_params=params,
            standardize=quickmodel.standardize or False,
            cv10=quickmodel.cv10 or False,
            balance_classes=quickmodel.balance_classes or False,
            exclude_labels=quickmodel.exclude_labels,
            retrain=retrain,
            texts=self.data.train["text"] if self.data.train is not None else None,
        )
        self.monitoring.register_process(
            process_name=process_id,
            kind="train_quickmodel",
            parameters={},
            user_name=username,
        )
        return process_id

    def retrain_quickmodel(self, name: str, scheme: str, username: str) -> None:
        """
        Retrain a quickmodel
        """
        # Get old model parameters in a QuickModelInModel
        model = self.quickmodels.get(name)
        quickmodel = QuickModelInModel(
            name=name,
            scheme=scheme,
            model=model.model_type,
            features=model.features,
            params=model.model_params,
            standardize=model.standardize,
            dichotomize=model.model_params.get("dichotomize", None),
            cv10=model.cv10,
            balance_classes=model.balance_classes,
            exclude_labels=model.exclude_labels,
        )
        self.train_quickmodel(quickmodel, username, retrain=True)

    def get_model_prediction(self, type: str, name: str) -> pd.DataFrame:
        """
        Get prediction of a model or raise an error
        - quickmodel
        - languagemodel
        """
        if type == "quickmodel":
            if not self.quickmodels.exists(name):
                raise Exception("Quickmodel doesn't exist")
            else:
                prediction = self.quickmodels.get_prediction(name)
        elif type == "languagemodel":
            if not self.languagemodels.exists(name):
                raise Exception("Languagemodel doesn't exist")
            else:
                prediction = self.languagemodels.get_prediction(name)
        else:
            raise Exception("Model type not recognized")
        return prediction

    def get_prediction_element(self, kind: str, name: str, element_id: str) -> PredictedLabel:
        """
        Get the prediction for a specific element
        - quickmodel
        - languagemodel
        """
        prediction = self.get_model_prediction(kind, name)
        predicted_label = str(prediction.loc[element_id, "prediction"])
        predicted_proba = float(cast(float, prediction.loc[element_id, predicted_label]))
        predicted_entropy = float(cast(float, prediction.loc[element_id, "entropy"]))
        return PredictedLabel(
            label=predicted_label,
            proba=round(predicted_proba, 2) if not math.isnan(predicted_proba) else None,
            entropy=round(predicted_entropy, 2) if not math.isnan(predicted_entropy) else None,
        )

    def get_next(
        self,
        next: NextInModel,
        username: str = "user",
    ) -> ElementOutModel:
        """
        Get next item for a specific scheme with a specific selection method
        - fixed
        - random
        - active
        - maxprob
        - minprob
        - test

        history : previous selected elements
        frame is the use of projection coordinates to limit the selection
        filter is a regex to use on the corpus
        """
        if next.scheme not in self.schemes.available():
            raise ValueError("Scheme doesn't exist")

        # select the current dataset
        if next.dataset == "test":
            if self.data.test is None:
                raise ValueError("No test dataset available")
            df = self.schemes.get_scheme(next.scheme, complete=True, datasets=["test"])
        elif next.dataset == "valid":
            if self.data.valid is None:
                raise ValueError("No valid dataset available")
            df = self.schemes.get_scheme(next.scheme, complete=True, datasets=["valid"])
        elif next.dataset == "train":
            df = self.schemes.get_scheme(next.scheme, complete=True, datasets=["train"])
        else:
            raise ValueError("Dataset should be test, valid or train")

        # check conditions for active learning and get proba
        proba = None
        predict = PredictedLabel(label=None, proba=None, entropy=None)
        if next.model_active is not None:
            prediction = self.get_model_prediction(next.model_active.type, next.model_active.value)
            proba = prediction

        # filter based on the labels
        if next.sample == "untagged":
            f = df["labels"].isna()
        elif next.sample == "tagged":
            # on specific labels
            if next.on_labels is not None and len(next.on_labels) > 0:
                f = df["labels"].isin(next.on_labels)
            else:
                f = df["labels"].notna()

            # on specific users
            if next.on_users is not None and len(next.on_users) > 0:
                f_user = df["user"].isin(next.on_users)
                f = f & f_user
        elif next.sample == "predicted":
            if next.model_active is None:
                raise ValueError("An active model is required for predicted sample")
            pred_reindexed = prediction.reindex(df.index)
            if next.on_labels is not None and len(next.on_labels) > 0:
                f = pred_reindexed["prediction"].isin(next.on_labels)
            else:
                f = pred_reindexed["prediction"].notna()
        else:
            f = pd.Series(True, index=df.index)

        # filter based on the text (field, context)

        if next.filter:
            # add context in the dataframe (it is ugly but ...)
            if next.dataset == "train":
                existing_cols_contexts = self.params.cols_context
                df = df.join(self.data.train[existing_cols_contexts])
            elif next.dataset == "valid" and self.data.valid is not None:
                existing_cols_contexts = [
                    i for i in self.params.cols_context if i in self.data.valid.columns
                ]
                df = df.join(self.data.valid[existing_cols_contexts])
            elif next.dataset == "test" and self.data.test is not None:
                existing_cols_contexts = [
                    i for i in self.params.cols_context if i in self.data.test.columns
                ]
                df = df.join(self.data.test[existing_cols_contexts])
            else:
                raise ValueError("Dataset should be test, valid or train")

            # sanitize
            df["ID"] = df.index  # duplicate the id column
            filter_san = clean_regex(next.filter)
            if "CONTEXT=" in filter_san:  # case to search in the context
                f_regex = (
                    df[existing_cols_contexts + ["ID"]]
                    .apply(lambda row: " ".join(row.values.astype(str)), axis=1)
                    .str.contains(
                        filter_san.replace("CONTEXT=", ""),
                        regex=True,
                        case=True,
                        na=False,
                    )
                )
            elif "QUERY=" in filter_san:  # case to use a query
                f_regex = cast(
                    pd.Series, df[existing_cols_contexts].eval(filter_san.replace("QUERY=", ""))
                )
            else:
                f_regex = df["text"].str.contains(filter_san, regex=True, case=True, na=False)
            f = f & f_regex

        # filter with a frame (projection coordinates)
        if next.frame and len(next.frame) == 4:
            if username in self.projections.available:
                if self.projections.available[username].data is not None:
                    projection = self.projections.available[username].data
                    f_frame = (
                        (projection[0] > next.frame[0])
                        & (projection[0] < next.frame[1])
                        & (projection[1] > next.frame[2])
                        & (projection[1] < next.frame[3])
                    )
                    f = f & f_frame
                else:
                    raise ValueError("No vizualisation data available")
            else:
                raise ValueError("No vizualisation available")

        # test if there is at least one element available
        if f.sum() == 0:
            raise ValueError("No element available with this selection mode.")

        # filter by history
        ss = df[f].drop(next.history, errors="ignore")
        if len(ss) == 0:
            raise ValueError(
                "No element available with this selection mode and history. Clear the history to access previous elements."
            )
        indicator = None
        n_sample = f.sum()  # use len(ss) for adding history

        # select an element based on the method

        if next.selection == "fixed":  # next row
            element_id = ss.index[0]

        if next.selection == "random":  # random row
            element_id = ss.sample(n=1).index[0]

        # be sure that the model has been trained
        if next.selection in ["maxprob", "active", "minprob"] and next.model_active is None:
            raise Exception("An active model is required for this selection method")

        # class on proba prob for the label_prob, only possible if the model has been trained
        if next.selection in ["maxprob", "minprob"] and proba is not None:
            if next.label_prob is None:  # default label to first
                raise Exception("Label maxprob is required")
            # use the history to not send already tagged data
            if next.selection == "maxprob":
                ascending = False
            else:
                ascending = True
            ss = (
                proba[f][next.label_prob]
                .drop(next.history, errors="ignore")
                .sort_values(ascending=ascending)
            )  # get max proba id
            element_id = ss.index[0]
            n_sample = f.sum()
            indicator = f"probability: {round(proba.loc[element_id, next.label_prob], 2)}"

        # higher entropy, only possible if the model has been trained
        if next.selection == "active" and proba is not None:
            ss_active = (
                proba[f]["entropy"].drop(next.history, errors="ignore").sort_values(ascending=False)
            )  # get max entropy id
            element_id = ss_active.index[0]
            n_sample = f.sum()
            indicator = round(proba.loc[element_id, "entropy"], 2)
            indicator = f"entropy: {indicator}"

        if (
            next.model_active is not None
            and next.model_active.type is not None
            and next.model_active.value is not None
            and next.dataset == "train"
        ):
            predict = self.get_prediction_element(
                next.model_active.type, next.model_active.value, element_id
            )

        # get all tags already existing for the element selected
        previous = self.schemes.projects_service.get_annotations_by_element(
            self.params.project_slug, next.scheme, element_id
        )

        if next.dataset in ["test", "valid"]:
            context = {}
        else:
            if self.data.train is None:
                raise Exception("Train dataset is not defined")
            # get context for the single selected element
            context = dict(
                self.data.train.loc[element_id, self.params.cols_context].fillna("NA").apply(str)
            )

        text = df.loc[element_id, "text"]
        if pd.isna(text):
            text = "NA"

        return ElementOutModel(
            element_id=element_id,
            text=text,
            context=context,
            selection=next.selection,
            info=indicator,
            predict=predict,
            frame=next.frame,
            limit=None,
            history=previous,
            n_sample=n_sample,
        )

    def get_element(
        self,
        element: ElementInModel,
        user: str | None = None,
    ) -> ElementOutModel:
        """
        Get an element of the database
        Separate train/test dataset

        TODO : get next and get element could be merged
        """

        text = None
        predict = PredictedLabel(label=None, proba=None, entropy=None)
        context = {}
        history = None
        if element.scheme is not None:
            history = self.schemes.projects_service.get_annotations_by_element(
                self.params.project_slug, element.scheme, element.element_id
            )

        if element.dataset == "valid":
            if self.data.valid is None:
                raise Exception("Valid dataset is not defined")
            if element.element_id not in self.data.valid.index:
                raise Exception("Element does not exist.")
            text = str(self.data.valid.loc[element.element_id, "text"])

        if element.dataset == "test":
            if self.data.test is None:
                raise Exception("Test dataset is not defined")
            if element.element_id not in self.data.test.index:
                raise Exception("Element does not exist.")
            text = str(self.data.test.loc[element.element_id, "text"])

        # case for train with more information
        if element.dataset == "train":
            if self.data.train is None:
                raise Exception("Train dataset is not defined")
            if element.element_id not in self.data.train.index:
                raise Exception("Element does not exist.")

            text = str(self.data.train.loc[element.element_id, "text"])

            # get prediction if it exists
            predict = PredictedLabel(label=None, proba=None, entropy=None)
            try:
                if element.active_model is not None:
                    predict = self.get_prediction_element(
                        element.active_model.type, element.active_model.value, element.element_id
                    )
            except Exception as e:
                # TODO: warn user to retrain the model
                print(
                    (
                        f"No prediction found for element {element.element_id}."
                        f"Please retrain the model.\n"
                        f"Error: \n{e}"
                    )
                )

            # extract context
            context = cast(
                dict[str, Any],
                self.data.train.loc[element.element_id, self.params.cols_context]  # type: ignore[index]
                .fillna("NA")
                .astype(str)
                .to_dict(),
            )
            context = {i.replace("dataset_", ""): str(context[i]) for i in context}

        if text is None:
            raise Exception(
                (f"Element {element.element_id} was not found in dataset {element.dataset}")
            )

        return ElementOutModel(
            element_id=element.element_id,
            text=text,
            context=context,
            selection="request",
            predict=predict,
            info="get specific",
            frame=None,
            limit=None,
            history=history,
        )

    def get_params(self) -> ProjectModel:
        """
        Send parameters
        """
        return self.params

    @staticmethod
    def compute_annotations_distribution(df: DataFrame, kind: str) -> dict[str, int]:
        if kind == "multiclass":
            return json.loads(df["labels"].value_counts().to_json())
        elif kind == "multilabel":
            return json.loads(df["labels"].str.split("|").explode().value_counts().to_json())
        elif kind == "span":
            r = (
                df["labels"]
                .apply(lambda x: json.loads(x) if pd.notna(x) else [])
                .explode()
                .apply(lambda x: x["tag"] if isinstance(x, dict) and "tag" in x else None)
            )
            return json.loads(r.value_counts().to_json())
        else:
            raise Exception("Not implemented for this kind of scheme")

    def get_statistics(self, scheme: str | None) -> ProjectDescriptionModel:
        """
        Generate a description of a current project/scheme/user
        """
        if scheme is None:
            raise Exception("Scheme is required")

        schemes = self.schemes.available()
        if scheme not in schemes:
            raise Exception("Scheme not available")
        kind = schemes[scheme].kind

        users = self.db_manager.users_service.get_coding_users(scheme, self.params.project_slug)
        df_annotable = self.schemes.get_scheme(scheme, datasets=["train", "valid", "test"])

        # train
        df_train = df_annotable[df_annotable["dataset"] == "train"]
        train_annotated_distribution = self.compute_annotations_distribution(df_train, kind)
        train_annotated_n = len(df_train.dropna(subset=["labels"]))
        train_set_n = len(self.data.train) if self.data.train is not None else 0

        # valid
        if self.params.valid and (self.data.valid is not None):
            df_valid = df_annotable[df_annotable["dataset"] == "valid"]
            valid_set_n = len(self.data.valid)
            valid_annotated_n = len(df_valid.dropna(subset=["labels"]))
            valid_annotated_distribution = self.compute_annotations_distribution(df_valid, kind)
        else:
            valid_set_n = None
            valid_annotated_n = None
            valid_annotated_distribution = None

        # test
        if self.params.test and (self.data.test is not None):
            df_test = df_annotable[df_annotable["dataset"] == "test"]
            test_set_n = len(self.data.test)
            test_annotated_n = len(df_test.dropna(subset=["labels"]))
            test_annotated_distribution = self.compute_annotations_distribution(df_test, kind)
        else:
            test_set_n = None
            test_annotated_n = None
            test_annotated_distribution = None

        return ProjectDescriptionModel(
            users=users,
            train_set_n=train_set_n,
            train_annotated_n=train_annotated_n,
            train_annotated_distribution=train_annotated_distribution,
            valid_set_n=valid_set_n,
            valid_annotated_n=valid_annotated_n,
            valid_annotated_distribution=valid_annotated_distribution,
            test_set_n=test_set_n,
            test_annotated_n=test_annotated_n,
            test_annotated_distribution=test_annotated_distribution,
            sm_10cv=None,
        )

    def get_projection(
        self,
        username: str,
        scheme: str,
        active_model: ActiveModel | None = None,
    ) -> ProjectionOutModel | None:
        """
        Get projection if computed
        """
        projection = self.projections.get(username)
        if projection is None:
            return None
        # get annotations - use copy to avoid mutating stored projection data
        df = self.schemes.get_scheme(scheme, complete=True, datasets=["train"])
        data = projection.data.copy()
        data["labels"] = df["labels"].reindex(data.index).fillna("NA")

        # get & add predictions if available
        if active_model is not None and active_model.type == "quickmodel":
            if not self.quickmodels.exists(active_model.value):
                raise Exception("Quickmodel doesn't exist")
            data["prediction"] = self.quickmodels.get_prediction(active_model.value)["prediction"]
        elif active_model is not None and active_model.type == "languagemodel":
            if not self.languagemodels.exists(active_model.value):
                raise Exception("Languagemodel doesn't exist")
            data["prediction"] = self.languagemodels.get_prediction(active_model.value)[
                "prediction"
            ]

        if "prediction" in data:
            predictions = data["prediction"].to_list()
        else:
            predictions = [None] * len(data)

        return ProjectionOutModel(
            nodes=[
                ProjectionOutModelNode(
                    node_id=node_id,
                    x=x,
                    y=y,
                    label=label,
                    predictions=[prediction] if prediction else None,
                )
                for node_id, x, y, label, prediction in zip(
                    data.index.to_list(),
                    data[0].to_list(),
                    data[1].to_list(),
                    data["labels"].to_list(),
                    predictions,
                )
            ],
            status=projection.id,
            parameters=projection.parameters,
            active_model=active_model,
        )

    def _get_cached_memory(self) -> float:
        """
        Return cached project directory size (MB).
        Refreshes at most every _memory_cache_interval seconds.
        """
        now = time.time()
        if (now - self._memory_cache_time) >= self._memory_cache_interval:
            self._memory_cache = get_dir_size(str(self.params.dir))
            self._memory_cache_time = now
        return self._memory_cache

    def state(self) -> ProjectStateModel:
        """
        State of the project
        Collecting states for submodules
        Cached with a short TTL so multiple users polling don't each trigger
        the full computation (DB queries, file reads, pandas work).
        """
        now = time.time()
        if (
            self._state_cache is not None
            and (now - self._state_cache_time) < self._state_cache_interval
        ):
            return self._state_cache

        result = ProjectStateModel(
            params=self.params,
            next=NextProjectStateModel(
                methods_min=["fixed", "random"],
                methods=["fixed", "random", "maxprob", "active", "minprob"],
                sample=["untagged", "all", "tagged", "predicted"],
            ),
            schemes=self.schemes.state(),
            features=self.features.state(),
            quickmodel=self.quickmodels.state(),
            languagemodels=self.languagemodels.state(),
            projections=self.projections.state(),
            generations=self.generations.state(),
            bertopic=self.bertopic.state(),
            errors=self.errors.state(),
            memory=self._get_cached_memory(),
            last_activity=self.db_manager.logs_service.get_last_activity_project(
                self.params.project_slug
            ),
            users=self.users.state(self.params.project_slug),
        )
        self._state_cache = result
        self._state_cache_time = now
        return result

    def export_features(self, features: list, format: str = "parquet") -> FileResponse:
        """
        Export features data in different formats
        """
        if len(features) == 0:
            raise ValueError("No feature selected")

        path = self.params.dir  # path of the data
        if path is None:
            raise ValueError("Problem of filesystem for project")

        data = self.features.get(features, dataset="annotable")

        file_name = f"extract_schemes_{self.name}.{format}"

        # create files
        if format == "csv":
            data.to_csv(path.joinpath(file_name))
        if format == "parquet":
            data.to_parquet(path.joinpath(file_name))
        if format == "xlsx":
            data.to_excel(path.joinpath(file_name))

        return FileResponse(path=path.joinpath(file_name), filename=file_name)

    def export_data(
        self, scheme: str, dataset: str = "train", format: str = "parquet", dropna: bool = True
    ) -> FileResponse:
        """
        Export annotation data in different formats
        - for a scheme & dataset
        - for all schemes & every annotation
        """

        path = self.params.dir  # path of the data
        if path is None:
            raise ValueError("Problem of filesystem for project")

        # test dataset availability
        if dataset == "valid":
            if self.data.valid is None:
                raise Exception("No valid data available")
        if dataset == "test":
            if self.data.test is None:
                raise Exception("No test data available")

        # for a specific scheme and dataset
        if scheme != "all" and dataset in ["train", "test", "valid"]:
            data = self.schemes.get_scheme(
                scheme=scheme, complete=True, datasets=[dataset], id_external=True
            )
            file_name = f"export_tags_{self.name}_{scheme}.{format}"
        # for all the annotated data in the project, need to concate
        elif scheme == "all":
            schemes = self.schemes.available()
            data = pd.concat(
                [
                    self.schemes.get_scheme(
                        scheme_name,
                        complete=True,
                        datasets=["train", "valid", "test"],
                        id_external=True,
                    ).rename(columns=lambda col: f"{scheme_name}_{col}")
                    for scheme_name in schemes
                ],
                axis=1,
            )
            file_name = f"export_tags_{self.name}_all.{format}"
            dropna = False

            # Combine all columns id_internal into one
            columns_id_external = [col for col in data.columns if col.endswith("id_external")]
            id_external_serie = data[columns_id_external[0]].copy()
            for column_external in columns_id_external:
                id_external_serie = id_external_serie.combine(
                    data[column_external], lambda a, b: a
                )  # if 2 elements exist take the first one
            data = data.drop(columns=columns_id_external)
            data.loc[:, "id_external"] = id_external_serie
        else:
            raise Exception("Scheme or dataset not recognized")

        # transformation of the data
        if dropna:
            data = data.dropna(subset=["labels"])

        # select columns + order

        cols = [col for col in data.columns if not (col.endswith("id_internal"))]
        data = data[cols]
        if self.params.col_id is not None:
            data.rename(columns={"id_external": self.params.col_id}, inplace=True)

        # write file in the folder
        if format == "csv":
            data.to_csv(path.joinpath(file_name))
        if format == "parquet":
            data.to_parquet(path.joinpath(file_name))
        if format == "xlsx":
            if "timestamp" in data.columns:
                data["timestamp"] = data["timestamp"].dt.tz_localize(None)
            data.to_excel(path.joinpath(file_name))

        return FileResponse(path.joinpath(file_name), filename=file_name)

    def export_generations(
        self, project_slug: str, username: str, params: ExportGenerationsParams
    ) -> DataFrame:
        # get the elements
        table = self.generations.get_generated(
            project_slug=project_slug,
            user_name=username,
            params=params,
        )

        # apply filters on the generated
        table["answer"] = self.generations.filter(table["answer"], params.filters)

        # join the text
        if self.data.train is None:
            raise Exception("No train data available")
        table = table.join(self.data.train["text"], on="index")

        return table

    def get_process(
        self, kind: str | list, user: str
    ) -> list[FeatureComputing | LMComputing | QuickModelComputing]:
        """
        Get current processes
        """
        if isinstance(kind, str):
            kind = [kind]
        return [e for e in self.computing if e.user == user and e.kind in kind]

    def export_raw(self, project_slug: str) -> StaticFileModel:
        """
        Export raw data
        To be able to export, need to copy in the static folder
        """
        target_dir = self.params.dir if self.params.dir is not None else Path(".")
        path_origin = target_dir.joinpath("data_all.parquet")
        folder_target = f"{config.data_path}/projects/static/{project_slug}"
        if not Path(folder_target).exists():
            os.makedirs(folder_target)
        files = [i for i in os.listdir(folder_target) if "_data_all_" in i]
        # file already exists
        if len(files) > 0:
            name = files[0]
            path_target = f"{config.data_path}/projects/static/{project_slug}/{name}"
        # create the file with a unique id
        else:
            name = f"{project_slug}_data_all_{uuid.uuid4()}.parquet"
            path_target = f"{config.data_path}/projects/static/{project_slug}/{name}"
            shutil.copyfile(path_origin, path_target)
        return StaticFileModel(name=name, path=f"{project_slug}/{name}")

    def start_update_project(self, update: ProjectUpdateModel, username: str) -> None:
        """
        Update project parameters

        For text/contexts/expand, it needs to draw from raw data
        - direct small modification
        - bigger modification (texts/contexts/expand) with the queue
        """

        if not self.params.dir:
            raise ValueError("No directory for project")
        if self.data.train is None:
            raise ValueError("No train data for project")

        # update the name
        if update.project_name and update.project_name != self.params.project_name:
            self.params.project_name = update.project_name

        # update the language
        if update.language and update.language != self.params.language:
            self.params.language = update.language

        # for other updates, add task to the queue
        unique_id = self.queue.add_task(
            kind="update_datasets",
            project_slug=self.name,
            task=UpdateDatasets(
                project_params=self.params,
                update=update,
            ),
            queue="cpu",
        )
        self.computing.append(
            UpdateComputing(
                unique_id=unique_id,
                user=username,
                time=datetime.now(),
                kind="update_datasets",
                update=update,
            )
        )

    def start_languagemodel_training(self, bert: BertModelModel, username: str) -> None:
        """
        Launch a training process
        """
        # Check if there is no other competing processes : 1 active process by user
        if len(self.languagemodels.current_user_processes(username)) > 0:
            raise Exception(
                "User already has a process launched, please wait before launching another one"
            )
        # get data
        df = self.schemes.get_scheme(bert.scheme, datasets=["train"], complete=True)
        df = df[["text", "labels"]].dropna()

        # management for multilabels / dichotomize
        if bert.dichotomize is not None:
            df["labels"] = df["labels"].apply(
                lambda x: self.schemes.dichotomize(x, bert.dichotomize)
            )
            bert.name = f"{bert.name}_multilabel_on_{bert.dichotomize}"

        # remove class under the threshold
        label_counts = df["labels"].value_counts()
        df = df[df["labels"].isin(label_counts[label_counts >= bert.class_min_freq].index)]

        # remove class requested by the user
        if len(bert.exclude_labels) > 0:
            df = df[~df["labels"].isin(bert.exclude_labels)]
            bert.name = f"{bert.name}_exclude_labels_"

        # balance the dataset based on the min class
        if bert.class_balance:
            min_freq = df["labels"].value_counts().sort_values().min()
            df = (
                df.groupby("labels")
                .apply(lambda x: x.sample(min_freq))
                .reset_index(level=0, drop=True)
            )

        # launch training process
        process_id = self.languagemodels.start_training_process(
            name=bert.name,
            project=self.name,
            user=username,
            scheme=bert.scheme,
            df=df,
            col_text=df.columns[0],
            col_label=df.columns[1],
            base_model=bert.base_model,
            params=bert.params,
            test_size=bert.test_size,
            loss=bert.loss,
            max_length=bert.max_length,
            auto_max_length=bert.auto_max_length,
            class_balance=bert.class_balance,
        )
        self.monitoring.register_process(
            process_name=process_id,
            kind="train_languagemodel",
            parameters={},
            user_name=username,
        )

    def start_generation(self, request: GenerationRequest, username: str) -> None:
        """
        Start a generation process
        """
        extract = self.schemes.get_sample(request.scheme, request.n_batch, request.mode)
        if len(extract) == 0:
            raise Exception("No elements available for generation")
        model = self.generations.generations_service.get_gen_model(request.model_id)
        # add task to the queue
        unique_id = self.queue.add_task(
            "generation",
            self.name,
            GenerateCall(
                path_process=self.params.dir,
                username=username,
                project_slug=self.name,
                df=extract,
                prompt=request.prompt,
                model=GenerationModel(**model.__dict__),
                cols_context=self.params.cols_context,
            ),
        )
        self.computing.append(
            GenerationComputing(
                unique_id=unique_id,
                prompt_name=request.prompt_name if request.prompt_name else "",
                user=username,
                project=self.name,
                model_id=request.model_id,
                number=request.n_batch,
                time=datetime.now(),
                kind="generation",
                get_progress=GenerateCall.get_progress_callback(
                    self.params.dir.joinpath(unique_id) if self.params.dir is not None else None
                ),
            )
        )

    def clean_process(self, e: ProcessComputing) -> None:
        """
        Clean a process from computing and queue
        """
        self.computing.remove(e)
        self.queue.delete(e.unique_id)

    def update_processes(self) -> None:
        """
        Update completed processes and do specific operations regarding their kind
        - get the result from the queue
        - add the result if needed
        - manage error if needed
        """
        add_predictions = {}

        # loop on the current process
        for e in self.computing.copy():
            # get the process
            process = self.queue.get(e.unique_id)
            if process is None:
                self.clean_process(e)
                continue

            # check if the process is done, else continue
            if process.future is None or not process.future.done():
                continue

            # log error if exists in the process execution
            exception = process.future.exception()
            if exception:
                print(f"Error in {e.kind} : {exception}")
                self.errors.add(f"Error for process {e.kind} : {exception}")

                # specific case for project creation ; delete the project
                if e.kind == "create_project":
                    print("Error in project creation")
                    self.status = "error"

                self.clean_process(e)
                continue

            # get the result and do specific operations, if it fails, log the error
            try:
                results = process.future.result()
                match e.kind:
                    case "create_project":
                        e = cast(ProjectCreatingModel, e)
                        if results is None:
                            print("No result from project creation")
                            raise Exception("No result from project creation")
                        self.finish_project_creation(
                            e.username, results[0], results[1], results[2], results[3]
                        )
                    case "update_datasets":
                        e = cast(UpdateComputing, e)
                        self.db_manager.projects_service.update_project(
                            self.params.project_slug, jsonable_encoder(results[0])
                        )
                        self.params = results[0]
                        # reload the data in memory
                        self.data.load_dataset("all")
                        # reset the features file and load the dataset again
                        if results[1]:
                            self.features.reset_features_file()
                            self.bertopic.clear_bertopic()
                            self.projections.clear_projections()

                    case "train_bert":
                        model = cast(LMComputing, e)
                        events = cast(EventsModel, results)
                        self.languagemodels.add(model)
                        self.monitoring.close_process(model.unique_id, events)
                    case "predict_bert":
                        prediction = cast(LMComputing, e)
                        if (
                            results is not None
                            and results.path
                            and "predict_annotable.parquet" in results.path
                        ):
                            add_predictions["predict_" + prediction.model_name] = results.path
                        self.languagemodels.add(prediction)
                    case "train_quickmodel":
                        sm = cast(QuickModelComputing, e)

                        # Retrieve the additional events if they exist
                        events = cast(EventsModel, results)
                        self.quickmodels.add(sm)
                        self.monitoring.close_process(sm.unique_id, events)
                    case "predict_quickmodel":
                        sm = cast(QuickModelComputing, e)
                    case "feature":
                        feature_computation = cast(FeatureComputing, e)
                        self.features.add(
                            feature_computation.name,
                            feature_computation.type,
                            feature_computation.user,
                            feature_computation.parameters,
                            results,
                        )
                    case "projection":
                        projection = cast(ProjectionComputing, e)
                        self.projections.add(projection, results)
                    case "generation":
                        e = cast(GenerationComputing, e)
                        r = cast(
                            list[GenerationResult],
                            results,
                        )
                        batch = str(e.prompt_name) + "_" + e.unique_id
                        for row in r:
                            self.generations.add(
                                user=row.user,
                                project_slug=row.project_slug,
                                element_id=row.element_id,
                                model_id=row.model_id,
                                prompt=row.prompt,
                                answer=row.answer,
                                batch=batch,
                            )
                    case "bertopic":
                        bertopic_model = cast(BertopicComputing, e)
                        events = cast(EventsModel, results)
                        self.bertopic.add(bertopic_model)
                        self.monitoring.close_process(bertopic_model.unique_id, events)
            except Exception as ex:
                print(f"Error in {e.kind} : {ex}")
                self.errors.add(f"Error in {e.kind} : {str(ex)}")
                match e.kind:
                    case "create_project":
                        self.status = "error"
                    case "train_bert":
                        bert_task = cast(LMComputing, e)
                        self.db_manager.language_models_service.delete_model(
                            self.name, bert_task.model_name
                        )
            # clean the process from the list and the queue
            finally:
                self.clean_process(e)

        # if there are predictions, add them
        if len(add_predictions) > 0:
            errors = self.features.add_predictions(add_predictions)
            for err in errors:
                self.errors.add(err)

    # def dump(self, with_files=True) -> None:
    #     """
    #     Dump the project in a archive
    #     - keep the files
    #     - do not keep the models

    #     Ideally, to be able to rerun everything
    #     """
    #     if self.params.dir is None:
    #         raise Exception("No directory for project")
    #     os.mkdir(self.params.dir.joinpath("dump"))

    #     # save the project parameters
    #     # - features computed

    #     # save the annotations

    #     # save the data (train + test + all)
    #     if with_files:
    #         shutil.copyfile(
    #             self.params.dir.joinpath("data_all.parquet"),
    #             self.params.dir.joinpath("dump").joinpath("data_all.parquet"),
    #         )
    #         shutil.copyfile(
    #             self.params.dir.joinpath("train.parquet"),
    #             self.params.dir.joinpath("dump").joinpath("data_train.parquet"),
    #         )
    #         if self.params.test:
    #             shutil.copyfile(
    #                 self.params.dir.joinpath("test.parquet"),
    #                 self.params.dir.joinpath("dump").joinpath("data_test.parquet"),
    #             )

    #     # save the codebook

    #     # create the archive
    #     shutil.make_archive(
    #         f"dump_{self.project_slug}", "zip", self.params.dir.joinpath("dump"), self.params.dir
    #     )

    #     # delete the dump folder
    #     shutil.rmtree(self.params.dir.joinpath("dump"))
    #     return None
