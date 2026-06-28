import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from fastapi.responses import FileResponse
from slugify import slugify

from activetigger.config import config
from activetigger.datamodels import (
    BertopicComputing,
    BERTopicDescriptionModel,
    BertopicOutModelParameters,
    BertopicParamsModel,
    BertopicProjectionData,
    BertopicProjectionNode,
    BertopicProjectStateModel,
    TopicsOutModel,
)
from activetigger.db.languagemodels import ModelsService
from activetigger.db.manager import DatabaseManager
from activetigger.features import Features
from activetigger.queue_manager import Queue
from activetigger.tasks.compute_bertopic import ComputeBertopic

# TODO : put params in database
# TODO : Implement the get_topics and get_projection methods
# TODO : Richer state with defined typemodels


class Bertopic:
    """
    Class to handle BERTopic computations.
    """

    models_service: ModelsService

    def __init__(
        self,
        project_slug: str,
        path: Path,
        queue: Queue,
        computing: list,
        features: Features,
        db_manager: DatabaseManager,
    ) -> None:
        self.cache: dict[str, BERTopicDescriptionModel] = {}
        self.project_slug = project_slug
        self.queue = queue
        self.computing = computing
        self.path: Path = path.joinpath("bertopic")
        self.path.mkdir(parents=True, exist_ok=True)
        if not self.path.joinpath("runs").exists():
            self.path.joinpath("runs").mkdir(parents=True, exist_ok=True)
        self.features = features
        self.models_service = db_manager.language_models_service

    @staticmethod
    def get_params(folder_path):
        with open(folder_path / "params.json", "r") as file:
            return json.load(file)

    def compute(
        self,
        path_data: Path,
        col_id: str | None,
        col_text: str,
        parameters: BertopicParamsModel,
        name: str,
        user: str,
        scheme: str,  # This is a dummy necessary to save the model in the database, it will not be used afterwards — Axel
    ) -> str:
        """
        Compute BERTopic model.

        BERTopic always reuses embeddings from an existing project feature
        (parameters.existing_feature). Embeddings are never recomputed here —
        the feature must already have been built in the project's Features page.
        """

        name = slugify(name)

        if len(self.current_user_processes(user)) > 0:
            raise ValueError("You already have computation in progress.")

        if not parameters.existing_feature:
            raise ValueError(
                "existing_feature is required: pick a sentence-embeddings feature "
                "from the project's Features page."
            )
        self._materialize_feature_embeddings(parameters)

        args = ComputeBertopic(
            path_bertopic=self.path,
            path_data=path_data,
            col_id=col_id,
            col_text=col_text,
            parameters=parameters,
            name=name,
            force_compute_embeddings=False,
            random_seed=config.random_seed,
        )
        unique_id = self.queue.add_task("bertopic", self.project_slug, args, queue="cpu")
        self.computing.append(
            BertopicComputing(
                user=user,
                unique_id=unique_id,
                name=name,
                path_data=path_data,
                col_id=col_id,
                col_text=col_text,
                parameters=parameters,
                time=datetime.now(timezone.utc),
                kind="bertopic",
                force_compute_embeddings=False,
                get_progress=self.get_progress(name),
                scheme=scheme,
            )
        )
        return unique_id

    def add(self, element: BertopicComputing) -> None:
        """
        Add a trained BERTopic in the database
        """
        model_path = self.path.joinpath("runs").joinpath(element.name)
        self.models_service.add_model(
            kind="bertopic",
            project=self.project_slug,
            name=element.name,
            user=element.user,
            status="trained",
            scheme=element.scheme,
            params=element.parameters.model_dump(),
            path=str(model_path),
        )

    def training(self) -> dict[str, dict[str, str | int | float | None]]:
        """
        Get available BERTopic models in the current process
        """
        return {
            e.user: {
                "name": e.name,
                "status": "training",
                "progress": e.get_progress() if e.get_progress else None,
            }
            for e in self.computing
            if e.kind == "bertopic"
        }

    def get_model(self, name: str) -> BERTopicDescriptionModel:
        """
        Get a BERTopic model parameters.
        """
        if name in self.cache:
            return self.cache[name]
        else:
            model = BERTopicDescriptionModel(
                name=name, time=self.get_params(self.path.joinpath("runs") / name)["timestamp"]
            )
            self.cache[name] = model
            return model

    def available(self) -> dict[str, BERTopicDescriptionModel]:
        """
        Get available BERTopic models.
        """

        return {
            p.name: self.get_model(p.name)
            for p in self.path.joinpath("runs").iterdir()
            if p.is_dir() & p.joinpath("params.json").exists()
        }

    def name_available(self, name: str) -> bool:
        """
        Check if a BERTopic model name is available.
        """
        return slugify(name) not in self.available()

    def state(self) -> BertopicProjectStateModel:
        return BertopicProjectStateModel(
            available=self.available(),
            training=self.training(),
            bindable_features=self._bindable_features(),
        )

    def _bindable_features(self) -> list[str]:
        """
        Project features that can be reused as BERTopic embeddings.
        Currently restricted to sentence-embeddings features.
        """
        try:
            available = self.features.get_available()
        except Exception:
            return []
        return [name for name, feat in available.items() if feat.kind == "sentence-embeddings"]

    def _materialize_feature_embeddings(self, parameters: BertopicParamsModel) -> None:
        """
        Copy the selected project feature into the BERTopic embeddings folder
        so the compute task picks it up via its standard caching path.

        Mutates parameters.embedding_model so that the path computed by the task
        matches the file produced here. The original feature name is preserved
        via parameters.existing_feature for traceability in params.json.
        """
        feature_name = parameters.existing_feature
        if feature_name is None:
            return
        if parameters.input_datasets == "complete":
            raise ValueError(
                "Existing features cover only train/valid/test rows; "
                "input_datasets='complete' is not supported with existing_feature."
            )
        if not self.features.exists(feature_name):
            raise ValueError(f"Feature '{feature_name}' does not exist.")
        feat_info = self.features.get_available().get(feature_name)
        if feat_info is None or feat_info.kind != "sentence-embeddings":
            raise ValueError(f"Feature '{feature_name}' is not a sentence-embeddings feature.")

        datasets = ["train"] if parameters.input_datasets == "train" else ["train", "valid", "test"]
        df = self.features.get([feature_name], dataset=datasets)

        synthetic_model = f"feature-{slugify(feature_name)}"
        parameters.embedding_model = synthetic_model

        embeddings_dir = self.path.joinpath("embeddings")
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        target = embeddings_dir.joinpath(
            f"bertopic_embeddings_{parameters.input_datasets}_{slugify(synthetic_model)}.parquet"
        )
        # Always overwrite to avoid stale data if the feature was recomputed.
        df.to_parquet(target)

    def current_user_processes(self, user: str) -> list:
        """
        Get current user processes
        """
        return [e for e in self.computing if e.user == user]

    def get_progress(self, name) -> Callable[[], float | None]:
        """
        Access the log progress.
        During embedding computation, reads from a separate SBERT progress file
        and scales it to the 10-60% range of the overall BERTopic progress.
        """
        path_run = self.path.joinpath("runs").joinpath(name)
        path_progress = path_run.joinpath("progress")
        path_sbert_progress = path_run.joinpath("progress_sbert")

        def progress():
            # If SBERT is computing embeddings, scale its 0-100 to 10-60
            if path_sbert_progress.exists():
                try:
                    sbert_val = float(path_sbert_progress.read_text().strip())
                    return 10 + (sbert_val / 100) * 50
                except (ValueError, OSError):
                    pass
            if path_progress.exists():
                try:
                    return float(path_progress.read_text().strip())
                except (ValueError, OSError):
                    pass
            return None

        return progress

    def delete(self, name: str) -> None:
        """
        Delete a BERTopic model.
        """

        # on disk
        path_model = self.path.joinpath("runs").joinpath(name)
        # In database
        self.models_service.delete_model(self.project_slug, name)
        if path_model.exists():
            shutil.rmtree(path_model)
        else:
            raise FileNotFoundError(f"Model {name} does not exist.")

    def clear_bertopic(self) -> None:
        """
        Delete all bertopics for a scheme
        """
        for name in self.available():
            self.delete(name)

    def get_topics(self, name: str) -> list[TopicsOutModel]:
        """
        Get topics from a BERTopic model.
        Return a list of dictionaries where a dictionary is a row in the dataframe
        (alike TopicsOutModel).
        """
        path_model = self.path.joinpath("runs").joinpath(name)
        if path_model.exists():
            df = pd.read_csv(path_model.joinpath("bertopic_topics.csv"), index_col=0)
            df.columns = df.columns.astype(str)
            df_list = df.reset_index().to_dict(orient="records")
            return [TopicsOutModel(**item) for item in df_list]  # ty: ignore[invalid-argument-type]
        else:
            raise FileNotFoundError(f"Model {name} does not exist.")

    def get_clusters(self, name: str) -> dict[str, int]:
        """
        Get clusters from a BERTopic model.
        Return a list of dictionaries where a dictionary is a row in the dataframe
        (structure: {'id' : cluster}).
        """
        path_model = self.path.joinpath("runs").joinpath(name)
        if path_model.exists():
            return pd.read_csv(path_model.joinpath("bertopic_clusters.csv"), index_col=0).to_dict()[
                "cluster"
            ]
        else:
            raise FileNotFoundError(f"Model {name} does not exist.")

    def get_parameters(self, name: str) -> BertopicOutModelParameters:
        """
        Get parameters file from a BERTopic model
        TODO : cache ?
        """
        path_model = self.path.joinpath("runs").joinpath(name)
        if not path_model.exists():
            raise FileNotFoundError(f"Model {name} does not exist.")
        params_path = path_model.joinpath("params.json")
        if not params_path.exists():
            raise FileNotFoundError(f"Parameters for model {name} do not exist.")
        with open(params_path) as f:
            r = json.load(f)
            return BertopicOutModelParameters(**r)

    def get_projection(self, name: str) -> BertopicProjectionData:
        """
        Open the project and the cluster
        """
        path_clusters = self.path.joinpath("runs").joinpath(name).joinpath("bertopic_clusters.csv")
        path_projection = self.path.joinpath("runs").joinpath(name).joinpath("projection2D.parquet")
        if not path_clusters.exists() or not path_projection.exists():
            raise FileNotFoundError(f"Projection for model {name} does not exist.")
        clusters = pd.read_csv(path_clusters, index_col=0)
        clusters.index = clusters.index.astype(str)
        projection = pd.read_parquet(path_projection)
        projection["cluster"] = clusters["cluster"]
        path_model = self.path.joinpath("runs").joinpath(name)
        cluster_id_label_mapper = {}
        if path_model.exists():
            df = pd.read_csv(path_model.joinpath("bertopic_topics.csv"), index_col=0)
            cluster_id_label_mapper = dict(df["Name"])

        nodes_info = [
            BertopicProjectionNode(
                node_id=str(node_id),
                x=x,
                y=y,
                cluster_id=cluster_id,
                label=str(cluster_id_label_mapper[cluster_id]),
            )
            for x, y, cluster_id, node_id in zip(
                projection["x"], projection["y"], projection["cluster"], projection.index.to_list()
            )
        ]
        return BertopicProjectionData(
            nodes=nodes_info, cluster_id_label_mapper=cluster_id_label_mapper
        )

    def export_topics(self, name: str) -> FileResponse:
        """
        Export topics from a BERTopic model.
        """
        path_model = self.path.joinpath("runs").joinpath(name)
        if path_model.exists():
            topics_path = path_model.joinpath("bertopic_topics.csv")
            if topics_path.exists():
                return FileResponse(path=topics_path, filename=f"bertopic_topics_{name}.csv")
            else:
                raise FileNotFoundError(f"Topics for model {name} do not exist.")
        else:
            raise FileNotFoundError(f"Model {name} does not exist.")

    def export_clusters(self, name: str, col_id: str | None = None) -> FileResponse:
        """
        Export clusters from a BERTopic model.
        Rename id_external to original column name for consistency with other exports.
        """
        path_model = self.path.joinpath("runs").joinpath(name)
        if not path_model.exists():
            raise FileNotFoundError(f"Model {name} does not exist.")
        clusters_path = path_model.joinpath("bertopic_clusters.csv")
        if not clusters_path.exists():
            raise FileNotFoundError(f"Clusters for model {name} do not exist.")

        if col_id is not None:
            df = pd.read_csv(clusters_path)
            if "id_external" in df.columns:
                df.rename(
                    columns={"id_external": col_id.removeprefix("dataset_")},
                    inplace=True,
                )
            export_path = path_model.joinpath("bertopic_clusters_export.csv")
            df.to_csv(export_path, index=False)
            return FileResponse(path=export_path, filename=f"bertopic_clusters_{name}.csv")

        return FileResponse(path=clusters_path, filename=f"bertopic_clusters_{name}.csv")

    def export_report(self, name: str) -> FileResponse:
        """
        Export clusters from a BERTopic model.
        """
        path_model = self.path.joinpath("runs").joinpath(name)
        if path_model.exists():
            report_path = path_model.joinpath("report.html")
            if report_path.exists():
                return FileResponse(path=report_path, filename=f"bertopic_report_{name}.html")
            else:
                raise FileNotFoundError(f"Report for model {name} do not exist.")
        else:
            raise FileNotFoundError(f"Model {name} does not exist.")

    def export_embeddings(self, name: str) -> FileResponse:
        """
        Export embeddings used for BERTopic model.
        """

        path_params = self.path.joinpath("runs").joinpath(name).joinpath("params.json")
        if path_params.exists():
            with open(path_params, "r") as file:
                path_embeddings = json.load(file)["path_embeddings"]
            path_embeddings = Path(path_embeddings)
            if path_embeddings.exists():
                return FileResponse(
                    path=path_embeddings, filename=f"bertopic_embeddings_{name}.parquet"
                )
            else:
                raise FileNotFoundError(f"Embeddings for model {name} do not exist.")
        else:
            raise FileNotFoundError(f"Model {name} does not exist.")

    def export_to_scheme(self, name: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
        """
        Export topics and clusters from a BERTopic model as a scheme.
        """
        topics = self.get_topics(name)

        def get_topic_id(t: str) -> int:
            return int(t.split("_")[0])

        topic_id_to_topic_name = {
            get_topic_id(topic.Name): topic.Name
            for topic in topics
            if get_topic_id(topic.Name) != -1
        }
        clusters = self.get_clusters(name)
        labels = [topic.Name for topic in topics if get_topic_id(topic.Name) != -1]
        return labels, clusters, topic_id_to_topic_name
