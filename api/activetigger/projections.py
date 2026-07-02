import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.responses import FileResponse
from pandas import DataFrame

from activetigger.datamodels import (
    ProjectionComputing,
    ProjectionDataModel,
    ProjectionParametersModel,
    ProjectionsProjectStateModel,
)
from activetigger.queue_manager import Queue
from activetigger.tasks.compute_projection import ComputeProjection


class Projections:
    """
    Manage projections.

    Projections are project-level and keyed by a user-provided name (like
    BERTopic models), so a project can hold several named visualisations
    computed with different parameters and users can switch between them.
    """

    path: Path
    available: dict[str, ProjectionDataModel]
    options: dict[str, dict[str, Any]]
    computing: list
    queue: Queue

    def __init__(self, path: Path, computing: list, queue: Queue) -> None:
        self.path = path
        self.computing = computing
        self.queue = queue
        self.available = {}
        self.options = {
            "umap": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "n_components": 2,
                "metric": ["cosine", "euclidean"],
            },
            "tsne": {
                "n_components": 2,
                "learning_rate": "auto",
                "init": "random",
                "perplexity": 3,
            },
        }
        self.load()

    def load(self) -> None:
        """
        Load available projections in pickle file.

        Old pickle files were keyed by username with ProjectionDataModel
        objects that didn't carry a `name` field. We keep that mapping
        (treating the username as the projection name) so existing projects
        keep their visualisation after upgrade.
        """
        pickle_path = self.path.joinpath("projections.pickle")
        if not pickle_path.exists():
            return
        try:
            with open(pickle_path, "rb") as file:
                loaded = pickle.load(file)
        except Exception as e:
            print(e)
            return

        migrated: dict[str, ProjectionDataModel] = {}
        for key, projection in loaded.items():
            if not isinstance(projection, ProjectionDataModel):
                continue
            if not getattr(projection, "name", ""):
                projection.name = key
            migrated[projection.name] = projection
        self.available = migrated

    def _save(self) -> None:
        try:
            with open(self.path.joinpath("projections.pickle"), "wb") as f:
                pickle.dump(self.available, f)
        except Exception as e:
            print("Error in saving projections", e)

    def current_computing(self):
        return [e.name for e in self.computing if e.kind == "projection"]

    def training(self) -> dict[str, str]:
        """
        Currently under training, keyed by projection name.
        """
        return {e.name: e.method for e in self.computing if e.kind == "projection"}

    def add(self, element: ProjectionComputing, results: DataFrame) -> None:
        """
        Add projection after computation, keyed by the projection name
        provided by the user at compute time.
        """
        name = element.params.name or element.name
        self.available[name] = ProjectionDataModel(
            id=element.unique_id,
            name=name,
            data=results,
            parameters=element.params,
        )
        self._save()

    def delete(self, name: str) -> None:
        """
        Delete a projection by name.
        """
        if name not in self.available:
            raise Exception(f"Projection '{name}' does not exist")
        del self.available[name]
        self._save()

    def compute(
        self,
        project_slug: str,
        username: str,
        projection: ProjectionParametersModel,
        features: DataFrame,
        normalize_features: bool = False,
    ) -> None:
        """
        Launch the projection computation in the queue.
        """
        if not projection.name:
            raise Exception("A projection name is required")
        if projection.name in self.available:
            raise Exception(f"A projection named '{projection.name}' already exists")
        if projection.name in [e.name for e in self.computing if e.kind == "projection"]:
            raise Exception(f"A projection named '{projection.name}' is already being computed")

        unique_id = self.queue.add_task(
            "projection",
            project_slug,
            ComputeProjection(
                kind=projection.method,
                features=features,
                params=projection.parameters,
                normalize_features=normalize_features,
            ),
        )
        self.computing.append(
            ProjectionComputing(
                unique_id=unique_id,
                name=projection.name,
                user=username,
                time=datetime.now(timezone.utc),
                kind="projection",
                method=projection.method,
                params=projection,
                normalize_features=normalize_features,
            )
        )

    def get(self, name: str) -> ProjectionDataModel | None:
        """
        Get a projection by name.
        """
        return self.available.get(name)

    def state(self) -> ProjectionsProjectStateModel:
        return ProjectionsProjectStateModel(
            options=self.options,
            available={name: projection.id for name, projection in self.available.items()},
            training=self.training(),
        )

    def export(
        self,
        name: str,
        format: str = "csv",
        col_id: str | None = None,
        id_mapping: DataFrame | None = None,
    ) -> FileResponse:
        """
        Export a named projection.

        The projection is internally indexed on id_internal (slugified). When
        an id_mapping is supplied (project.data.index), expose the original
        id values under col_id, for consistency with other exports.
        """
        if name not in self.available:
            raise Exception("No projection available")
        data = self.available[name].data.copy()

        if id_mapping is not None and "id_external" in id_mapping.columns:
            column_name = (col_id or "id").removeprefix("dataset_")
            data[column_name] = data.index.map(id_mapping["id_external"].to_dict())
            data = data.set_index(column_name)

        file_name = f"projection_{name}.{format}"
        if format == "csv":
            data.to_csv(self.path.joinpath(file_name))
        if format == "parquet":
            data.to_parquet(self.path.joinpath(file_name))
        if format == "xlsx":
            data.to_excel(self.path.joinpath(file_name))

        return FileResponse(
            path=self.path.joinpath(file_name),
            filename=file_name,
        )

    def clear_projections(self):
        """
        Clear the projections
        """
        self.available = {}
        self._save()
