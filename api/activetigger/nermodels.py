import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple, cast

import pandas as pd
from fastapi.responses import FileResponse
from pandas import DataFrame
from task_manager.tasks.predict_ner_task import PredictNerTask, PredictNerTaskInputs
from task_manager.tasks.train_ner_task import TrainNerTask, TrainNerTaskInputs

import activetigger.functions as functions
from activetigger.config import config
from activetigger.datamodels import (
    LMComputingOutModel,
    LMParametersModel,
    LMStatusModel,
    ModelInformationsModel,
    ModelScoresModel,
    NerModelsProjectStateModel,
    StaticFileModel,
    TaskComputing,
    TextDatasetModel,
)
from activetigger.db.languagemodels import ModelsService
from activetigger.db.manager import DatabaseManager
from activetigger.errors import AlreadyExistsError, NotFoundError
from activetigger.functions import get_model_metrics
from activetigger.task_registry import TaskRegistry


class NerModels:
    """
    Manage fine-tuned token-classification (NER) models for span schemes.

    Mirrors LanguageModels but with a dedicated `ner/` subdirectory on
    disk and a dedicated "ner" DB kind so the two never collide — NER
    models can share a project with BERT classifiers without name clashes.

    TODO: if validated, mutualize code with languagemodel.py
    """

    project_slug: str
    path: Path
    task_registry: TaskRegistry
    models_service: ModelsService
    db_manager: DatabaseManager
    base_models: list[dict[str, Any]]
    params_default: LMParametersModel
    cache_predictions: dict[str, Tuple[datetime, pd.DataFrame]]

    def __init__(
        self,
        project_slug: str,
        path: Path,
        task_registry: TaskRegistry,
        db_manager: DatabaseManager,
        list_models: str | None = None,
    ) -> None:
        self.params_default = LMParametersModel()
        self.project_slug = project_slug

        self.task_registry = task_registry
        self.models_service = db_manager.language_models_service
        self.path = Path(path).joinpath("ner")
        self.cache_predictions = {}
        self._loss_cache: dict[str, Tuple[float, dict | None]] = {}
        self._loss_cache_interval: float = 5

        if list_models is not None and Path(list_models).exists():
            self.base_models = cast(
                list[dict[str, Any]], pd.read_csv(list_models).to_dict(orient="records")
            )
        else:
            # Reuse the BERT base-model defaults — token classification works
            # with the same encoders as sequence classification.
            self.base_models = [
                {
                    "name": "answerdotai/ModernBERT-base",
                    "priority": 10,
                    "comment": "",
                    "language": "en",
                },
                {
                    "name": "camembert/camembert-base",
                    "priority": 0,
                    "comment": "",
                    "language": "fr",
                },
            ]

        os.makedirs(self.path, exist_ok=True)

    def available(self) -> dict[str, dict[str, LMStatusModel]]:
        models = self.models_service.available_models(self.project_slug, "ner")
        r: dict = {}
        for m in models:
            if m.scheme not in r:
                r[m.scheme] = {}
            r[m.scheme][m.name] = LMStatusModel(
                predicted=m.parameters.get("predicted", False),
                predicted_all=m.parameters.get("predicted_all", False),
                tested=m.parameters.get("tested", False),
                predicted_external=m.parameters.get("predicted_external", False),
                name=m.name,
                time=m.time,
                exclude_labels=m.parameters.get("exclude_labels", []),
            )
        return r

    def exists(self, name: str) -> bool:
        return self.models_service.model_exists(self.project_slug, name)

    def training(self) -> dict[str, LMComputingOutModel]:
        return {
            e.user: LMComputingOutModel(
                name=e.inputs.model_name,
                status=e.inputs.status,
                progress=self.task_registry.task_progress(e.id),
                loss=self.get_loss(e.inputs.model_name),
                epochs=e.inputs.params.epochs if isinstance(e.inputs, TrainNerTaskInputs) else None,
            )
            for e in cast(
                list[TaskComputing[TrainNerTaskInputs | PredictNerTaskInputs]],
                self.task_registry.project_computing_tasks(
                    self.project_slug, kinds=["train_ner", "predict_ner"]
                ),
            )
        }

    def estimate_memory_use(self, model: str, kind: str = "train") -> int:
        if kind == "train":
            return 4
        if kind == "predict":
            return 3
        return 0

    def start_training_task(
        self,
        name: str,
        project: str,
        user: str,
        scheme: str,
        df: DataFrame,
        scheme_labels: list[str],
        col_text: str,
        col_label: str,
        params: LMParametersModel,
        base_model: str = "almanach/camembert-base",
        test_size: float = 0.2,
        num_min_annotations: int = 10,
        max_length: int = 512,
    ) -> str:
        if len(df.dropna()) < num_min_annotations:
            raise Exception(f"Less than {num_min_annotations} elements annotated")
        if self.models_service.model_exists(project, name):
            raise AlreadyExistsError("A model with this name already exists")
        if config.cpu_only:
            params.gpu = False
        if params.gpu:
            mem = functions.get_gpu_memory_info()
            if self.estimate_memory_use(name, kind="train") > mem.available_memory:
                raise Exception("Not enough GPU memory available. Wait or reduce batch.")

        # write df on disk on a temporary file read by the worker
        path_df = self.path.joinpath("train_ner_" + uuid.uuid4().hex + ".parquet")
        df.to_parquet(path_df)

        # queue celery task
        inputs = TrainNerTaskInputs(
            user=user,
            path=self.path,
            path_df=path_df,
            project_slug=project,
            model_name=name,
            scheme_labels=scheme_labels,
            col_label=col_label,
            col_text=col_text,
            base_model=base_model,
            params=params,
            test_size=test_size,
            max_length=max_length,
            scheme=scheme,
        )

        del df
        return self.task_registry.start_task(TrainNerTask.name, inputs)

    def start_predicting_process(
        self,
        project_slug: str,
        name: str,
        user: str,
        df: DataFrame | None,
        dataset: str,
        scheme_labels: list[str],
        col_label: str | None = None,
        batch_size: int = 16,
        status: str = "predicting",
        statistics: list | None = None,
        path_data: Path | None = None,
        external_dataset: TextDatasetModel | None = None,
        path_train: Path | None = None,
        path_valid: Path | None = None,
        path_test: Path | None = None,
    ) -> str:
        if not (self.path.joinpath(name)).exists():
            raise NotFoundError("The model does not exist")

        if df is None and dataset not in ["all", "external"]:
            raise Exception("Dataframe is required for this dataset")

        file_name = f"predict_{dataset}.parquet"

        # write df on disk on a temporary file read by the worker
        path_df = None
        if df is not None:
            path_df = self.path.joinpath(name, "predict_ner_" + uuid.uuid4().hex + ".parquet")
            df.to_parquet(path_df)

        # queue celery task
        inputs = PredictNerTaskInputs(
            user=user,
            path=self.path.joinpath(name),
            path_df=path_df,
            project_slug=project_slug,
            model_name=name,
            dataset=dataset,
            scheme_labels=scheme_labels,
            col_text="text",
            col_label=col_label,
            col_id_external="id_external",
            col_datasets="dataset",
            file_name=file_name,
            batch=batch_size,
            statistics=statistics,
            path_data=path_data,
            external_dataset=external_dataset,
            path_train=path_train,
            path_valid=path_valid,
            path_test=path_test,
        )
        # do we need :
        # status=status,
        # training_kind="ner",
        del df
        return self.task_registry.start_task(PredictNerTask.name, inputs)

    def delete(self, name: str) -> None:
        if not name:
            raise ValueError("Model name is empty")
        model_dir = self.path.joinpath(name)
        tar_path = f"{config.data_path}/projects/static/{self.project_slug}/{name}.tar.gz"
        had_files = model_dir.exists() or os.path.exists(tar_path)
        errors = []
        if model_dir.exists():
            try:
                shutil.rmtree(model_dir)
            except Exception as e:
                errors.append(f"model directory: {e}")
        try:
            os.remove(tar_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            errors.append(f"tar archive: {e}")
        db_removed = self.models_service.delete_model(self.project_slug, name)
        if not db_removed and not had_files:
            raise NotFoundError("Model does not exist")
        if errors:
            raise Exception(f"Problem to delete model files : {'; '.join(errors)}")

    def rename(self, former_name: str, new_name: str) -> None:
        model = self.models_service.get_model(self.project_slug, former_name)
        if model is None:
            raise NotFoundError("Model does not exist")
        if (Path(model.path) / "status.log").exists():
            raise Exception("Model is currently computing")
        self.models_service.rename_model(self.project_slug, former_name, new_name)
        model_path = Path(model.path)
        new_path = model_path.parent / model_path.name.replace(former_name, new_name)
        os.rename(model_path, new_path)

    def add_prediction(self, computing_task: TaskComputing[PredictNerTaskInputs]):
        if computing_task.inputs.dataset == "annotable":
            self.models_service.set_model_params(
                self.project_slug, computing_task.inputs.model_name, "predicted", True
            )
        if computing_task.inputs.dataset == "all":
            self.models_service.set_model_params(
                self.project_slug, computing_task.inputs.model_name, "predicted_all", True
            )
        if computing_task.inputs.dataset == "external":
            self.models_service.set_model_params(
                self.project_slug, computing_task.inputs.model_name, "predicted_external", True
            )
        print("NER prediction finished")

    def add_train(self, computing_task: TaskComputing[TrainNerTaskInputs]) -> None:
        params_db = {**computing_task.inputs.params.model_dump(), "exclude_labels": []}
        self.models_service.add_model(
            kind="ner",
            name=computing_task.inputs.model_name,
            user=computing_task.inputs.user,
            project=self.project_slug,
            scheme=computing_task.inputs.scheme or "default",
            params=params_db,
            path=str(self.path.joinpath(computing_task.inputs.model_name)),
            status="trained",
        )
        self.models_service.set_model_params(
            self.project_slug, computing_task.inputs.model_name, "compressed", True
        )
        print("NER model trained")

    def export_prediction(
        self,
        name: str,
        file_name: str = "predict.parquet",
        format: str = "parquet",
        col_id: str | None = None,
    ) -> FileResponse:
        path = self.path.joinpath(name).joinpath(file_name)
        if not path.exists():
            raise FileNotFoundError(
                f"The file {file_name} does not exist for this model, please run prediction again."
            )
        if format not in ("parquet", "csv", "xlsx"):
            raise Exception("Format not supported")
        if format == "parquet":
            return FileResponse(path=path, filename=file_name)
        ext = "csv" if format == "csv" else "xlsx"
        out_name = f"{file_name}.{ext}"
        out_path = self.path.joinpath(name).joinpath(out_name)
        if not out_path.exists() or out_path.stat().st_mtime < path.stat().st_mtime:
            df = pd.read_parquet(path)
            target_col = None
            meta_path = self.path.joinpath(name).joinpath(f"{file_name}.meta.json")
            if meta_path.exists():
                with open(meta_path) as f:
                    target_col = json.load(f).get("col_id")
            elif col_id is not None:
                target_col = col_id.removeprefix("dataset_")
            if target_col is not None and "id_external" in df.columns:
                df.rename(columns={"id_external": target_col}, inplace=True)
                df = df[[target_col] + [c for c in df.columns if c != target_col]]
            if format == "csv":
                df.to_csv(out_path, index=False)
            else:
                df.to_excel(out_path, index=False)
        return FileResponse(path=out_path, filename=out_name)

    def export_ner(self, name: str) -> StaticFileModel:
        file = f"{config.data_path}/projects/static/{self.project_slug}/{name}.tar.gz"
        if not Path(file).exists():
            raise NotFoundError("file does not exist")
        return StaticFileModel(name=f"{name}.tar.gz", path=f"{self.project_slug}/{name}.tar.gz")

    def get_loss(self, model_name: str) -> dict | None:
        now = time.time()
        if model_name in self._loss_cache:
            cached_time, cached_loss = self._loss_cache[model_name]
            if (now - cached_time) < self._loss_cache_interval:
                return cached_loss
        try:
            with open(self.path.joinpath(model_name).joinpath("log_history.txt"), "r") as f:
                log = json.load(f)
            loss = pd.DataFrame(
                [
                    [
                        log[2 * i]["epoch"],
                        log[2 * i]["loss"],
                        log[2 * i + 1]["eval_loss"],
                    ]
                    for i in range(0, int(len(log) / 2))
                ],
                columns=["epoch", "val_loss", "val_eval_loss"],
            ).to_json()
            result = json.loads(loss)
        except Exception:
            result = None
        self._loss_cache[model_name] = (now, result)
        return result

    def get_parameters(self, model_name: str) -> dict | None:
        path = self.path.joinpath(model_name).joinpath("parameters.json")
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    def get_informations(self, model_name: str) -> ModelInformationsModel:
        if not self.exists(model_name):
            raise NotFoundError(f"The model {model_name} does not exist")
        metrics = get_model_metrics(self.path.joinpath(model_name)) or {}
        db_params = self.models_service.get_model_db_parameters(self.project_slug, model_name) or {}
        return ModelInformationsModel(
            params=self.get_parameters(model_name),
            loss=self.get_loss(model_name),
            scores=ModelScoresModel(
                train_scores=metrics.get("train", None),
                internalvalid_scores=metrics.get("trainvalid", None),
                valid_scores=metrics.get("valid", None),
                test_scores=metrics.get("test", None),
                outofsample_scores=metrics.get("outofsample", None),
            ),
            predicted=bool(db_params.get("predicted", False)),
        )

    def get_base_model(self, model_name: str) -> str:
        with open(self.path.joinpath(model_name).joinpath("parameters.json"), "r") as f:
            data = json.load(f)
            if "base_model" in data:
                return data["base_model"]
            raise ValueError("No base_model found in parameters.json.")

    def get_prediction(self, model_name: str, cache_time: int = 120) -> pd.DataFrame:
        for key in list(self.cache_predictions.keys()):
            timestamp, _ = self.cache_predictions[key]
            if (datetime.now(timezone.utc) - timestamp).total_seconds() > cache_time:
                del self.cache_predictions[key]
        if model_name in self.cache_predictions:
            _, cached_df = self.cache_predictions[model_name]
            return cached_df
        path = self.path.joinpath(model_name).joinpath("predict_annotable.parquet")
        if not path.exists():
            raise FileNotFoundError("Prediction file does not exist")
        df = pd.read_parquet(path)
        self.cache_predictions[model_name] = (datetime.now(timezone.utc), df)
        return df

    def state(self) -> NerModelsProjectStateModel:
        return NerModelsProjectStateModel(
            options=self.base_models,
            available=self.available(),
            training=self.training(),
            base_parameters=self.params_default,
        )
