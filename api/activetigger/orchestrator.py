"""
Define the orchestrator and launch the instance that will span projects
Each project share a few elements:
- database manager
- users management
- messages management
"""

import asyncio
import os
import shutil
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import pyarrow.parquet as pq
from fastapi.encoders import jsonable_encoder
from jose import jwt
from task_manager.tasks.create_project_task import CreateProjectTask, CreateProjectTaskInputs
from task_manager.tasks.duplicate_project_task import (
    DuplicateProjectTask,
    DuplicateProjectTaskInputs,
)

from activetigger import __version__
from activetigger.config import config
from activetigger.datamodels import (
    AuthUserModel,
    DatasetModel,
    ProjectBaseModel,
    ProjectModel,
    ProjectStatus,
    ServerStateModel,
    TaskEndedEvent,
)
from activetigger.db import DBException
from activetigger.db.manager import DatabaseManager
from activetigger.errors import (
    AlreadyExistsError,
    InvalidInputError,
    NotFoundError,
    ServerBusyError,
)
from activetigger.functions import (
    get_dir_size,
    get_gpu_memory_info,
    sanitize_uploaded_filename,
    slugify,
)
from activetigger.generation.generations import Generations
from activetigger.messages import Messages
from activetigger.monitoring import Monitoring
from activetigger.project import Project, _detect_span_scheme
from activetigger.task_registry import TaskRegistry
from activetigger.tasks.task_manager_client import (
    purge_celery_tasks,
)
from activetigger.toolbox import Toolbox
from activetigger.uploads import get_upload_staging
from activetigger.users import Users


class Orchestrator:
    """
    Process to manage projects
    Use the config object to get parameters
    """

    starting_time: float
    db_name: str
    jwt_algorithm: str
    path: Path
    path_models: Path
    projects: dict[str, Project]
    db_manager: DatabaseManager
    users: Users
    messages: Messages
    max_projects: int

    task_registry: TaskRegistry
    monitoring: Monitoring
    toolbox: Toolbox

    def __init__(self) -> None:
        """
        Start the server
        """
        self.starting_time = time.time()

        self.max_projects = config.max_loaded_projects
        self.jwt_algorithm = config.jwt_algorithm

        # Define path
        self.path: Path = Path(config.data_path) / "projects"
        self.path_models: Path = Path(config.data_path) / "models"
        self.path_toy_datasets: Path = Path(self.path) / "toy-datasets"

        # create directories parent/static/models
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            (self.path.joinpath("static")).mkdir(parents=True, exist_ok=True)
            self.path_models.mkdir(exist_ok=True)
            self.path_toy_datasets.mkdir(exist_ok=True)
        except Exception as e:
            print(e)
            raise PermissionError(
                f"Cannot create directory: {e}. "
                f"Please ensure the data directory '{config.data_path}' is writable "
                f"by the current user (uid={os.getuid()}, gid={os.getgid()}). "
                f"Fix with: chown -R {os.getuid()}:{os.getgid()} {config.data_path}"
            ) from e

        # attributes of the server
        self.db_manager = DatabaseManager()
        self.messages = Messages(self.db_manager)
        self.users = Users(self.db_manager, self.messages)
        self.monitoring = Monitoring(self.db_manager)

        # projects in memory
        self.projects = {}
        self.task_registry = TaskRegistry(monitoring=self.monitoring)
        self.toolbox = Toolbox(self.task_registry)
        # subscribe orchestrator-level bookkeeping in task registry
        self.task_registry.subscribe_task_end(self._on_task_ended)
        # store creation errors so they survive project cleanup
        self.creation_errors: dict[str, str] = {}

        # cached heavy system stats (refreshed every _heavy_stats_interval seconds)
        self._heavy_stats_interval = 30
        self._heavy_stats_last_update = 0.0
        self._heavy_stats_cache: dict = self._collect_heavy_stats()

        # update the projects asynchronously; if no loop is running yet
        # (module-import-time init from main.py), defer to ensure_update_task()
        # which the FastAPI lifespan will call once the loop is up.
        self._running = True
        self._update_task: asyncio.Task | None = None
        self.ensure_update_task()

        # drop tasks left by a previous API process
        try:
            counts = purge_celery_tasks()
            print(f"Celery tasks purged at startup: {counts}")
        except Exception as e:
            print(f"Error while purging celery tasks at startup: {e}")

        # create the demo project if not existing at startup
        try:
            if "demo" not in self.existing_projects():
                self.create_demo_project("demo", "demo")
        except Exception as e:
            print(f"Error while creating demo project: {e}")
        self.server_state = self.get_server_state()

    def ensure_update_task(self) -> None:
        """
        Schedule the background _update loop if it isn't already running.
        Safe to call repeatedly. Used by the FastAPI lifespan to (re)start
        the loop when the orchestrator was instantiated outside one
        (module-import-time init, tests).
        """
        if self._update_task is not None and not self._update_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._update_task = None
            return
        self._update_task = loop.create_task(self._update(timeout=config.update_timeout))

    def __del__(self):
        """
        Close the server
        """
        print("Ending the server")
        if self._update_task:
            self._update_task.cancel()
        self._running = False
        print("Server off")

    def reset(self, username: str = "system"):
        """
        Stop every task and erase the projects in memory
        """
        self.task_registry.stop_all_tasks()
        # explicit del to reset memory
        for p in self.projects.values():
            del p
        self.projects = {}

    def _new_project(self, project_slug: str) -> Project:
        """
        Instantiate a project on its orchestrator-owned computing list
        """
        p = Project(
            project_slug,
            self.db_manager,
            task_registry=self.task_registry,
            path_models=self.path_models,
            users=self.users,
            messages=self.messages,
        )
        return p

    def starting_project_creation(self, project: ProjectBaseModel, username: str) -> str:
        """
        Start the project creation
        """
        project_slug = self.check_project_name(project.project_name)
        if project_slug in ["new", "logs"]:
            raise InvalidInputError("This project name is not valid - reserved word")
        # the upload endpoint sanitized the filename on disk; resolve the
        # client-provided name to the same form or the task won't find the file
        if project.filename is not None:
            project.filename = sanitize_uploaded_filename(project.filename)

        # test if the name of the column is specified
        if project.col_id is None or project.col_id == "":
            raise InvalidInputError("No column selected for the id")
        if project.cols_text is None or len(project.cols_text) == 0:
            raise InvalidInputError("No column selected for the text")

        project.dir = self.path.joinpath(project_slug)

        # start a create project task
        task_inputs = CreateProjectTaskInputs(
            image_project=getattr(project, "kind", "text") == "image",
            project_slug=project_slug,
            params=project,
            user=username,
            data_all=config.data_all,
            train_file=config.train_file,
            valid_file=config.valid_file,
            test_file=config.test_file,
            features_file=config.features_file,
            random_seed=project.seed,
        )

        self.task_registry.start_task(CreateProjectTask.name, task_inputs)
        return project_slug

    def _sync_update_processes(self, project_lifetime: int) -> None:
        """
        Synchronous housekeeping for the update loop (runs in a thread to avoid
        blocking the async event loop); task results arrive through the callbacks.
        """
        timer = time.time()

        # evict idle projects, never one with tasks in flight
        to_del = []
        for p, project in list(self.projects.items()):
            if (timer - project.starting_time) > project_lifetime:
                if p in self.task_registry.projects_with_running_tasks():
                    project.starting_time = timer
                else:
                    to_del.append(p)
        for p in to_del:
            del self.projects[p]

        # drop abandoned chunked-upload staging sessions
        try:
            get_upload_staging().clean_old()
        except Exception as e:
            print(f"Error while cleaning staged uploads: {e}")

        try:
            self.task_registry.maybe_reconcile()
        except Exception as e:
            print(f"Error while reconciling tasks with the workers: {e}")

    def _collect_heavy_stats(self) -> dict:
        """
        Collect expensive system metrics (CPU, memory, disk, GPU).
        Designed to run in a background thread.
        """
        cpu = psutil.cpu_percent()
        cpu_count = psutil.cpu_count()
        memory_info = psutil.virtual_memory()
        # measure the filesystem holding the data (external volume in Docker),
        # not the container root
        disk_info = psutil.disk_usage(config.data_path)
        at_memory = get_dir_size(config.data_path + "/projects")
        gpu = get_gpu_memory_info()
        return {
            "cpu": {"proportion": cpu, "total": cpu_count},
            "memory": {
                "proportion": memory_info.percent,
                "total": memory_info.total / (1024**3),
                "available": memory_info.available / (1024**3),
            },
            "disk": {
                "activetigger": at_memory,
                "proportion": disk_info.percent,
                "total": disk_info.total / (1024**3),
            },
            "gpu": gpu,
        }

    async def _update(self, timeout: int = 1, project_lifetime: int = 7200) -> None:
        """
        Update each project in memory every X seconds.
        - loaded project
        - project in creation
        Remove projects that are older than project_lifetime seconds.
        Heavy system stats are refreshed every _heavy_stats_interval seconds.
        All synchronous work runs in a thread to avoid blocking the event loop.
        """
        try:
            while self._running:
                try:
                    # run synchronous update_processes in a thread
                    await asyncio.to_thread(self._sync_update_processes, project_lifetime)

                    # refresh heavy system stats periodically in a thread
                    now = time.time()
                    if (now - self._heavy_stats_last_update) >= self._heavy_stats_interval:
                        self._heavy_stats_cache = await asyncio.to_thread(self._collect_heavy_stats)
                        self._heavy_stats_last_update = now

                    # build server state (cheap parts only, heavy stats come from cache)
                    self.server_state = self.get_server_state()
                except Exception as e:
                    print(f"Error in update loop iteration: {e}")
                    traceback.print_exc()

                await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            print("Update task cancelled.")
        finally:
            print("Update task finished.")

    def log_action(
        self,
        user: str,
        action: str,
        project: str = "general",
        connect="not implemented",
    ) -> None:
        """
        Log action in the database
        """
        self.db_manager.logs_service.add_log(user, action, project, connect)

    def get_logs(self, project_slug: str, limit: int, partial: bool = True) -> pd.DataFrame:
        """
        Get logs for a user/project
        project_slug: project slug or "all"
        """
        logs = self.db_manager.logs_service.get_logs("all", project_slug, limit)
        df = pd.DataFrame(logs, columns=["id", "time", "user", "project", "action", "NA"])
        if partial:
            return df[~df["action"].str.contains("INFO ")]
        return df

    def get_server_state(self) -> ServerStateModel:
        """
        Build server state.
        Light data (active projects, queue) is computed inline.
        Heavy system stats (CPU, memory, disk, GPU) come from the
        periodically refreshed cache (_heavy_stats_cache).
        """
        # active projects in the orchestrator
        active_projects = {
            p: [
                {"unique_id": c.id, "user": c.user, "kind": c.kind, "time": c.inputs.time}
                for c in computing_tasks_by_id.values()
            ]
            for p, computing_tasks_by_id in self.task_registry.projects_computing_tasks().items()
            if computing_tasks_by_id
        }
        # every process in flight on the celery workers
        queue = {
            c.id: {"unique_id": c.id, "kind": c.kind, "state": "running"}
            for c in self.task_registry.all_computing_tasks()
        }

        # use cached heavy stats
        stats = self._heavy_stats_cache

        return ServerStateModel(
            version=__version__,
            mode=str(config.mode),
            cpu_only=config.cpu_only,
            active_projects=active_projects,
            queue=queue,
            gpu=stats["gpu"],
            cpu=stats["cpu"],
            memory=stats["memory"],
            disk=stats["disk"],
            mail_available=self.messages.mail_available,
            messages=self.messages.get_messages_system(),
        )

    def exists(self, project_name: str, include_toy_datasets: bool = False) -> bool:
        """
        Test if a project exists in the database
        with a sluggified form (to be able to use it in URL)
        """
        return slugify(project_name) in self.existing_projects()

    def check_project_name(self, project_name: str) -> str:
        """
        Check if a project name is valid
        """
        project_slug = slugify(project_name)
        if self.exists(project_slug):
            raise AlreadyExistsError("This project already exists")
        if project_slug == "":
            raise InvalidInputError("The project name is not valid - empty")
        return project_slug

    def check_project_status(self, project_slug: str) -> ProjectStatus:
        """
        Check project status
        """
        if not self.exists(project_slug):
            return "not existing"
        if project_slug in self.task_registry.project_duplication_ongoing_computing_tasks():
            return "duplicating"

        if project_slug in self.task_registry.project_creation_ongoing_computing_tasks():
            return "creating"

        # down
        if project_slug not in self.projects:
            return "down"

        return "up"

    def create_access_token(self, data: dict, expires_min: int = 60) -> str:
        """
        Create access token
        """
        # create the token
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(minutes=expires_min)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, config.secret_key, algorithm=self.jwt_algorithm)

        # add it in the database as active
        self.db_manager.projects_service.add_token(encoded_jwt, "active")

        # logins are rare enough to piggyback the tokens-table cleanup here
        try:
            self.db_manager.projects_service.prune_old_tokens()
        except Exception as e:
            print(f"Error while pruning old tokens: {e}")

        return encoded_jwt

    def revoke_access_token(self, token) -> None:
        """
        Revoke existing access token
        """
        self.db_manager.projects_service.revoke_token(token)

    def decode_access_token(self, token: str) -> dict:
        """
        Decode access token
        """
        # get status
        try:
            status = self.db_manager.projects_service.get_token_status(token)
        except DBException as e:
            raise Exception from e

        if status != "active":
            raise Exception("Token is invalid")

        # decode payload
        payload = jwt.decode(token, config.secret_key, algorithms=[self.jwt_algorithm])
        return payload

    def manage_fifo_queue(self) -> None:
        """
        Manage the current projects in memory for an orchestrator
        """
        if len(self.projects) >= self.max_projects:
            # only idle projects are stopable
            idle_projects = sorted(
                [
                    p
                    for p in self.projects
                    # check if project has computing_tasks
                    if p not in self.task_registry.projects_with_running_tasks()
                    # check if the project has a least ten minutes old to avoid destroying current projects
                    and self.projects[p].starting_time < time.time() - 600
                ],
                key=lambda p: self.projects[p].starting_time,
            )
            if len(idle_projects) > 0:
                self.stop_project(idle_projects[0])
            else:
                print("Too many projects in the current memory")
                raise ServerBusyError(
                    "There is too many projects currently loaded in this server. Please wait"
                )

    def start_project(self, project_slug: str) -> None:
        """
        Load project in server
        """
        if not self.exists(project_slug):
            raise NotFoundError("This project does not exist")

        try:
            self.projects[project_slug] = self._new_project(project_slug)
        except Exception as e:
            raise Exception(
                f"Error while loading project {project_slug}: {e} - {traceback.format_exc()}"
            ) from e

    def stop_project(self, project_slug: str) -> None:
        """
        Unload a project from memory; refused while it has tasks in flight
        """
        if project_slug not in self.projects:
            return None
        if project_slug in self.task_registry.projects_with_running_tasks():
            raise InvalidInputError(
                "This project has tasks in progress, stop them before closing it"
            )
        print(f"Stop project {project_slug} to gain memory")

        del self.projects[project_slug]

    def _on_task_ended(self, event: TaskEndedEvent) -> None:
        """Project-level side effects of any task leaving the registry."""
        ct = event.computing_task
        username = ct.user  # or track the stopping user separately

        # something went wrong, let's clean
        if event.reason in ("stopped", "lost", "error"):
            self.log_action(
                username, f"TASK {event.reason.upper()}: {ct.id}", ct.project_slug or "all"
            )

            #  creation / duplication lifecycle
            if ct.kind == "create_project" and ct.project_slug:
                if event.reason in ("stopped", "lost", "error"):
                    self.creation_errors[ct.project_slug] = (
                        "Project creation stopped by user"
                        if event.stopped_by_user
                        else f"Creation lost ({event.error_message})"
                    )
                    project = self.projects.get(ct.project_slug)
                    if project is not None:
                        project.status = "error"
                        project.errors.add(self.creation_errors[ct.project_slug])
                self.clean_unfinished_project(project_slug=ct.project_slug)

            #  training lifecycle: killed/lost training leaves a phantom row
            elif ct.kind in ("train_bert", "train_image", "train_ner") and ct.project_slug:
                if event.reason in ("stopped", "lost"):
                    try:
                        self.db_manager.language_models_service.delete_model(
                            ct.project_slug, ct.inputs.model_name
                        )
                    except Exception as e:
                        print(f"Could not delete model after {event.reason} {ct.kind}: {e}")
            # Surface the error

        #  generic surfacing for lost tasks
        if event.error_message:
            project = self.projects.get(ct.project_slug, None)
            if project is not None:
                project.errors.add(f"Task {ct.kind} {event.reason}: {event.error_message}")

        # TODO: here would be a goof place to generate a success message

    def existing_projects(self) -> list:
        """
        Get existing projects
        """
        return self.db_manager.projects_service.existing_projects()

    def delete_project(self, project_slug: str, username: str = "system") -> None:
        """
        Delete a project
        """

        # test if the project exists
        if not self.exists(project_slug):
            raise NotFoundError("This project does not exist")

        # load the project
        if project_slug in self.projects:
            project = self.projects[project_slug]
        else:
            self.start_project(project_slug)
            project = self.projects[project_slug]

        # workers must not keep writing into the deleted directory
        self.task_registry.stop_project_tasks(project_slug, username)

        # clean current memory
        self.stop_project(project_slug)

        # delete the project
        try:
            project.delete()
        except Exception as e:
            raise Exception from e

    def duplicate_project(self, source_slug: str, username: str) -> str:
        """
        Kick off the duplication of an existing project. Non-blocking.

        Sends a celery task that copies the source project directory
        (data, features, trained model files…) and any static export files.
        The task callback clones the DB rows (see DuplicateProjectCallback).
        Returns the target slug immediately so callers can poll `/projects/status`.
        """
        if not self.exists(source_slug):
            raise NotFoundError("Source project does not exist")

        source_record = self.db_manager.projects_service.get_project(source_slug)
        if source_record is None:
            raise NotFoundError("Source project not found in database")
        source_params: dict[str, Any] = dict(source_record["parameters"])
        source_dir = source_params.get("dir")
        if not source_dir:
            raise Exception("Source project has no directory")
        source_dir_str = str(source_dir)
        if not Path(source_dir_str).exists():
            raise Exception(f"Source project directory missing on disk: {source_dir_str}")

        # find an available target slug, accounting for in-flight duplications
        existing = set(self.existing_projects())
        existing.update(self.task_registry.project_duplication_ongoing_computing_tasks())
        existing.update(self.task_registry.project_creation_ongoing_computing_tasks())

        base = f"{source_slug}-copy"
        target_slug = base
        n = 2
        while target_slug in existing or self.path.joinpath(target_slug).exists():
            target_slug = f"{base}-{n}"
            n += 1

        target_dir = self.path.joinpath(target_slug)
        target_dir_str = str(target_dir)

        # build target parameters: keep everything from source but retarget
        # slug/name/dir so the new project loads against the new directory
        target_params = dict(source_params)
        target_params["project_slug"] = target_slug
        source_name = source_params.get("project_name") or source_slug
        target_params["project_name"] = f"{source_name} (copy)"
        target_params["dir"] = target_dir_str

        source_static = f"{config.data_path}/projects/static/{source_slug}"
        target_static = f"{config.data_path}/projects/static/{target_slug}"
        has_static = Path(source_static).exists()

        try:
            task_id = self.task_registry.start_task(
                DuplicateProjectTask.name,
                DuplicateProjectTaskInputs(
                    user=username,
                    project_slug=target_slug,
                    source_slug=source_slug,
                    target_slug=target_slug,
                    username=username,
                    source_dir=Path(source_dir_str),
                    target_dir=target_dir,
                    source_static_dir=Path(source_static) if has_static else None,
                    target_static_dir=Path(target_static) if has_static else None,
                    target_params=target_params,
                ),
            )

        except Exception as e:
            raise Exception(f"Failed to start duplication: {e}") from e

        return target_slug

    def clean_unfinished_project(
        self, project_slug: str | None = None, project_name: str | None = None
    ) -> None:
        """
        Clean a project that is not loadable
        Careful : this will delete all the data of the project
        """

        if project_slug is not None:
            project_slug_verif = project_slug
        elif project_name is not None:
            project_slug_verif = slugify(project_name)
        else:
            raise Exception("No project specified")

        self.task_registry.stop_project_tasks(project_slug_verif)

        # remove from database
        self.db_manager.projects_service.delete_project(project_slug_verif)

        # remove from the server
        if project_slug_verif and project_slug_verif != "":
            shutil.rmtree(self.path.joinpath(project_slug_verif), ignore_errors=True)

            ## remove static files
            if Path(f"{config.data_path}/projects/static/{project_slug_verif}").exists():
                shutil.rmtree(f"{config.data_path}/projects/static/{project_slug_verif}")

    def create_demo_project(self, project_name: str, username: str) -> None:
        """
        Create a demo project for a specific user
        """
        # TODO : put those elements in the config file
        path_data = Path("../frontend/public/gwsd_train_test.csv")
        col_id = "guid"
        col_text = "sentence"
        col_label = "label"

        if not path_data.exists():
            raise NotFoundError("The demo dataset is not available")

        # create name of the project and the directory
        project_name = project_name
        project_slug = slugify(project_name)
        project_path = self.path.joinpath(project_slug)
        os.makedirs(project_path, exist_ok=True)

        # create and save a toy dataset
        df = pd.read_csv(path_data)
        df.to_csv(project_path.joinpath("dataset.csv"), index=False)

        # parameters of the project and creation
        project = ProjectBaseModel(
            project_name=project_name,
            language="en",
            filename="dataset.csv",
            col_id=col_id,
            cols_text=[col_text],
            cols_label=[col_label],
            n_train=1000,
            n_test=500,
        )
        self.starting_project_creation(project, username)

    def available_storage(self, username: str) -> bool:
        """
        Check if the user storage is not exceeded
        """
        limit = self.users.get_storage_limit(username)
        if self.users.get_storage(username) > limit:
            return False
        return True

    def get_toy_datasets(self) -> list[DatasetModel]:
        """
        Get the name of available toy datasets
        """
        toy_datasets = []
        for file in os.listdir(self.path_toy_datasets):
            if file.endswith(".parquet"):
                toy_dataset_name = file.removesuffix(".parquet")
                pq_file = pq.ParquetFile(self.path_toy_datasets.joinpath(file))
                toy_datasets += [
                    DatasetModel(
                        project_slug=toy_dataset_name,
                        columns=[col.name for col in list(pq_file.schema)],
                        n_rows=pq_file.metadata.num_rows,
                    )
                ]
        return toy_datasets

    def finish_project_creation(
        self,
        username: str,
        project: ProjectModel,
        import_trainset_labels: pd.DataFrame | None = None,
        import_validset_labels: pd.DataFrame | None = None,
        import_testset_labels: pd.DataFrame | None = None,
    ) -> None:
        """
        Finish the creation from the worker result (called by the task callback)
        """
        # add the project to the database
        self.db_manager.projects_service.add_project(
            project.project_slug, jsonable_encoder(project), username
        )

        # add the default scheme if needed
        if import_trainset_labels is None or len(import_trainset_labels.columns) == 0:
            self.db_manager.projects_service.add_scheme(
                project.project_slug, config.default_scheme, [], "multiclass", "system"
            )

        # if labels/schemes to import, add them to the database
        else:
            for col in import_trainset_labels.columns:
                scheme_name = col.replace("dataset_", "")
                non_null = import_trainset_labels[col].dropna().astype(str)
                span_kind, span_labels = _detect_span_scheme(non_null)
                if span_kind:
                    scheme_type = "span"
                    scheme_labels = span_labels
                else:
                    delimiters = non_null.str.contains("|", regex=False).sum()
                    if delimiters < 5:
                        scheme_type = "multiclass"
                        scheme_labels = list(non_null.unique())
                    else:
                        scheme_type = "multilabel"
                        scheme_labels = list(non_null.str.split("|").explode().unique())
                self.db_manager.projects_service.add_scheme(
                    project.project_slug,
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
                    project_slug=project.project_slug,
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
                        project_slug=project.project_slug,
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
                        project_slug=project.project_slug,
                        scheme=scheme_name,
                        elements=elements,
                    )

        # add user authorizations
        self.users.set_auth(
            AuthUserModel(username=username, project_slug=project.project_slug, status="manager")
        )

        # pre-populate the project with the generative models declared in
        # generative.yaml (no-op if the file is missing or empty)
        try:
            Generations(
                project.project_slug, self.db_manager, self.task_registry
            ).add_default_models(project.project_slug, username)
        except Exception as e:
            print(f"Failed to add default generative models for {project.project_slug}: {e}")


_orchestrator: Orchestrator | None = None


# TODO: remove get orchestrator pass orchestrator instance in controllers from fastApi dependency or nested in project
def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
