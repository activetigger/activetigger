import inspect
from abc import ABC, abstractmethod
from importlib import util
from pathlib import Path
from typing import Any

from task_manager.utils import TaskFailureReportForCallback

from activetigger.datamodels import ProcessComputing
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project
from activetigger.tasks.task_manager_client import was_stopped


class ProjectNotTrackedError(Exception):
    """The project of a callback is neither loaded nor being created (deleted, or API restarted)."""


# TaskCallback is a generic class to specify task callback definitions
class TaskCallback(ABC):
    task_name: str

    @classmethod
    def get_project_slug(cls, task_result: Any) -> str:
        """Where the task result stores the project slug (override if not top-level)."""
        return str(task_result["project_slug"])

    @staticmethod
    def get_project_manager(project_slug: str) -> Project:
        orchestrator = get_orchestrator()
        if project_slug in orchestrator.projects:
            return orchestrator.projects[project_slug]
        # a project still being created is not yet in orchestrator.projects
        if project_slug in orchestrator.project_creation_ongoing:
            return orchestrator.project_creation_ongoing[project_slug]
        raise ProjectNotTrackedError(
            f"Project {project_slug} not listed in orchestrator, we can't finish task"
        )

    @classmethod
    def on_complete(cls, task_id: str, task_result: Any) -> Any:
        """Callback called by the orchestrator once the task succeeds."""
        print(f"Completion of {cls.task_name} task {task_id} in orchestrator {task_result}")
        project_manager = cls.get_project_manager(cls.get_project_slug(task_result))
        # copy: clean_process mutates the list
        for process in list(project_manager.computing):
            if process.unique_id == task_id:
                # always drop the entry, a raising hook must not leave a stale process
                try:
                    cls.on_complete_task_specific(task_id, task_result, project_manager, process)
                finally:
                    project_manager.clean_process(process)

    @classmethod
    @abstractmethod
    def on_complete_task_specific(
        cls, task_id: str, task_result: Any, project_manager: Project, process: ProcessComputing
    ) -> None:
        """Specific Callback for one task called by the generic on_complete method."""
        raise NotImplementedError

    @classmethod
    def on_failure(cls, task_id: str, task_report: TaskFailureReportForCallback):
        print(f"Error on {cls.task_name} task {task_id} in orchestrator {task_report}")
        # cast first args as Task Input type
        task_input = task_report.task_args[0]
        project_slug = task_input["project_slug"]
        project_manager = cls.get_project_manager(project_slug)
        for e in list(project_manager.computing):
            if e.unique_id == task_id:
                project_manager.clean_process(e)
        specific_message = cls.on_failure_task_specific(task_id, task_report, project_manager)
        # a stopped task is not an error (same as the in-process "cancelled" state)
        if "Process interrupted by user" in task_report.exception or was_stopped(task_id):
            # except for a creation: the loop only cleans a creating project on "error"
            if project_slug in get_orchestrator().project_creation_ongoing:
                project_manager.errors.add("Project creation stopped by user")
                project_manager.status = "error"
            return
        project_manager.status = "error"
        if specific_message:
            project_manager.errors.add(specific_message)
        # a pool child killed by the kernel: almost always RAM exhaustion
        if "WorkerLostError" in task_report.exception or "SIGKILL" in task_report.exception:
            project_manager.errors.add(
                (
                    f"Error for task {cls.task_name} : the worker process was killed, "
                    "most likely because the machine ran out of memory (RAM). "
                    "Try a smaller model, a smaller batch size or a shorter max length. "
                    f"Details: {task_report.exception}"
                )
            )
        # generic message for GPU errors
        elif any(
            s in task_report.exception
            for s in [
                "CUDA",
                "CUDACachingAllocator",
                "out of memory",
                "NVML",
                "cuda",
            ]
        ):
            project_manager.errors.add(
                (
                    f"Error for task {cls.task_name} : GPU error — not enough GPU memory available. "
                    "Try reducing the batch size, the max sequence length, or using a smaller model. "
                    f"Details: {task_report.exception}"
                )
            )
        elif specific_message is None:
            project_manager.errors.add(
                f"Error for process {cls.task_name} : {task_report.exception}"
            )

    @classmethod
    @abstractmethod
    def on_failure_task_specific(
        cls, task_id: str, task_report: TaskFailureReportForCallback, project_manager: Project
    ) -> str | None:
        """Specific callback for failure case. If a specific message is return it will replace the generic exception."""
        return None


# discover all TaskCallback implementations to collect on_complete methods
def discover_task_callbacks(directory_path: str):
    discovered_tasks: list[TaskCallback] = []
    for file in Path(directory_path).glob("*.py"):
        if file.name.startswith("_"):
            continue

        module_name = file.stem
        spec = util.spec_from_file_location(module_name, file)
        if spec and spec.loader:
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, TaskCallback) and obj is not TaskCallback:
                    discovered_tasks.append(obj)  # ty:ignore[invalid-argument-type]
    return discovered_tasks
