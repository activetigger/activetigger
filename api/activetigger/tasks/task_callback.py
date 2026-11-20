import inspect
from abc import ABC, abstractmethod
from importlib import util
from pathlib import Path
from typing import Any

from task_manager.task_names import TaskNames
from task_manager.utils import (
    FailureCallbackPayload,
    SuccessCallbackPayload,
    TaskFailureReportForCallback,
)

from activetigger.datamodels import EventsModel, TaskComputing, TaskEndedReason
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project
from activetigger.task_registry import TaskRegistry


class ProjectNotTrackedError(Exception):
    """The project of a callback is neither loaded nor being created (deleted, or API restarted)."""


# TaskCallback is a generic class to specify task callback definitions
# TODO: we need two different interface one for Project scope and one for Orchestrator scope
class TaskCallback(ABC):
    task_name: TaskNames

    @classmethod
    def get_project_manager(cls, project_slug: str | None) -> Project | None:
        orchestrator = get_orchestrator()
        if (
            project_slug
            and TaskRegistry.task_scope(cls.task_name) == "project"
            and project_slug in orchestrator.projects
        ):
            return orchestrator.projects[project_slug]
        # TODO: what if the project has been taken down meanwhile
        return None

    @classmethod
    def on_complete(cls, success_payload: SuccessCallbackPayload) -> Any:
        """Callback called by the orchestrator once the task succeeds."""
        task_id = success_payload.task_id
        project_slug = success_payload.project_slug
        task_results = success_payload.results

        print(f"Completion of {cls.task_name} task {task_id} in orchestrator {task_results}")
        orchestrator = get_orchestrator()
        # the reconciliation must not take a task for lost while its result is stored
        orchestrator.task_registry.mark_callback_started(task_id)
        try:
            project_manager = (
                cls.get_project_manager(project_slug)
                if project_slug and TaskRegistry.task_scope(cls.task_name) == "project"
                else None
            )
            reason: TaskEndedReason = "finished"
            error_message: str | None = None
            events: EventsModel | None = None
            # always drop the entry, a raising hook must not leave a stale process
            try:
                computing_task = orchestrator.task_registry.get_computing_task(
                    task_id, project_slug
                )
                cls.on_complete_task_specific(
                    task_id, task_results, project_manager, computing_task
                )
                events = (
                    EventsModel(events=task_results["events"]) if "events" in task_results else None
                )

            except Exception as ex:
                reason = "error"
                # surface the error to the user instead of a 500 the worker would retry
                error_message = f"Error in Task callback {cls.task_name} ({task_id}): {ex}"
                print(error_message)

            finally:
                orchestrator.task_registry.on_task_end(
                    task_id, project_slug, reason, events, error_message=error_message
                )
        finally:
            orchestrator.task_registry.mark_callback_finished(task_id)

    @classmethod
    @abstractmethod
    def on_complete_task_specific(
        cls,
        task_id: str,
        task_result: Any,
        project_manager: Project | None,
        computing_task: TaskComputing,
    ) -> None:
        """Specific Callback for one task called by the generic on_complete method."""
        raise NotImplementedError

    @classmethod
    def on_failure(cls, failure_payload: FailureCallbackPayload):
        task_id = failure_payload.task_id
        task_report = failure_payload.report
        print(f"Error on {cls.task_name} task {task_id} in orchestrator {task_report}")
        orchestrator = get_orchestrator()
        orchestrator.task_registry.mark_callback_started(task_id)

        project_slug = failure_payload.project_slug
        try:
            project_manager = cls.get_project_manager(project_slug)
            specific_message = cls.on_failure_task_specific(task_id, task_report, project_manager)

            # TODO: we have two different place where we trigger error message for task here and in orchestrator._on_task_ended
            # TODO: a stopped task is not an error (same as the in-process "cancelled" state)
            # if "Process interrupted by user" in task_report.exception or was_stopped(task_id):
            if project_manager:
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
                # worker misconfiguration, not a memory problem
                elif "Cannot re-initialize CUDA in forked subprocess" in task_report.exception:
                    project_manager.errors.add(
                        (
                            f"Error for task {cls.task_name} : CUDA was initialised in the worker "
                            "parent process, so GPU tasks cannot run in the prefork children. "
                            "Check the worker logs (TASKMANAGER_TRACE_CUDA_INIT=1 shows the cause) "
                            "or run the GPU worker with --pool=solo."
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
        finally:
            orchestrator.task_registry.mark_callback_finished(task_id)
            orchestrator.task_registry.on_task_end(task_id, project_slug, reason="error")

    @classmethod
    @abstractmethod
    def on_failure_task_specific(
        cls,
        task_id: str,
        task_report: TaskFailureReportForCallback,
        project_manager: Project | None,
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
