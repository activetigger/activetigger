import shutil

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.duplicate_project_task import (
    DuplicateProjectTask,
    DuplicateProjectTaskResult,
)
from task_manager.utils import TaskFailureReportForCallback

from activetigger.orchestrator import get_orchestrator

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class DuplicateProjectCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = DuplicateProjectTask.name

    # the target project does not exist yet, so the generic project hooks do not apply
    @classmethod
    def on_complete(cls, task_id: str, task_result: DuplicateProjectTaskResult):
        orchestrator = get_orchestrator()
        target_slug = task_result["target_slug"]
        try:
            # filesystem copy succeeded — clone the DB rows
            orchestrator.db_manager.projects_service.duplicate_project(
                source_slug=task_result["source_slug"],
                target_slug=target_slug,
                target_parameters=task_result["target_params"],
                user_name=task_result["username"],
                source_dir=task_result["source_dir"],
                target_dir=task_result["target_dir"],
            )
            orchestrator.log_action(
                task_result["username"],
                f"DUPLICATE PROJECT: {task_result['source_slug']} -> {target_slug}",
                target_slug,
            )
        except Exception as e:
            cls._clean_target(task_result["target_dir"], task_result["target_static_dir"])
            orchestrator.creation_errors[target_slug] = f"DB duplication failed: {e}"
        finally:
            orchestrator.project_duplication_ongoing.pop(target_slug, None)

    @classmethod
    def on_failure(cls, task_id: str, task_report: TaskFailureReportForCallback):
        orchestrator = get_orchestrator()
        task_input = task_report.task_args[0]
        target_slug = task_input["target_slug"]
        cls._clean_target(task_input["target_dir"], task_input.get("target_static_dir"))
        if "Process interrupted by user" in task_report.exception:
            orchestrator.creation_errors[target_slug] = "Duplication cancelled"
        else:
            orchestrator.creation_errors[target_slug] = (
                f"Duplication failed: {task_report.exception}"
            )
        orchestrator.project_duplication_ongoing.pop(target_slug, None)

    @staticmethod
    def _clean_target(target_dir: str, target_static_dir: str | None) -> None:
        shutil.rmtree(target_dir, ignore_errors=True)
        if target_static_dir:
            shutil.rmtree(target_static_dir, ignore_errors=True)

    @classmethod
    def on_complete_task_specific(cls, task_id, task_result, project_manager, process):
        pass

    @classmethod
    def on_failure_task_specific(cls, task_id, task_report, project_manager):
        return None
