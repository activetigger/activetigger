from pathlib import Path
from typing import Any, TypedDict

from task_manager.active_tigger_task import AbortableActiveTiggerTask, QueueName, TaskScope
from task_manager.celery import celery_app
from task_manager.task_input import TaskInputs
from task_manager.task_names import TaskNames
from task_manager.tasks.duplicate_project import duplicate_project_files


class DuplicateProjectTaskInputs(TaskInputs[str]):
    # everything the callback needs to clone the DB rows, so it survives an API restart
    source_slug: str
    target_slug: str
    username: str
    source_dir: Path
    target_dir: Path
    source_static_dir: Path | None = None
    target_static_dir: Path | None = None
    target_params: dict[str, Any]


# Return type must be a dict, a BaseModel would not be serialized by Celery
class DuplicateProjectTaskResult(TypedDict):
    source_slug: str
    target_slug: str
    username: str
    source_dir: str
    target_dir: str
    target_static_dir: str | None
    target_params: dict[str, Any]


class DuplicateProjectTask(AbortableActiveTiggerTask):
    name: TaskNames = "duplicate_project"
    queue = QueueName.CPU
    scope: TaskScope = "orchestrator"


@celery_app.task(
    bind=True,
    name=DuplicateProjectTask.name,
    queue=DuplicateProjectTask.queue,
    base=DuplicateProjectTask,
    pydantic=True,
)
def duplicate_project_task(
    self: DuplicateProjectTask, inputs: DuplicateProjectTaskInputs
) -> DuplicateProjectTaskResult:
    # a copy cannot be interrupted midway, only skipped before it starts
    if self.is_aborted():
        raise Exception("Process interrupted by user")
    duplicate_project_files(
        inputs.source_dir,
        inputs.target_dir,
        inputs.source_static_dir,
        inputs.target_static_dir,
    )
    return {
        "source_slug": inputs.source_slug,
        "target_slug": inputs.target_slug,
        "username": inputs.username,
        "source_dir": str(inputs.source_dir),
        "target_dir": str(inputs.target_dir),
        "target_static_dir": str(inputs.target_static_dir) if inputs.target_static_dir else None,
        "target_params": inputs.target_params,
    }
