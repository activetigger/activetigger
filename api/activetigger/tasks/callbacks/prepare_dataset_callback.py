import logging

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.prepare_dataset_task import PrepareDatasetTask
from task_manager.utils import TaskFailureReportForCallback

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback

logger = logging.getLogger("activetigger.fastapi.tasks")


class PrepareDatasetCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = PrepareDatasetTask.name

    # toolbox tasks have no project: the result is on disk and Toolbox.status
    # reads the celery state, so both hooks only log
    @classmethod
    def on_complete(cls, task_id, task_result):
        logger.info(f"prepare dataset {task_id} done: {task_result}")

    @classmethod
    def on_failure(cls, task_id, task_report: TaskFailureReportForCallback):
        logger.info(f"prepare dataset {task_id} failed: {task_report.exception}")

    @classmethod
    def on_complete_task_specific(cls, task_id, task_result, project_manager, process):
        pass

    @classmethod
    def on_failure_task_specific(cls, task_id, task_report, project_manager):
        return None
