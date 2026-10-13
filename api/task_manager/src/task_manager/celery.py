import os

from billiard.exceptions import WorkerLostError
from celery import Celery
from celery.signals import task_failure

from task_manager.auto_callback_task import AutoCallbackTask
from task_manager.config import redis_url
from task_manager.utils import TaskFailureReportForCallback, task_failure_callback

celery_app = Celery("activetigger", broker=redis_url, backend=redis_url)


# A pool child killed by the kernel (OOM, SIGKILL) never runs Task.on_failure:
# the failure is only seen by the main process through this signal. Report it
# to the orchestrator or the process would stay displayed as running forever.
@task_failure.connect
def report_lost_worker(sender=None, task_id=None, exception=None, args=None, kwargs=None, **_):
    if (
        task_id is None
        or not isinstance(exception, WorkerLostError)
        or not isinstance(sender, AutoCallbackTask)
    ):
        return
    report = TaskFailureReportForCallback(
        exception=str(exception),
        task_name=sender.name,
        task_args=args,
        task_kwargs=kwargs,
    )
    task_failure_callback(task_id, report)


# Task discovery: automatically discover tasks in the "tasks" folder
tasks_path = os.path.join(os.path.dirname(__file__), "tasks")
tasks_files = [
    f"task_manager.tasks.{f}"
    for f in os.listdir(tasks_path)
    if os.path.isfile(os.path.join(tasks_path, f))
]
celery_app.autodiscover_tasks(tasks_files)

# Optional configuration, see the application user guide.
celery_app.conf.update(
    task_default_queue="CPU",
    result_expires=3600,
)

if __name__ == "__main__":
    celery_app.start()
