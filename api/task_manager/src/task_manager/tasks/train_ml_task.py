from activetigger.datamodels import EventsDict

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.train_ml import TrainMLMultiClass, TrainMLMultiClassInput


class TrainMLTaskResult(EventsDict):
    project_slug: str


class TrainMLTask(AbortableAutoCallbackTask):
    name = "train ml"
    # sklearn models never use the GPU
    queue = QueueName.CPU


@celery_app.task(
    bind=True,
    name=TrainMLTask.name,
    queue=TrainMLTask.queue,
    base=TrainMLTask,
    pydantic=True,
)
def train_ml(self: TrainMLTask, inputs: TrainMLMultiClassInput) -> TrainMLTaskResult:
    train_ml = TrainMLMultiClass(self.request.id, inputs, is_aborted=self.is_aborted)
    r = train_ml.run()
    return {"events": r["events"], "project_slug": inputs.project_slug}
