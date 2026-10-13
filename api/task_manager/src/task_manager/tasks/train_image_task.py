from activetigger.datamodels import EventsDict

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.train_image import TrainImage, TrainImageTaskInput


# Return type must be a dict, a BaseModel would not be serialized by Celery
class TrainImageTaskResult(EventsDict):
    project_slug: str


class TrainImageTask(AbortableAutoCallbackTask):
    name = "train image"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU


@celery_app.task(
    bind=True,
    name=TrainImageTask.name,
    queue=TrainImageTask.queue,
    base=TrainImageTask,
    pydantic=True,
)
def train_image(self: TrainImageTask, inputs: TrainImageTaskInput) -> TrainImageTaskResult:
    train_image = TrainImage(self.request.id, inputs, is_aborted=self.is_aborted)
    r = train_image.run()
    return {"events": r["events"], "project_slug": inputs.project_slug}
