from activetigger.datamodels import EventsDict

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.train_ner import TrainNer, TrainNerTaskInput


# Return type must be a dict, a BaseModel would not be serialized by Celery
class TrainNerTaskResult(EventsDict):
    project_slug: str


class TrainNerTask(AbortableAutoCallbackTask):
    name = "train ner"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU


@celery_app.task(
    bind=True,
    name=TrainNerTask.name,
    queue=TrainNerTask.queue,
    base=TrainNerTask,
    pydantic=True,
)
def train_ner(self: TrainNerTask, inputs: TrainNerTaskInput) -> TrainNerTaskResult:
    train_ner = TrainNer(self.request.id, inputs, is_aborted=self.is_aborted)
    r = train_ner.run()
    return {"events": r["events"], "project_slug": inputs.project_slug}
