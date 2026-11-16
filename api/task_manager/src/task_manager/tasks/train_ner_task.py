from activetigger.datamodels import EventsDict, TaskComputing

from task_manager.active_tigger_task import AbortableActiveTiggerTask, QueueName
from task_manager.celery import celery_app
from task_manager.task_names import TaskNames
from task_manager.tasks.train_ner import TrainNer, TrainNerTaskInputs


# Return type must be a dict, a BaseModel would not be serialized by Celery
class TrainNerTaskResult(EventsDict):
    project_slug: str


class TrainNerTask(AbortableActiveTiggerTask):
    name: TaskNames = "train_ner"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU

    @classmethod
    def clean_temp_files(cls, computing_task: TaskComputing[TrainNerTaskInputs]):
        computing_task.inputs.path_df.unlink(missing_ok=True)


@celery_app.task(
    bind=True,
    name=TrainNerTask.name,
    queue=TrainNerTask.queue,
    base=TrainNerTask,
    pydantic=True,
)
def train_ner(self: TrainNerTask, inputs: TrainNerTaskInputs) -> TrainNerTaskResult:
    train_ner = TrainNer(
        self.request.id, inputs, is_aborted=self.is_aborted, update_progress=self.update_progress
    )
    r = train_ner.run()
    return {"events": r["events"], "project_slug": inputs.project_slug}
