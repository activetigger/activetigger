from activetigger.datamodels import EventsDict, TaskComputing

from task_manager.active_tigger_task import AbortableActiveTiggerTask, QueueName
from task_manager.celery import celery_app
from task_manager.task_names import TaskNames
from task_manager.tasks.train_image import TrainImage, TrainImageTaskInputs


# Return type must be a dict, a BaseModel would not be serialized by Celery
class TrainImageTaskResult(EventsDict):
    project_slug: str


class TrainImageTask(AbortableActiveTiggerTask[TrainImageTaskInputs]):
    name: TaskNames = "train_image"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU

    @classmethod
    def clean_temp_files(cls, computing_task: TaskComputing[TrainImageTaskInputs]):
        if computing_task.inputs.path_df:
            computing_task.inputs.path_df.unlink(missing_ok=True)


@celery_app.task(
    bind=True,
    name=TrainImageTask.name,
    queue=TrainImageTask.queue,
    base=TrainImageTask,
    pydantic=True,
)
def train_image(self: TrainImageTask, inputs: TrainImageTaskInputs) -> TrainImageTaskResult:
    train_image = TrainImage(
        self.request.id, inputs, is_aborted=self.is_aborted, update_progress=self.update_progress
    )
    r = train_image.run()
    return {"events": r["events"], "project_slug": inputs.project_slug}
