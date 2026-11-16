from activetigger.datamodels import EventsDict, TaskComputing

from task_manager.active_tigger_task import AbortableActiveTiggerTask, QueueName
from task_manager.celery import celery_app
from task_manager.task_names import TaskNames
from task_manager.tasks.train_ml import TrainMLMultiClass, TrainMLMultiClassInputs


class TrainMLTaskResult(EventsDict):
    project_slug: str


class TrainMLTask(AbortableActiveTiggerTask):
    name: TaskNames = "train_ml"
    # sklearn models never use the GPU
    queue = QueueName.CPU

    @classmethod
    def clean_temp_files(cls, computing_task: TaskComputing[TrainMLMultiClassInputs]):
        computing_task.inputs.path_df.unlink(missing_ok=True)


@celery_app.task(
    bind=True,
    name=TrainMLTask.name,
    queue=TrainMLTask.queue,
    base=TrainMLTask,
    pydantic=True,
)
def train_ml(self: TrainMLTask, inputs: TrainMLMultiClassInputs) -> TrainMLTaskResult:
    train_ml = TrainMLMultiClass(self.request.id, inputs, is_aborted=self.is_aborted)
    r = train_ml.run()
    return {"events": r["events"], "project_slug": inputs.project_slug}
