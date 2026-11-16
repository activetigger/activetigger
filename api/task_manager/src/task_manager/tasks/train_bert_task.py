from typing import cast

from activetigger.datamodels import EventsDict, TaskComputing

from task_manager.active_tigger_task import AbortableActiveTiggerTask, QueueName, TaskNames
from task_manager.celery import celery_app
from task_manager.tasks.train_bert import TrainBert, TrainBertTaskInputs


# Return type must be a dict, a BaseModel would not be serialized by Celery
class TrainBertTaskResult(EventsDict):
    project_slug: str


# Task definition using the auto callback generic parent task class
class TrainBertTask(AbortableActiveTiggerTask[TrainBertTaskInputs]):
    name: TaskNames = "train_bert"
    # GPU task unless CPU_only mode
    queue: QueueName = QueueName.GPU  # if os.environ.get("GPU") == "true" else QueueName.CPU

    @classmethod
    def clean_temp_files(cls, computing_task: TaskComputing[TrainBertTaskInputs]):
        computing_task.inputs.path_df.unlink(missing_ok=True)


# Task registration
@celery_app.task(
    bind=True,
    name=TrainBertTask.name,
    queue=TrainBertTask.queue,
    base=TrainBertTask,
    pydantic=True,
)
def train_bert(self: TrainBertTask, inputs: TrainBertTaskInputs) -> TrainBertTaskResult:
    train_bert = TrainBert(
        self.request.id, inputs, is_aborted=self.is_aborted, update_progress=self.update_progress
    )
    r = train_bert.run()
    return {"events": r["events"], "project_slug": inputs.project_slug}
