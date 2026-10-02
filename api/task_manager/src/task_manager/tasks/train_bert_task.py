from activetigger.datamodels import EventsDict

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.train_bert import TrainBert, TrainBertTaskInput


# Return type must be a dict, a BaseModel would not be serialized by Celery
class TrainBertTaskResult(EventsDict):
    project_slug:str

# Task definition using the auto callback generic parent task class
class TrainBertTask(AbortableAutoCallbackTask):
    name = "train bert"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU #if os.environ.get("GPU") == "true" else QueueName.CPU
   
# Task registration
@celery_app.task(
    bind=True,
    name=TrainBertTask.name,
    queue=TrainBertTask.queue,
    base=TrainBertTask,
    pydantic=True,
)
def train_bert(self:TrainBertTask, inputs: TrainBertTaskInput)->TrainBertTaskResult:
    train_bert = TrainBert(self.request.id,  inputs, is_aborted=self.is_aborted)
    return {**train_bert.run(), 'project_slug': inputs.project_slug}