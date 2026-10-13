from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.predict_bert import (
    PredictBertMultiClass,
    PredictBertResult,
    PredictBertTaskInput,
)


# Return type must be a dict, a BaseModel would not be serialized by Celery
class PredictBertTaskResult(PredictBertResult):
    project_slug: str


class PredictBertTask(AbortableAutoCallbackTask):
    name = "predict bert"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU


@celery_app.task(
    bind=True,
    name=PredictBertTask.name,
    queue=PredictBertTask.queue,
    base=PredictBertTask,
    pydantic=True,
)
def predict_bert(self: PredictBertTask, inputs: PredictBertTaskInput) -> PredictBertTaskResult:
    predict_bert = PredictBertMultiClass(self.request.id, inputs, is_aborted=self.is_aborted)
    r = predict_bert.run()
    return {"project_slug": inputs.project_slug, 
            "path": r["path"], 
            "events": r["events"],
            "metrics":r["metrics"]}