from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.predict_ner import PredictNer, PredictNerResult, PredictNerTaskInput


# Return type must be a dict, a BaseModel would not be serialized by Celery
class PredictNerTaskResult(PredictNerResult):
    project_slug: str


class PredictNerTask(AbortableAutoCallbackTask):
    name = "predict ner"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU


@celery_app.task(
    bind=True,
    name=PredictNerTask.name,
    queue=PredictNerTask.queue,
    base=PredictNerTask,
    pydantic=True,
)
def predict_ner(self: PredictNerTask, inputs: PredictNerTaskInput) -> PredictNerTaskResult:
    predict_ner = PredictNer(self.request.id, inputs, is_aborted=self.is_aborted)
    r = predict_ner.run()
    return {"project_slug": inputs.project_slug, "path":r["path"],
     "metrics":r["metrics"], "events":r["events"]}
