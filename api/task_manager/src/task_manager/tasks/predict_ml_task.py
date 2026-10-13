from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.predict_ml import PredictMLMultiClass, PredictMLResult, PredictMLTaskInput


# Return type must be a dict, a BaseModel would not be serialized by Celery
class PredictMLTaskResult(PredictMLResult):
    project_slug: str


class PredictMLTask(AbortableAutoCallbackTask):
    name = "predict ml"
    # sklearn models never use the GPU
    queue = QueueName.CPU


@celery_app.task(
    bind=True,
    name=PredictMLTask.name,
    queue=PredictMLTask.queue,
    base=PredictMLTask,
    pydantic=True,
)
def predict_ml(self: PredictMLTask, inputs: PredictMLTaskInput) -> PredictMLTaskResult:
    predict_ml = PredictMLMultiClass(self.request.id, inputs, is_aborted=self.is_aborted)
    r = predict_ml.run()
    return {"project_slug": inputs.project_slug, "path":r["path"],
     "metrics":r["metrics"], "events":r["events"]}
