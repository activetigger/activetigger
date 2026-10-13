from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.predict_image import PredictImage, PredictImageResult, PredictImageTaskInput


# Return type must be a dict, a BaseModel would not be serialized by Celery
class PredictImageTaskResult(PredictImageResult):
    project_slug: str


class PredictImageTask(AbortableAutoCallbackTask):
    name = "predict image"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU


@celery_app.task(
    bind=True,
    name=PredictImageTask.name,
    queue=PredictImageTask.queue,
    base=PredictImageTask,
    pydantic=True,
)
def predict_image(self: PredictImageTask, inputs: PredictImageTaskInput) -> PredictImageTaskResult:
    predict_image = PredictImage(self.request.id, inputs, is_aborted=self.is_aborted)
    r = predict_image.run()
    return {"project_slug": inputs.project_slug, "metrics":r["metrics"],"path":r["path"],}

