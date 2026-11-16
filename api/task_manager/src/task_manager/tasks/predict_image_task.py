from activetigger.datamodels import TaskComputing

from task_manager.active_tigger_task import AbortableActiveTiggerTask, QueueName
from task_manager.celery import celery_app
from task_manager.task_names import TaskNames
from task_manager.tasks.predict_image import (
    PredictImage,
    PredictImageResult,
    PredictImageTaskInputs,
)


# Return type must be a dict, a BaseModel would not be serialized by Celery
class PredictImageTaskResult(PredictImageResult):
    project_slug: str


class PredictImageTask(AbortableActiveTiggerTask[PredictImageTaskInputs]):
    name: TaskNames = "predict_image"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU

    @classmethod
    def clean_temp_files(cls, computing_task: TaskComputing[PredictImageTaskInputs]):
        if computing_task.inputs.path_df:
            computing_task.inputs.path_df.unlink(missing_ok=True)


@celery_app.task(
    bind=True,
    name=PredictImageTask.name,
    queue=PredictImageTask.queue,
    base=PredictImageTask,
    pydantic=True,
)
def predict_image(self: PredictImageTask, inputs: PredictImageTaskInputs) -> PredictImageTaskResult:
    predict_image = PredictImage(
        self.request.id, inputs, is_aborted=self.is_aborted, update_progress=self.update_progress
    )
    r = predict_image.run()
    return {
        "project_slug": inputs.project_slug,
        "metrics": r["metrics"],
        "path": r["path"],
    }
