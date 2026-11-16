from activetigger.datamodels import TaskComputing

from task_manager.active_tigger_task import AbortableActiveTiggerTask, QueueName
from task_manager.celery import celery_app
from task_manager.task_names import TaskNames
from task_manager.tasks.predict_ml import PredictMLMultiClass, PredictMLResult, PredictMLTaskInputs


# Return type must be a dict, a BaseModel would not be serialized by Celery
class PredictMLTaskResult(PredictMLResult):
    project_slug: str


class PredictMLTask(AbortableActiveTiggerTask):
    name: TaskNames = "predict_ml"
    # sklearn models never use the GPU
    queue = QueueName.CPU

    @classmethod
    def clean_temp_files(cls, computing_task: TaskComputing[PredictMLTaskInputs]):
        computing_task.inputs.path_df.unlink(missing_ok=True)


@celery_app.task(
    bind=True,
    name=PredictMLTask.name,
    queue=PredictMLTask.queue,
    base=PredictMLTask,
    pydantic=True,
)
def predict_ml(self: PredictMLTask, inputs: PredictMLTaskInputs) -> PredictMLTaskResult:
    predict_ml = PredictMLMultiClass(self.request.id, inputs, is_aborted=self.is_aborted)
    r = predict_ml.run()
    return {
        "project_slug": inputs.project_slug,
        "path": r["path"],
        "metrics": r["metrics"],
        "events": r["events"],
    }
