from activetigger.datamodels import TaskComputing

from task_manager.active_tigger_task import AbortableActiveTiggerTask, QueueName
from task_manager.celery import celery_app
from task_manager.task_names import TaskNames
from task_manager.tasks.predict_ner import PredictNer, PredictNerResult, PredictNerTaskInputs


# Return type must be a dict, a BaseModel would not be serialized by Celery
class PredictNerTaskResult(PredictNerResult):
    project_slug: str


class PredictNerTask(AbortableActiveTiggerTask[PredictNerTaskInputs]):
    name: TaskNames = "predict_ner"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU

    @classmethod
    def clean_temp_files(cls, computing_task: TaskComputing[PredictNerTaskInputs]):
        if computing_task.inputs.path_df:
            computing_task.inputs.path_df.unlink(missing_ok=True)


@celery_app.task(
    bind=True,
    name=PredictNerTask.name,
    queue=PredictNerTask.queue,
    base=PredictNerTask,
    pydantic=True,
)
def predict_ner(self: PredictNerTask, inputs: PredictNerTaskInputs) -> PredictNerTaskResult:
    predict_ner = PredictNer(
        self.request.id, inputs, is_aborted=self.is_aborted, update_progress=self.update_progress
    )
    r = predict_ner.run()
    return {
        "project_slug": inputs.project_slug,
        "path": r["path"],
        "metrics": r["metrics"],
        "events": r["events"],
    }
