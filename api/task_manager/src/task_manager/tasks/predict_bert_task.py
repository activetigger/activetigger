from task_manager.active_tigger_task import AbortableActiveTiggerTask, QueueName
from task_manager.celery import celery_app
from task_manager.task_names import TaskNames
from task_manager.tasks.predict_bert import (
    PredictBertMultiClass,
    PredictBertResult,
    PredictBertTaskInputs,
)


# Return type must be a dict, a BaseModel would not be serialized by Celery
class PredictBertTaskResult(PredictBertResult):
    project_slug: str


class PredictBertTask(AbortableActiveTiggerTask):
    name: TaskNames = "predict_bert"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU


@celery_app.task(
    bind=True,
    name=PredictBertTask.name,
    queue=PredictBertTask.queue,
    base=PredictBertTask,
    pydantic=True,
)
def predict_bert(self: PredictBertTask, inputs: PredictBertTaskInputs) -> PredictBertTaskResult:
    predict_bert = PredictBertMultiClass(
        self.request.id, inputs, is_aborted=self.is_aborted, update_progress=self.update_progress
    )
    r = predict_bert.run()
    return {
        "project_slug": inputs.project_slug,
        "path": r["path"],
        "events": r["events"],
        "metrics": r["metrics"],
    }
