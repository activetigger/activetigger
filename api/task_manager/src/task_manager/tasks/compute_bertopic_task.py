from activetigger.datamodels import EventsDict

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.compute_bertopic import ComputeBertopic, ComputeBertopicTaskInput


# Return type must be a dict, a BaseModel would not be serialized by Celery
class ComputeBertopicTaskResult(EventsDict):
    project_slug: str


class ComputeBertopicTask(AbortableAutoCallbackTask):
    name = "compute bertopic"
    # embeddings come from an existing feature; UMAP/HDBSCAN run on CPU
    queue = QueueName.CPU


@celery_app.task(
    bind=True,
    name=ComputeBertopicTask.name,
    queue=ComputeBertopicTask.queue,
    base=ComputeBertopicTask,
    pydantic=True,
)
def compute_bertopic(
    self: ComputeBertopicTask, inputs: ComputeBertopicTaskInput
) -> ComputeBertopicTaskResult:
    task = ComputeBertopic(self.request.id, inputs, is_aborted=self.is_aborted)
    r = task.run()
    return {"events": r["events"], "project_slug": inputs.project_slug}
