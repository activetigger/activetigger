# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.predict_ml_task import PredictMLTask, PredictMLTaskResult

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class PredictMLCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = PredictMLTask.name

    # the prediction and metrics files are written by the worker in the model
    # directory and read on demand: nothing else to persist on completion
    @classmethod
    def on_complete_task_specific(
        cls, task_id: str, task_result: PredictMLTaskResult, project_manager, process
    ):
        pass
