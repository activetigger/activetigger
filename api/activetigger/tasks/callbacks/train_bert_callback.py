from typing import cast

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.train_bert_task import (
    TrainBertTask,
    TrainBertTaskInput,
    TrainBertTaskResult,
)

from activetigger.datamodels import EventsModel, LMComputing

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class TrainBertCallback(TaskCallback):

    # task_name is used to identify which callback to execute 
    task_name = TrainBertTask.name

    # the on_complete method will be executed when a task succeeds
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    def on_complete_task_specific(cls, task_id:str, task_result:TrainBertTaskResult, project_manager, process):    
        model = cast(LMComputing, process)
        events = EventsModel(events=task_result['events'])
        project_manager.languagemodels.add(model)
        project_manager.monitoring.close_process(model.unique_id, events)
    
