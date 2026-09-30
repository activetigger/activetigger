from typing import cast

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.train_bert_task import (
    TrainBertTask,
    TrainBertTaskInput,
    TrainBertTaskResult,
)
from task_manager.utils import TaskFailureReportForCallback

from activetigger.datamodels import EventsModel, LMComputing
from activetigger.orchestrator import get_orchestrator

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class TrainBertCallback(TaskCallback):

    # task_name is used to identify which callback to execute 
    task_name = TrainBertTask.name

    # the on_complete method will be executed when a task succeeds
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    def on_complete(cls, task_id:str, task_result:TrainBertTaskResult):
        print(f"Completion of {cls.task_name} task {task_id} in orchestrator {task_result}")
        
       
        orchestrator = get_orchestrator()
        try:
            project_manager = orchestrator.projects[task_result['project_slug']]
            for e in project_manager.computing:
                if e.unique_id == task_id:
                    model = cast(LMComputing, e)
                    events = EventsModel(events=task_result['events'])
                    project_manager.languagemodels.add(model)
                    project_manager.monitoring.close_process(model.unique_id, events)
                    project_manager.clean_process(e)
            
        except KeyError:
            raise Exception(f"Project {task_result['project_slug']} not listed in orchestrator, we can't finish creation process")

    # the on_failure method will be executed when a task fails
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    def on_failure(cls, task_id:str, task_report:TaskFailureReportForCallback):
        print(f"Error on {cls.task_name} task {task_id} in orchestrator {task_report}")
        # cast first args as Task Input type
        orchestrator = get_orchestrator()
        try:
            task_input = TrainBertTaskInput(**task_report['task_args'][0])

            project_manager = orchestrator.projects[task_input.project_slug]
            for e in project_manager.computing:
                if e.unique_id == task_id:
                    bert_task = cast(LMComputing, e)
                    project_manager.db_manager.language_models_service.delete_model(
                            task_input.project_slug, bert_task.model_name
                    )
                    project_manager.clean_process(e)
            project_manager.status = 'error'
            # TODO: this message is generic on any GPU task
            message = (
                        f"Error for task {TrainBertTask.name} : GPU error — not enough GPU memory available. "
                        "Try reducing the batch size, the max sequence length, or using a smaller model. "
                        f"Details: {task_report['exception']}"
                    ) if any(
                    s in task_report['exception']
                    for s in [
                        "CUDA",
                        "CUDACachingAllocator",
                        "out of memory",
                        "NVML",
                        "cuda",
                    ]
                ) else f"Error for process {TrainBertTask.name} : {task_report['exception']}"
            project_manager.errors.add(message)
        except KeyError:
            raise Exception(f"Project {task_input.project_slug} not listed in orchestrator, we can't finish creation process")

