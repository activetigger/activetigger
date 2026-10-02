import inspect
from abc import ABC, abstractmethod
from importlib import util
from pathlib import Path
from typing import Any, cast

from task_manager.utils import TaskFailureReportForCallback

from activetigger.datamodels import ProcessComputing
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project


# TaskCallback is a generic class to specify task callback definitions
class TaskCallback(ABC):
    
    task_name:str
  
    @classmethod
    def on_complete(cls, task_id:str, task_result: Any) -> Any:
        """Callback called by the orchestrator once the task succeeds."""
        print(f"Completion of {cls.task_name} task {task_id} in orchestrator {task_result}")
        
        orchestrator = get_orchestrator()
        try:
            project_manager = orchestrator.projects[task_result['project_slug']]
            for process in project_manager.computing:
                if process.unique_id == task_id:
                    cls.on_complete_task_specific(task_id, task_result, project_manager, process)
                    project_manager.clean_process(process)
        except KeyError:
            raise Exception(f"Project {task_result['project_slug']} not listed in orchestrator, we can't finish creation process")

    @classmethod
    @abstractmethod
    def on_complete_task_specific(cls, task_id:str, task_result: Any, project_manager:Project, process: ProcessComputing) -> None:
        """Specific Callback for one task called by the generic on_complete method."""
        raise NotImplementedError

    @classmethod
    def on_failure(cls, task_id:str, task_report: TaskFailureReportForCallback):
        print(f"Error on {cls.task_name} task {task_id} in orchestrator {task_report}")
        # cast first args as Task Input type
        orchestrator = get_orchestrator()
        task_input = task_report['task_args'][0]
        project_slug = task_input['project_slug']
        try:
            project_manager = orchestrator.projects[project_slug]
            for e in project_manager.computing:
                if e.unique_id == task_id:
                    project_manager.clean_process(e)
            project_manager.status = 'error'
            specific_message = cls.on_failure_task_specific(task_id, task_report, project_manager)
            if specific_message:
                project_manager.errors.add(specific_message)
            #generic message for GPU errors
            if any(s in task_report['exception'] for s in [
                        "CUDA",
                        "CUDACachingAllocator",
                        "out of memory",
                        "NVML",
                        "cuda",
                    ]):
                project_manager.errors.add((
                        f"Error for task {cls.task_name} : GPU error — not enough GPU memory available. "
                        "Try reducing the batch size, the max sequence length, or using a smaller model. "
                        f"Details: {task_report['exception']}"
                    ))
            elif specific_message is None:
                project_manager.errors.add(f"Error for process {cls.task_name} : {task_report['exception']}")

        except KeyError:
            raise Exception(f"Project {project_slug} not listed in orchestrator, we can't finish creation process")



    @classmethod
    @abstractmethod
    def on_failure_task_specific(cls, task_id:str, task_report: TaskFailureReportForCallback, project_manager:Project) -> str | None:
        """Specific callback for failure case. If a specific message is return it will replace the generic exception."""
        return None
    
    
# discover all TaskCallback implementations to collect on_complete methods
def discover_task_callbacks(directory_path: str):
    discovered_tasks:list[TaskCallback] = []
    for file in Path(directory_path).glob("*.py"):
        if file.name.startswith("_"): 
            continue
        
        module_name = file.stem
        spec = util.spec_from_file_location(module_name, file)
        if spec and spec.loader:
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
        
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, TaskCallback) and obj is not TaskCallback:
                    discovered_tasks.append(obj)  # ty:ignore[invalid-argument-type]
    return discovered_tasks
