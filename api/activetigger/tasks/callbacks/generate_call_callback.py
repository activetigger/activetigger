from pathlib import Path

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.generate_call_task import GenerateCallTask, GenerateCallTaskResult

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class GenerateCallCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = GenerateCallTask.name

    # every answer is in the jsonl checkpoint: insert them and drop the file
    @classmethod
    def on_complete_task_specific(
        cls, task_id: str, task_result: GenerateCallTaskResult, project_manager, process
    ):
        project_manager._recover_generations_from_jsonl(Path(task_result["jsonl_path"]))

    # keep the answers already paid for, whether the batch failed or was stopped
    @classmethod
    def on_failure_task_specific(cls, task_id, task_report, project_manager):
        path_process = Path(task_report.task_args[0]["path_process"])
        project_manager._recover_generations_from_jsonl(
            path_process.joinpath(f"gen_{task_id}.jsonl")
        )
        return None
