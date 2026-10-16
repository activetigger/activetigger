import json
from pathlib import Path
from typing import TypedDict

import pandas as pd
from activetigger.datamodels import EvalSetDataModel, EvalSetImageModel
from pydantic import BaseModel

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.add_evalset import add_evalset, add_evalset_image


class AddEvalsetTaskInput(BaseModel):
    project_slug: str
    dataset: str
    username: str
    path_project: Path
    # pickled Index of every id already in the project
    index_path: Path
    # exactly one of the two payloads is set, by project kind
    evalset_data: EvalSetDataModel | None = None
    evalset_image: EvalSetImageModel | None = None
    # text projects: labels allowed per scheme column; image projects: labels of the scheme
    schemes: dict[str, list[str]] = {}
    scheme: list[str] | None = None


# Return type must be a dict, a BaseModel would not be serialized by Celery
class AddEvalsetTaskResult(TypedDict):
    project_slug: str
    dataset: str
    username: str
    n_rows: int
    # JSON file with [(scheme_name, elements), ...]
    annotations_path: str


class AddEvalsetTask(AbortableAutoCallbackTask):
    name = "add evalset"
    queue = QueueName.CPU


@celery_app.task(
    bind=True,
    name=AddEvalsetTask.name,
    queue=AddEvalsetTask.queue,
    base=AddEvalsetTask,
    pydantic=True,
)
def add_evalset_task(self: AddEvalsetTask, inputs: AddEvalsetTaskInput) -> AddEvalsetTaskResult:
    try:
        index_all = pd.read_pickle(inputs.index_path)
        if inputs.evalset_image is not None:
            n_rows, schemes_elements = add_evalset_image(
                inputs.dataset,
                inputs.evalset_image,
                inputs.path_project,
                index_all,
                scheme=inputs.scheme,
                is_aborted=self.is_aborted,
            )
        elif inputs.evalset_data is not None:
            n_rows, schemes_elements = add_evalset(
                inputs.dataset,
                inputs.evalset_data,
                inputs.path_project,
                index_all,
                schemes=inputs.schemes,
                is_aborted=self.is_aborted,
            )
        else:
            raise Exception("No evalset payload provided")
        annotations_path = inputs.path_project.joinpath(f"add_evalset_{self.request.id}.json")
        with open(annotations_path, "w") as f:
            json.dump(schemes_elements, f)
    finally:
        inputs.index_path.unlink(missing_ok=True)
    return {
        "project_slug": inputs.project_slug,
        "dataset": inputs.dataset,
        "username": inputs.username,
        "n_rows": n_rows,
        "annotations_path": str(annotations_path),
    }
