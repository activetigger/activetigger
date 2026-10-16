from pathlib import Path
from typing import TypedDict

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel

from task_manager.auto_callback_task import AbortableAutoCallbackTask, QueueName
from task_manager.celery import celery_app
from task_manager.tasks.compute_features_from_specs import (
    FeatureSpec,
    compute_features_from_specs,
    write_progress,
)


class ExtendFeaturesTaskInput(BaseModel):
    project_slug: str
    specs: list[FeatureSpec]
    # pickled Series covering exactly the new eval rows (texts)
    data_path: Path
    # pickled (new_index, full_index) tuple
    indexes_path: Path
    path_process: Path
    path_models: Path
    language: str
    path_features: Path
    eval_dataset: str


# Return type must be a dict, a BaseModel would not be serialized by Celery
class ExtendFeaturesTaskResult(TypedDict):
    project_slug: str
    eval_dataset: str


class ExtendFeaturesTask(AbortableAutoCallbackTask):
    name = "extend features"
    queue = QueueName.CPU


@celery_app.task(
    bind=True,
    name=ExtendFeaturesTask.name,
    queue=ExtendFeaturesTask.queue,
    base=ExtendFeaturesTask,
    pydantic=True,
)
def extend_features(
    self: ExtendFeaturesTask, inputs: ExtendFeaturesTaskInput
) -> ExtendFeaturesTaskResult:
    """
    Rewrite the features parquet after an eval set is imported (recompute the
    existing features for the new rows and append them) or dropped (empty
    specs: the set's rows are filtered out).
    """
    path_progress = inputs.path_process.joinpath(self.request.id)
    try:
        data = pd.read_pickle(inputs.data_path)
        new_index, full_index = pd.read_pickle(inputs.indexes_path)
        # one extra progress unit for the merge/write step
        computed = compute_features_from_specs(
            inputs.specs,
            data,
            path_process=inputs.path_process,
            path_models=inputs.path_models,
            language=inputs.language,
            path_progress=path_progress,
            is_aborted=self.is_aborted,
            total=len(inputs.specs) + 1,
        )

        new_rows = DataFrame(index=new_index)
        if not computed.empty:
            new_rows = new_rows.join(computed)

        # read the current file as late as possible (the API blocks feature
        # add/delete while an extension is running, so this is defensive)
        old_df = pd.read_parquet(inputs.path_features)
        kept = old_df[old_df["dataset"] != inputs.eval_dataset]

        # pin the new rows to the stored column set: no-op for embeddings, but
        # dfm vocabularies depend on their corpus (absent terms 0, new terms dropped)
        expected = [c for c in old_df.columns if c != "dataset"]
        new_rows = new_rows.reindex(columns=expected, fill_value=0)
        new_rows["dataset"] = inputs.eval_dataset

        # skip the concat when shrinking (empty new_rows would upcast dtypes)
        merged = (pd.concat([kept, new_rows]) if len(new_rows) else kept).loc[full_index]

        if self.is_aborted():
            raise Exception("Process interrupted by user")
        tmp = inputs.path_features.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp, index=True)
        tmp.replace(inputs.path_features)
        write_progress(path_progress, 100.0)
    finally:
        path_progress.unlink(missing_ok=True)
        inputs.data_path.unlink(missing_ok=True)
        inputs.indexes_path.unlink(missing_ok=True)
    return {"project_slug": inputs.project_slug, "eval_dataset": inputs.eval_dataset}
