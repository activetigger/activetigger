from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame

from activetigger.tasks.compute_features_from_specs import ComputeFeaturesFromSpecs


class ExtendFeatures(ComputeFeaturesFromSpecs):
    """
    Rewrite the features parquet after an eval set (test or valid) is
    imported or dropped, so the operation does not force dropping the
    features already computed on the train set:

    - import: recompute the existing features for the new rows
      (`specs` + `new_index`) and append them to the kept rows;
    - drop: pass empty `specs`/`new_index` — the dropped set's rows are
      simply filtered out, no recomputation involved.

    Every spec's `data` series must cover exactly `new_index` (the rows
    of the new eval set). Runs in a worker process, so it only touches
    files: the parent project refreshes `Features.map`/`n` on success
    and falls back to `Features.reset_features_file()` on failure or
    cancellation (see `Project.update_processes`).
    """

    kind = "extend_features"

    def __init__(
        self,
        specs: list[dict[str, Any]],
        path_process: Path,
        path_models: Path,
        language: str,
        path_features: Path,
        eval_dataset: str,
        new_index: pd.Index,
        full_index: pd.Index,
        path_progress: Path | None = None,
    ):
        super().__init__(
            specs=specs,
            path_process=path_process,
            path_models=path_models,
            language=language,
            path_progress=path_progress,
        )
        self.path_features = path_features
        self.eval_dataset = eval_dataset
        self.new_index = new_index
        self.full_index = full_index

    def __call__(self) -> None:
        self._refresh_progress_path()
        try:
            # Reserve one extra progress unit for the merge/write step.
            computed = self._compute_all(total=len(self.specs) + 1)

            new_rows = DataFrame(index=self.new_index)
            if not computed.empty:
                new_rows = new_rows.join(computed)

            # Read the current file as late as possible to minimize
            # staleness (the parent blocks feature add/delete while an
            # extension is running, so this is defensive).
            old_df = pd.read_parquet(self.path_features)
            kept = old_df[old_df["dataset"] != self.eval_dataset]

            # Pin the new rows to the stored column set: a no-op for
            # embeddings (fixed dimensions), but dfm vocabularies depend
            # on the corpus they were fitted on — terms absent from the
            # eval corpus get 0, terms new to it are dropped.
            expected = [c for c in old_df.columns if c != "dataset"]
            new_rows = new_rows.reindex(columns=expected, fill_value=0)
            new_rows["dataset"] = self.eval_dataset

            # skip the concat when shrinking (empty new_rows would upcast
            # dtypes / trigger pandas empty-entries warnings)
            merged = (pd.concat([kept, new_rows]) if len(new_rows) else kept).loc[self.full_index]

            self._raise_if_cancelled()
            tmp = self.path_features.with_suffix(".parquet.tmp")
            merged.to_parquet(tmp, index=True)
            tmp.replace(self.path_features)
            self._write_overall(100.0)
        finally:
            self._cleanup_progress_file()
