from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import entropy

from activetigger.datamodels import QuickModelComputed
from activetigger.tasks.compute_features_from_specs import ComputeFeaturesFromSpecs


class PredictWithFeatures(ComputeFeaturesFromSpecs):
    """
    End-to-end prediction task: (re)compute a batch of features and
    apply a trained quickmodel to them, saving the resulting proba
    DataFrame (label columns + `prediction` + `entropy`) to
    `path_output / file_name`.

    See `ComputeFeaturesFromSpecs` for the spec format and the progress
    reporting convention. The final prediction step counts as one more
    progress unit so the percentages stay meaningful.
    """

    kind = "predict_with_features"

    def __init__(
        self,
        specs: list[dict[str, Any]],
        path_process: Path,
        path_models: Path,
        language: str,
        quickmodel: QuickModelComputed,
        path_output: Path,
        file_name: str,
        path_progress: Path | None = None,
    ):
        super().__init__(
            specs=specs,
            path_process=path_process,
            path_models=path_models,
            language=language,
            path_progress=path_progress,
        )
        self.quickmodel = quickmodel
        self.path_output = path_output
        self.file_name = file_name

    def __call__(self) -> DataFrame:
        self._refresh_progress_path()
        try:
            # Reserve one extra progress unit for the final prediction step.
            features = self._compute_all(total=len(self.specs) + 1)
            proba = self._predict(features)
            self._save(proba)
            self._write_overall(100.0)
            return proba
        finally:
            self._cleanup_progress_file()

    def _predict(self, features: DataFrame) -> DataFrame:
        """
        Apply `self.quickmodel.model` to the computed feature matrix and
        return a proba DataFrame shaped like the one stored on the
        trained quickmodel (label columns + `prediction` + `entropy`).
        """
        sm = self.quickmodel
        expected = list(sm.model.feature_names_in_)  # ty: ignore[unresolved-attribute]
        missing = [c for c in expected if c not in features.columns]
        if missing:
            raise ValueError(
                f"Feature columns required by quickmodel are missing: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        X = features.reindex(columns=expected)

        proba_values = sm.model.predict_proba(X)  # ty: ignore[unresolved-attribute]
        labels = list(sm.model.classes_)  # ty: ignore[unresolved-attribute]
        proba = pd.DataFrame(proba_values, columns=labels, index=X.index)
        proba["prediction"] = proba[labels].idxmax(axis=1)
        proba["entropy"] = entropy(proba_values, axis=1)
        # Match the per-label entropy scheme stored on training-time proba.
        for label in labels:
            try:
                prob_A_not_A = np.column_stack([proba[label], 1 - proba[label]])
                proba[f"entropy-{label}"] = entropy(prob_A_not_A, axis=1)
            except Exception:
                pass
        return proba

    def _save(self, proba: DataFrame) -> None:
        """
        Persist the proba DataFrame under `path_output / file_name` in
        parquet (matches the convention of the other prediction tasks).
        Creates the directory if it does not yet exist.
        """
        self.path_output.mkdir(parents=True, exist_ok=True)
        proba.to_parquet(self.path_output.joinpath(self.file_name), index=True)
