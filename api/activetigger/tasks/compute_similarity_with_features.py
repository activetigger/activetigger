from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from activetigger.tasks.compute_features_from_specs import ComputeFeaturesFromSpecs


class ComputeSimilarityWithFeatures(ComputeFeaturesFromSpecs):
    """
    End-to-end prompt-similarity task: (re)compute the embedding feature a
    prompt is bound to and rank every element by cosine similarity against
    the prompt vector, saving the resulting DataFrame (`similarity` +
    `rank` columns) to `path_output / file_name`.

    Mirrors `PredictWithFeatures` (same spec format and progress
    convention) with the trained quickmodel replaced by a dot product
    against `prompt_vector`. The final similarity step counts as one more
    progress unit so the percentages stay meaningful.
    """

    kind = "prompt_similarity"

    def __init__(
        self,
        specs: list[dict[str, Any]],
        path_process: Path,
        path_models: Path,
        language: str,
        prompt_vector: np.ndarray,
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
        self.prompt_vector = np.asarray(prompt_vector, dtype=float).reshape(-1)
        self.path_output = path_output
        self.file_name = file_name

    def __call__(self) -> DataFrame:
        self._refresh_progress_path()
        try:
            # Reserve one extra progress unit for the final similarity step.
            features = self._compute_all(total=len(self.specs) + 1)
            similarity = self._similarity(features)
            self._save(similarity)
            self._write_overall(100.0)
            return similarity
        finally:
            self._cleanup_progress_file()

    def _similarity(self, features: DataFrame) -> DataFrame:
        """
        Cosine similarity between `self.prompt_vector` and each row of the
        computed embedding matrix (same math as `Prompts.get_ranking`).
        """
        mat = features.to_numpy(dtype=float)
        if mat.shape[1] != self.prompt_vector.shape[0]:
            raise ValueError(
                f"Dimension mismatch between prompt ({self.prompt_vector.shape[0]}) "
                f"and computed feature ({mat.shape[1]}). "
                "The feature may have been recomputed with a different model."
            )
        row_norms = np.linalg.norm(mat, axis=1)
        prompt_norm = float(np.linalg.norm(self.prompt_vector))
        sims = (mat @ self.prompt_vector) / (row_norms * prompt_norm + 1e-12)
        out = pd.DataFrame({"similarity": sims}, index=features.index)
        out["rank"] = out["similarity"].rank(ascending=False, method="first").astype(int)
        return out

    def _save(self, similarity: DataFrame) -> None:
        """
        Persist the similarity DataFrame under `path_output / file_name` in
        parquet (matches the convention of the prediction tasks).
        """
        self.path_output.mkdir(parents=True, exist_ok=True)
        similarity.to_parquet(self.path_output.joinpath(self.file_name), index=True)
