# Experimental image projects — see docs/multimodal-prompt-selection.md
"""
Prompt management for multimodal image projects.

A prompt is a short text query that is embedded with the same
sentence-transformer model used for a `multimodal-embeddings` feature; the
prompt vector and the image vectors then live in the same space, and
get_next can rank candidates by cosine similarity.

Storage lives in a dedicated `prompts.parquet` (one row per prompt, with
the embedding columns inline), separate from `features.parquet`.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from activetigger.datamodels import PromptComputing, PromptOutModel, PromptsProjectStateModel
from activetigger.features import Features
from activetigger.queue_manager import Queue
from activetigger.tasks.compute_multimodal_prompt import ComputeMultimodalPrompt

PROMPTS_FILE = "prompts.parquet"

METADATA_COLUMNS = ["text", "feature_name", "user", "created_at"]


class Prompts:
    """Per-project store of embedded natural-language prompts."""

    def __init__(
        self,
        project_slug: str,
        path_dir: Path,
        queue: Queue,
        computing: list,
        features: Features,
    ) -> None:
        self.project_slug = project_slug
        self.path_dir = path_dir
        self.path_file = path_dir.joinpath(PROMPTS_FILE)
        self.queue = queue
        self.computing = computing
        self.features = features
        # Cache of sorted rankings keyed by (prompt_id, dataset). Each entry
        # is a pd.Series indexed by element_id with cosine similarity values,
        # sorted descending. Populated lazily by get_ranking and invalidated
        # on prompt delete / feature cascade.
        self._ranking_cache: dict[str, dict[str, pd.Series]] = {}

    # ---------- helpers ----------

    def _read(self) -> pd.DataFrame:
        if not self.path_file.exists():
            return pd.DataFrame(columns=METADATA_COLUMNS)
        return pd.read_parquet(self.path_file)

    def _write(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.path_file, index=True)

    def _resolve_feature(self, feature_name: str) -> str:
        """Return the HF model name for a multimodal-embeddings feature."""
        available = self.features.get_available()
        feat = available.get(feature_name)
        if feat is None:
            raise ValueError(f"Feature '{feature_name}' does not exist")
        if feat.kind != "multimodal-embeddings":
            raise ValueError(
                f"Feature '{feature_name}' is not a multimodal-embeddings feature "
                f"(kind={feat.kind})"
            )
        hf_name = (feat.parameters or {}).get("hf_name")
        if not hf_name:
            raise ValueError(f"Feature '{feature_name}' has no stored HF model name")
        return hf_name

    # ---------- public API ----------

    def add(self, text: str, feature_name: str, user: str) -> str:
        """
        Queue the encoding of a new prompt. Returns the task unique_id.
        The actual parquet row is appended by `receive_result` once the
        GPU worker completes.
        """
        text = text.strip()
        if not text:
            raise ValueError("Prompt text cannot be empty")
        hf_name = self._resolve_feature(feature_name)
        prompt_id = str(uuid.uuid4())

        unique_id = self.queue.add_task(
            "prompt",
            self.project_slug,
            ComputeMultimodalPrompt(
                text=text,
                model_name=hf_name,
                path_process=self.path_dir,
            ),
            queue="gpu",
        )

        self.computing.append(
            PromptComputing(
                user=user,
                unique_id=unique_id,
                time=datetime.now(timezone.utc),
                kind="prompt",
                prompt_id=prompt_id,
                text=text,
                feature_name=feature_name,
                hf_name=hf_name,
            )
        )
        return unique_id

    def receive_result(self, computing: PromptComputing, vec: np.ndarray) -> None:
        """Persist a completed prompt to prompts.parquet."""
        # The feature might have been deleted while the task was queued.
        if not self.features.exists(computing.feature_name):
            return

        vec = np.asarray(vec).reshape(-1)
        row: dict[str, Any] = {
            "text": computing.text,
            "feature_name": computing.feature_name,
            "user": computing.user,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        for i, v in enumerate(vec):
            row[f"dim_{i}"] = float(v)

        df = self._read()
        new_row = pd.DataFrame([row], index=pd.Index([computing.prompt_id], name="prompt_id"))

        # Align columns: existing rows may have fewer/more dim_* columns if
        # the feature changed. Union columns, fill missing with NaN.
        combined = pd.concat([df, new_row], axis=0)
        combined.index.name = "prompt_id"
        self._write(combined)

    def list(self, user: str | None = None) -> list[PromptOutModel]:
        df = self._read()
        if df.empty:
            return []
        if user is not None:
            df = df[df["user"] == user]
        out: list[PromptOutModel] = []
        for prompt_id, row in df.iterrows():
            out.append(
                PromptOutModel(
                    prompt_id=str(prompt_id),
                    text=str(row["text"]),
                    feature_name=str(row["feature_name"]),
                    user=str(row["user"]),
                    created_at=str(row["created_at"]),
                )
            )
        return out

    def get_embedding_and_feature(self, prompt_id: str) -> tuple[np.ndarray, str]:
        df = self._read()
        if prompt_id not in df.index:
            raise ValueError(f"Prompt '{prompt_id}' not found")
        row = df.loc[prompt_id]
        dim_cols = [c for c in df.columns if c.startswith("dim_")]
        if not dim_cols:
            raise ValueError(f"Prompt '{prompt_id}' has no embedding")
        vec = row[dim_cols].to_numpy(dtype=float)
        return vec, str(row["feature_name"])

    def delete(self, prompt_id: str) -> None:
        df = self._read()
        if prompt_id not in df.index:
            raise ValueError(f"Prompt '{prompt_id}' not found")
        df = df.drop(index=prompt_id)
        self._write(df)
        self._ranking_cache.pop(prompt_id, None)

    def reset_all(self) -> None:
        """
        Drop every saved prompt and clear the ranking cache.
        Called from the Features on_reset cascade when all features are wiped.
        """
        if self.path_file.exists():
            try:
                self.path_file.unlink()
            except OSError as ex:
                print(f"Could not delete prompts file: {ex}")
        self._ranking_cache.clear()

    def delete_by_feature(self, feature_name: str) -> int:
        """Drop every prompt bound to a feature (cascade from feature delete)."""
        df = self._read()
        if df.empty or "feature_name" not in df.columns:
            return 0
        mask = df["feature_name"] == feature_name
        n = int(mask.sum())
        if n:
            dropped_ids = [str(i) for i in df.index[mask]]
            self._write(df.loc[~mask])
            for pid in dropped_ids:
                self._ranking_cache.pop(pid, None)
        return n

    def get_ranking(self, prompt_id: str, dataset: str) -> pd.Series:
        """
        Return element_ids sorted by descending cosine similarity between the
        prompt embedding and the bound feature's image embeddings, over the
        full given dataset. Cached per (prompt_id, dataset) — subsequent
        get_next calls on the same prompt only do an index intersection.
        """
        cache_by_ds = self._ranking_cache.setdefault(prompt_id, {})
        cached = cache_by_ds.get(dataset)
        if cached is not None:
            return cached

        prompt_vec, feature_name = self.get_embedding_and_feature(prompt_id)
        img_df = self.features.get([feature_name], dataset=[dataset])
        if img_df.empty:
            raise ValueError(
                f"No image embeddings for feature '{feature_name}' in dataset '{dataset}'"
            )
        mat = img_df.to_numpy(dtype=float)
        prompt_vec = np.asarray(prompt_vec, dtype=float).reshape(-1)
        if mat.shape[1] != prompt_vec.shape[0]:
            raise ValueError(
                f"Dimension mismatch between prompt ({prompt_vec.shape[0]}) "
                f"and feature '{feature_name}' ({mat.shape[1]}). "
                "The feature may have been recomputed with a different model."
            )
        img_norms = np.linalg.norm(mat, axis=1)
        prompt_norm = float(np.linalg.norm(prompt_vec))
        sims = (mat @ prompt_vec) / (img_norms * prompt_norm + 1e-12)
        ranked = pd.Series(sims, index=img_df.index).sort_values(ascending=False)

        cache_by_ds[dataset] = ranked
        return ranked

    def current_computing(self) -> dict[str, dict[str, str | None]]:
        out: dict[str, dict[str, str | None]] = {}
        for e in self.computing:
            if e.kind != "prompt":
                continue
            e = e  # type: PromptComputing
            progress_file = self.path_dir.joinpath(e.unique_id)
            progress: str | None = None
            try:
                if progress_file.exists():
                    progress = progress_file.read_text().strip()
            except OSError:
                progress = None
            out[e.prompt_id] = {
                "prompt_id": e.prompt_id,
                "text": e.text,
                "feature_name": e.feature_name,
                "progress": progress,
            }
        return out

    def state(self) -> PromptsProjectStateModel:
        try:
            available = self.features.get_available()
        except Exception:
            available = {}
        bindable = [
            name for name, feat in available.items() if feat.kind == "multimodal-embeddings"
        ]
        return PromptsProjectStateModel(
            available=self.list(),
            bindable_features=bindable,
            training=self.current_computing(),
        )

    def _is_multimodal_feature(self, name: str) -> bool:
        try:
            available = self.features.get_available()
        except Exception:
            return False
        feat = available.get(name)
        return feat is not None and feat.kind == "multimodal-embeddings"
