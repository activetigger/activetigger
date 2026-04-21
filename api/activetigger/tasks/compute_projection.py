from pandas import DataFrame

from activetigger.tasks.base_task import BaseTask

# accelerate UMAP
try:
    import cuml  # ty: ignore[unresolved-import]

    CUMl_AVAILABLE = True
except Exception:
    print("CuML not available")
    cuml = None
    CUMl_AVAILABLE = False
import pandas as pd
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from activetigger.datamodels import TsneModel, UmapModel


class ComputeProjection(BaseTask):
    """
    Compute projection
    """

    kind = "projection"
    features: DataFrame
    params: dict

    def __init__(
        self, kind: str, features: DataFrame, params: dict, normalize_features: bool = False
    ):
        super().__init__()
        self.kind = kind
        self.features = features
        self.params = params
        self.normalize_features = normalize_features

    def __call__(self) -> DataFrame:
        """
        Compute projection
        """
        if self.kind == "umap":
            self.params = UmapModel(**self.params).model_dump()
            return self.compute_umap()
        elif self.kind == "tsne":
            self.params = TsneModel(**self.params).model_dump()
            return self.compute_tsne()
        else:
            raise ValueError(f"Unknown kind {self.kind}")

    def compute_umap(self) -> DataFrame:
        """
        Compute UMAP
        """
        if self.normalize_features:
            features = StandardScaler().fit_transform(self.features)
        else:
            features = self.features.values

        # Check if cuML is available for GPU acceleration
        try:
            reducer = cuml.UMAP(**self.params, metric="cosine")  # ty: ignore[unresolved-attribute]
            print("Using cuML for UMAP computation")
        except Exception:
            reducer = umap.UMAP(**self.params, metric="cosine")
            print("Using standard UMAP for computation")

        reduced_features = reducer.fit_transform(features)
        df = pd.DataFrame(reduced_features, index=self.features.index)
        df_scaled = 2 * (df - df.min()) / (df.max() - df.min()) - 1
        return df_scaled

    def compute_tsne(self) -> DataFrame:
        """
        Compute TSNE
        """
        if self.normalize_features:
            features = StandardScaler().fit_transform(self.features)
        else:
            features = self.features.values

        reduced_features = TSNE(**self.params).fit_transform(features)
        df = pd.DataFrame(reduced_features, index=self.features.index)
        df_scaled = 2 * (df - df.min()) / (df.max() - df.min()) - 1
        return df_scaled
