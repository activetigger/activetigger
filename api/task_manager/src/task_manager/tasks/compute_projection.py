import pandas as pd
import umap
from activetigger.datamodels import TsneModel, UmapModel
from pandas import DataFrame
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def compute_projection(
    features: DataFrame, kind: str, params: dict, normalize_features: bool = False
) -> DataFrame:
    """
    2D/3D projection of a features matrix (umap or tsne), axes rescaled to [-1, 1].
    Shared by the celery task and any in-process caller.
    """
    if kind == "umap":
        return compute_umap(features, UmapModel(**params).model_dump(), normalize_features)
    elif kind == "tsne":
        return compute_tsne(features, TsneModel(**params).model_dump(), normalize_features)
    else:
        raise ValueError(f"Unknown kind {kind}")


def _scale(reduced, index) -> DataFrame:
    df = pd.DataFrame(reduced, index=index)
    return 2 * (df - df.min()) / (df.max() - df.min()) - 1


def compute_umap(features: DataFrame, params: dict, normalize_features: bool) -> DataFrame:
    values = StandardScaler().fit_transform(features) if normalize_features else features.values

    # cuML (GPU) when available; imported here and not at module level: the
    # prefork worker parent imports every task module and must never touch CUDA
    try:
        import cuml  # ty: ignore[unresolved-import]

        reducer = cuml.UMAP(**params, metric="cosine")
        print("Using cuML for UMAP computation")
    except Exception:
        reducer = umap.UMAP(**params, metric="cosine")
        print("Using standard UMAP for computation")

    return _scale(reducer.fit_transform(values), features.index)


def compute_tsne(features: DataFrame, params: dict, normalize_features: bool) -> DataFrame:
    values = StandardScaler().fit_transform(features) if normalize_features else features.values
    return _scale(TSNE(**params).fit_transform(values), features.index)
