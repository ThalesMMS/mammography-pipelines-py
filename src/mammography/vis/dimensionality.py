"""Shared dimensionality reduction helpers for visualization modules."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mammography.utils.numpy_warnings import (
    resolve_pca_svd_solver,
    suppress_numpy_matmul_warnings,
)


def build_pca(
    features: np.ndarray,
    n_components: int,
    seed: int,
    svd_solver: str | None = "auto",
) -> PCA:
    """Build a PCA reducer with component count clipped to the input shape."""
    max_components = min(n_components, features.shape[0], features.shape[1])
    max_components = max(1, max_components)
    solver = resolve_pca_svd_solver(
        features.shape[0], features.shape[1], max_components, svd_solver
    )
    return PCA(n_components=max_components, random_state=seed, svd_solver=solver)


def resolve_tsne_perplexity(n_samples: int, perplexity: float) -> float:
    """Clamp t-SNE perplexity to a value valid for the sample count."""
    if n_samples <= 1:
        return 1.0
    max_perplexity = max(1.0, (n_samples - 1) / 3)
    return min(perplexity, max_perplexity)


def project_pca(
    features: np.ndarray,
    *,
    n_components: int = 2,
    seed: int = 42,
    pca_params: dict[str, Any] | None = None,
    model: PCA | None = None,
) -> tuple[np.ndarray, PCA]:
    """Fit and transform features with PCA."""
    if model is not None:
        if pca_params:
            raise ValueError("pca_params cannot be supplied with a pre-fitted PCA model")
        with suppress_numpy_matmul_warnings():
            embedding = model.transform(features)
        return embedding, model

    params = dict(pca_params or {})
    requested_components = int(params.pop("n_components", n_components))
    seed = int(params.pop("random_state", seed))
    svd_solver = params.pop("svd_solver", "auto")
    pca = build_pca(features, requested_components, seed, svd_solver)
    with suppress_numpy_matmul_warnings():
        embedding = pca.fit_transform(features)
    return embedding, pca


def project_tsne(
    features: np.ndarray,
    *,
    n_components: int = 2,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    learning_rate: str | float = "auto",
    seed: int = 42,
) -> tuple[np.ndarray, TSNE]:
    """Fit and transform features with t-SNE."""
    tsne = TSNE(
        n_components=n_components,
        perplexity=resolve_tsne_perplexity(features.shape[0], perplexity),
        max_iter=max_iter,
        learning_rate=learning_rate,
        random_state=seed,
        init="pca",
    )
    with suppress_numpy_matmul_warnings():
        embedding = tsne.fit_transform(features)
    return embedding, tsne


def build_umap(
    *,
    n_components: int = 2,
    seed: int = 42,
    umap_params: dict[str, Any] | None = None,
):
    """Build a UMAP reducer, importing the optional dependency lazily."""
    from umap import UMAP

    params = dict(umap_params or {})
    params["n_components"] = n_components
    params.setdefault("random_state", seed)
    return UMAP(**params)


def project_umap(
    features: np.ndarray,
    *,
    n_components: int = 2,
    seed: int = 42,
    umap_params: dict[str, Any] | None = None,
    model: Any | None = None,
) -> tuple[np.ndarray, Any]:
    """Fit and transform features with UMAP."""
    if model is not None:
        reducer = model
        with suppress_numpy_matmul_warnings():
            embedding = reducer.transform(features)
        return embedding, reducer

    reducer = build_umap(
        n_components=n_components,
        seed=seed,
        umap_params=umap_params,
    )
    with suppress_numpy_matmul_warnings():
        embedding = reducer.fit_transform(features)
    return embedding, reducer
