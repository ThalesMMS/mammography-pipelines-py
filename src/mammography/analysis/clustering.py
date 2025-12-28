#
# clustering.py
# mammography-pipelines
#
# Implements lightweight PCA/t-SNE/UMAP projections and k-means utilities for embedding analysis.
#
# Thales Matheus Mendonça Santos - November 2025
#
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, Any, Tuple, Optional

from ..utils.numpy_warnings import suppress_numpy_matmul_warnings, resolve_pca_svd_solver

def run_pca(
    features: np.ndarray,
    n_components: int = 2,
    seed: int = 42,
    svd_solver: str | None = "auto",
) -> np.ndarray:
    """Standard PCA wrapper that preserves the configured random seed."""
    solver = resolve_pca_svd_solver(
        features.shape[0], features.shape[1], n_components, svd_solver
    )
    pca = PCA(n_components=n_components, random_state=seed, svd_solver=solver)
    with suppress_numpy_matmul_warnings():
        return pca.fit_transform(features)

def run_tsne(features: np.ndarray, n_components: int = 2, perplexity: float = 30.0, seed: int = 42) -> np.ndarray:
    # t-SNE is slow; sklearn keeps dependencies minimal even if openTSNE is faster.
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=seed, init="pca", learning_rate="auto")
    with suppress_numpy_matmul_warnings():
        return tsne.fit_transform(features)

def run_umap(features: np.ndarray, n_components: int = 2, seed: int = 42) -> np.ndarray:
    try:
        from umap import UMAP  # type: ignore
    except Exception as exc:
        raise ImportError(f"umap-learn não disponível: {exc}")
    umap_model = UMAP(n_components=n_components, random_state=seed)
    with suppress_numpy_matmul_warnings():
        return umap_model.fit_transform(features)

def find_optimal_k(features: np.ndarray, k_range: range = range(2, 10), seed: int = 42) -> Dict[str, Any]:
    """Sweep K and report the configuration that maximizes silhouette score."""
    results = []
    best_k = 2
    best_score = -1.0
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        with suppress_numpy_matmul_warnings():
            labels = kmeans.fit_predict(features)
            sil = silhouette_score(features, labels)
            db = davies_bouldin_score(features, labels)
        
        results.append({
            "k": k,
            "silhouette": sil,
            "davies_bouldin": db
        })
        
        if sil > best_score:
            best_score = sil
            best_k = k
            
    return {
        "best_k": best_k,
        "best_score": best_score,
        "history": results
    }

def run_kmeans(features: np.ndarray, k: int, seed: int = 42) -> Tuple[np.ndarray, Any]:
    """Fit k-means and return both the labels and fitted estimator."""
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    with suppress_numpy_matmul_warnings():
        labels = kmeans.fit_predict(features)
    return labels, kmeans
