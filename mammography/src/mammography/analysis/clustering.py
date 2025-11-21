import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, Any, Tuple, Optional

def run_pca(features: np.ndarray, n_components: int = 2, seed: int = 42) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(features)

def run_tsne(features: np.ndarray, n_components: int = 2, perplexity: float = 30.0, seed: int = 42) -> np.ndarray:
    # t-SNE is slow, consider using openTSNE if available, but sklearn is standard
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=seed, init="pca", learning_rate="auto")
    return tsne.fit_transform(features)

def find_optimal_k(features: np.ndarray, k_range: range = range(2, 10), seed: int = 42) -> Dict[str, Any]:
    results = []
    best_k = 2
    best_score = -1.0
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
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
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(features)
    return labels, kmeans
