#
# advanced_embeddings.py
# mammography-pipelines
#
# Advanced visualization functions: t-SNE plots, heatmaps, scatterplots, and more.
# DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
# It must NOT be used for clinical or medical diagnostic purposes.
# No medical decision should be based on these results.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""
Advanced visualization utilities for mammography pipeline analysis.

Provides:
- t-SNE, UMAP, PCA 2D/3D scatter plots
- Feature correlation heatmaps
- Confusion matrix heatmaps
- Distribution plots (histograms, KDE, violin)
- Interactive embedding exploration
- Pair plots and feature relationships
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

from ..utils.numpy_warnings import suppress_numpy_matmul_warnings
from ._seaborn_compat import _SeabornGrid, _SeabornStub, sns  # noqa: F401
from .dimensionality import (
    project_pca,
    project_tsne,
    project_umap,
)
from .primitives import (
    DPI,
    FIGSIZE_DEFAULT,
    FIGSIZE_SQUARE,
    FIGSIZE_WIDE,
    PALETTE,
)
from .primitives import (
    ensure_dir as _ensure_dir,
)


def _validate_labels_for_plot(
    labels: np.ndarray, n_samples: int, function_name: str
) -> np.ndarray:
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1 or labels_array.shape[0] != n_samples:
        raise ValueError(
            f"{function_name} requires labels to be a 1D array with length "
            "matching the number of plotted samples."
        )
    return labels_array


def plot_tsne_2d(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    learning_rate: Union[str, float] = "auto",
    seed: int = 42,
    title: str = "t-SNE Visualization",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = FIGSIZE_DEFAULT,
    palette: str = PALETTE,
    alpha: float = 0.7,
    point_size: int = 50,
    label_names: Optional[Dict[int, str]] = None,
) -> Tuple[np.ndarray, plt.Figure]:
    """
    Compute and plot 2D t-SNE embedding.

    Args:
        features: (N, D) array of embeddings
        labels: Optional (N,) array of class labels for coloring
        perplexity: t-SNE perplexity (typically 5-50)
        max_iter: Maximum number of iterations
        learning_rate: Learning rate or 'auto'
        seed: Random seed
        title: Plot title
        out_path: Save path (optional)
        figsize: Figure size
        palette: Color palette
        alpha: Point transparency
        point_size: Marker size
        label_names: Map from int labels to string names

    Returns:
        Tuple of (tsne_embedding, figure)
    """
    embedding, _ = project_tsne(
        features,
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        learning_rate=learning_rate,
        seed=seed,
    )

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        labels = _validate_labels_for_plot(labels, embedding.shape[0], "plot_tsne_2d")
        unique_labels = np.unique(labels)
        colors = sns.color_palette(palette, len(unique_labels))
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            name = label_names.get(lab, str(lab)) if label_names else str(lab)
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[colors[i]],
                label=name,
                alpha=alpha,
                s=point_size,
                edgecolors="white",
                linewidth=0.5,
            )
        ax.legend(title="Class", loc="best", framealpha=0.9)
    else:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            alpha=alpha,
            s=point_size,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")

    return embedding, fig


def plot_tsne_3d(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    seed: int = 42,
    title: str = "t-SNE 3D Visualization",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = FIGSIZE_SQUARE,
    palette: str = PALETTE,
    alpha: float = 0.7,
    point_size: int = 30,
    label_names: Optional[Dict[int, str]] = None,
    elevation: float = 20,
    azimuth: float = 45,
) -> Tuple[np.ndarray, plt.Figure]:
    """
    Create a 3D t-SNE embedding from feature vectors and render it as a 3D scatter plot.

    Parameters:
        features (np.ndarray): Array of shape (n_samples, n_features) containing input feature vectors.
        labels (Optional[np.ndarray]): Optional 1D array of class labels used to color points by class.
        perplexity (float): Perplexity parameter passed to t-SNE.
        max_iter (int): Maximum number of optimization iterations for t-SNE.
        seed (int): Random seed used for projection reproducibility.
        title (str): Plot title.
        out_path (Optional[str]): If provided, path where the figure will be saved.
        figsize (Tuple[int, int]): Figure size in inches.
        palette (str): Color palette name used for class colors.
        alpha (float): Marker transparency.
        point_size (int): Marker size for scatter points.
        label_names (Optional[Dict[int, str]]): Optional mapping from label values to display names for the legend.
        elevation (float): Elevation angle (in degrees) for the 3D view.
        azimuth (float): Azimuth angle (in degrees) for the 3D view.

    Returns:
        Tuple[np.ndarray, plt.Figure]:
            embedding — Array of shape (n_samples, 3) with the t-SNE coordinates.
            fig — Matplotlib Figure containing the rendered 3D scatter plot.
    """
    embedding, _ = project_tsne(
        features,
        n_components=3,
        perplexity=perplexity,
        max_iter=max_iter,
        learning_rate="auto",
        seed=seed,
    )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if labels is not None:
        labels = _validate_labels_for_plot(labels, embedding.shape[0], "plot_tsne_3d")
        unique_labels = np.unique(labels)
        colors = sns.color_palette(palette, len(unique_labels))
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            name = label_names.get(lab, str(lab)) if label_names else str(lab)
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                embedding[mask, 2],
                c=[colors[i]],
                label=name,
                alpha=alpha,
                s=point_size,
            )
        ax.legend(title="Class", loc="best")
    else:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            alpha=alpha,
            s=point_size,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.view_init(elev=elevation, azim=azimuth)
    fig.tight_layout()

    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")

    return embedding, fig


def compare_two_embeddings(
    features: np.ndarray,
    embedding2: np.ndarray,
    class_labels: Optional[np.ndarray] = None,
    *,
    method1_name: str = "Method 1",
    method2_name: str = "Method 2",
    figsize: Tuple[int, int] = (15, 5),
    alpha: float = 0.7,
    point_size: int = 30,
    palette: str = PALETTE,
    title: str = "Embedding Comparison",
    out_path: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], plt.Figure]:
    """Plot two precomputed 2D embeddings side-by-side for legacy callers."""
    features = np.asarray(features)
    embedding2 = np.asarray(embedding2)
    legacy_labels = np.asarray(class_labels) if class_labels is not None else None
    if legacy_labels is not None and (
        legacy_labels.ndim != 1 or legacy_labels.shape[0] != features.shape[0]
    ):
        raise ValueError(
            "class_labels must be a 1D array with length matching features.shape[0]."
        )
    embeddings_dict = {
        method1_name: features,
        method2_name: embedding2,
    }
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, emb, method_name in zip(
        axes,
        [features, embedding2],
        [method1_name, method2_name],
        strict=True,
    ):
        if legacy_labels is not None:
            unique_labels = np.unique(legacy_labels)
            colors = sns.color_palette(palette, len(unique_labels))
            for i, lab in enumerate(unique_labels):
                mask = legacy_labels == lab
                ax.scatter(
                    emb[mask, 0],
                    emb[mask, 1],
                    c=[colors[i]],
                    label=str(lab),
                    alpha=alpha,
                    s=point_size,
                    edgecolors="white",
                    linewidth=0.3,
                )
            ax.legend(title="Class", loc="best", fontsize=8)
        else:
            ax.scatter(emb[:, 0], emb[:, 1], alpha=alpha, s=point_size)
        ax.set_xlabel("Dim 1", fontsize=10)
        ax.set_ylabel("Dim 2", fontsize=10)
        ax.set_title(method_name, fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    return embeddings_dict, fig


def plot_embedding_comparison(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    methods: Optional[List[str]] = None,
    method1_name: str = "Method 1",
    method2_name: str = "Method 2",
    seed: int = 42,
    title: str = "Embedding Comparison",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    palette: str = PALETTE,
    alpha: float = 0.7,
    point_size: int = 30,
    label_names: Optional[Dict[int, str]] = None,
    legacy_mode: bool = False,
) -> Tuple[Dict[str, np.ndarray], plt.Figure]:
    """
    Compare multiple 2D embeddings produced by dimensionality-reduction methods and plot them side-by-side.

    Supported `methods` are `"pca"`, `"tsne"`, and `"umap"`. If UMAP is not installed, its subplot is marked as unavailable and skipped. In legacy mode, `features` and `labels` are treated as two precomputed 2D embeddings and `methods` is treated as class labels for coloring.

    Parameters:
        features: Feature matrix with shape (N, D).
        labels: Optional 1D array of class labels for coloring points, or a 2D embedding when `legacy_mode=True`.
        methods: Sequence of embedding method names to compute and plot. Order determines subplot arrangement.
        method1_name: Name to display for the first embedding when using the legacy branch.
        method2_name: Name to display for the second embedding when using the legacy branch.
        seed: Random seed passed to projection helpers.
        title: Overall figure title.
        out_path: If provided, path where the figure will be saved.
        figsize: Figure size in inches.
        palette: Color palette identifier passed to seaborn for class colors.
        alpha: Marker transparency.
        point_size: Marker size for scatter points.
        label_names: Optional mapping from label values to display names used in the legend.
        legacy_mode: Explicitly enable the historical precomputed-embedding comparison behavior. Passing a 2D `labels` array with scalar `methods` still enables legacy mode for compatibility, but now emits a deprecation warning.

    Returns:
        A tuple (embeddings_dict, fig) where `embeddings_dict` maps method identifiers (strings) to their (N, 2) numpy embeddings (for the legacy branch the provided embeddings are returned as-is), and `fig` is the Matplotlib Figure containing the plotted subplots.
    """
    features = np.asarray(features)
    labels_array = np.asarray(labels) if labels is not None else None
    methods_array = np.asarray(methods) if methods is not None else None
    implicit_legacy = False
    if (
        labels_array is not None
        and labels_array.ndim == 2
        and methods is not None
        and not isinstance(methods, list)
    ):
        if not isinstance(methods, (np.ndarray, tuple)):
            raise ValueError(
                "Legacy embedding comparison requires methods to be an array "
                "or tuple of labels matching the first dimension of labels; "
                "otherwise pass methods as a list of projection names."
            )
        implicit_legacy = (
            methods_array is not None
            and methods_array.ndim == 1
            and methods_array.shape[0] == labels_array.shape[0]
        )
        if not implicit_legacy and not legacy_mode:
            raise ValueError(
                "Ambiguous legacy embedding comparison call: labels is a 2D "
                "embedding, but methods is not a 1D label array with matching "
                "length. Pass methods as a list of projection names or set "
                "legacy_mode=True with valid class labels."
            )
    if implicit_legacy and not legacy_mode:
        warnings.warn(
            "Passing labels as a 2D embedding with methods as class labels is "
            "deprecated; pass legacy_mode=True to make this behavior explicit.",
            DeprecationWarning,
            stacklevel=2,
        )
        legacy_mode = True
    if legacy_mode:
        if labels_array is None or labels_array.ndim != 2:
            raise ValueError(
                "legacy_mode=True requires labels to be a 2D embedding array."
            )
        embedding2 = labels_array
        if methods is not None and not isinstance(methods, (np.ndarray, tuple)):
            raise ValueError(
                "legacy_mode=True requires methods to be an array or tuple of "
                "class labels, or None."
            )
        if methods_array is not None and (
            methods_array.ndim != 1 or methods_array.shape[0] != embedding2.shape[0]
        ):
            raise ValueError(
                "legacy_mode=True requires class labels to be a 1D array with "
                "length matching the second embedding."
            )
        legacy_labels = methods_array if methods_array is not None else None
        return compare_two_embeddings(
            features,
            embedding2,
            class_labels=legacy_labels,
            method1_name=method1_name,
            method2_name=method2_name,
            figsize=figsize,
            alpha=alpha,
            point_size=point_size,
            palette=palette,
            title=title,
            out_path=out_path,
        )

    if methods is None:
        methods = ["pca", "tsne", "umap"]
    if labels_array is not None:
        labels = _validate_labels_for_plot(
            labels_array, features.shape[0], "plot_embedding_comparison"
        )

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]

    embeddings = {}

    for ax, method in zip(axes, methods, strict=True):
        if method.lower() == "pca":
            emb, pca = project_pca(features, n_components=2, seed=seed)
            method_title = f"PCA (var: {pca.explained_variance_ratio_.sum():.1%})"
        elif method.lower() == "tsne":
            emb, _ = project_tsne(
                features,
                n_components=2,
                seed=seed,
                learning_rate="auto",
            )
            method_title = "t-SNE"
        elif method.lower() == "umap":
            try:
                emb, _ = project_umap(features, n_components=2, seed=seed)
                method_title = "UMAP"
            except (ImportError, ModuleNotFoundError):
                ax.text(
                    0.5,
                    0.5,
                    "UMAP not installed",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("UMAP (unavailable)")
                continue
        else:
            continue

        embeddings[method] = emb

        if labels is not None:
            unique_labels = np.unique(labels)
            colors = sns.color_palette(palette, len(unique_labels))
            for i, lab in enumerate(unique_labels):
                mask = labels == lab
                name = label_names.get(lab, str(lab)) if label_names else str(lab)
                ax.scatter(
                    emb[mask, 0],
                    emb[mask, 1],
                    c=[colors[i]],
                    label=name,
                    alpha=alpha,
                    s=point_size,
                    edgecolors="white",
                    linewidth=0.3,
                )
            ax.legend(title="Class", loc="best", fontsize=8)
        else:
            ax.scatter(emb[:, 0], emb[:, 1], alpha=alpha, s=point_size)

        ax.set_xlabel("Dim 1", fontsize=10)
        ax.set_ylabel("Dim 2", fontsize=10)
        ax.set_title(method_title, fontsize=12)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")

    return embeddings, fig


def plot_class_separation(
    features: np.ndarray,
    labels: np.ndarray,
    title: str = "Class Separation Analysis",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = FIGSIZE_WIDE,
    label_names: Optional[Dict[int, str]] = None,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create a 3-panel diagnostic figure showing class separation in feature space and return summary metrics.

    The figure contains:
    - A heatmap of pairwise Euclidean distances between class centroids (inter-class distances).
    - A bar chart of mean per-feature variance within each class (intra-class variance).
    - A bar chart of mean silhouette score per class with reference lines for common thresholds.

    Parameters:
        features (np.ndarray): Array of shape (n_samples, n_features) containing feature vectors.
        labels (np.ndarray): One-dimensional array of class labels for each sample.
        title (str): Figure super-title.
        out_path (Optional[str]): If provided, path where the figure will be saved.
        figsize (Tuple[int, int]): Figure size passed to matplotlib.
        label_names (Optional[Dict[int, str]]): Mapping from label value to display name used on axes and legends.

    Returns:
        Tuple[plt.Figure, Dict[str, Any]]: A tuple with the matplotlib Figure and a metrics dictionary containing:
            - "silhouette" (float): Global silhouette score, or 0.0 if not computable.
            - "davies_bouldin" (float): Davies-Bouldin score, or 0.0 if not computable.
            - "intra_class_variance" (float): Mean of per-class mean variances.
    """
    try:
        features = np.asarray(features, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "plot_class_separation requires features to contain numeric values."
        ) from exc
    labels = np.asarray(labels)
    if features.ndim != 2:
        raise ValueError("plot_class_separation requires features to be a 2D array.")
    if labels.ndim != 1:
        raise ValueError("plot_class_separation requires labels to be a 1D array.")
    if features.shape[0] == 0 or labels.size == 0:
        raise ValueError(
            "plot_class_separation requires non-empty features and labels."
        )
    if labels.shape[0] != features.shape[0]:
        raise ValueError(
            "plot_class_separation requires len(labels) to match features.shape[0]."
        )
    if not np.all(np.isfinite(features)):
        raise ValueError(
            "plot_class_separation requires finite feature values without NaN or inf."
        )

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    if n_classes <= 1:
        raise ValueError("plot_class_separation requires more than one unique label.")

    # Compute class centroids
    centroids = np.array(
        [features[labels == label].mean(axis=0) for label in unique_labels]
    )

    # Inter-class distance matrix
    inter_dist = cdist(centroids, centroids, metric="euclidean")

    # Intra-class variance
    intra_var = [
        features[labels == label].var(axis=0).mean() for label in unique_labels
    ]

    n_samples = len(labels)
    if 1 < n_classes < n_samples:
        with suppress_numpy_matmul_warnings():
            silhouette_vals = silhouette_samples(features, labels)
            silhouette = float(silhouette_score(features, labels))
            davies_bouldin = float(davies_bouldin_score(features, labels))
    else:
        silhouette_vals = np.array([], dtype=float)
        silhouette = 0.0
        davies_bouldin = 0.0

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Inter-class distance heatmap
    names = [
        label_names.get(label, str(label)) if label_names else str(label)
        for label in unique_labels
    ]
    sns.heatmap(
        inter_dist,
        ax=axes[0],
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=names,
        yticklabels=names,
        square=True,
    )
    axes[0].set_title("Inter-class Distances", fontsize=12)

    # 2. Intra-class variance bar
    axes[1].bar(names, intra_var, color=sns.color_palette("viridis", n_classes))
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Mean Variance")
    axes[1].set_title("Intra-class Variance", fontsize=12)
    axes[1].tick_params(axis="x", rotation=45)

    # 3. Silhouette scores per class
    if silhouette_vals.size:
        silhouette_per_class = [
            silhouette_vals[labels == label].mean() for label in unique_labels
        ]
    else:
        silhouette_per_class = [0.0 for _ in unique_labels]
    colors = [
        "green" if s > 0.5 else "orange" if s > 0.25 else "red"
        for s in silhouette_per_class
    ]
    axes[2].bar(names, silhouette_per_class, color=colors)
    axes[2].axhline(
        y=0.5, color="green", linestyle="--", alpha=0.5, label="Good (>0.5)"
    )
    axes[2].axhline(
        y=0.25, color="orange", linestyle="--", alpha=0.5, label="Fair (>0.25)"
    )
    axes[2].set_xlabel("Class")
    axes[2].set_ylabel("Silhouette Score")
    axes[2].set_title("Silhouette per Class", fontsize=12)
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend(fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")

    return fig, {
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "intra_class_variance": float(np.mean(intra_var)),
    }
