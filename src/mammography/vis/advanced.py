#
# advanced.py
# mammography-pipelines-py
#
# Advanced visualization functions: t-SNE plots, heatmaps, scatterplots, and more.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# Default style configuration
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = "viridis"
FIGSIZE_DEFAULT = (10, 8)
FIGSIZE_WIDE = (14, 8)
FIGSIZE_SQUARE = (10, 10)
DPI = 150


def _ensure_dir(path: Union[str, Path]) -> Path:
    """Create output directory if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_tsne_2d(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    perplexity: float = 30.0,
    n_iter: int = 1000,
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
        n_iter: Number of iterations
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
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        learning_rate=learning_rate,
        random_state=seed,
        init="pca",
    )
    embedding = tsne.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
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
    """Compute and plot 3D t-SNE embedding."""
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    embedding = tsne.fit_transform(features)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    
    if labels is not None:
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


def plot_heatmap_correlation(
    features: np.ndarray,
    feature_names: Optional[List[str]] = None,
    method: str = "pearson",
    title: str = "Feature Correlation Heatmap",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = FIGSIZE_SQUARE,
    cmap: str = "RdBu_r",
    annot: bool = True,
    max_features: int = 50,
) -> plt.Figure:
    """
    Plot correlation heatmap of features.
    
    Args:
        features: (N, D) array
        feature_names: Names for each dimension
        method: 'pearson', 'spearman', or 'kendall'
        title: Plot title
        out_path: Save path
        figsize: Figure size
        cmap: Colormap
        annot: Show correlation values
        max_features: Limit features for readability
    """
    if features.shape[1] > max_features:
        # Use PCA to reduce for visualization
        pca = PCA(n_components=max_features, random_state=42)
        features = pca.fit_transform(features)
        feature_names = [f"PC{i+1}" for i in range(max_features)]
    
    df = pd.DataFrame(features, columns=feature_names or [f"F{i}" for i in range(features.shape[1])])
    corr = df.corr(method=method)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Adjust annotation based on size
    annot_kws = {"size": 8} if features.shape[1] <= 20 else {"size": 6}
    show_annot = annot and features.shape[1] <= 30
    
    sns.heatmap(
        corr,
        ax=ax,
        cmap=cmap,
        annot=show_annot,
        annot_kws=annot_kws,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": f"{method.capitalize()} Correlation"},
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    
    return fig


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: str = "true",
    title: str = "Confusion Matrix",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 7),
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for each class
        normalize: 'true', 'pred', 'all', or None
        title: Plot title
        out_path: Save path
        figsize: Figure size
        cmap: Colormap
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    fmt = ".2f" if normalize else "d"
    sns.heatmap(
        cm,
        ax=ax,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    
    return fig


def plot_feature_heatmap(
    features: np.ndarray,
    sample_ids: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    title: str = "Feature Heatmap",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = FIGSIZE_WIDE,
    cmap: str = "viridis",
    max_samples: int = 100,
    max_features: int = 50,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
) -> plt.Figure:
    """
    Plot heatmap of feature values across samples.
    
    Useful for seeing patterns in high-dimensional embeddings.
    """
    # Subsample for visualization
    if features.shape[0] > max_samples:
        idx = np.random.choice(features.shape[0], max_samples, replace=False)
        features = features[idx]
        if sample_ids:
            sample_ids = [sample_ids[i] for i in idx]
    
    if features.shape[1] > max_features:
        pca = PCA(n_components=max_features, random_state=42)
        features = pca.fit_transform(features)
        feature_names = [f"PC{i+1}" for i in range(max_features)]
    
    df = pd.DataFrame(
        features,
        index=sample_ids or [f"S{i}" for i in range(features.shape[0])],
        columns=feature_names or [f"F{i}" for i in range(features.shape[1])],
    )
    
    g = sns.clustermap(
        df,
        figsize=figsize,
        cmap=cmap,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        dendrogram_ratio=0.15,
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        linewidths=0,
    )
    g.fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    
    if out_path:
        _ensure_dir(out_path)
        g.fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    
    return g.fig


def plot_scatter_matrix(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    title: str = "Scatter Matrix",
    out_path: Optional[str] = None,
    max_features: int = 6,
    palette: str = PALETTE,
    alpha: float = 0.6,
    label_names: Optional[Dict[int, str]] = None,
) -> plt.Figure:
    """
    Create pairwise scatter plot matrix.
    
    Args:
        features: (N, D) array
        labels: Optional class labels
        feature_names: Feature dimension names
        title: Plot title
        out_path: Save path
        max_features: Max dimensions to show (uses PCA if needed)
        palette: Color palette
        alpha: Point transparency
        label_names: Map from int labels to string names
    """
    if features.shape[1] > max_features:
        pca = PCA(n_components=max_features, random_state=42)
        features = pca.fit_transform(features)
        feature_names = [f"PC{i+1}" for i in range(max_features)]
    
    df = pd.DataFrame(features, columns=feature_names or [f"D{i}" for i in range(features.shape[1])])
    
    if labels is not None:
        if label_names:
            df["Class"] = [label_names.get(l, str(l)) for l in labels]
        else:
            df["Class"] = labels.astype(str)
        g = sns.pairplot(df, hue="Class", palette=palette, plot_kws={"alpha": alpha, "s": 20})
    else:
        g = sns.pairplot(df, plot_kws={"alpha": alpha, "s": 20})
    
    g.fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")
    
    if out_path:
        _ensure_dir(out_path)
        g.fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    
    return g.fig


def plot_distribution(
    values: np.ndarray,
    labels: Optional[np.ndarray] = None,
    kind: str = "hist",
    title: str = "Distribution",
    xlabel: str = "Value",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = FIGSIZE_DEFAULT,
    palette: str = PALETTE,
    bins: int = 50,
    label_names: Optional[Dict[int, str]] = None,
) -> plt.Figure:
    """
    Plot distribution of values.
    
    Args:
        values: 1D array of values
        labels: Optional class labels for grouping
        kind: 'hist', 'kde', 'violin', 'box'
        title: Plot title
        xlabel: X-axis label
        out_path: Save path
        figsize: Figure size
        palette: Color palette
        bins: Number of histogram bins
        label_names: Map from int labels to string names
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        df = pd.DataFrame({"value": values, "label": labels})
        if label_names:
            df["label"] = df["label"].map(lambda x: label_names.get(x, str(x)))
        
        if kind == "hist":
            for lab in df["label"].unique():
                subset = df[df["label"] == lab]["value"]
                ax.hist(subset, bins=bins, alpha=0.6, label=str(lab))
            ax.legend(title="Class")
        elif kind == "kde":
            for lab in df["label"].unique():
                subset = df[df["label"] == lab]["value"]
                sns.kdeplot(subset, ax=ax, label=str(lab), fill=True, alpha=0.3)
            ax.legend(title="Class")
        elif kind == "violin":
            sns.violinplot(data=df, x="label", y="value", ax=ax, palette=palette)
        elif kind == "box":
            sns.boxplot(data=df, x="label", y="value", ax=ax, palette=palette)
    else:
        if kind == "hist":
            ax.hist(values, bins=bins, alpha=0.7, edgecolor="white")
        elif kind == "kde":
            sns.kdeplot(values, ax=ax, fill=True, alpha=0.5)
        elif kind == "violin":
            sns.violinplot(y=values, ax=ax)
        elif kind == "box":
            sns.boxplot(y=values, ax=ax)
    
    ax.set_xlabel(xlabel if kind in ["hist", "kde"] else "Class", fontsize=12)
    ax.set_ylabel("Density" if kind in ["hist", "kde"] else xlabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    
    return fig


def plot_embedding_comparison(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    methods: List[str] = ["pca", "tsne", "umap"],
    seed: int = 42,
    title: str = "Embedding Comparison",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    palette: str = PALETTE,
    alpha: float = 0.7,
    point_size: int = 30,
    label_names: Optional[Dict[int, str]] = None,
) -> Tuple[Dict[str, np.ndarray], plt.Figure]:
    """
    Compare multiple dimensionality reduction methods side by side.
    
    Args:
        features: (N, D) feature array
        labels: Optional class labels
        methods: List of methods: 'pca', 'tsne', 'umap'
        seed: Random seed
        title: Overall title
        out_path: Save path
        figsize: Figure size
        palette: Color palette
        alpha: Point transparency
        point_size: Marker size
        label_names: Map from int labels to string names
    
    Returns:
        Tuple of (embeddings_dict, figure)
    """
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    
    embeddings = {}
    
    for ax, method in zip(axes, methods):
        if method.lower() == "pca":
            pca = PCA(n_components=2, random_state=seed)
            emb = pca.fit_transform(features)
            method_title = f"PCA (var: {pca.explained_variance_ratio_.sum():.1%})"
        elif method.lower() == "tsne":
            tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
            emb = tsne.fit_transform(features)
            method_title = "t-SNE"
        elif method.lower() == "umap":
            try:
                from umap import UMAP
                umap_model = UMAP(n_components=2, random_state=seed)
                emb = umap_model.fit_transform(features)
                method_title = "UMAP"
            except ImportError:
                ax.text(0.5, 0.5, "UMAP not installed", ha="center", va="center", transform=ax.transAxes)
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
) -> plt.Figure:
    """
    Analyze and visualize class separation in feature space.
    
    Shows:
    - Inter-class distances
    - Intra-class variance
    - Silhouette scores per class
    """
    from sklearn.metrics import silhouette_samples
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Compute class centroids
    centroids = np.array([features[labels == l].mean(axis=0) for l in unique_labels])
    
    # Inter-class distance matrix
    from scipy.spatial.distance import cdist
    inter_dist = cdist(centroids, centroids, metric="euclidean")
    
    # Intra-class variance
    intra_var = [features[labels == l].var(axis=0).mean() for l in unique_labels]
    
    # Silhouette scores
    silhouette_vals = silhouette_samples(features, labels)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Inter-class distance heatmap
    names = [label_names.get(l, str(l)) if label_names else str(l) for l in unique_labels]
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
    silhouette_per_class = [silhouette_vals[labels == l].mean() for l in unique_labels]
    colors = ["green" if s > 0.5 else "orange" if s > 0.25 else "red" for s in silhouette_per_class]
    axes[2].bar(names, silhouette_per_class, color=colors)
    axes[2].axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="Good (>0.5)")
    axes[2].axhline(y=0.25, color="orange", linestyle="--", alpha=0.5, label="Fair (>0.25)")
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
    
    return fig


def plot_feature_importance(
    importance_scores: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 20,
    title: str = "Feature Importance",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    color: str = "steelblue",
) -> plt.Figure:
    """
    Plot feature importance scores as horizontal bar chart.
    """
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance_scores))]
    
    # Sort by importance
    indices = np.argsort(importance_scores)[-top_k:]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importance_scores[indices], color=color, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Add value labels
    for i, (idx, score) in enumerate(zip(indices, importance_scores[indices])):
        ax.text(score + 0.01, i, f"{score:.3f}", va="center", fontsize=9)
    
    fig.tight_layout()
    
    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    
    return fig


def plot_learning_curves(
    history: List[Dict[str, Any]],
    metrics: List[str] = ["loss", "acc"],
    title: str = "Learning Curves",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = FIGSIZE_WIDE,
) -> plt.Figure:
    """
    Plot training and validation learning curves.
    
    Args:
        history: List of dicts with keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        metrics: Which metrics to plot
        title: Plot title
        out_path: Save path
        figsize: Figure size
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    epochs = list(range(1, len(history) + 1))
    
    for ax, metric in zip(axes, metrics):
        train_key = f"train_{metric}"
        val_key = f"val_{metric}"
        
        if train_key in history[0]:
            train_vals = [h.get(train_key, np.nan) for h in history]
            ax.plot(epochs, train_vals, "b-", label="Train", linewidth=2, marker="o", markersize=4)
        
        if val_key in history[0]:
            val_vals = [h.get(val_key, np.nan) for h in history]
            ax.plot(epochs, val_vals, "r-", label="Validation", linewidth=2, marker="s", markersize=4)
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f"{metric.capitalize()} Curves", fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    
    return fig


def generate_visualization_report(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    metadata: Optional[pd.DataFrame] = None,
    output_dir: Union[str, Path] = "vis_report",
    seed: int = 42,
    label_names: Optional[Dict[int, str]] = None,
) -> Dict[str, str]:
    """
    Generate a comprehensive visualization report.
    
    Creates multiple visualizations and saves them to the output directory.
    
    Args:
        features: (N, D) feature array
        labels: Optional class labels
        metadata: Optional metadata DataFrame
        output_dir: Output directory
        seed: Random seed
        label_names: Map from int labels to string names
    
    Returns:
        Dict mapping visualization names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_paths = {}
    
    # 1. Embedding comparison (PCA, t-SNE, UMAP)
    print("Generating embedding comparison...")
    _, _ = plot_embedding_comparison(
        features,
        labels,
        methods=["pca", "tsne"],
        seed=seed,
        out_path=str(output_dir / "embedding_comparison.png"),
        label_names=label_names,
    )
    report_paths["embedding_comparison"] = str(output_dir / "embedding_comparison.png")
    
    # 2. t-SNE 2D detailed
    print("Generating t-SNE 2D plot...")
    _, _ = plot_tsne_2d(
        features,
        labels,
        seed=seed,
        out_path=str(output_dir / "tsne_2d.png"),
        label_names=label_names,
    )
    report_paths["tsne_2d"] = str(output_dir / "tsne_2d.png")
    
    # 3. Correlation heatmap
    print("Generating correlation heatmap...")
    _ = plot_heatmap_correlation(
        features,
        out_path=str(output_dir / "correlation_heatmap.png"),
    )
    report_paths["correlation_heatmap"] = str(output_dir / "correlation_heatmap.png")
    
    # 4. Feature heatmap
    print("Generating feature heatmap...")
    _ = plot_feature_heatmap(
        features,
        out_path=str(output_dir / "feature_heatmap.png"),
    )
    report_paths["feature_heatmap"] = str(output_dir / "feature_heatmap.png")
    
    # 5. Scatter matrix
    print("Generating scatter matrix...")
    _ = plot_scatter_matrix(
        features,
        labels,
        out_path=str(output_dir / "scatter_matrix.png"),
        label_names=label_names,
    )
    report_paths["scatter_matrix"] = str(output_dir / "scatter_matrix.png")
    
    # 6. Class separation (if labels provided)
    if labels is not None:
        print("Generating class separation analysis...")
        _ = plot_class_separation(
            features,
            labels,
            out_path=str(output_dir / "class_separation.png"),
            label_names=label_names,
        )
        report_paths["class_separation"] = str(output_dir / "class_separation.png")
    
    # 7. First PC distribution
    print("Generating PC1 distribution...")
    pca = PCA(n_components=1, random_state=seed)
    pc1 = pca.fit_transform(features).ravel()
    _ = plot_distribution(
        pc1,
        labels,
        kind="kde",
        title="First Principal Component Distribution",
        xlabel="PC1",
        out_path=str(output_dir / "pc1_distribution.png"),
        label_names=label_names,
    )
    report_paths["pc1_distribution"] = str(output_dir / "pc1_distribution.png")
    
    print(f"Visualization report saved to {output_dir}")
    return report_paths


# Export all public functions
__all__ = [
    "plot_tsne_2d",
    "plot_tsne_3d",
    "plot_heatmap_correlation",
    "plot_confusion_matrix_heatmap",
    "plot_feature_heatmap",
    "plot_scatter_matrix",
    "plot_distribution",
    "plot_embedding_comparison",
    "plot_class_separation",
    "plot_feature_importance",
    "plot_learning_curves",
    "generate_visualization_report",
]
