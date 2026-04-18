#
# advanced_heatmaps.py
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

from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ._seaborn_compat import _SeabornGrid, _SeabornStub, sns  # noqa: F401
from .dimensionality import (
    project_pca,
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
    Compute and render a correlation heatmap for the given feature matrix.

    If the number of features exceeds `max_features`, the features are reduced with PCA for visualization and feature names are set to `"PC1", "PC2", ..."`. Annotation values are shown only when `annot` is True and the (possibly reduced) feature count is <= 30.

    Parameters:
        features (np.ndarray): Array of shape (n_samples, n_features) containing feature vectors.
        feature_names (Optional[List[str]]): Optional names for each feature; when PCA reduction occurs, names are replaced with `"PCi"` labels.
        method (str): Correlation method to compute (`"pearson"`, `"spearman"`, or `"kendall"`).
        title (str): Plot title.
        out_path (Optional[str]): If provided, path to save the resulting figure (directory will be created if needed).
        figsize (Tuple[int, int]): Figure size in inches.
        cmap (str): Matplotlib colormap name for the heatmap.
        annot (bool): Whether to display numeric correlation annotations (suppressed automatically for large feature counts).
        max_features (int): Maximum number of features shown; larger feature sets are reduced via PCA for readability.

    Returns:
        matplotlib.figure.Figure: The Matplotlib Figure containing the rendered heatmap.
    """
    if features.shape[1] > max_features:
        # Use PCA to reduce for visualization
        features, _ = project_pca(features, n_components=max_features, seed=42)
        feature_names = [f"PC{i + 1}" for i in range(features.shape[1])]

    df = pd.DataFrame(
        features, columns=feature_names or [f"F{i}" for i in range(features.shape[1])]
    )
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
    Render a confusion matrix as an annotated heatmap.

    Parameters:
        y_true (np.ndarray): Ground-truth class labels.
        y_pred (np.ndarray): Predicted class labels.
        class_names (Optional[List[str]]): Labels for classes; if omitted, names are generated as "Class 0", "Class 1", ....
        normalize (str | None): One of 'true', 'pred', 'all', or None. When set to 'true', 'pred', or 'all' the matrix is normalized accordingly; when None the raw counts are shown.
        title (str): Figure title.
        out_path (Optional[str]): If provided, the figure is saved to this path.
        figsize (Tuple[int, int]): Size of the created figure.
        cmap (str): Matplotlib colormap name to use for the heatmap.

    Returns:
        matplotlib.figure.Figure: The figure containing the plotted confusion matrix.
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
    seed: Optional[int] = None,
) -> plt.Figure:
    """
    Render a clustered heatmap of sample-by-feature values.

    Constructs a DataFrame from `features` (rows = samples, columns = features), optionally subsampling rows when the number of samples exceeds `max_samples`, and optionally reducing feature dimensionality via PCA when the number of features exceeds `max_features`. The result is shown as a seaborn clustermap with configurable row/column clustering and colormap; the figure is saved to `out_path` when provided.

    Parameters:
        features (np.ndarray): 2D array of shape (n_samples, n_features).
        sample_ids (Optional[List[str]]): Optional list of sample identifiers to use as row labels. When omitted, rows are labeled "S0", "S1", ....
        feature_names (Optional[List[str]]): Optional list of feature names to use as column labels. When omitted, columns are labeled "F0", "F1", ...; if dimensionality is reduced, names become "PC1", "PC2", ....
        title (str): Figure title.
        out_path (Optional[str]): File path to save the resulting figure; directory will be created if needed. If `None`, the figure is not saved.
        figsize (Tuple[int,int]): Figure size passed to the clustermap.
        cmap (str): Colormap name for the heatmap.
        max_samples (int): Maximum number of rows to display; when `features.shape[0] > max_samples` a random subset of rows is selected for plotting.
        max_features (int): Maximum number of columns to display; when `features.shape[1] > max_features` PCA is applied to reduce dimensionality for visualization.
        cluster_rows (bool): Whether to perform hierarchical clustering on rows.
        cluster_cols (bool): Whether to perform hierarchical clustering on columns.
        seed (Optional[int]): Optional seed for reproducible row subsampling.

    Returns:
        matplotlib.figure.Figure: The Matplotlib Figure object containing the clustermap.
    """
    # Subsample for visualization
    if features.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(features.shape[0], max_samples, replace=False)
        features = features[idx]
        if sample_ids is not None:
            sample_ids = [sample_ids[i] for i in idx]

    if features.shape[1] > max_features:
        features, _ = project_pca(features, n_components=max_features, seed=42)
        feature_names = [f"PC{i + 1}" for i in range(features.shape[1])]

    df = pd.DataFrame(
        features,
        index=sample_ids
        if sample_ids is not None
        else [f"S{i}" for i in range(features.shape[0])],
        columns=feature_names
        if feature_names is not None
        else [f"F{i}" for i in range(features.shape[1])],
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
    Create a pairwise scatter plot matrix for feature dimensions, optionally colored by class.

    If the feature dimensionality exceeds `max_features`, the features are reduced to `max_features` components for visualization. When `labels` is provided, points are colored by class; `label_names` can map label ids to display names.

    Parameters:
        features (np.ndarray): Array of shape (N, D) containing N samples and D features.
        labels (Optional[np.ndarray]): Optional array of length N with class labels for coloring.
        feature_names (Optional[List[str]]): Optional names for feature dimensions; defaults to generated names.
        title (str): Title placed above the pairplot.
        out_path (Optional[str]): File path to save the figure; directory will be created if needed.
        max_features (int): Maximum number of feature dimensions to display; higher dimensional inputs are reduced.
        palette (str): Color palette name or palette object for class coloring.
        alpha (float): Marker transparency for scatter points.
        label_names (Optional[Dict[int, str]]): Mapping from label ids to display strings used when `labels` is provided.

    Returns:
        matplotlib.figure.Figure: Figure object containing the generated scatter matrix.
    """
    if features.shape[1] > max_features:
        features, _ = project_pca(features, n_components=max_features, seed=42)
        feature_names = [f"PC{i + 1}" for i in range(max_features)]

    df = pd.DataFrame(
        features, columns=feature_names or [f"D{i}" for i in range(features.shape[1])]
    )

    if labels is not None:
        if label_names:
            df["Class"] = [label_names.get(label, str(label)) for label in labels]
        else:
            df["Class"] = labels.astype(str)
        g = sns.pairplot(
            df, hue="Class", palette=palette, plot_kws={"alpha": alpha, "s": 20}
        )
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
    feature_idx: Optional[int] = None,
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
    Plot a distribution for a single variable (or a selected feature column) with optional grouping by class.

    If `values` is multi-dimensional, the function selects a single feature column: when `feature_idx` is None the first column (index 0) is used; when provided it must be an integer and supports negative indexing. Raises ValueError if `feature_idx` has an invalid type or is out of range.

    Supported `kind` values:
    - "hist": histogram
    - "kde": kernel density estimate
    - "violin": violin plot
    - "box": box plot

    If `labels` is provided, plots are grouped by class; `label_names` can map label identifiers to display names. If `out_path` is given the figure is saved to that path (the directory will be created if needed).

    Parameters:
        values (np.ndarray): 1D array of values or 2D array of shape (n_samples, n_features).
        labels (Optional[np.ndarray]): Optional per-sample class labels for grouped plots.
        feature_idx (Optional[int]): Index of the feature to plot when `values` is 2D; supports negative indices.
        kind (str): One of "hist", "kde", "violin", "box" selecting the plot type.
        title (str): Plot title.
        xlabel (str): Label for the x axis (used for ungrouped plots or as the numeric label).
        out_path (Optional[str]): File path to save the rendered figure.
        figsize (Tuple[int, int]): Figure size in inches.
        palette (str): Color palette name used for categorical plots.
        bins (int): Number of bins for histogram plots.
        label_names (Optional[Dict[int, str]]): Mapping from label id to display name.

    Returns:
        matplotlib.figure.Figure: Figure containing the rendered distribution plot.
    """
    values = np.asarray(values)
    if values.ndim > 1:
        if feature_idx is None:
            idx = 0
        elif isinstance(feature_idx, bool) or not isinstance(
            feature_idx,
            (int, np.integer),
        ):
            raise ValueError(
                "feature_idx must be an integer for multidimensional values"
            )
        else:
            idx = int(feature_idx)
        if idx < 0:
            idx += values.shape[1]
        if idx < 0 or idx >= values.shape[1]:
            raise ValueError(
                f"feature_idx {feature_idx} is out of range for {values.shape[1]} features"
            )
        values = values[:, idx]

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
    Render a horizontal bar chart of the top-k feature importance scores.

    The function selects the highest `top_k` scores from `importance_scores`, plots them as a horizontal bar chart (largest to smallest along the axis of selected entries), annotates each bar with its numeric value, and optionally saves the figure to `out_path`.

    Parameters:
        importance_scores (np.ndarray): Array of numeric importance scores, one per feature.
        feature_names (Optional[List[str]]): Names for each feature. If omitted, defaults to "Feature 0", "Feature 1", ...
        top_k (int): Number of top-scoring features to display (selection is based on highest scores).
        title (str): Chart title.
        out_path (Optional[str]): Filesystem path to save the rendered figure; directory is created if necessary. If `None`, the figure is not saved.
        figsize (Tuple[int, int]): Figure size in inches.
        color (str): Color used for the bars.

    Returns:
        matplotlib.figure.Figure: The created Matplotlib Figure containing the bar chart.
    """
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance_scores))]
    elif len(feature_names) != len(importance_scores):
        raise ValueError(
            "feature_names length must match importance_scores length "
            f"({len(feature_names)} != {len(importance_scores)})"
        )

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
    x_min, x_max = ax.get_xlim()
    offset = max((x_max - x_min) * 0.01, 1e-6)
    ax.set_xlim(x_min, x_max + offset * 6)
    for i, score in enumerate(importance_scores[indices]):
        ax.text(score + offset, i, f"{score:.3f}", va="center", fontsize=9)

    fig.tight_layout()

    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")

    return fig
