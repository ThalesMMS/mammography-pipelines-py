# ruff: noqa
#
# advanced_learning.py
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

from typing import Optional, List, Dict, Any, Union, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ._seaborn_compat import _SeabornGrid, _SeabornStub, sns

from .primitives import (
    DPI,
    FIGSIZE_WIDE,
    ensure_dir as _ensure_dir,
)


def plot_learning_curves_from_arrays(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: Optional[np.ndarray] = None,
    *,
    title: str = "Learning Curves",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = FIGSIZE_WIDE,
) -> plt.Figure:
    """
    Create a learning-curve figure from arrays of training sizes and corresponding scores.
    
    Plots training scores and, if provided, validation scores against training size; sets axis labels, title, legend, and grid, applies a tight layout, and optionally saves the figure to `out_path` (parent directories will be created).
    
    Parameters:
        train_sizes: Array-like of training set sizes corresponding to each score point.
        train_scores: Array-like of training scores to plot against `train_sizes`.
        val_scores: Optional array-like of validation scores to plot against `train_sizes`.
        title: Title to use for the figure.
        out_path: If provided, path where the figure will be saved.
        figsize: Figure size passed to Matplotlib.
    
    Returns:
        fig: The Matplotlib Figure containing the plotted learning curves.
    """
    train_sizes = np.asarray(train_sizes)
    train_scores = np.asarray(train_scores)
    val_scores = None if val_scores is None else np.asarray(val_scores)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train_sizes, train_scores, "b-", label="Train", linewidth=2, marker="o", markersize=4)
    if val_scores is not None:
        ax.plot(train_sizes, val_scores, "r-", label="Validation", linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Training Size", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    return fig

def plot_learning_curves(
    history: Union[List[Dict[str, Any]], np.ndarray, List[float]],
    metrics: Optional[Union[List[str], np.ndarray, List[float]]] = None,
    title: str = "Learning Curves",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = FIGSIZE_WIDE,
    validation_scores: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Plot training and validation learning curves.

    Supports two input modes:
    - Epoch history mode: ``history`` is a list of dicts containing keys like
      ``train_loss``, ``val_loss``, ``train_acc``, and ``val_acc``; ``metrics``
      is a list of metric names to plot.
    - Array mode: ``history`` is train sizes and ``metrics`` is train scores;
      pass validation scores explicitly via ``validation_scores``.

    Args:
        history: Epoch history dictionaries or training-size array
        metrics: Metric names for history mode or train-score array for array mode
        title: Plot title
        out_path: Save path
        figsize: Figure size
        validation_scores: Optional validation scores for array mode
    """
    if len(history) == 0:
        raise ValueError("history must not be empty")

    if len(history) > 0 and not isinstance(history[0], dict):
        if not isinstance(title, str):
            raise TypeError(
                "title must be a string; pass validation scores with validation_scores="
            )
        if metrics is None:
            raise ValueError("metrics must contain train scores in array mode")
        return plot_learning_curves_from_arrays(
            np.asarray(history),
            np.asarray(metrics),
            validation_scores,
            title=title,
            out_path=out_path,
            figsize=figsize,
        )

    metrics = ["loss", "acc"] if metrics is None else metrics
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
