# ruff: noqa
#
# model_comparison.py
# mammography-pipelines
#
# Model comparison utilities for loading checkpoints and aggregating metrics across multiple models.
# DISCLAIMER: Educational project only - NOT for clinical or medical diagnostic purposes.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""
Model comparison utilities for comparing multiple trained models.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.

Provides utilities for loading multiple model checkpoints, extracting metrics,
and preparing data structures for comparison visualizations and statistical testing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as exc:  # pragma: no cover - optional dashboard deps
    px = None
    go = None
    make_subplots = None
    _PLOTLY_IMPORT_ERROR = exc
else:
    _PLOTLY_IMPORT_ERROR = None

from mammography.utils.statistics import (
    aggregate_cv_metrics,
    compute_confidence_interval,
    effect_size_cohen_d,
)
from mammography.vis.model_comparison_engine import ModelComparisonEngine

LOGGER = logging.getLogger("mammography")

from mammography.vis.model_comparison_stats import ModelMetrics

def _require_plotly() -> None:
    """Raise ImportError if plotly is not available."""
    if px is None or go is None or make_subplots is None:
        raise ImportError(
            "Plotly is required for visualization features. "
            "Install with: pip install plotly"
        ) from _PLOTLY_IMPORT_ERROR

def create_metrics_comparison_chart(
    metrics_list: List[ModelMetrics],
    metrics: Optional[List[str]] = None,
    chart_type: str = "grouped",
    title: str = "Model Metrics Comparison",
    show_values: bool = True,
) -> go.Figure:
    """
    Render an interactive Plotly bar chart comparing selected performance metrics across multiple models.
    
    Supports "grouped" (side-by-side) and "stacked" bar layouts. By default compares core metrics ['accuracy', 'kappa', 'macro_f1', 'balanced_accuracy'] (excludes inverse metrics such as val_loss). Charts display metric values and sort models by macro_f1 when available.
    
    Parameters:
        metrics_list: List of ModelMetrics objects to include in the chart.
        metrics: Optional list of metric names to plot. Defaults to ['accuracy', 'kappa', 'macro_f1', 'balanced_accuracy'].
        chart_type: Either "grouped" or "stacked" to select bar layout. Default is "grouped".
        title: Chart title.
        show_values: If True, annotate bars with numeric values.
    
    Returns:
        Plotly Figure containing the interactive bar chart.
    
    Raises:
        ImportError: If plotly is not installed.
        ValueError: If metrics_list is empty or chart_type is not one of "grouped" or "stacked".
    """
    _require_plotly()

    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")

    if chart_type not in ["grouped", "stacked"]:
        raise ValueError(f"chart_type must be 'grouped' or 'stacked', got '{chart_type}'")

    if metrics is None:
        # Default metrics (exclude val_loss as it's inverse)
        metrics = ["accuracy", "kappa", "macro_f1", "balanced_accuracy"]

    # Prepare data
    data = []
    for m in metrics_list:
        row = {"model_name": m.model_name}
        for metric_name in metrics:
            value = getattr(m, metric_name, None)
            row[metric_name] = value if value is not None else 0.0
        data.append(row)

    df = pd.DataFrame(data)

    # Sort by macro_f1 if available
    if "macro_f1" in df.columns:
        df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    # Create bar chart
    fig = go.Figure()

    # Color palette for metrics
    colors = px.colors.qualitative.Set2

    if chart_type == "grouped":
        # Grouped bar chart (side-by-side bars for each metric)
        for idx, metric_name in enumerate(metrics):
            if metric_name not in df.columns:
                LOGGER.warning("Metric '%s' not found in data, skipping", metric_name)
                continue

            values = df[metric_name].values
            color = colors[idx % len(colors)]

            fig.add_trace(
                go.Bar(
                    name=metric_name.replace("_", " ").title(),
                    x=df["model_name"],
                    y=values,
                    marker_color=color,
                    text=[f"{v:.3f}" for v in values] if show_values else None,
                    textposition="outside" if show_values else None,
                    textfont=dict(size=10),
                )
            )

        fig.update_layout(
            barmode="group",
            bargap=0.15,
            bargroupgap=0.1,
        )

    else:  # stacked
        # Stacked bar chart
        for idx, metric_name in enumerate(metrics):
            if metric_name not in df.columns:
                LOGGER.warning("Metric '%s' not found in data, skipping", metric_name)
                continue

            values = df[metric_name].values
            color = colors[idx % len(colors)]

            fig.add_trace(
                go.Bar(
                    name=metric_name.replace("_", " ").title(),
                    x=df["model_name"],
                    y=values,
                    marker_color=color,
                    text=[f"{v:.3f}" for v in values] if show_values else None,
                    textposition="inside" if show_values else None,
                    textfont=dict(size=9),
                )
            )

        fig.update_layout(
            barmode="stack",
        )

    # Update layout
    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "family": "Arial Black"},
        },
        xaxis_title="Model",
        yaxis_title="Metric Value",
        yaxis=dict(range=[0, 1.05]),  # Metrics are typically in [0, 1] range
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=60, r=150, t=80, b=60),
        height=500,
        hovermode="x unified",
    )

    # Update axes
    fig.update_xaxes(
        tickangle=-45 if len(metrics_list) > 5 else 0,
        tickfont=dict(size=10),
    )

    LOGGER.info(
        "Created %s bar chart with %d models and %d metrics",
        chart_type,
        len(metrics_list),
        len(metrics),
    )

    return fig

def create_per_class_comparison(
    metrics_list: List[ModelMetrics],
    metric: str = "f1-score",
    chart_type: str = "grouped",
    title: Optional[str] = None,
    show_values: bool = True,
) -> go.Figure:
    """
    Create an interactive Plotly bar chart comparing a per-class metric across multiple models.
    
    Generates either grouped (side-by-side) or stacked bars showing per-class values for `precision`, `recall`, or `f1-score` extracted from each ModelMetrics object's classification report. Missing class/model combinations are filled with 0.0 to preserve alignment; classes are sorted numerically. An automatic title is created when `title` is None.
    
    Parameters:
        metrics_list (List[ModelMetrics]): Models to include in the comparison.
        metric (str): Per-class metric to plot. One of "f1-score", "precision", "recall".
        chart_type (str): "grouped" for side-by-side bars or "stacked" for stacked bars.
        title (Optional[str]): Custom chart title; auto-generated if None.
        show_values (bool): If True, display numeric values on the bars.
    
    Returns:
        go.Figure: A Plotly Figure containing the per-class comparison chart.
    
    Raises:
        ImportError: If Plotly is not available.
        ValueError: If `metrics_list` is empty, `metric` is invalid, or no per-class metrics are present.
    """
    _require_plotly()

    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")

    valid_metrics = ["f1-score", "precision", "recall"]
    if metric not in valid_metrics:
        raise ValueError(
            f"Invalid metric '{metric}'. Must be one of: {valid_metrics}"
        )

    # Extract per-class metrics using static method
    per_class_df = ModelComparisonEngine.get_per_class_metrics(metrics_list)

    if per_class_df.empty:
        raise ValueError(
            "No per-class metrics available. Ensure models have classification_report data."
        )

    # Filter to selected metric
    metric_df = per_class_df[["model_name", "class", metric]].copy()

    # Sort classes for consistent ordering
    metric_df["class"] = pd.to_numeric(metric_df["class"], errors="coerce")
    metric_df = metric_df.dropna(subset=["class"])
    metric_df = metric_df.sort_values(["class", "model_name"]).reset_index(drop=True)

    # Get unique classes and models
    unique_classes = sorted(metric_df["class"].unique())
    unique_models = metric_df["model_name"].unique().tolist()

    # Auto-generate title if not provided
    if title is None:
        title = f"Per-Class {metric.replace('-', ' ').title()} Comparison"

    # Create figure
    fig = go.Figure()

    # Color palette for models
    colors = px.colors.qualitative.Set2

    if chart_type == "grouped":
        # Grouped bar chart (side-by-side bars for each model)
        for idx, model_name in enumerate(unique_models):
            model_data = metric_df[metric_df["model_name"] == model_name]

            # Ensure all classes are represented (fill missing with 0)
            class_values = []
            for class_label in unique_classes:
                class_row = model_data[model_data["class"] == class_label]
                if not class_row.empty:
                    class_values.append(class_row[metric].iloc[0])
                else:
                    class_values.append(0.0)

            color = colors[idx % len(colors)]

            fig.add_trace(
                go.Bar(
                    name=model_name,
                    x=[f"Class {int(c)}" for c in unique_classes],
                    y=class_values,
                    marker_color=color,
                    text=[f"{v:.3f}" for v in class_values] if show_values else None,
                    textposition="outside" if show_values else None,
                    textfont=dict(size=10),
                    hovertemplate=(
                        f"<b>{model_name}</b><br>"
                        f"Class: %{{x}}<br>"
                        f"{metric.title()}: %{{y:.4f}}<br>"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            barmode="group",
            bargap=0.15,
            bargroupgap=0.1,
        )

    else:  # stacked
        # Stacked bar chart
        for idx, model_name in enumerate(unique_models):
            model_data = metric_df[metric_df["model_name"] == model_name]

            # Ensure all classes are represented (fill missing with 0)
            class_values = []
            for class_label in unique_classes:
                class_row = model_data[model_data["class"] == class_label]
                if not class_row.empty:
                    class_values.append(class_row[metric].iloc[0])
                else:
                    class_values.append(0.0)

            color = colors[idx % len(colors)]

            fig.add_trace(
                go.Bar(
                    name=model_name,
                    x=[f"Class {int(c)}" for c in unique_classes],
                    y=class_values,
                    marker_color=color,
                    text=[f"{v:.3f}" for v in class_values] if show_values else None,
                    textposition="inside" if show_values else None,
                    textfont=dict(size=9),
                    hovertemplate=(
                        f"<b>{model_name}</b><br>"
                        f"Class: %{{x}}<br>"
                        f"{metric.title()}: %{{y:.4f}}<br>"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            barmode="stack",
        )

    # Update layout
    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "family": "Arial Black"},
        },
        xaxis_title="Density Class (BI-RADS)",
        yaxis_title=metric.replace("-", " ").title(),
        yaxis=dict(range=[0, 1.05]),  # Metrics are typically in [0, 1] range
        legend=dict(
            title="Model",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=60, r=150, t=80, b=60),
        height=500,
        hovermode="x unified",
    )

    # Update axes
    fig.update_xaxes(
        tickfont=dict(size=11),
    )

    LOGGER.info(
        "Created per-class %s comparison with %d models and %d classes",
        metric,
        len(unique_models),
        len(unique_classes),
    )

    return fig

def create_confusion_matrix_comparison(
    metrics_list: List[ModelMetrics],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix Comparison",
    colorscale: str = "Blues",
) -> go.Figure:
    """
    Builds a grid of confusion-matrix heatmaps comparing multiple models.
    
    Each subplot shows one model's confusion matrix; when `normalize` is True each matrix is row-normalized (per-true-label rates), otherwise raw counts are shown. Class labels default to stringified indices when `class_names` is None. All input confusion matrices must be square and share the same shape.
    
    Parameters:
        metrics_list (List[ModelMetrics]): Sequence of models to include; each must provide a square `confusion_matrix`.
        class_names (Optional[List[str]]): Labels for classes in display order. If None, labels "0", "1", ... are used.
        normalize (bool): If True, normalize each confusion matrix by true-label row sums (showing per-class recall); if False, show raw counts.
        title (str): Overall figure title.
        colorscale (str): Plotly colorscale name used for the heatmaps.
    
    Returns:
        go.Figure: Plotly Figure containing a subplot grid of confusion-matrix heatmaps (one subplot per model).
    
    Raises:
        ImportError: If Plotly is not available.
        ValueError: If `metrics_list` is empty, a model lacks a `confusion_matrix`, confusion matrices are not square, matrices differ in size, or `class_names` length does not match the number of classes.
    """
    _require_plotly()

    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")

    # Extract confusion matrices and validate
    matrices = []
    model_names = []
    num_classes = None

    for m in metrics_list:
        if not m.confusion_matrix:
            raise ValueError(
                f"Model '{m.model_name}' has no confusion matrix. "
                "Ensure models were trained with metrics tracking enabled."
            )

        cm = np.array(m.confusion_matrix, dtype=np.float64)

        # Validate squareness
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            raise ValueError(
                f"Confusion matrix for model '{m.model_name}' is not square: "
                f"got shape {cm.shape}"
            )

        # Validate shape consistency
        if num_classes is None:
            num_classes = cm.shape[0]
        elif cm.shape[0] != num_classes:
            raise ValueError(
                f"Inconsistent confusion matrix sizes: expected {num_classes}x{num_classes}, "
                f"got {cm.shape[0]}x{cm.shape[0]} for model '{m.model_name}'"
            )

        # Normalize if requested (by rows = true labels)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm = cm / row_sums

        matrices.append(cm)
        model_names.append(m.model_name)

    # Prepare class labels
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    elif len(class_names) != num_classes:
        raise ValueError(
            f"Number of class_names ({len(class_names)}) must match "
            f"number of classes ({num_classes})"
        )

    # Determine subplot grid dimensions (try to keep roughly square)
    n_models = len(metrics_list)
    if n_models <= 2:
        n_cols = n_models
        n_rows = 1
    elif n_models <= 4:
        n_cols = 2
        n_rows = (n_models + 1) // 2
    elif n_models <= 6:
        n_cols = 3
        n_rows = (n_models + 2) // 3
    else:
        n_cols = 3
        n_rows = (n_models + 2) // 3

    # Create subplot grid
    subplot_titles = [f"{name}" for name in model_names]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15 / n_rows if n_rows > 1 else 0.1,
        horizontal_spacing=0.12 / n_cols if n_cols > 1 else 0.1,
    )

    # Add heatmap for each model
    for idx, (cm, model_name) in enumerate(zip(matrices, model_names)):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Format hover text with appropriate precision
        if normalize:
            hover_text = [
                [
                    f"True: {class_names[i]}<br>"
                    f"Pred: {class_names[j]}<br>"
                    f"Rate: {cm[i, j]:.3f}"
                    for j in range(num_classes)
                ]
                for i in range(num_classes)
            ]
            text_values = [[f"{val:.2f}" for val in row] for row in cm]
        else:
            hover_text = [
                [
                    f"True: {class_names[i]}<br>"
                    f"Pred: {class_names[j]}<br>"
                    f"Count: {int(cm[i, j])}"
                    for j in range(num_classes)
                ]
                for i in range(num_classes)
            ]
            text_values = [[f"{int(val)}" for val in row] for row in cm]

        # Create heatmap trace
        heatmap = go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            text=text_values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale=colorscale,
            showscale=(idx == 0),  # Only show colorbar for first subplot
            hovertext=hover_text,
            hoverinfo="text",
            colorbar=dict(
                title="Proportion" if normalize else "Count",
                len=0.7,
                x=1.02,
            ) if idx == 0 else None,
        )

        fig.add_trace(heatmap, row=row, col=col)

        # Update axes for this subplot
        fig.update_xaxes(
            title_text="Predicted Label" if row == n_rows else "",
            tickmode="array",
            tickvals=list(range(num_classes)),
            ticktext=class_names,
            side="bottom",
            row=row,
            col=col,
        )

        fig.update_yaxes(
            title_text="True Label" if col == 1 else "",
            tickmode="array",
            tickvals=list(range(num_classes)),
            ticktext=class_names,
            autorange="reversed",  # Standard confusion matrix orientation
            row=row,
            col=col,
        )

    # Update overall layout
    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "family": "Arial Black"},
        },
        height=max(400, 300 * n_rows),
        width=max(600, 400 * n_cols),
        showlegend=False,
        margin=dict(l=80, r=120, t=100, b=60),
    )

    LOGGER.info(
        "Created confusion matrix comparison for %d models (%dx%d grid, normalize=%s)",
        n_models,
        n_rows,
        n_cols,
        normalize,
    )

    return fig
