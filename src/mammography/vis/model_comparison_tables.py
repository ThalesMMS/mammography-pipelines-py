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

LOGGER = logging.getLogger("mammography")

from mammography.vis.model_comparison_stats import ModelMetrics

def _require_plotly() -> None:
    """
    Ensure Plotly visualization dependencies are available.
    
    Raises:
        ImportError: If required Plotly symbols (`plotly.express`, `plotly.graph_objects`,
        `plotly.subplots`) are not importable. The raised error recommends installing
        `plotly` and chains the original import exception.
    """
    if px is None or go is None or make_subplots is None:
        raise ImportError(
            "Plotly is required for visualization features. "
            "Install with: pip install plotly"
        ) from _PLOTLY_IMPORT_ERROR

def format_statistical_summary(
    aggregated: Dict[str, Dict[str, float]],
    decimal_places: int = 4,
    title: str = "Model Comparison Statistical Summary",
) -> str:
    """
    Produce a human-readable multi-line summary of aggregated metrics including 95% confidence intervals.
    
    Formats an aggregated-metrics mapping into lines of the form:
        metric_name:  mean ± std  [ci_lower, ci_upper]
    Lines are aligned by metric name and the header includes the provided title with "(95% CI)".
    
    Parameters:
        aggregated (dict): Mapping from metric name to statistics dict with keys
            "mean", "std", "ci_lower", and "ci_upper". Example:
            {
                "accuracy": {"mean": 0.82, "std": 0.015, "ci_lower": 0.805, "ci_upper": 0.835},
                ...
            }
        decimal_places (int): Number of decimal places to display for numeric values (default: 4).
        title (str): Header title for the summary (default: "Model Comparison Statistical Summary").
    
    Returns:
        formatted_summary (str): Multi-line string containing the titled header and one aligned line per metric.
    """
    if not aggregated:
        return f"{title}:\n{'=' * len(title)}\n(No metrics available)"

    # Build header
    header = f"{title} (95% CI):"
    separator = "=" * len(header)
    lines = [header, separator]

    # Find longest metric name for alignment
    max_name_len = max((len(name) for name in aggregated.keys()), default=0)

    # Format each metric with alignment
    for metric_name in sorted(aggregated.keys()):
        stats = aggregated[metric_name]

        mean = stats["mean"]
        std = stats["std"]
        ci_lower = stats["ci_lower"]
        ci_upper = stats["ci_upper"]

        # Format with proper alignment
        name_padded = metric_name.ljust(max_name_len)
        mean_str = f"{mean:.{decimal_places}f}"
        std_str = f"{std:.{decimal_places}f}"
        ci_str = f"[{ci_lower:.{decimal_places}f}, {ci_upper:.{decimal_places}f}]"

        line = f"{name_padded}:  {mean_str} ± {std_str}  {ci_str}"
        lines.append(line)

    return "\n".join(lines)

def create_metrics_comparison_table(
    metrics_list: List[ModelMetrics],
    metrics: Optional[List[str]] = None,
    show_rank: bool = True,
    title: str = "Model Metrics Comparison",
) -> go.Figure:
    """
    Create an interactive Plotly table that compares selected performance metrics for multiple models side-by-side.
    
    Parameters:
        metrics_list (List[ModelMetrics]): Models to include in the comparison.
        metrics (Optional[List[str]]): Metric names to display. Default: ['accuracy', 'kappa', 'macro_f1', 'auc', 'balanced_accuracy', 'val_loss'].
        show_rank (bool): If True, include a rank column (sorted by macro_f1). Default: True.
        title (str): Title displayed above the table. Default: "Model Metrics Comparison".
    
    Returns:
        go.Figure: A Plotly Figure containing a styled, interactive table of the requested metrics.
    
    Raises:
        ImportError: If Plotly is not available.
        ValueError: If metrics_list is empty.
    """
    _require_plotly()

    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")

    if metrics is None:
        metrics = [
            "accuracy",
            "kappa",
            "macro_f1",
            "auc",
            "balanced_accuracy",
            "val_loss",
        ]

    # Prepare data
    data = []
    for m in metrics_list:
        row = {"model_name": m.model_name, "arch": m.arch}
        for metric_name in metrics:
            value = getattr(m, metric_name, None)
            row[metric_name] = value
        data.append(row)

    df = pd.DataFrame(data)

    # Sort by macro_f1 descending (best models first)
    if "macro_f1" in df.columns:
        df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    # Add rank column if requested
    if show_rank:
        df.insert(0, "rank", range(1, len(df) + 1))

    # Prepare table data
    header_values = [col.replace("_", " ").title() for col in df.columns]
    cell_values = []

    for col in df.columns:
        if col in ["model_name", "arch", "rank"]:
            # String columns - no formatting
            cell_values.append(df[col].tolist())
        else:
            # Numeric columns - format to 4 decimal places
            formatted = []
            for val in df[col]:
                if val is None:
                    formatted.append("N/A")
                elif isinstance(val, (int, float)):
                    formatted.append(f"{val:.4f}")
                else:
                    formatted.append(str(val))
            cell_values.append(formatted)

    # Color coding for metrics (normalize to [0, 1] range)
    cell_colors = []
    for col in df.columns:
        if col in metrics and col != "val_loss":
            # Higher is better - green gradient
            values = df[col].fillna(0).values
            if values.max() > values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
            else:
                normalized = np.ones_like(values)
            # Green color gradient
            colors = [f"rgba(144, 238, 144, {0.3 + 0.7 * v})" for v in normalized]
            cell_colors.append(colors)
        elif col == "val_loss":
            # Lower is better - red gradient (inverted)
            values = df[col].fillna(0).values
            if values.max() > values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
            else:
                normalized = np.ones_like(values)
            # Red color gradient (inverted)
            colors = [f"rgba(255, 182, 193, {0.3 + 0.7 * (1 - v)})" for v in normalized]
            cell_colors.append(colors)
        else:
            # No color coding for non-metric columns
            cell_colors.append(["white"] * len(df))

    # Create table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header_values,
                    fill_color="paleturquoise",
                    align="center",
                    font=dict(size=12, color="black", family="Arial Black"),
                    height=30,
                ),
                cells=dict(
                    values=cell_values,
                    fill_color=cell_colors,
                    align=["center"] * len(df.columns),
                    font=dict(size=11, color="black"),
                    height=25,
                ),
            )
        ]
    )

    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "family": "Arial Black"},
        },
        margin=dict(l=20, r=20, t=60, b=20),
        height=min(400, 100 + 30 * len(df)),
    )

    LOGGER.info("Created metrics comparison table with %d models", len(metrics_list))

    return fig

def _format_markdown_value(value: Any, float_format: str) -> str:
    """
    Format a single DataFrame cell value for inclusion in a Markdown table.
    
    Parameters:
        value (Any): Cell value to format. Floats (including NumPy floating types) are formatted using `float_format`; NaN-like values produce an empty string.
        float_format (str): printf-style format string applied to float values (e.g. "%.4f").
    
    Returns:
        str: The formatted cell as a string, an empty string for NaN-like values, or the stringified original value when formatting is not applicable.
    """
    if isinstance(value, (float, np.floating)):
        try:
            return float_format % float(value)
        except (TypeError, ValueError):
            return str(value)

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value)

def _dataframe_to_markdown(df: pd.DataFrame, index: bool, float_format: str) -> str:
    """
    Render a pandas DataFrame as a GitHub-flavored Markdown table.
    
    Parameters:
        df (pd.DataFrame): DataFrame to convert.
        index (bool): If True, reset the DataFrame index into columns before rendering.
        float_format (str): A printf-style float format (e.g. `"%.4f"`) used to format floating-point cells.
    
    Returns:
        str: The Markdown table as a string, with a header row, separator row (`---`), and one row per DataFrame record.
    """
    table = df.reset_index() if index else df
    headers = [str(col) for col in table.columns]
    separator = ["---"] * len(headers)
    rows = [
        [_format_markdown_value(value, float_format) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)

def export_comparison_table(
    df: pd.DataFrame,
    base_path: Union[str, Path],
    formats: Optional[List[str]] = None,
    index: bool = True,
    float_format: str = "%.4f",
) -> List[Path]:
    """Export comparison table to multiple publication-ready formats.

    Exports pandas DataFrame containing model comparison metrics to various
    table formats suitable for publications, reports, and documentation.

    Args:
        df: Pandas DataFrame to export (typically from get_metrics_dataframe or
            get_side_by_side_comparison)
        base_path: Base path without extension (e.g., "outputs/comparison_table")
        formats: List of formats to export. Defaults to ['csv', 'xlsx', 'json', 'md']
                 Supported: 'csv', 'xlsx', 'json', 'md' (markdown), 'tex' (LaTeX)
        index: Whether to include DataFrame index in exported files (default: True)
        float_format: Format string for floating point numbers (default: "%.4f")

    Returns:
        List of Path objects for successfully exported files

    Raises:
        ValueError: If df is empty or formats list is empty

    Example:
        >>> metrics_df = engine.get_metrics_dataframe(metrics_list)
        >>> paths = export_comparison_table(
        ...     metrics_df,
        ...     "outputs/model_comparison",
        ...     formats=['csv', 'xlsx', 'md']
        ... )
        >>> print(f"Exported to: {[str(p) for p in paths]}")
        Exported to: ['outputs/model_comparison.csv', ...]

    Note:
        - CSV: Plain text, widely compatible
        - XLSX: Excel format, requires openpyxl
        - JSON: Structured data, machine-readable
        - MD: Markdown table for documentation
        - TEX: LaTeX table for academic papers
    """
    if df.empty:
        raise ValueError("Cannot export empty DataFrame")

    if formats is None:
        formats = ['csv', 'xlsx', 'json', 'md']

    if not formats:
        raise ValueError("formats list cannot be empty")

    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    exported_paths: List[Path] = []

    for fmt in formats:
        fmt_lower = fmt.lower()
        out_path = base_path.with_suffix(f".{fmt_lower}")

        try:
            if fmt_lower == 'csv':
                df.to_csv(out_path, index=index, float_format=float_format)
                exported_paths.append(out_path)
                LOGGER.info(f"Exported table to CSV: {out_path}")

            elif fmt_lower == 'xlsx':
                try:
                    df.to_excel(out_path, index=index, float_format=float_format)
                    exported_paths.append(out_path)
                    LOGGER.info(f"Exported table to Excel: {out_path}")
                except ImportError:
                    LOGGER.warning(
                        "openpyxl not available, skipping XLSX export. "
                        "Install with: pip install openpyxl"
                    )

            elif fmt_lower == 'json':
                # Export as records format for better readability
                df.to_json(out_path, orient='records', indent=2)
                exported_paths.append(out_path)
                LOGGER.info(f"Exported table to JSON: {out_path}")

            elif fmt_lower == 'md' or fmt_lower == 'markdown':
                # Use .md extension for markdown
                if fmt_lower == 'markdown':
                    out_path = base_path.with_suffix('.md')

                try:
                    md_content = df.to_markdown(index=index, floatfmt=float_format)
                except ImportError:
                    md_content = _dataframe_to_markdown(
                        df,
                        index=index,
                        float_format=float_format,
                    )
                out_path.write_text(md_content, encoding='utf-8')
                exported_paths.append(out_path)
                LOGGER.info(f"Exported table to Markdown: {out_path}")

            elif fmt_lower == 'tex' or fmt_lower == 'latex':
                # Use .tex extension for LaTeX
                if fmt_lower == 'latex':
                    out_path = base_path.with_suffix('.tex')

                latex_content = df.to_latex(
                    index=index,
                    float_format=float_format,
                    caption="Model Comparison Table",
                    label="tab:model_comparison",
                )
                out_path.write_text(latex_content, encoding='utf-8')
                exported_paths.append(out_path)
                LOGGER.info(f"Exported table to LaTeX: {out_path}")

            else:
                LOGGER.warning(f"Unsupported format '{fmt}', skipping. "
                             f"Supported: csv, xlsx, json, md, tex")

        except Exception as e:
            LOGGER.error(f"Failed to export table as {fmt}: {e}")

    if not exported_paths:
        LOGGER.warning("No tables were successfully exported")

    return exported_paths
