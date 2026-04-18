# ruff: noqa: F401
# DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
# It must NOT be used for clinical or medical diagnostic purposes.
# No medical decision should be based on these results.
"""Compatibility facade for model comparison utilities."""

from __future__ import annotations

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

from mammography.vis import model_comparison_plots as _plots
from mammography.vis import model_comparison_tables as _tables
from mammography.vis.model_comparison_engine import ModelComparisonEngine
from mammography.vis.model_comparison_stats import (
    ModelMetrics,
    _resolve_metrics_json_path,
    compute_mcnemar_test,
)
from mammography.vis.model_comparison_tables import (
    _dataframe_to_markdown,
    _format_markdown_value,
    export_comparison_table,
    format_statistical_summary,
)


def _sync_plotly() -> None:
    """
    Synchronize Plotly bindings and the Plotly import error state into the internal visualization modules.

    Assigns the locally resolved Plotly objects (`px`, `go`, `make_subplots`) and `_PLOTLY_IMPORT_ERROR` onto the `_plots` and `_tables` modules so those modules use the current optional-import results.
    """
    for module in (_plots, _tables):
        module.px = px
        module.go = go
        module.make_subplots = make_subplots
        module._PLOTLY_IMPORT_ERROR = _PLOTLY_IMPORT_ERROR


def _require_plotly() -> None:
    """
    Ensure internal visualization modules have up-to-date Plotly bindings and enforce Plotly availability.

    Synchronizes the locally resolved Plotly objects into the internal plotting/table modules, then delegates to the underlying plotting module to validate that Plotly is available.

    Raises:
        Exception: The original Plotly import exception if Plotly is not available.
    """
    _sync_plotly()
    _plots._require_plotly()


def create_metrics_comparison_table(
    metrics_list: list[ModelMetrics],
    metrics: list[str] | None = None,
    show_rank: bool = True,
    title: str = "Model Metrics Comparison",
) -> go.Figure:
    """
    Create a metrics comparison table for one or more models.

    Returns:
        The table object produced by the underlying tables implementation (format and type are determined by that implementation).
    """
    _sync_plotly()
    return _tables.create_metrics_comparison_table(
        metrics_list,
        metrics=metrics,
        show_rank=show_rank,
        title=title,
    )


def create_metrics_comparison_chart(
    metrics_list: list[ModelMetrics],
    metrics: list[str] | None = None,
    chart_type: str = "grouped",
    title: str = "Model Metrics Comparison",
    show_values: bool = True,
) -> go.Figure:
    """
    Create a Plotly figure comparing metrics across models.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure representing the metrics comparison.
    """
    _sync_plotly()
    return _plots.create_metrics_comparison_chart(
        metrics_list,
        metrics=metrics,
        chart_type=chart_type,
        title=title,
        show_values=show_values,
    )


def create_per_class_comparison(
    metrics_list: list[ModelMetrics],
    metric: str = "f1-score",
    chart_type: str = "grouped",
    title: str | None = None,
    show_values: bool = True,
) -> go.Figure:
    """
    Create a per-class metrics comparison chart for multiple models.

    All positional and keyword arguments are forwarded to the underlying implementation.

    Returns:
        plotly.graph_objects.Figure: Plotly Figure showing per-class metric comparisons across models.
    """
    _sync_plotly()
    return _plots.create_per_class_comparison(
        metrics_list,
        metric=metric,
        chart_type=chart_type,
        title=title,
        show_values=show_values,
    )


def create_confusion_matrix_comparison(
    metrics_list: list[ModelMetrics],
    class_names: list[str] | None = None,
    normalize: bool = True,
    title: str = "Confusion Matrix Comparison",
    colorscale: str = "Blues",
) -> go.Figure:
    """
    Create a comparative confusion matrix visualization for one or more models.

    Generates a Plotly figure that arranges confusion matrices for the provided models side-by-side (or in a layout defined by the underlying implementation), facilitating visual comparison of true/false positive/negative distributions across models.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure containing the comparison of confusion matrices.
    """
    _sync_plotly()
    return _plots.create_confusion_matrix_comparison(
        metrics_list,
        class_names=class_names,
        normalize=normalize,
        title=title,
        colorscale=colorscale,
    )


__all__ = [
    "ModelComparisonEngine",
    "ModelMetrics",
    "compute_mcnemar_test",
    "create_confusion_matrix_comparison",
    "create_metrics_comparison_chart",
    "create_metrics_comparison_table",
    "create_per_class_comparison",
    "export_comparison_table",
    "format_statistical_summary",
]
