#
# __init__.py
# mammography-pipelines
#
# Components package for web UI dashboard.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""Reusable components for the Streamlit web UI dashboard."""

from __future__ import annotations

__all__ = [
    "DatasetViewer",
    "MetricsMonitor",
    "ReportExporter",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "render_confusion_matrix",
    "render_roc_curves",
    "render_results_summary",
]

try:
    from .dataset_viewer import DatasetViewer
    from .metrics_monitor import MetricsMonitor
    from .report_exporter import ReportExporter
    from .results_visualizer import (
        plot_confusion_matrix,
        plot_roc_curves,
        render_confusion_matrix,
        render_roc_curves,
        render_results_summary,
    )
except ImportError:  # pragma: no cover - optional UI dependency
    DatasetViewer = None  # type: ignore[misc,assignment]
    MetricsMonitor = None  # type: ignore[misc,assignment]
    ReportExporter = None  # type: ignore[misc,assignment]
    plot_confusion_matrix = None  # type: ignore[misc,assignment]
    plot_roc_curves = None  # type: ignore[misc,assignment]
    render_confusion_matrix = None  # type: ignore[misc,assignment]
    render_roc_curves = None  # type: ignore[misc,assignment]
    render_results_summary = None  # type: ignore[misc,assignment]
