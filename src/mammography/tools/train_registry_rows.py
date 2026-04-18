"""CSV and Markdown row formatting helpers for training registry entries."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from pathlib import Path


def _format_metric(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return str(value)


def _extract_tuned_params(
    summary: Mapping[str, Any],
    tuned_keys: Sequence[str] | None,
) -> dict[str, Any]:
    if not tuned_keys:
        return {}
    params: dict[str, Any] = {}
    for key in tuned_keys:
        key_clean = key.strip()
        if not key_clean:
            continue
        if key_clean in summary:
            params[key_clean] = summary.get(key_clean)
    return params


def _format_delta(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value}"


def _format_tuned_params(params: Mapping[str, Any]) -> str:
    if not params:
        return "n/a"
    items = []
    for key, value in params.items():
        items.append(f"{key}={value}")
    return ", ".join(items)


def _format_hparam_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, ensure_ascii=True, default=str)


_MARKDOWN_HPARAM_KEYS = (
    "arch",
    "classes",
    "epochs",
    "batch_size",
    "img_size",
    "lr",
    "backbone_lr",
    "class_weights",
    "sampler_weighted",
    "warmup_epochs",
    "early_stop_patience",
    "augment",
    "amp",
    "seed",
    "tracker",
    "tracker_run_name",
)


def _summarize_hyperparameters(summary: Mapping[str, Any]) -> str:
    items: list[str] = []
    for key in _MARKDOWN_HPARAM_KEYS:
        if key in summary:
            items.append(f"{key}={_format_hparam_value(summary.get(key))}")
    return ", ".join(items)


def build_registry_row(
    *,
    dataset: str,
    workflow: str,
    run_name: str,
    command: str,
    model: str,
    batch_size: int,
    img_size: int,
    results_dir: Path,
    summary_path: Path,
    train_history_path: Path,
    val_metrics_path: Path,
    confusion_matrix_path: Path | None,
    checkpoint_path: Path,
    val_predictions_path: Path | None,
    hyperparameters: str,
    metrics: Mapping[str, float | None],
    mlflow_run_id: str | None,
    baseline_run_name: str | None = None,
    baseline_metrics: Mapping[str, float | None] | None = None,
    baseline_deltas: Mapping[str, float | None] | None = None,
    tuned_hyperparameters: str | None = None,
    timestamp: str | None = None,
) -> dict[str, str | int]:
    baseline_metrics = baseline_metrics or {}
    baseline_deltas = baseline_deltas or {}
    return {
        "timestamp": timestamp or datetime.now(tz=timezone.utc).isoformat(),
        "dataset": dataset,
        "workflow": workflow,
        "run_name": run_name,
        "command": command,
        "model": model,
        "batch_size": batch_size,
        "img_size": img_size,
        "outdir": str(results_dir),
        "summary_path": str(summary_path),
        "train_history_path": str(train_history_path),
        "val_metrics_path": str(val_metrics_path),
        "confusion_matrix_path": str(confusion_matrix_path)
        if confusion_matrix_path
        else "",
        "checkpoint_path": str(checkpoint_path),
        "val_predictions_path": str(val_predictions_path)
        if val_predictions_path
        else "",
        "train_loss": _format_metric(metrics.get("train_loss")),
        "train_acc": _format_metric(metrics.get("train_acc")),
        "val_loss": _format_metric(metrics.get("val_loss")),
        "val_acc": _format_metric(metrics.get("val_acc")),
        "val_f1": _format_metric(metrics.get("val_f1")),
        "val_auc": _format_metric(metrics.get("val_auc")),
        "val_sensitivity": _format_metric(metrics.get("val_sensitivity")),
        "val_specificity": _format_metric(metrics.get("val_specificity")),
        "hyperparameters": hyperparameters,
        "baseline_run_name": baseline_run_name or "",
        "baseline_val_loss": _format_metric(baseline_metrics.get("val_loss")),
        "baseline_val_acc": _format_metric(baseline_metrics.get("val_acc")),
        "baseline_val_f1": _format_metric(baseline_metrics.get("val_f1")),
        "baseline_val_auc": _format_metric(baseline_metrics.get("val_auc")),
        "delta_val_loss": _format_metric(baseline_deltas.get("val_loss")),
        "delta_val_acc": _format_metric(baseline_deltas.get("val_acc")),
        "delta_val_f1": _format_metric(baseline_deltas.get("val_f1")),
        "delta_val_auc": _format_metric(baseline_deltas.get("val_auc")),
        "tuned_hyperparameters": tuned_hyperparameters or "",
        "mlflow_run_id": mlflow_run_id or "",
    }


def build_markdown_entry(
    *,
    dataset: str,
    workflow: str,
    run_name: str,
    command: str,
    results_dir: Path,
    outputs: Sequence[Path],
    metrics: Mapping[str, float | None],
    hyperparameters_summary: str | None = None,
    confusion_matrix_path: Path | None = None,
    baseline_run_name: str | None = None,
    baseline_metrics: Mapping[str, float | None] | None = None,
    baseline_deltas: Mapping[str, float | None] | None = None,
    tuned_params: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
) -> list[str]:
    """
    Builds a Markdown-formatted list of lines describing a single training run.

    Parameters:
        dataset (str): Dataset identifier.
        workflow (str): Workflow or experiment identifier.
        run_name (str): Human-readable run name.
        command (str): Command line used to start the run.
        results_dir (Path): Path to the run's results directory.
        outputs (Sequence[Path]): Paths to important output files/artifacts to include.
        metrics (Mapping[str, float | None]): Mapping of metric names to values. Expected keys include
            'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1', 'val_auc', and optionally
            'val_sensitivity' and 'val_specificity'.
        hyperparameters_summary (str | None): Short comma-separated hyperparameter summary to include.
        confusion_matrix_path (Path | None): Optional path to a confusion matrix image to reference.
        baseline_run_name (str | None): Optional baseline run identifier to list.
        baseline_metrics (Mapping[str, float | None] | None): Baseline metrics mapping (same keys as `metrics`).
        baseline_deltas (Mapping[str, float | None] | None): Differences between current and baseline metrics.
        tuned_params (Mapping[str, Any] | None): Optional tuned hyperparameters to render.
        timestamp (str | None): ISO-format date string to use for the entry header; uses today's date if omitted.

    Returns:
        list[str]: Lines of Markdown representing the run entry, starting with a header line and ending with a blank line.
    """
    stamp = timestamp or datetime.now().date().isoformat()
    outputs_str = ", ".join(f"`{path}`" for path in outputs)
    baseline_metrics = baseline_metrics or {}
    baseline_deltas = baseline_deltas or {}
    tuned_text = _format_tuned_params(tuned_params or {})
    baseline_lines: list[str] = []
    if baseline_run_name:
        baseline_lines.append(f"- Baseline run: {baseline_run_name}")
    if baseline_metrics:
        baseline_lines.extend(
            [
                (
                    "- Baseline val loss: {value} (Δ {delta})".format(
                        value=baseline_metrics.get("val_loss"),
                        delta=_format_delta(baseline_deltas.get("val_loss")),
                    )
                ),
                (
                    "- Baseline val acc: {value} (Δ {delta})".format(
                        value=baseline_metrics.get("val_acc"),
                        delta=_format_delta(baseline_deltas.get("val_acc")),
                    )
                ),
                (
                    "- Baseline val F1: {value} (Δ {delta})".format(
                        value=baseline_metrics.get("val_f1"),
                        delta=_format_delta(baseline_deltas.get("val_f1")),
                    )
                ),
                (
                    "- Baseline val AUC: {value} (Δ {delta})".format(
                        value=baseline_metrics.get("val_auc"),
                        delta=_format_delta(baseline_deltas.get("val_auc")),
                    )
                ),
            ]
        )
    return [
        f"## {stamp} - {run_name}",
        f"- Dataset: {dataset}",
        f"- Workflow: {workflow}",
        f"- Run name: {run_name}",
        f"- Command: `{command}`",
        f"- Outdir: `{results_dir}`",
        f"- Outputs: {outputs_str}",
        *(
            [f"- Hyperparameters: {hyperparameters_summary}"]
            if hyperparameters_summary
            else []
        ),
        *(
            [f"- Confusion matrix: `{confusion_matrix_path}`"]
            if confusion_matrix_path
            else []
        ),
        *([f"- Tuned hyperparameters: {tuned_text}"] if tuned_params else []),
        f"- Train loss: {_format_metric(metrics.get('train_loss'))}",
        f"- Train acc: {_format_metric(metrics.get('train_acc'))}",
        f"- Val loss: {_format_metric(metrics.get('val_loss'))}",
        f"- Val acc: {_format_metric(metrics.get('val_acc'))}",
        f"- Val F1: {_format_metric(metrics.get('val_f1'))}",
        f"- Val AUC: {_format_metric(metrics.get('val_auc'))}",
        *(
            [f"- Val sensitivity: {_format_metric(metrics.get('val_sensitivity'))}"]
            if metrics.get("val_sensitivity") is not None
            else []
        ),
        *(
            [f"- Val specificity: {_format_metric(metrics.get('val_specificity'))}"]
            if metrics.get("val_specificity") is not None
            else []
        ),
        *baseline_lines,
        "",
    ]
