# ruff: noqa
# DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
# It must NOT be used for clinical or medical diagnostic purposes.
# No medical decision should be based on these results.
"""Ensemble registration helpers for training registry."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mammography.tools import train_registry as _base
from mammography.tools.data_audit_registry import (
    append_registry_csv,
    append_registry_markdown,
)


def _resolve_ensemble_confusion_matrix(metrics_dir: Path) -> Path | None:
    """
    Locate an ensemble confusion-matrix or metrics image file inside a metrics directory.

    Searches the provided directory for common ensemble image artifact names
    (e.g., "ensemble_metrics.png" or "ensemble_confusion_matrix.png") and, if
    those are absent, for the newest file matching "ensemble*_metrics.png".

    Parameters:
        metrics_dir (Path): Directory to search for ensemble image artifacts.

    Returns:
        Path | None: Path to the first matching image file if found, `None` otherwise.
    """
    candidates = [
        metrics_dir / "ensemble_metrics.png",
        metrics_dir / "ensemble_confusion_matrix.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(metrics_dir.glob("ensemble*_metrics.png"))
    if not matches:
        return None
    try:
        return max(matches, key=lambda path: path.stat().st_mtime)
    except OSError:
        return matches[-1]


def _collect_ensemble_metrics(
    val_metrics: Mapping[str, Any],
) -> dict[str, float | None]:
    """
    Extract validation metrics from a raw metrics mapping and normalize them to a fixed set of optional numeric values.

    Parameters:
        val_metrics (Mapping[str, Any]): Mapping produced by ensemble validation (e.g., parsed JSON). Expected keys include common metric names such as "acc", "loss", "f1_macro", "macro_f1" or "f1", "auc_ovr" or "auc", and entries required to compute sensitivity/specificity (confusion counts) when present.

    Returns:
        dict[str, float | None]: Dictionary with the following keys:
            - "val_loss": validation loss as a float, or `None` if unavailable.
            - "val_acc": validation accuracy as a float, or `None` if unavailable.
            - "val_f1": validation macro F1 (falls back to `f1`) as a float, or `None`.
            - "val_auc": validation AUC one-vs-rest (falls back to `auc`) as a float, or `None`.
            - "val_sensitivity": sensitivity (recall) computed from available confusion data, or `None` if not computable.
            - "val_specificity": specificity computed from available confusion data, or `None` if not computable.
    """
    val_acc = _base._parse_optional_float(val_metrics.get("acc"))
    val_loss = _base._parse_optional_float(val_metrics.get("loss"))
    val_f1 = _base._parse_optional_float(val_metrics.get("f1_macro"))
    if val_f1 is None:
        val_f1 = _base._parse_optional_float(val_metrics.get("macro_f1"))
    if val_f1 is None:
        val_f1 = _base._parse_optional_float(val_metrics.get("f1"))
    val_auc = _base._parse_optional_float(val_metrics.get("auc_ovr"))
    if val_auc is None:
        val_auc = _base._parse_optional_float(val_metrics.get("auc"))

    val_sensitivity, val_specificity = _base._compute_sensitivity_specificity(
        val_metrics
    )

    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_auc": val_auc,
        "val_sensitivity": val_sensitivity,
        "val_specificity": val_specificity,
    }


def build_ensemble_registry_row(
    *,
    dataset: str,
    workflow: str,
    run_name: str,
    command: str,
    model: str,
    batch_size: int,
    img_size: int,
    outdir: Path,
    summary_path: Path | None,
    metrics_path: Path,
    confusion_matrix_path: Path | None,
    hyperparameters: str,
    metrics: Mapping[str, float | None],
    mlflow_run_id: str | None,
    timestamp: str | None = None,
) -> dict[str, str | int]:
    """
    Builds a single registry-row mapping representing an ensemble run for CSV storage.

    Parameters:
        outdir: Output directory for the run; stored as a string in the row.
        summary_path: Optional path to the run summary JSON; stored as a string or empty when not provided.
        metrics_path: Path to the ensemble validation metrics JSON; stored as `val_metrics_path`.
        confusion_matrix_path: Optional path to an ensemble confusion-matrix image; stored as a string or empty.
        hyperparameters: JSON-serialized string of the run hyperparameters (or "{}") stored verbatim in the row.
        metrics: Mapping of metric names to optional float values. Recognized keys used in the row:
            - "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "val_auc",
              "val_sensitivity", "val_specificity".
            Values may be None; metrics are formatted via the module's metric formatter before insertion.
        mlflow_run_id: Optional MLflow run identifier to record; empty string when not provided.
        dataset, workflow, run_name, command, model, batch_size, img_size, timestamp:
            Stored directly into corresponding row fields. If `timestamp` is omitted, the current UTC
            ISO-8601 timestamp is used.

    Returns:
        dict: A mapping of registry column names to values (strings or integers) suitable for appending to
        the CSV registry. The dictionary includes standardized fields for paths, identifiers, formatted
        metric strings, hyperparameters, and the MLflow run id.
    """
    return {
        "timestamp": timestamp or datetime.now(tz=timezone.utc).isoformat(),
        "dataset": dataset,
        "workflow": workflow,
        "run_name": run_name,
        "command": command,
        "model": model,
        "batch_size": batch_size,
        "img_size": img_size,
        "outdir": str(outdir),
        "summary_path": str(summary_path) if summary_path else "",
        "train_history_path": "",
        "val_metrics_path": str(metrics_path),
        "confusion_matrix_path": str(confusion_matrix_path)
        if confusion_matrix_path
        else "",
        "checkpoint_path": "",
        "val_predictions_path": "",
        "train_loss": _base._format_metric(metrics.get("train_loss")),
        "train_acc": _base._format_metric(metrics.get("train_acc")),
        "val_loss": _base._format_metric(metrics.get("val_loss")),
        "val_acc": _base._format_metric(metrics.get("val_acc")),
        "val_f1": _base._format_metric(metrics.get("val_f1")),
        "val_auc": _base._format_metric(metrics.get("val_auc")),
        "val_sensitivity": _base._format_metric(metrics.get("val_sensitivity")),
        "val_specificity": _base._format_metric(metrics.get("val_specificity")),
        "hyperparameters": hyperparameters,
        "mlflow_run_id": mlflow_run_id or "",
    }


def build_ensemble_markdown_entry(
    *,
    dataset: str,
    workflow: str,
    run_name: str,
    command: str,
    outdir: Path,
    outputs: Sequence[Path],
    metrics: Mapping[str, float | None],
    timestamp: str | None = None,
) -> list[str]:
    """
    Build a Markdown entry (as lines) summarizing an ensemble run for registry files.

    Parameters:
        dataset (str): Dataset identifier.
        workflow (str): Workflow name.
        run_name (str): Run identifier used in the header.
        command (str): Command used to launch the run; rendered inline in backticks.
        outdir (Path): Output directory path; rendered inline in backticks.
        outputs (Sequence[Path]): Paths to produced artifacts; each will be shown as a backticked, comma-separated list.
        metrics (Mapping[str, float | None]): Validation metrics; expected keys include
            'val_loss', 'val_acc', 'val_f1', 'val_auc', and optionally 'val_sensitivity', 'val_specificity'.
        timestamp (str | None): ISO timestamp to use for the entry header; when omitted the current UTC time is used.

    Returns:
        list[str]: Lines composing the Markdown entry; the sequence ends with an empty string.
    """
    stamp = timestamp or datetime.now(tz=timezone.utc).isoformat()
    outputs_str = ", ".join(f"`{path}`" for path in outputs)
    return [
        f"## {stamp} - {run_name} (ensemble)",
        f"- Dataset: {dataset}",
        f"- Workflow: {workflow}",
        f"- Run name: {run_name}",
        f"- Command: `{command}`",
        f"- Outdir: `{outdir}`",
        f"- Outputs: {outputs_str}",
        f"- Val loss: {metrics.get('val_loss')}",
        f"- Val acc: {metrics.get('val_acc')}",
        f"- Val F1: {metrics.get('val_f1')}",
        f"- Val AUC: {metrics.get('val_auc')}",
        *(
            [f"- Val sensitivity: {metrics.get('val_sensitivity')}"]
            if metrics.get("val_sensitivity") is not None
            else []
        ),
        *(
            [f"- Val specificity: {metrics.get('val_specificity')}"]
            if metrics.get("val_specificity") is not None
            else []
        ),
        "",
    ]


def register_ensemble_run(
    *,
    outdir: Path,
    metrics_path: Path,
    summary_path: Path | None,
    dataset: str,
    workflow: str,
    run_name: str,
    command: str,
    registry_csv: Path,
    registry_md: Path,
    tracking_uri: str | None = None,
    experiment: str | None = None,
) -> str:
    """
    Register an ensemble training run by logging metrics and artifacts to MLflow and appending entries to local CSV and Markdown registries.

    Parameters:
        outdir (Path): Output directory for the run (used in registry entries).
        metrics_path (Path): Path to the ensemble metrics JSON file; must exist and contain a JSON object.
        summary_path (Path | None): Optional path to a run summary JSON file; if provided it must exist.
        dataset (str): Dataset identifier to record in the registry (fallback to summary dataset if empty).
        workflow (str): Workflow name to record in the registry.
        run_name (str): Human-readable run name used for the MLflow run.
        command (str): Command or invocation string to record in the registry.
        registry_csv (Path): Path to the CSV registry file to append a new row.
        registry_md (Path): Path to the Markdown registry file to append a new entry.
        tracking_uri (str | None): Optional MLflow tracking server URI.
        experiment (str | None): Optional MLflow experiment name or ID.

    Returns:
        run_id (str): The MLflow run identifier created by logging this ensemble run.

    Raises:
        FileNotFoundError: If `metrics_path` does not exist, or if `summary_path` is provided but does not exist.
        ValueError: If the JSON loaded from `metrics_path` is not a JSON object (mapping).
    """
    if not metrics_path.exists():
        raise FileNotFoundError(f"ensemble_metrics.json ausente: {metrics_path}")
    val_metrics = _base._load_json(metrics_path)
    if not isinstance(val_metrics, Mapping):
        raise ValueError("ensemble_metrics.json invalido: esperado objeto JSON.")

    summary: dict[str, Any] = {}
    if summary_path:
        if not summary_path.exists():
            raise FileNotFoundError(f"summary.json ausente: {summary_path}")
        summary = _base._load_json(summary_path)

    metrics = _collect_ensemble_metrics(val_metrics)

    params: dict[str, str] = {}
    if summary:
        params.update(_base._collect_params(summary))
    params["view"] = "ensemble"
    params["ensemble_metrics_path"] = str(metrics_path)
    if summary_path:
        params["summary_path"] = str(summary_path)

    metric_payload: dict[str, float] = {}
    for key, value in val_metrics.items():
        if isinstance(value, (int, float)):
            numeric = float(value)
            if math.isfinite(numeric):
                metric_payload[key] = numeric
    for key, value in metrics.items():
        if value is None or not math.isfinite(value):
            continue
        metric_payload.setdefault(key, float(value))

    artifacts: list[Path] = [metrics_path]
    confusion_matrix_path = _resolve_ensemble_confusion_matrix(metrics_path.parent)
    if confusion_matrix_path:
        artifacts.append(confusion_matrix_path)
    if summary_path:
        artifacts.append(summary_path)

    run_id = _base.log_mlflow_run(
        run_name=run_name,
        params=params,
        metrics=metric_payload,
        artifacts=artifacts,
        tracking_uri=tracking_uri,
        experiment=experiment,
    )

    dataset_value = dataset or str(summary.get("dataset") or "unknown")
    model_value = str(summary.get("arch") or "ensemble")
    batch_size = int(summary.get("batch_size") or 0)
    img_size = int(summary.get("img_size") or 0)
    hyperparameters = (
        json.dumps(summary, ensure_ascii=True, default=str) if summary else "{}"
    )

    row = build_ensemble_registry_row(
        dataset=dataset_value,
        workflow=workflow,
        run_name=run_name,
        command=command,
        model=model_value,
        batch_size=batch_size,
        img_size=img_size,
        outdir=outdir,
        summary_path=summary_path,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        hyperparameters=hyperparameters,
        metrics=metrics,
        mlflow_run_id=run_id,
    )
    append_registry_csv(registry_csv, row)

    outputs: list[Path] = [metrics_path]
    if confusion_matrix_path:
        outputs.append(confusion_matrix_path)
    if summary_path:
        outputs.append(summary_path)

    entry_lines = build_ensemble_markdown_entry(
        dataset=dataset_value,
        workflow=workflow,
        run_name=run_name,
        command=command,
        outdir=outdir,
        outputs=outputs,
        metrics=metrics,
    )
    append_registry_markdown(registry_md, entry_lines)
    return run_id
