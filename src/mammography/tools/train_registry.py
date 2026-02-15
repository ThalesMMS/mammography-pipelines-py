#!/usr/bin/env python3
"""Register training outputs in MLflow and local registry files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mammography.tools.data_audit_registry import (
    _resolve_tracking_root,
    _write_meta_yaml,
    _write_metric,
    _write_param,
    append_registry_csv,
    append_registry_markdown,
)


@dataclass(frozen=True)
class TrainingArtifacts:
    results_dir: Path
    summary_path: Path
    train_history_path: Path
    val_metrics_path: Path
    confusion_matrix_path: Path | None
    checkpoint_path: Path
    val_predictions_path: Path | None
    train_history_plot_path: Path | None


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON invalido: esperado objeto. Recebido: {path}")
    return data


def _parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _normalize_view(value: object) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned.lower()
    return None


def _compute_sensitivity_specificity(
    val_metrics: Mapping[str, Any],
) -> tuple[float | None, float | None]:
    cm = val_metrics.get("confusion_matrix")
    if isinstance(cm, Sequence) and len(cm) == 2:
        try:
            row0 = cm[0]
            row1 = cm[1]
            if isinstance(row0, Sequence) and isinstance(row1, Sequence) and len(row0) == 2 and len(row1) == 2:
                tn = float(row0[0])
                fp = float(row0[1])
                fn = float(row1[0])
                tp = float(row1[1])
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
                specificity = tn / (tn + fp) if (tn + fp) > 0 else None
                return sensitivity, specificity
        except (TypeError, ValueError):
            pass

    report = val_metrics.get("classification_report")
    if isinstance(report, Mapping):
        pos = report.get("1") or report.get(1)
        neg = report.get("0") or report.get(0)
        sensitivity = _parse_optional_float(
            pos.get("recall") if isinstance(pos, Mapping) else None
        )
        specificity = _parse_optional_float(
            neg.get("recall") if isinstance(neg, Mapping) else None
        )
        return sensitivity, specificity

    return None, None


def _find_results_dir(outdir: Path) -> Path:
    if (outdir / "summary.json").exists():
        return outdir
    candidates: list[tuple[int, Path]] = []
    for path in outdir.glob("results*"):
        if not path.is_dir():
            continue
        if not (path / "summary.json").exists():
            continue
        suffix = path.name.replace("results", "", 1)
        if suffix.startswith("_"):
            suffix = suffix[1:]
        try:
            index = int(suffix) if suffix else 0
        except ValueError:
            index = 0
        candidates.append((index, path))
    if not candidates:
        raise FileNotFoundError(f"Nenhum results com summary.json encontrado em {outdir}")
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _pick_best_model(results_dir: Path) -> Path:
    best_model = results_dir / "best_model.pt"
    if best_model.exists():
        return best_model
    matches = sorted(results_dir.glob("best_model*.pt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Checkpoint best_model*.pt ausente em {results_dir}")


def _resolve_metrics_json(metrics_dir: Path, view: str | None) -> Path:
    candidates = [metrics_dir / "val_metrics.json"]
    if view:
        candidates.append(metrics_dir / f"val_metrics_{view}.json")
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(metrics_dir.glob("val_metrics_*.json"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise FileNotFoundError(
            f"Multiplos val_metrics_*.json encontrados em {metrics_dir}; "
            "especifique a view no summary.json."
        )
    raise FileNotFoundError(f"val_metrics.json ausente: {metrics_dir}")


def _resolve_metrics_fallback(results_dir: Path, view: str | None) -> Path | None:
    if not view:
        return None
    suffix = f"_{view}"
    name = results_dir.name
    if name.lower().endswith(suffix):
        base_name = name[: -len(suffix)]
        base_dir = results_dir.parent / base_name
        fallback = base_dir / "metrics"
        if fallback.exists():
            return fallback
    return None


def _resolve_confusion_matrix(
    metrics_dir: Path,
    figures_dir: Path,
    view: str | None,
) -> Path | None:
    suffix = f"_{view}" if view else ""
    candidates = [
        metrics_dir / f"val_metrics{suffix}.png",
        metrics_dir / f"best_metrics{suffix}.png",
        figures_dir / f"val_metrics{suffix}.png",
        figures_dir / f"best_metrics{suffix}.png",
        metrics_dir / "val_metrics.png",
        metrics_dir / "best_metrics.png",
        figures_dir / "val_metrics.png",
        figures_dir / "best_metrics.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    for pattern in ("val_metrics_*.png", "best_metrics_*.png"):
        matches = sorted(metrics_dir.glob(pattern))
        if len(matches) == 1:
            return matches[0]
    return None


def _collect_artifacts(outdir: Path) -> TrainingArtifacts:
    results_dir = _find_results_dir(outdir)
    summary_path = results_dir / "summary.json"
    train_history_path = results_dir / "train_history.csv"
    metrics_dir = results_dir / "metrics"

    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json ausente: {summary_path}")
    if not train_history_path.exists():
        raise FileNotFoundError(f"train_history.csv ausente: {train_history_path}")

    summary = _load_json(summary_path)
    view = _normalize_view(summary.get("view"))
    metrics_root = metrics_dir
    try:
        val_metrics_path = _resolve_metrics_json(metrics_dir, view)
    except FileNotFoundError:
        fallback_dir = _resolve_metrics_fallback(results_dir, view)
        if fallback_dir is None:
            raise
        metrics_root = fallback_dir
        val_metrics_path = _resolve_metrics_json(metrics_root, view)

    checkpoint_path = _pick_best_model(results_dir)
    confusion_matrix_path = _resolve_confusion_matrix(
        metrics_root,
        results_dir / "figures",
        view,
    )
    val_predictions_path = results_dir / "val_predictions.csv"
    if not val_predictions_path.exists():
        val_predictions_path = None
    train_history_plot_path = results_dir / "train_history.png"
    if not train_history_plot_path.exists():
        train_history_plot_path = None

    return TrainingArtifacts(
        results_dir=results_dir,
        summary_path=summary_path,
        train_history_path=train_history_path,
        val_metrics_path=val_metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        checkpoint_path=checkpoint_path,
        val_predictions_path=val_predictions_path,
        train_history_plot_path=train_history_plot_path,
    )


def _load_train_history(train_history_path: Path) -> dict[str, float | None]:
    with train_history_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("train_history.csv sem dados")
    last_row = rows[-1]
    return {
        "train_loss": _parse_optional_float(last_row.get("train_loss")),
        "train_acc": _parse_optional_float(last_row.get("train_acc")),
        "val_loss": _parse_optional_float(last_row.get("val_loss")),
        "val_acc": _parse_optional_float(last_row.get("val_acc")),
        "val_f1": _parse_optional_float(last_row.get("val_macro_f1")),
        "val_auc": _parse_optional_float(last_row.get("val_auc")),
    }


def _load_val_metrics(val_metrics_path: Path) -> dict[str, Any]:
    return _load_json(val_metrics_path)


def _format_metric(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return str(value)


def _collect_metrics(
    history_metrics: Mapping[str, float | None],
    val_metrics: Mapping[str, Any],
) -> dict[str, float | None]:
    report = val_metrics.get("classification_report")
    report_map = report if isinstance(report, Mapping) else {}

    val_acc = history_metrics.get("val_acc")
    if val_acc is None:
        val_acc = _parse_optional_float(val_metrics.get("acc"))
    if val_acc is None:
        val_acc = _parse_optional_float(val_metrics.get("accuracy"))
    if val_acc is None:
        val_acc = _parse_optional_float(report_map.get("accuracy"))

    val_loss = history_metrics.get("val_loss")
    if val_loss is None:
        val_loss = _parse_optional_float(val_metrics.get("loss"))

    val_f1 = history_metrics.get("val_f1")
    if val_f1 is None:
        val_f1 = _parse_optional_float(val_metrics.get("macro_f1"))
    if val_f1 is None:
        val_f1 = _parse_optional_float(val_metrics.get("f1"))
    if val_f1 is None:
        macro_avg = report_map.get("macro avg")
        if isinstance(macro_avg, Mapping):
            val_f1 = _parse_optional_float(
                macro_avg.get("f1-score") or macro_avg.get("f1")
            )

    val_auc = history_metrics.get("val_auc")
    if val_auc is None:
        val_auc = _parse_optional_float(val_metrics.get("auc_ovr"))
    if val_auc is None:
        val_auc = _parse_optional_float(val_metrics.get("auc"))
    if val_auc is None:
        val_auc = _parse_optional_float(val_metrics.get("roc_auc"))

    val_sensitivity, val_specificity = _compute_sensitivity_specificity(val_metrics)

    return {
        "train_loss": history_metrics.get("train_loss"),
        "train_acc": history_metrics.get("train_acc"),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_auc": val_auc,
        "val_sensitivity": val_sensitivity,
        "val_specificity": val_specificity,
    }


def _compute_metric_delta(
    value: float | None, baseline: float | None
) -> float | None:
    if value is None or baseline is None:
        return None
    if not math.isfinite(value) or not math.isfinite(baseline):
        return None
    return value - baseline


def _collect_baseline_metrics(outdir: Path) -> dict[str, float | None]:
    artifacts = _collect_artifacts(outdir)
    history_metrics = _load_train_history(artifacts.train_history_path)
    val_metrics = _load_val_metrics(artifacts.val_metrics_path)
    return _collect_metrics(history_metrics, val_metrics)


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


def _collect_params(summary: Mapping[str, Any]) -> dict[str, str]:
    params: dict[str, str] = {}
    for key, value in summary.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            params[key] = str(value)
        else:
            params[key] = json.dumps(value, ensure_ascii=True, default=str)
    return params


def _log_local_mlflow_run(
    *,
    run_name: str,
    params: Mapping[str, str],
    metrics: Mapping[str, float],
    artifacts: Sequence[Path],
    tracking_uri: str | None,
    experiment: str | None,
) -> str:
    tracking_root = _resolve_tracking_root(tracking_uri)
    experiment_id = "0"
    run_id = uuid.uuid4().hex
    run_dir = tracking_root / experiment_id / run_id
    artifacts_dir = run_dir / "artifacts"
    params_dir = run_dir / "params"
    metrics_dir = run_dir / "metrics"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    artifact_uri = f"file:{artifacts_dir.resolve()}"
    _write_meta_yaml(
        meta_path=run_dir / "meta.yaml",
        run_id=run_id,
        run_name=run_name,
        experiment_id=experiment_id,
        artifact_uri=artifact_uri,
        start_time_ms=timestamp_ms,
        end_time_ms=timestamp_ms,
    )
    for key, value in params.items():
        _write_param(params_dir, key, value)
    if experiment:
        _write_param(params_dir, "experiment", experiment)
    for key, value in metrics.items():
        _write_metric(metrics_dir, key, value, timestamp_ms)
    for path in artifacts:
        shutil.copy2(path, artifacts_dir / path.name)
    return run_id


def log_mlflow_run(
    *,
    run_name: str,
    params: Mapping[str, str],
    metrics: Mapping[str, float],
    artifacts: Sequence[Path],
    tracking_uri: str | None = None,
    experiment: str | None = None,
) -> str:
    for path in artifacts:
        if not path.exists():
            raise FileNotFoundError(f"Artefato ausente: {path}")
    try:
        import mlflow  # type: ignore
    except Exception:
        return _log_local_mlflow_run(
            run_name=run_name,
            params=params,
            metrics=metrics,
            artifacts=artifacts,
            tracking_uri=tracking_uri,
            experiment=experiment,
        )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment:
        mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(dict(params))
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        for path in artifacts:
            mlflow.log_artifact(str(path))
        return run.info.run_id


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
        "confusion_matrix_path": str(confusion_matrix_path) if confusion_matrix_path else "",
        "checkpoint_path": str(checkpoint_path),
        "val_predictions_path": str(val_predictions_path) if val_predictions_path else "",
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
        *(
            [f"- Tuned hyperparameters: {tuned_text}"]
            if tuned_params
            else []
        ),
        f"- Train loss: {metrics.get('train_loss')}",
        f"- Train acc: {metrics.get('train_acc')}",
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
        *baseline_lines,
        "",
    ]


def _resolve_ensemble_confusion_matrix(metrics_dir: Path) -> Path | None:
    candidates = [
        metrics_dir / "ensemble_metrics.png",
        metrics_dir / "ensemble_confusion_matrix.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(metrics_dir.glob("ensemble*_metrics.png"))
    if len(matches) == 1:
        return matches[0]
    return None


def _collect_ensemble_metrics(
    val_metrics: Mapping[str, Any],
) -> dict[str, float | None]:
    val_acc = _parse_optional_float(val_metrics.get("acc"))
    val_loss = _parse_optional_float(val_metrics.get("loss"))
    val_f1 = _parse_optional_float(val_metrics.get("macro_f1"))
    if val_f1 is None:
        val_f1 = _parse_optional_float(val_metrics.get("f1"))
    val_auc = _parse_optional_float(val_metrics.get("auc_ovr"))
    if val_auc is None:
        val_auc = _parse_optional_float(val_metrics.get("auc"))

    val_sensitivity, val_specificity = _compute_sensitivity_specificity(val_metrics)

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
        "confusion_matrix_path": str(confusion_matrix_path) if confusion_matrix_path else "",
        "checkpoint_path": "",
        "val_predictions_path": "",
        "train_loss": _format_metric(metrics.get("train_loss")),
        "train_acc": _format_metric(metrics.get("train_acc")),
        "val_loss": _format_metric(metrics.get("val_loss")),
        "val_acc": _format_metric(metrics.get("val_acc")),
        "val_f1": _format_metric(metrics.get("val_f1")),
        "val_auc": _format_metric(metrics.get("val_auc")),
        "val_sensitivity": _format_metric(metrics.get("val_sensitivity")),
        "val_specificity": _format_metric(metrics.get("val_specificity")),
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
    stamp = timestamp or datetime.now().date().isoformat()
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


def register_training_run(
    *,
    outdir: Path,
    dataset: str,
    workflow: str,
    run_name: str,
    command: str,
    registry_csv: Path,
    registry_md: Path,
    tracking_uri: str | None = None,
    experiment: str | None = None,
    baseline_outdir: Path | None = None,
    baseline_run_name: str | None = None,
    tuned_keys: Sequence[str] | None = None,
) -> str:
    artifacts = _collect_artifacts(outdir)
    summary = _load_json(artifacts.summary_path)
    history_metrics = _load_train_history(artifacts.train_history_path)
    val_metrics = _load_val_metrics(artifacts.val_metrics_path)
    metrics = _collect_metrics(history_metrics, val_metrics)
    tuned_params = _extract_tuned_params(summary, tuned_keys)
    tuned_params_json = (
        json.dumps(tuned_params, ensure_ascii=True, sort_keys=True, default=str)
        if tuned_params
        else ""
    )

    baseline_metrics: dict[str, float | None] = {}
    baseline_deltas: dict[str, float | None] = {}
    if baseline_outdir:
        baseline_metrics = _collect_baseline_metrics(baseline_outdir)
        for key in ("val_loss", "val_acc", "val_f1", "val_auc"):
            baseline_deltas[key] = _compute_metric_delta(
                metrics.get(key), baseline_metrics.get(key)
            )

    params = _collect_params(summary)

    artifact_list: list[Path] = [
        artifacts.summary_path,
        artifacts.train_history_path,
        artifacts.val_metrics_path,
        artifacts.checkpoint_path,
    ]
    if artifacts.confusion_matrix_path:
        artifact_list.append(artifacts.confusion_matrix_path)
    if artifacts.val_predictions_path:
        artifact_list.append(artifacts.val_predictions_path)
    if artifacts.train_history_plot_path:
        artifact_list.append(artifacts.train_history_plot_path)

    metric_payload: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None or not math.isfinite(value):
            continue
        metric_payload[key] = float(value)

    run_id = log_mlflow_run(
        run_name=run_name,
        params=params,
        metrics=metric_payload,
        artifacts=artifact_list,
        tracking_uri=tracking_uri,
        experiment=experiment,
    )

    dataset_value = dataset or str(summary.get("dataset") or "unknown")
    model_value = str(summary.get("arch") or "unknown")
    batch_size = int(summary.get("batch_size") or 0)
    img_size = int(summary.get("img_size") or 0)
    hyperparameters = json.dumps(summary, ensure_ascii=True, default=str)

    row = build_registry_row(
        dataset=dataset_value,
        workflow=workflow,
        run_name=run_name,
        command=command,
        model=model_value,
        batch_size=batch_size,
        img_size=img_size,
        results_dir=artifacts.results_dir,
        summary_path=artifacts.summary_path,
        train_history_path=artifacts.train_history_path,
        val_metrics_path=artifacts.val_metrics_path,
        confusion_matrix_path=artifacts.confusion_matrix_path,
        checkpoint_path=artifacts.checkpoint_path,
        val_predictions_path=artifacts.val_predictions_path,
        hyperparameters=hyperparameters,
        metrics=metrics,
        baseline_run_name=baseline_run_name,
        baseline_metrics=baseline_metrics,
        baseline_deltas=baseline_deltas,
        tuned_hyperparameters=tuned_params_json,
        mlflow_run_id=run_id,
    )
    append_registry_csv(registry_csv, row)

    outputs: list[Path] = [
        artifacts.summary_path,
        artifacts.train_history_path,
        artifacts.val_metrics_path,
        artifacts.checkpoint_path,
    ]
    if artifacts.confusion_matrix_path:
        outputs.append(artifacts.confusion_matrix_path)
    if artifacts.val_predictions_path:
        outputs.append(artifacts.val_predictions_path)

    entry_lines = build_markdown_entry(
        dataset=dataset_value,
        workflow=workflow,
        run_name=run_name,
        command=command,
        results_dir=artifacts.results_dir,
        outputs=outputs,
        metrics=metrics,
        hyperparameters_summary=_summarize_hyperparameters(summary),
        confusion_matrix_path=artifacts.confusion_matrix_path,
        baseline_run_name=baseline_run_name,
        baseline_metrics=baseline_metrics,
        baseline_deltas=baseline_deltas,
        tuned_params=tuned_params,
    )
    append_registry_markdown(registry_md, entry_lines)
    return run_id


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
    if not metrics_path.exists():
        raise FileNotFoundError(f"ensemble_metrics.json ausente: {metrics_path}")
    val_metrics = _load_json(metrics_path)
    if not isinstance(val_metrics, Mapping):
        raise ValueError("ensemble_metrics.json invalido: esperado objeto JSON.")

    summary: dict[str, Any] = {}
    if summary_path:
        if not summary_path.exists():
            raise FileNotFoundError(f"summary.json ausente: {summary_path}")
        summary = _load_json(summary_path)

    metrics = _collect_ensemble_metrics(val_metrics)

    params: dict[str, str] = {}
    if summary:
        params.update(_collect_params(summary))
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

    run_id = log_mlflow_run(
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
    hyperparameters = json.dumps(summary, ensure_ascii=True, default=str) if summary else "{}"

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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Registra resultados de treinamento em MLflow e no registry local."
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--workflow", default="train-density")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument(
        "--ensemble-metrics",
        type=Path,
        default=None,
        help="Path para ensemble_metrics.json (ativa registro de ensemble)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path para summary.json (usado no registro de ensemble)",
    )
    parser.add_argument("--registry-csv", type=Path, default=Path("results/registry.csv"))
    parser.add_argument("--registry-md", type=Path, default=Path("results/registry.md"))
    parser.add_argument("--tracking-uri", default="")
    parser.add_argument("--experiment", default="")
    parser.add_argument(
        "--baseline-outdir",
        type=Path,
        default=None,
        help="Diretorio de resultados base para comparar desempenho (opcional).",
    )
    parser.add_argument(
        "--baseline-run-name",
        default="",
        help="Nome do run baseline para referencia no registry (opcional).",
    )
    parser.add_argument(
        "--tuned-keys",
        default="",
        help=(
            "Lista separada por virgulas de hiperparametros para registrar como tuned."
        ),
    )
    args = parser.parse_args(argv)

    if args.ensemble_metrics:
        register_ensemble_run(
            outdir=args.outdir,
            metrics_path=args.ensemble_metrics,
            summary_path=args.summary,
            dataset=args.dataset,
            workflow=args.workflow,
            run_name=args.run_name,
            command=args.command,
            registry_csv=args.registry_csv,
            registry_md=args.registry_md,
            tracking_uri=args.tracking_uri or None,
            experiment=args.experiment or None,
        )
    else:
        tuned_keys = [item.strip() for item in args.tuned_keys.split(",") if item.strip()]
        register_training_run(
            outdir=args.outdir,
            dataset=args.dataset,
            workflow=args.workflow,
            run_name=args.run_name,
            command=args.command,
            registry_csv=args.registry_csv,
            registry_md=args.registry_md,
            tracking_uri=args.tracking_uri or None,
            experiment=args.experiment or None,
            baseline_outdir=args.baseline_outdir,
            baseline_run_name=args.baseline_run_name or None,
            tuned_keys=tuned_keys or None,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
