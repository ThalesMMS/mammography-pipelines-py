#!/usr/bin/env python3
"""Register tuning outputs in MLflow and local registry files."""

from __future__ import annotations

import json
import math
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from mammography.tools.data_audit_registry import (
    _resolve_tracking_root,
    _write_meta_yaml,
    _write_metric,
    _write_param,
    append_registry_csv,
    append_registry_markdown,
)
from mammography.utils.class_modes import normalize_classes_mode


def _format_metric(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def _serialize_params(params: Mapping[str, object]) -> str:
    return json.dumps(dict(params), ensure_ascii=True, sort_keys=True, default=str)


def _build_mlflow_params(
    *,
    dataset: str,
    arch: str,
    classes: str,
    study_name: str,
    storage: str | None,
    best_params: Mapping[str, object],
) -> dict[str, str]:
    payload = {
        "dataset": dataset,
        "arch": arch,
        "classes": classes,
        "study_name": study_name,
    }
    if storage:
        payload["storage"] = storage
    for key, value in best_params.items():
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            payload[f"best_{key}"] = json.dumps(value, ensure_ascii=True, default=str)
        else:
            payload[f"best_{key}"] = str(value)
    return payload


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

    try:
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
    except Exception:
        return _log_local_mlflow_run(
            run_name=run_name,
            params=params,
            metrics=metrics,
            artifacts=artifacts,
            tracking_uri=tracking_uri,
            experiment=experiment,
        )


def build_registry_row(
    *,
    dataset: str,
    workflow: str,
    run_name: str,
    command: str,
    model: str,
    img_size: int,
    outdir: Path,
    hyperparameters: str,
    n_trials: int,
    completed_trials: int,
    pruned_trials: int,
    best_trial: int,
    best_value: float,
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
        "img_size": img_size,
        "outdir": str(outdir),
        "hyperparameters": hyperparameters,
        "tune_n_trials": n_trials,
        "tune_completed_trials": completed_trials,
        "tune_pruned_trials": pruned_trials,
        "tune_best_trial": best_trial,
        "tune_best_value": _format_metric(best_value),
        "mlflow_run_id": mlflow_run_id or "",
    }


def build_markdown_entry(
    *,
    dataset: str,
    workflow: str,
    run_name: str,
    command: str,
    outdir: Path,
    model: str,
    classes: str,
    img_size: int,
    n_trials: int,
    completed_trials: int,
    pruned_trials: int,
    best_trial: int,
    best_value: float,
    best_params: Mapping[str, object],
    outputs: Iterable[Path],
    timestamp: str | None = None,
) -> list[str]:
    stamp = timestamp or datetime.now().date().isoformat()
    outputs_str = ", ".join(f"`{path}`" for path in outputs) if outputs else "n/a"
    best_params_text = _serialize_params(best_params)
    return [
        f"## {stamp} - {run_name}",
        f"- Dataset: {dataset}",
        f"- Workflow: {workflow}",
        f"- Run name: {run_name}",
        f"- Command: `{command}`",
        f"- Model: {model}",
        f"- Classes: {classes}",
        f"- Img size: {img_size}",
        f"- Output dir: `{outdir}`",
        f"- Outputs: {outputs_str}",
        (
            "- Trials: total={total}, completed={completed}, pruned={pruned}".format(
                total=n_trials,
                completed=completed_trials,
                pruned=pruned_trials,
            )
        ),
        f"- Best trial: {best_trial} (value={best_value:.4f})",
        f"- Best params: `{best_params_text}`",
        "",
    ]


def register_tune_run(
    *,
    outdir: Path,
    dataset: str,
    arch: str,
    classes: str,
    img_size: int,
    run_name: str,
    command: str,
    study_name: str,
    n_trials: int,
    completed_trials: int,
    pruned_trials: int,
    best_trial: int,
    best_value: float,
    best_params: Mapping[str, object],
    storage: str | None,
    best_params_path: Path | None,
    stats_path: Path | None,
    optuna_db_path: Path | None,
    registry_csv: Path,
    registry_md: Path,
    tracking_uri: str | None = None,
    experiment: str | None = None,
) -> str:
    classes = normalize_classes_mode(classes, allow_unknown=True)
    params = _build_mlflow_params(
        dataset=dataset,
        arch=arch,
        classes=classes,
        study_name=study_name,
        storage=storage,
        best_params=best_params,
    )
    metrics = {
        "best_value": float(best_value),
        "best_trial": float(best_trial),
        "n_trials": float(n_trials),
        "completed_trials": float(completed_trials),
        "pruned_trials": float(pruned_trials),
    }

    artifacts: list[Path] = []
    for path in (best_params_path, stats_path, optuna_db_path):
        if path is not None and path.exists():
            artifacts.append(path)

    run_id = log_mlflow_run(
        run_name=run_name,
        params=params,
        metrics=metrics,
        artifacts=artifacts,
        tracking_uri=tracking_uri,
        experiment=experiment,
    )

    hyperparameters = _serialize_params(best_params)
    row = build_registry_row(
        dataset=dataset,
        workflow="tune",
        run_name=run_name,
        command=command,
        model=arch,
        img_size=img_size,
        outdir=outdir,
        hyperparameters=hyperparameters,
        n_trials=n_trials,
        completed_trials=completed_trials,
        pruned_trials=pruned_trials,
        best_trial=best_trial,
        best_value=best_value,
        mlflow_run_id=run_id,
    )
    append_registry_csv(registry_csv, row)

    outputs = artifacts or []
    entry_lines = build_markdown_entry(
        dataset=dataset,
        workflow="tune",
        run_name=run_name,
        command=command,
        outdir=outdir,
        model=arch,
        classes=classes,
        img_size=img_size,
        n_trials=n_trials,
        completed_trials=completed_trials,
        pruned_trials=pruned_trials,
        best_trial=best_trial,
        best_value=best_value,
        best_params=best_params,
        outputs=outputs,
    )
    append_registry_markdown(registry_md, entry_lines)
    return run_id
