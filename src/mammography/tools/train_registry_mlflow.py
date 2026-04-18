"""MLflow and local tracking helpers for training registry entries."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import shutil
from typing import TYPE_CHECKING, Any, Mapping, Sequence
import uuid

if TYPE_CHECKING:
    from pathlib import Path

from mammography.tools.data_audit_registry import (
    _is_plain_local_tracking_uri,
    _resolve_tracking_root,
    _write_meta_yaml,
    _write_metric,
    _write_param,
)


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
    if _is_plain_local_tracking_uri(tracking_uri):
        return _log_local_mlflow_run(
            run_name=run_name,
            params=params,
            metrics=metrics,
            artifacts=artifacts,
            tracking_uri=tracking_uri,
            experiment=experiment,
        )
    try:
        import mlflow  # type: ignore
    except ImportError:
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
