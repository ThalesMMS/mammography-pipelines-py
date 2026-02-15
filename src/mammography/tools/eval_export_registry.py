#!/usr/bin/env python3
"""Register evaluation export outputs in MLflow and local registry files."""

from __future__ import annotations

import argparse
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mammography.tools.data_audit_registry import (
    _resolve_tracking_root,
    _write_meta_yaml,
    _write_metric,
    _write_param,
    append_registry_csv,
    append_registry_markdown,
)


@dataclass(frozen=True)
class EvalExportOutputs:
    export_dir: Path
    output_paths: list[Path]


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "eval_export"


def infer_dataset_name(summary: Mapping[str, Any] | None, run_dir: Path) -> str:
    dataset_value = None
    if summary and summary.get("dataset") is not None:
        dataset_value = str(summary.get("dataset"))
    if not dataset_value:
        candidate = run_dir.parent.name if run_dir.name.startswith("results") else run_dir.name
        dataset_value = candidate or run_dir.name
    return _sanitize_name(dataset_value)


def default_run_name(dataset: str) -> str:
    return f"{_sanitize_name(dataset)}_eval_export"


def collect_export_outputs(export_dir: Path) -> EvalExportOutputs:
    if not export_dir.exists():
        raise FileNotFoundError(f"Diretorio de exportacao nao encontrado: {export_dir}")
    output_paths = sorted(
        (path for path in export_dir.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix(),
    )
    return EvalExportOutputs(export_dir=export_dir, output_paths=output_paths)


def _format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(str(path) for path in paths)


def _log_local_mlflow_run(
    *,
    run_name: str,
    params: Mapping[str, str],
    metrics: Mapping[str, float],
    artifact_dirs: Sequence[Path],
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
    for path in artifact_dirs:
        if path.exists():
            shutil.copytree(path, artifacts_dir / path.name, dirs_exist_ok=True)
    return run_id


def log_mlflow_run(
    *,
    run_name: str,
    params: Mapping[str, str],
    metrics: Mapping[str, float],
    artifact_dirs: Sequence[Path],
    tracking_uri: str | None = None,
    experiment: str | None = None,
) -> str:
    for path in artifact_dirs:
        if not path.exists():
            raise FileNotFoundError(f"Diretorio de artefatos ausente: {path}")
    try:
        import mlflow  # type: ignore
    except Exception:
        return _log_local_mlflow_run(
            run_name=run_name,
            params=params,
            metrics=metrics,
            artifact_dirs=artifact_dirs,
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
        for path in artifact_dirs:
            mlflow.log_artifacts(str(path), artifact_path=path.name)
        return run.info.run_id


def build_registry_row(
    *,
    dataset: str,
    run_name: str,
    command: str,
    export_dir: Path,
    summary_path: Path | None,
    train_history_path: Path | None,
    val_metrics_path: Path | None,
    confusion_matrix_path: Path | None,
    checkpoint_path: Path | None,
    val_predictions_path: Path | None,
    export_paths: Sequence[Path],
    mlflow_run_id: str | None,
    timestamp: str | None = None,
) -> dict[str, str | int]:
    return {
        "timestamp": timestamp or datetime.now(tz=timezone.utc).isoformat(),
        "dataset": dataset,
        "workflow": "eval-export",
        "run_name": run_name,
        "command": command,
        "outdir": str(export_dir),
        "summary_path": str(summary_path) if summary_path else "",
        "train_history_path": str(train_history_path) if train_history_path else "",
        "val_metrics_path": str(val_metrics_path) if val_metrics_path else "",
        "confusion_matrix_path": (
            str(confusion_matrix_path) if confusion_matrix_path else ""
        ),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "val_predictions_path": str(val_predictions_path) if val_predictions_path else "",
        "visualization_output_paths": _format_paths(export_paths),
        "mlflow_run_id": mlflow_run_id or "",
    }


def build_markdown_entry(
    *,
    dataset: str,
    run_name: str,
    command: str,
    export_dir: Path,
    export_paths: Sequence[Path],
    timestamp: str | None = None,
) -> list[str]:
    stamp = timestamp or datetime.now().date().isoformat()
    outputs_str = ", ".join(f"`{path}`" for path in export_paths)
    return [
        f"## {stamp} - {run_name}",
        f"- Dataset: {dataset}",
        "- Workflow: eval-export",
        f"- Run name: {run_name}",
        f"- Command: `{command}`",
        f"- Outdir: `{export_dir}`",
        f"- Outputs: {outputs_str}",
        "",
    ]


def register_eval_export_run(
    *,
    dataset: str,
    run_name: str,
    command: str,
    export_dir: Path,
    export_paths: Sequence[Path],
    summary_path: Path | None,
    train_history_path: Path | None,
    val_metrics_path: Path | None,
    confusion_matrix_path: Path | None,
    checkpoint_path: Path | None,
    val_predictions_path: Path | None,
    registry_csv: Path,
    registry_md: Path,
    tracking_uri: str | None = None,
    experiment: str | None = None,
    log_mlflow: bool = True,
) -> str | None:
    params = {
        "dataset": dataset,
        "workflow": "eval-export",
        "command": command,
        "export_dir": str(export_dir),
    }
    metrics = {"exported_artifacts": float(len(export_paths))}
    run_id: str | None = None
    if log_mlflow:
        run_id = log_mlflow_run(
            run_name=run_name,
            params=params,
            metrics=metrics,
            artifact_dirs=[export_dir],
            tracking_uri=tracking_uri,
            experiment=experiment,
        )

    row = build_registry_row(
        dataset=dataset,
        run_name=run_name,
        command=command,
        export_dir=export_dir,
        summary_path=summary_path,
        train_history_path=train_history_path,
        val_metrics_path=val_metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        checkpoint_path=checkpoint_path,
        val_predictions_path=val_predictions_path,
        export_paths=export_paths,
        mlflow_run_id=run_id,
    )
    append_registry_csv(registry_csv, row)
    entry_lines = build_markdown_entry(
        dataset=dataset,
        run_name=run_name,
        command=command,
        export_dir=export_dir,
        export_paths=export_paths,
    )
    append_registry_markdown(registry_md, entry_lines)
    return run_id


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Registra exportacoes de avaliacao em MLflow e registry local."
    )
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--train-history-path", type=Path)
    parser.add_argument("--val-metrics-path", type=Path)
    parser.add_argument("--confusion-matrix-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--val-predictions-path", type=Path)
    parser.add_argument("--registry-csv", type=Path, default=Path("results/registry.csv"))
    parser.add_argument("--registry-md", type=Path, default=Path("results/registry.md"))
    parser.add_argument("--tracking-uri", default="")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    artifacts = collect_export_outputs(args.export_dir)
    register_eval_export_run(
        dataset=args.dataset,
        run_name=args.run_name,
        command=args.command,
        export_dir=args.export_dir,
        export_paths=artifacts.output_paths,
        summary_path=args.summary_path,
        train_history_path=args.train_history_path,
        val_metrics_path=args.val_metrics_path,
        confusion_matrix_path=args.confusion_matrix_path,
        checkpoint_path=args.checkpoint_path,
        val_predictions_path=args.val_predictions_path,
        registry_csv=args.registry_csv,
        registry_md=args.registry_md,
        tracking_uri=args.tracking_uri or None,
        experiment=args.experiment or None,
        log_mlflow=not args.no_mlflow,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
