#!/usr/bin/env python3
"""Register explainability outputs in MLflow and local registry files."""

from __future__ import annotations

import argparse
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from mammography.tools.data_audit_registry import (
    _resolve_tracking_root,
    _write_meta_yaml,
    _write_metric,
    _write_param,
    append_registry_csv,
    append_registry_markdown,
)


@dataclass(frozen=True)
class ExplainMetrics:
    total_images: int
    loaded_images: int
    saved_images: int
    failed_images: int


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "explain"


def infer_dataset_name(input_path: Path) -> str:
    if input_path.is_file():
        name = input_path.parent.name
    else:
        name = input_path.name
    if not name:
        name = input_path.resolve().name
    return _sanitize_name(name or "explain")


def default_run_name(dataset: str, method: str) -> str:
    return f"{_sanitize_name(dataset)}_explain_{_sanitize_name(method)}"


def default_run_name_from_output(output_dir: Path, method: str) -> str:
    """Create a run name based on the output directory name and method."""
    return f"{_sanitize_name(output_dir.name)}_{_sanitize_name(method)}"


def _log_local_mlflow_run(
    *,
    run_name: str,
    params: Mapping[str, str],
    metrics: Mapping[str, float],
    artifacts: Sequence[Path],
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
    for path in artifacts:
        shutil.copy2(path, artifacts_dir / path.name)
    for path in artifact_dirs:
        if path.exists():
            shutil.copytree(path, artifacts_dir / path.name, dirs_exist_ok=True)
    return run_id


def log_mlflow_run(
    *,
    run_name: str,
    params: Mapping[str, str],
    metrics: Mapping[str, float],
    artifacts: Sequence[Path],
    artifact_dirs: Sequence[Path],
    tracking_uri: str | None = None,
    experiment: str | None = None,
) -> str:
    for path in artifacts:
        if not path.exists():
            raise FileNotFoundError(f"Artefato ausente: {path}")
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
            artifacts=artifacts,
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
        for path in artifacts:
            mlflow.log_artifact(str(path))
        for path in artifact_dirs:
            mlflow.log_artifacts(str(path), artifact_path=path.name)
        return run.info.run_id


def build_registry_row(
    *,
    dataset: str,
    run_name: str,
    command: str,
    arch: str,
    method: str,
    layer: str,
    batch_size: int,
    img_size: int,
    output_dir: Path,
    output_path: Path,
    checkpoint_path: Path,
    metrics: ExplainMetrics,
    mlflow_run_id: str | None,
    timestamp: str | None = None,
) -> dict[str, str | int]:
    return {
        "timestamp": timestamp or datetime.now(tz=timezone.utc).isoformat(),
        "dataset": dataset,
        "workflow": "explain",
        "run_name": run_name,
        "command": command,
        "model": arch,
        "layer": layer,
        "batch_size": batch_size,
        "img_size": img_size,
        "outdir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "explain_output_path": str(output_path),
        "explain_total_images": metrics.total_images,
        "explain_loaded_images": metrics.loaded_images,
        "explain_saved_images": metrics.saved_images,
        "explain_failed_images": metrics.failed_images,
        "explain_method": method,
        "mlflow_run_id": mlflow_run_id or "",
    }


def build_markdown_entry(
    *,
    dataset: str,
    run_name: str,
    command: str,
    arch: str,
    method: str,
    layer: str,
    batch_size: int,
    img_size: int,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    metrics: ExplainMetrics,
    timestamp: str | None = None,
) -> list[str]:
    stamp = timestamp or datetime.now().date().isoformat()
    return [
        f"## {stamp} - {run_name}",
        f"- Dataset: {dataset}",
        "- Workflow: explain",
        f"- Run name: {run_name}",
        f"- Command: `{command}`",
        f"- Model: {arch}",
        f"- Method: {method}",
        f"- Layer: {layer or 'auto'}",
        f"- Batch size: {batch_size}",
        f"- Image size: {img_size}",
        f"- Input: `{input_path}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Output: `{output_path}`",
        f"- Total images: {metrics.total_images}",
        f"- Loaded images: {metrics.loaded_images}",
        f"- Saved images: {metrics.saved_images}",
        f"- Failed images: {metrics.failed_images}",
        "",
    ]


def register_explain_run(
    *,
    input_path: Path,
    output_dir: Path,
    output_path: Path,
    checkpoint_path: Path,
    arch: str,
    method: str,
    layer: str,
    img_size: int,
    batch_size: int,
    metrics: ExplainMetrics,
    run_name: str,
    command: str,
    registry_csv: Path,
    registry_md: Path,
    artifacts: Sequence[Path] | None = None,
    artifact_dirs: Sequence[Path] | None = None,
    tracking_uri: str | None = None,
    experiment: str | None = None,
    log_mlflow: bool = True,
) -> str | None:
    dataset = infer_dataset_name(input_path)
    params = {
        "dataset": dataset,
        "workflow": "explain",
        "command": command,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "output_path": str(output_path),
        "checkpoint_path": str(checkpoint_path),
        "arch": arch,
        "method": method,
        "layer": layer,
        "batch_size": str(batch_size),
        "img_size": str(img_size),
    }
    metric_payload = {
        "total_images": float(metrics.total_images),
        "loaded_images": float(metrics.loaded_images),
        "saved_images": float(metrics.saved_images),
        "failed_images": float(metrics.failed_images),
    }
    artifacts = list(artifacts or [])
    artifact_dirs = list(artifact_dirs or [])
    run_id: str | None = None
    if log_mlflow:
        run_id = log_mlflow_run(
            run_name=run_name,
            params=params,
            metrics=metric_payload,
            artifacts=artifacts,
            artifact_dirs=artifact_dirs,
            tracking_uri=tracking_uri,
            experiment=experiment,
        )
    row = build_registry_row(
        dataset=dataset,
        run_name=run_name,
        command=command,
        arch=arch,
        method=method,
        layer=layer,
        batch_size=batch_size,
        img_size=img_size,
        output_dir=output_dir,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
        mlflow_run_id=run_id,
    )
    append_registry_csv(registry_csv, row)
    entry_lines = build_markdown_entry(
        dataset=dataset,
        run_name=run_name,
        command=command,
        arch=arch,
        method=method,
        layer=layer,
        batch_size=batch_size,
        img_size=img_size,
        input_path=input_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )
    append_registry_markdown(registry_md, entry_lines)
    return run_id


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Registra explainability outputs em MLflow e no registry local."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--layer", default="")
    parser.add_argument("--img-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--total-images", type=int, required=True)
    parser.add_argument("--loaded-images", type=int, required=True)
    parser.add_argument("--saved-images", type=int, required=True)
    parser.add_argument("--failed-images", type=int, required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--command", required=True)
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("results/registry.csv"),
    )
    parser.add_argument(
        "--registry-md",
        type=Path,
        default=Path("results/registry.md"),
    )
    parser.add_argument("--tracking-uri", default="")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    run_name = args.run_name or default_run_name(
        infer_dataset_name(args.input),
        args.method,
    )
    metrics = ExplainMetrics(
        total_images=args.total_images,
        loaded_images=args.loaded_images,
        saved_images=args.saved_images,
        failed_images=args.failed_images,
    )
    register_explain_run(
        input_path=args.input,
        output_dir=args.output_dir,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint,
        arch=args.arch,
        method=args.method,
        layer=args.layer,
        img_size=args.img_size,
        batch_size=args.batch_size,
        metrics=metrics,
        run_name=run_name,
        command=args.command,
        registry_csv=args.registry_csv,
        registry_md=args.registry_md,
        tracking_uri=args.tracking_uri or None,
        experiment=args.experiment or None,
        log_mlflow=not args.no_mlflow,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
