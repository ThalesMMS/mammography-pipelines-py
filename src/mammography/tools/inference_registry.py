#!/usr/bin/env python3
"""Register inference outputs in MLflow and local registry files."""

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
class InferenceMetrics:
    total_images: int
    images_per_sec: float
    duration_sec: float


ARCH_ALIASES = {
    "efficientnet_b0": "effnet",
    "resnet50": "resnet50",
}


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "inference"


def infer_dataset_name(input_path: Path) -> str:
    if input_path.is_file():
        name = input_path.parent.name
    else:
        name = input_path.name
    if not name:
        name = input_path.resolve().name
    return _sanitize_name(name or "inference")


def default_run_name(dataset: str, arch: str) -> str:
    alias = ARCH_ALIASES.get(arch, arch)
    return f"{_sanitize_name(dataset)}_inference_{_sanitize_name(alias)}"


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
    run_name: str,
    command: str,
    arch: str,
    batch_size: int,
    img_size: int,
    output_path: Path,
    checkpoint_path: Path,
    metrics: InferenceMetrics,
    mlflow_run_id: str | None,
    timestamp: str | None = None,
) -> dict[str, str | int | float]:
    return {
        "timestamp": timestamp or datetime.now(tz=timezone.utc).isoformat(),
        "dataset": dataset,
        "workflow": "inference",
        "run_name": run_name,
        "command": command,
        "model": arch,
        "layer": "",
        "batch_size": batch_size,
        "img_size": img_size,
        "outdir": str(output_path.parent),
        "checkpoint_path": str(checkpoint_path),
        "inference_output_path": str(output_path),
        "inference_total_images": metrics.total_images,
        "inference_images_per_sec": metrics.images_per_sec,
        "inference_duration_sec": metrics.duration_sec,
        "mlflow_run_id": mlflow_run_id or "",
        "hyperparameters": "",
    }


def build_markdown_entry(
    *,
    dataset: str,
    run_name: str,
    command: str,
    arch: str,
    classes: str,
    batch_size: int,
    img_size: int,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    metrics: InferenceMetrics,
    timestamp: str | None = None,
) -> list[str]:
    stamp = timestamp or datetime.now().date().isoformat()
    return [
        f"## {stamp} - {run_name}",
        f"- Dataset: {dataset}",
        "- Workflow: inference",
        f"- Run name: {run_name}",
        f"- Command: `{command}`",
        f"- Model: {arch}",
        f"- Classes: {classes}",
        f"- Batch size: {batch_size}",
        f"- Image size: {img_size}",
        f"- Input: `{input_path}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Output: `{output_path}`",
        f"- Total images: {metrics.total_images}",
        f"- Images/sec: {metrics.images_per_sec:.4f}",
        f"- Duration (sec): {metrics.duration_sec:.4f}",
        "",
    ]


def register_inference_run(
    *,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    arch: str,
    classes: str,
    img_size: int,
    batch_size: int,
    metrics: InferenceMetrics,
    run_name: str,
    command: str,
    registry_csv: Path,
    registry_md: Path,
    tracking_uri: str | None = None,
    experiment: str | None = None,
    log_mlflow: bool = True,
) -> str | None:
    params = {
        "dataset": infer_dataset_name(input_path),
        "workflow": "inference",
        "command": command,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "checkpoint_path": str(checkpoint_path),
        "arch": arch,
        "classes": classes,
        "batch_size": str(batch_size),
        "img_size": str(img_size),
    }
    metric_payload = {
        "total_images": float(metrics.total_images),
        "images_per_sec": metrics.images_per_sec,
        "inference_seconds": metrics.duration_sec,
    }
    artifacts = [output_path]
    run_id: str | None = None
    if log_mlflow:
        run_id = log_mlflow_run(
            run_name=run_name,
            params=params,
            metrics=metric_payload,
            artifacts=artifacts,
            tracking_uri=tracking_uri,
            experiment=experiment,
        )
    row = build_registry_row(
        dataset=params["dataset"],
        run_name=run_name,
        command=command,
        arch=arch,
        batch_size=batch_size,
        img_size=img_size,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
        mlflow_run_id=run_id,
    )
    append_registry_csv(registry_csv, row)
    entry_lines = build_markdown_entry(
        dataset=params["dataset"],
        run_name=run_name,
        command=command,
        arch=arch,
        classes=classes,
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
        description="Registra inferencias em MLflow e no registry local."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--img-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--total-images", type=int, required=True)
    parser.add_argument("--images-per-sec", type=float, required=True)
    parser.add_argument("--duration-sec", type=float, required=True)
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

    dataset = infer_dataset_name(args.input)
    run_name = args.run_name or default_run_name(dataset, args.arch)
    metrics = InferenceMetrics(
        total_images=args.total_images,
        images_per_sec=args.images_per_sec,
        duration_sec=args.duration_sec,
    )
    register_inference_run(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        arch=args.arch,
        classes=args.classes,
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
