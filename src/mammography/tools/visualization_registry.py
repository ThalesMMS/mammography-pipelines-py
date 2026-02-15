#!/usr/bin/env python3
"""Register visualization outputs in MLflow and local registry files."""

from __future__ import annotations

import argparse
import re
import shutil
import uuid
from dataclasses import dataclass
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


@dataclass(frozen=True)
class VisualizationArtifacts:
    output_dir: Path
    output_paths: list[Path]


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "visualize"


def infer_dataset_name(input_path: Path) -> str:
    if input_path.is_file():
        name = input_path.parent.name
    else:
        name = input_path.name
    if not name:
        name = input_path.resolve().name
    return _sanitize_name(name or "visualize")


def default_run_name(output_dir: Path) -> str:
    return _sanitize_name(output_dir.name or "visualizations")


def collect_visualization_outputs(output_dir: Path) -> VisualizationArtifacts:
    if not output_dir.exists():
        raise FileNotFoundError(f"Diretorio de saída não encontrado: {output_dir}")
    output_paths = sorted(
        (path for path in output_dir.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix(),
    )
    return VisualizationArtifacts(output_dir=output_dir, output_paths=output_paths)


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
    input_path: Path,
    labels_path: Path | None,
    output_dir: Path,
    output_paths: Sequence[Path],
    mlflow_run_id: str | None,
    timestamp: str | None = None,
) -> dict[str, str | int]:
    return {
        "timestamp": timestamp or datetime.now(tz=timezone.utc).isoformat(),
        "dataset": dataset,
        "workflow": "visualize",
        "run_name": run_name,
        "command": command,
        "outdir": str(output_dir),
        "features_path": str(input_path),
        "metadata_path": str(labels_path) if labels_path else "",
        "visualization_output_paths": _format_paths(output_paths),
        "mlflow_run_id": mlflow_run_id or "",
    }


def build_markdown_entry(
    *,
    dataset: str,
    run_name: str,
    command: str,
    input_path: Path,
    labels_path: Path | None,
    output_dir: Path,
    output_paths: Sequence[Path],
    timestamp: str | None = None,
) -> list[str]:
    stamp = timestamp or datetime.now().date().isoformat()
    outputs_str = ", ".join(f"`{path}`" for path in output_paths)
    lines = [
        f"## {stamp} - {run_name}",
        f"- Dataset: {dataset}",
        "- Workflow: visualize",
        f"- Run name: {run_name}",
        f"- Command: `{command}`",
        f"- Input: `{input_path}`",
    ]
    if labels_path:
        lines.append(f"- Labels: `{labels_path}`")
    lines.extend(
        [
            f"- Outdir: `{output_dir}`",
            f"- Outputs: {outputs_str}",
            "",
        ]
    )
    return lines


def register_visualization_run(
    *,
    input_path: Path,
    labels_path: Path | None,
    output_dir: Path,
    output_paths: Sequence[Path],
    run_name: str,
    command: str,
    registry_csv: Path,
    registry_md: Path,
    tracking_uri: str | None = None,
    experiment: str | None = None,
    log_mlflow: bool = True,
) -> str | None:
    dataset = infer_dataset_name(input_path)
    params = {
        "dataset": dataset,
        "workflow": "visualize",
        "command": command,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
    }
    if labels_path:
        params["labels_path"] = str(labels_path)
    if output_paths:
        params["output_paths"] = _format_paths(output_paths)

    metrics = {"visualization_count": float(len(output_paths))}
    run_id: str | None = None
    if log_mlflow:
        run_id = log_mlflow_run(
            run_name=run_name,
            params=params,
            metrics=metrics,
            artifact_dirs=[output_dir],
            tracking_uri=tracking_uri,
            experiment=experiment,
        )

    row = build_registry_row(
        dataset=dataset,
        run_name=run_name,
        command=command,
        input_path=input_path,
        labels_path=labels_path,
        output_dir=output_dir,
        output_paths=output_paths,
        mlflow_run_id=run_id,
    )
    append_registry_csv(registry_csv, row)
    entry_lines = build_markdown_entry(
        dataset=dataset,
        run_name=run_name,
        command=command,
        input_path=input_path,
        labels_path=labels_path,
        output_dir=output_dir,
        output_paths=output_paths,
    )
    append_registry_markdown(registry_md, entry_lines)
    return run_id


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Registra visualizações em MLflow e no registry local."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--command", required=True)
    parser.add_argument("--registry-csv", type=Path, default=Path("results/registry.csv"))
    parser.add_argument("--registry-md", type=Path, default=Path("results/registry.md"))
    parser.add_argument("--tracking-uri", default="")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    artifacts = collect_visualization_outputs(args.output_dir)
    run_name = args.run_name or default_run_name(args.output_dir)
    register_visualization_run(
        input_path=args.input,
        labels_path=args.labels,
        output_dir=args.output_dir,
        output_paths=artifacts.output_paths,
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
