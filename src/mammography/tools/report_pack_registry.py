#!/usr/bin/env python3
"""Register report-pack outputs in MLflow and local registry files."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from mammography.tools.data_audit_registry import (
    _is_plain_local_tracking_uri,
    _resolve_tracking_root,
    _write_meta_yaml,
    _write_metric,
    _write_param,
    append_registry_csv,
    append_registry_markdown,
)


@dataclass(frozen=True)
class ReportPackArtifacts:
    assets_dir: Path
    tex_path: Path | None
    output_paths: list[Path]


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "report_pack"


def _load_summary(run_dir: Path) -> Mapping[str, object] | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def infer_dataset_name(run_dirs: Sequence[Path]) -> str:
    for run_dir in run_dirs:
        summary = _load_summary(run_dir)
        if summary and summary.get("dataset"):
            return _sanitize_name(str(summary["dataset"]))
    if not run_dirs:
        return "report_pack"
    candidate = (
        run_dirs[0].parent.name
        if run_dirs[0].name.startswith("results")
        else run_dirs[0].name
    )
    return _sanitize_name(candidate or "report_pack")


def default_run_name(dataset: str) -> str:
    return f"{_sanitize_name(dataset)}_report_pack"


def collect_report_pack_outputs(
    assets_dir: Path,
    tex_path: Path | None = None,
    asset_names: Sequence[str] | None = None,
) -> ReportPackArtifacts:
    if not assets_dir.exists():
        raise FileNotFoundError(f"Diretorio de assets nao encontrado: {assets_dir}")
    output_paths: list[Path] = []
    if asset_names is not None:
        for name in asset_names:
            path = assets_dir / name
            if path.exists() and path.is_file():
                output_paths.append(path)
    else:
        output_paths.extend(
            path for path in assets_dir.rglob("*") if path.is_file()
        )
    if tex_path and tex_path.exists():
        output_paths.append(tex_path)
    unique_paths = sorted({path for path in output_paths}, key=lambda p: p.as_posix())
    return ReportPackArtifacts(
        assets_dir=assets_dir,
        tex_path=tex_path,
        output_paths=unique_paths,
    )


def _format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(str(path) for path in paths)


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
    assets_dir: Path,
    output_paths: Sequence[Path],
    mlflow_run_id: str | None,
    timestamp: str | None = None,
) -> dict[str, str | int]:
    return {
        "timestamp": timestamp or datetime.now(tz=timezone.utc).isoformat(),
        "dataset": dataset,
        "workflow": "report-pack",
        "run_name": run_name,
        "command": command,
        "outdir": str(assets_dir),
        "visualization_output_paths": _format_paths(output_paths),
        "mlflow_run_id": mlflow_run_id or "",
    }


def build_markdown_entry(
    *,
    dataset: str,
    run_name: str,
    command: str,
    run_paths: Sequence[Path],
    assets_dir: Path,
    tex_path: Path | None,
    output_paths: Sequence[Path],
    timestamp: str | None = None,
) -> list[str]:
    stamp = timestamp or datetime.now().date().isoformat()
    outputs_str = ", ".join(f"`{path}`" for path in output_paths)
    runs_str = ", ".join(f"`{path}`" for path in run_paths)
    lines = [
        f"## {stamp} - {run_name}",
        f"- Dataset: {dataset}",
        "- Workflow: report-pack",
        f"- Run name: {run_name}",
        f"- Command: `{command}`",
        f"- Runs: {runs_str}",
        f"- Assets dir: `{assets_dir}`",
    ]
    if tex_path:
        lines.append(f"- LaTeX: `{tex_path}`")
    lines.extend([f"- Outputs: {outputs_str}", ""])
    return lines


def register_report_pack_run(
    *,
    run_paths: Sequence[Path],
    assets_dir: Path,
    tex_path: Path | None,
    output_paths: Sequence[Path],
    run_name: str,
    command: str,
    registry_csv: Path,
    registry_md: Path,
    tracking_uri: str | None = None,
    experiment: str | None = None,
    log_mlflow: bool = True,
) -> str | None:
    dataset = infer_dataset_name(run_paths)
    params = {
        "dataset": dataset,
        "workflow": "report-pack",
        "command": command,
        "assets_dir": str(assets_dir),
    }
    if tex_path:
        params["tex_path"] = str(tex_path)
    if run_paths:
        params["run_paths"] = _format_paths(run_paths)
    if output_paths:
        params["output_paths"] = _format_paths(output_paths)

    metrics = {
        "report_pack_runs": float(len(run_paths)),
        "report_pack_assets": float(len(output_paths)),
    }
    run_id: str | None = None
    if log_mlflow:
        run_id = log_mlflow_run(
            run_name=run_name,
            params=params,
            metrics=metrics,
            artifacts=output_paths,
            tracking_uri=tracking_uri,
            experiment=experiment,
        )

    row = build_registry_row(
        dataset=dataset,
        run_name=run_name,
        command=command,
        assets_dir=assets_dir,
        output_paths=output_paths,
        mlflow_run_id=run_id,
    )
    append_registry_csv(registry_csv, row)
    entry_lines = build_markdown_entry(
        dataset=dataset,
        run_name=run_name,
        command=command,
        run_paths=run_paths,
        assets_dir=assets_dir,
        tex_path=tex_path,
        output_paths=output_paths,
    )
    append_registry_markdown(registry_md, entry_lines)
    return run_id


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Registra report-pack em MLflow e no registry local."
    )
    parser.add_argument("--run", dest="runs", action="append", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--tex", dest="tex_path", type=Path)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--command", required=True)
    parser.add_argument("--registry-csv", type=Path, default=Path("results/registry.csv"))
    parser.add_argument("--registry-md", type=Path, default=Path("results/registry.md"))
    parser.add_argument("--tracking-uri", default="")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    artifacts = collect_report_pack_outputs(args.assets_dir, args.tex_path)
    run_name = args.run_name or default_run_name(infer_dataset_name(args.runs))
    register_report_pack_run(
        run_paths=args.runs,
        assets_dir=args.assets_dir,
        tex_path=args.tex_path,
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
