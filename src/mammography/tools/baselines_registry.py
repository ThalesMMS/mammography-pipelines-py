#!/usr/bin/env python3
"""Register embeddings baseline outputs in MLflow and local registry files."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlparse

from mammography.tools.data_audit_registry import (
    _resolve_tracking_root,
    _write_meta_yaml,
    _write_metric,
    _write_param,
    append_registry_csv,
    append_registry_markdown,
)


@dataclass(frozen=True)
class BaselineEntry:
    feature_set: str
    model: str
    accuracy: float
    macro_f1: float
    auc: float


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "baselines"


def _extract_dataset_slug(name: str) -> str | None:
    lowered = name.lower()
    for marker in ("_embeddings", "-embeddings", "_baselines", "-baselines"):
        idx = lowered.find(marker)
        if idx <= 0:
            continue
        return name[:idx]
    return None


def infer_dataset_name(embeddings_dir: Path, outdir: Path | None = None) -> str:
    for candidate in (embeddings_dir, outdir):
        if candidate is None:
            continue
        slug = _extract_dataset_slug(candidate.name)
        if slug:
            return _sanitize_name(slug)
    fallback = embeddings_dir.name or (outdir.name if outdir else "")
    return _sanitize_name(fallback or "baselines")


def default_run_name(embeddings_dir: Path, outdir: Path | None = None) -> str:
    base_name = embeddings_dir.name
    if "_embeddings_" in base_name:
        candidate = base_name.replace("_embeddings_", "_baselines_")
    elif "_embeddings" in base_name:
        candidate = base_name.replace("_embeddings", "_baselines")
    elif outdir and outdir.name:
        candidate = outdir.name
    else:
        candidate = f"{infer_dataset_name(embeddings_dir, outdir)}_baselines"
    return _sanitize_name(candidate)


def _sanitize_metric_key(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    return cleaned.strip("_").lower()


def _parse_metric(metrics: Mapping[str, object], key: str) -> float:
    for candidate in (f"{key}_mean", key):
        if candidate not in metrics:
            continue
        value = metrics[candidate]
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    raise ValueError(f"Metrica ausente no report: {key}")


def _load_baselines_report(report_path: Path) -> dict[str, object]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Baselines report invalido: esperado objeto JSON.")
    return data


def _collect_entries(report: Mapping[str, object]) -> list[BaselineEntry]:
    feature_sets = report.get("feature_sets")
    if not isinstance(feature_sets, Mapping):
        raise ValueError("Baselines report invalido: feature_sets ausente.")
    entries: list[BaselineEntry] = []
    for feature_set, models in feature_sets.items():
        if not isinstance(models, Mapping):
            continue
        for model_name, metrics in models.items():
            if not isinstance(metrics, Mapping):
                continue
            entries.append(
                BaselineEntry(
                    feature_set=str(feature_set),
                    model=str(model_name),
                    accuracy=_parse_metric(metrics, "accuracy"),
                    macro_f1=_parse_metric(metrics, "macro_f1"),
                    auc=_parse_metric(metrics, "auc"),
                )
            )
    if not entries:
        raise ValueError("Baselines report invalido: nenhuma entrada encontrada.")
    return entries


def _format_metric(value: float) -> str:
    if math.isnan(value):
        return ""
    return str(value)


def _format_metric_display(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


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


def _is_plain_local_tracking_uri(tracking_uri: str | None) -> bool:
    if not tracking_uri:
        return False
    parsed = urlparse(tracking_uri)
    if parsed.scheme == "":
        return True
    return len(parsed.scheme) == 1 and tracking_uri[1:3] in {":\\", ":/"}


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
    workflow: str,
    run_name: str,
    command: str,
    outdir: Path,
    report_path: Path,
    entry: BaselineEntry,
    mlflow_run_id: str | None,
    timestamp: str | None = None,
) -> dict[str, str | int]:
    return {
        "timestamp": timestamp or datetime.now(tz=timezone.utc).isoformat(),
        "dataset": dataset,
        "workflow": workflow,
        "run_name": run_name,
        "command": command,
        "outdir": str(outdir),
        "baseline_feature_set": entry.feature_set,
        "baseline_model": entry.model,
        "baseline_accuracy": _format_metric(entry.accuracy),
        "baseline_macro_f1": _format_metric(entry.macro_f1),
        "baseline_auc": _format_metric(entry.auc),
        "baseline_report_path": str(report_path),
        "mlflow_run_id": mlflow_run_id or "",
    }


def build_markdown_entry(
    *,
    dataset: str,
    workflow: str,
    run_name: str,
    command: str,
    outdir: Path,
    report_path: Path,
    entry: BaselineEntry,
    timestamp: str | None = None,
) -> list[str]:
    stamp = timestamp or datetime.now().date().isoformat()
    return [
        f"## {stamp} - {run_name} ({entry.feature_set}/{entry.model})",
        f"- Dataset: {dataset}",
        f"- Workflow: {workflow}",
        f"- Run name: {run_name}",
        f"- Command: `{command}`",
        f"- Feature set: {entry.feature_set}",
        f"- Model: {entry.model}",
        f"- Accuracy: {_format_metric_display(entry.accuracy)}",
        f"- Macro-F1: {_format_metric_display(entry.macro_f1)}",
        f"- AUC: {_format_metric_display(entry.auc)}",
        f"- Outdir: `{outdir}`",
        f"- Report: `{report_path}`",
        "",
    ]


def register_baselines_run(
    *,
    report_path: Path,
    dataset: str,
    run_name: str,
    command: str,
    registry_csv: Path,
    registry_md: Path,
    tracking_uri: str | None = None,
    experiment: str | None = None,
    log_mlflow: bool = True,
) -> str | None:
    report = _load_baselines_report(report_path)
    entries = _collect_entries(report)
    outdir_raw = report.get("outdir")
    outdir = report_path.parent
    if isinstance(outdir_raw, str) and outdir_raw:
        outdir = Path(outdir_raw)

    metrics: dict[str, float] = {}
    for entry in entries:
        prefix = "__".join(
            [
                _sanitize_metric_key(entry.feature_set),
                _sanitize_metric_key(entry.model),
            ]
        )
        if not math.isnan(entry.accuracy):
            metrics[f"{prefix}__accuracy"] = entry.accuracy
        if not math.isnan(entry.macro_f1):
            metrics[f"{prefix}__macro_f1"] = entry.macro_f1
        if not math.isnan(entry.auc):
            metrics[f"{prefix}__auc"] = entry.auc

    params = {
        "dataset": dataset,
        "workflow": "embeddings-baselines",
        "command": command,
        "outdir": str(outdir),
        "report_path": str(report_path),
    }
    run_id: str | None = None
    if log_mlflow:
        run_id = log_mlflow_run(
            run_name=run_name,
            params=params,
            metrics=metrics,
            artifacts=[report_path],
            tracking_uri=tracking_uri,
            experiment=experiment,
        )

    for entry in entries:
        row = build_registry_row(
            dataset=dataset,
            workflow="embeddings-baselines",
            run_name=run_name,
            command=command,
            outdir=outdir,
            report_path=report_path,
            entry=entry,
            mlflow_run_id=run_id,
        )
        append_registry_csv(registry_csv, row)
        entry_lines = build_markdown_entry(
            dataset=dataset,
            workflow="embeddings-baselines",
            run_name=run_name,
            command=command,
            outdir=outdir,
            report_path=report_path,
            entry=entry,
        )
        append_registry_markdown(registry_md, entry_lines)
    return run_id


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Registra baselines de embeddings em MLflow e no registry local."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--registry-csv", type=Path, default=Path("results/registry.csv"))
    parser.add_argument("--registry-md", type=Path, default=Path("results/registry.md"))
    parser.add_argument("--tracking-uri", default="")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--no-mlflow", action="store_true", help="Nao registrar no MLflow")
    args = parser.parse_args(argv)

    register_baselines_run(
        report_path=args.report,
        dataset=args.dataset,
        run_name=args.run_name,
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
