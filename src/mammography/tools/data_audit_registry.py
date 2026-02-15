#!/usr/bin/env python3
"""Register data-audit outputs in MLflow and local registry files."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class DataAuditCounts:
    total_dicom_files: int
    valid_dicom_files: int
    invalid_dicom_files: int
    class_histogram: dict[str, int]
    generated_at: str | None


REGISTRY_FIELDS = [
    "timestamp",
    "dataset",
    "workflow",
    "run_name",
    "command",
    "manifest_path",
    "audit_csv_path",
    "log_path",
    "total_dicom_files",
    "valid_dicom_files",
    "invalid_dicom_files",
    "class_distribution",
    "model",
    "layer",
    "batch_size",
    "img_size",
    "outdir",
    "features_path",
    "metadata_path",
    "joined_csv_path",
    "session_info_path",
    "reduction_outputs",
    "clustering_outputs",
    "baseline_feature_set",
    "baseline_model",
    "baseline_accuracy",
    "baseline_macro_f1",
    "baseline_auc",
    "baseline_report_path",
    "mlflow_run_id",
    "summary_path",
    "train_history_path",
    "val_metrics_path",
    "confusion_matrix_path",
    "checkpoint_path",
    "val_predictions_path",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_f1",
    "val_auc",
    "val_sensitivity",
    "val_specificity",
    "hyperparameters",
    "tune_n_trials",
    "tune_completed_trials",
    "tune_pruned_trials",
    "tune_best_trial",
    "tune_best_value",
    "baseline_run_name",
    "baseline_val_loss",
    "baseline_val_acc",
    "baseline_val_f1",
    "baseline_val_auc",
    "delta_val_loss",
    "delta_val_acc",
    "delta_val_f1",
    "delta_val_auc",
    "tuned_hyperparameters",
    "inference_output_path",
    "inference_total_images",
    "inference_images_per_sec",
    "inference_duration_sec",
    "explain_output_path",
    "explain_total_images",
    "explain_loaded_images",
    "explain_saved_images",
    "explain_failed_images",
    "explain_method",
    "visualization_output_paths",
]


def _ensure_registry_header(registry_path: Path) -> None:
    if not registry_path.exists() or registry_path.stat().st_size == 0:
        return
    with registry_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return
    if header == REGISTRY_FIELDS:
        return
    with registry_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_manifest_counts(manifest_path: Path) -> DataAuditCounts:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Manifesto inválido: esperado um objeto JSON.")
    histogram_raw = data.get("class_histogram", {})
    if not isinstance(histogram_raw, Mapping):
        raise ValueError("Manifesto inválido: class_histogram ausente.")
    histogram = {str(key): int(value) for key, value in histogram_raw.items()}
    total_files = int(data.get("total_dicom_files", 0))
    valid_files = int(data.get("total_readable_files", 0))
    invalid_files = max(total_files - valid_files, 0)
    generated_at = data.get("generated_at")
    return DataAuditCounts(
        total_dicom_files=total_files,
        valid_dicom_files=valid_files,
        invalid_dicom_files=invalid_files,
        class_histogram=histogram,
        generated_at=generated_at,
    )


def format_class_distribution(histogram: Mapping[str, int]) -> str:
    def sort_key(item: tuple[str, int]) -> tuple[int, int | str]:
        key = item[0]
        if key == "missing":
            return (2, 0)
        if key.isdigit():
            return (0, int(key))
        return (1, key)

    ordered = sorted(histogram.items(), key=sort_key)
    return ", ".join(f"{key}={value}" for key, value in ordered)


def build_registry_row(
    *,
    dataset: str,
    workflow: str,
    run_name: str,
    command: str,
    manifest_path: Path,
    audit_csv_path: Path,
    log_path: Path,
    counts: DataAuditCounts,
    class_distribution: str,
    mlflow_run_id: str | None,
    timestamp: str | None = None,
) -> dict[str, str | int]:
    return {
        "timestamp": timestamp or datetime.now(tz=timezone.utc).isoformat(),
        "dataset": dataset,
        "workflow": workflow,
        "run_name": run_name,
        "command": command,
        "manifest_path": str(manifest_path),
        "audit_csv_path": str(audit_csv_path),
        "log_path": str(log_path),
        "total_dicom_files": counts.total_dicom_files,
        "valid_dicom_files": counts.valid_dicom_files,
        "invalid_dicom_files": counts.invalid_dicom_files,
        "class_distribution": class_distribution,
        "mlflow_run_id": mlflow_run_id or "",
    }


def append_registry_csv(registry_path: Path, row: Mapping[str, str | int]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not registry_path.exists() or registry_path.stat().st_size == 0
    if not write_header:
        _ensure_registry_header(registry_path)
    with registry_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_markdown_entry(
    *,
    dataset: str,
    workflow: str,
    run_name: str,
    command: str,
    outputs: Iterable[Path],
    counts: DataAuditCounts,
    class_distribution: str,
    timestamp: str | None = None,
) -> list[str]:
    stamp = timestamp or datetime.now().date().isoformat()
    outputs_str = ", ".join(f"`{path}`" for path in outputs)
    return [
        f"## {stamp} - {run_name}",
        f"- Dataset: {dataset}",
        f"- Workflow: {workflow}",
        f"- Run name: {run_name}",
        f"- Command: `{command}`",
        f"- Outputs: {outputs_str}",
        f"- Total DICOMs: {counts.total_dicom_files}",
        f"- Valid DICOMs: {counts.valid_dicom_files}",
        f"- Invalid DICOMs: {counts.invalid_dicom_files}",
        f"- Class distribution: {class_distribution}",
        "",
    ]


def append_registry_markdown(registry_path: Path, entry_lines: Iterable[str]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if not registry_path.exists() or registry_path.stat().st_size == 0:
        registry_path.write_text("# Results Registry\n\n", encoding="utf-8")
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(entry_lines))


def _resolve_tracking_root(tracking_uri: str | None) -> Path:
    if tracking_uri:
        if tracking_uri.startswith("file:"):
            return Path(tracking_uri.replace("file:", "", 1))
        return Path(tracking_uri)
    return Path("mlruns")


def _write_meta_yaml(
    *,
    meta_path: Path,
    run_id: str,
    run_name: str,
    experiment_id: str,
    artifact_uri: str,
    start_time_ms: int,
    end_time_ms: int,
    status: str = "FINISHED",
) -> None:
    meta_lines = [
        f"artifact_uri: {artifact_uri}",
        f"end_time: {end_time_ms}",
        "entry_point_name: ''",
        f"experiment_id: '{experiment_id}'",
        "lifecycle_stage: active",
        f"name: '{run_name}'",
        "run_id: {run_id}".format(run_id=run_id),
        "run_uuid: {run_id}".format(run_id=run_id),
        "source_name: ''",
        "source_type: 4",
        "source_version: ''",
        f"start_time: {start_time_ms}",
        f"status: {status}",
        "tags: []",
        "user_id: ''",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")


def _write_param(params_dir: Path, key: str, value: str) -> None:
    params_dir.mkdir(parents=True, exist_ok=True)
    (params_dir / key).write_text(value, encoding="utf-8")


def _write_metric(metrics_dir: Path, key: str, value: float, timestamp_ms: int) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    line = f"{timestamp_ms} {value} 0\n"
    (metrics_dir / key).write_text(line, encoding="utf-8")


def _log_local_mlflow_run(
    *,
    run_name: str,
    dataset: str,
    workflow: str,
    command: str,
    manifest_path: Path,
    audit_csv_path: Path,
    log_path: Path,
    counts: DataAuditCounts,
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
    _write_param(params_dir, "dataset", dataset)
    _write_param(params_dir, "workflow", workflow)
    _write_param(params_dir, "command", command)
    _write_param(params_dir, "manifest_path", str(manifest_path))
    _write_param(params_dir, "audit_csv_path", str(audit_csv_path))
    _write_param(params_dir, "log_path", str(log_path))
    if experiment:
        _write_param(params_dir, "experiment", experiment)
    _write_metric(
        metrics_dir,
        "total_dicom_files",
        float(counts.total_dicom_files),
        timestamp_ms,
    )
    _write_metric(
        metrics_dir,
        "valid_dicom_files",
        float(counts.valid_dicom_files),
        timestamp_ms,
    )
    _write_metric(
        metrics_dir,
        "invalid_dicom_files",
        float(counts.invalid_dicom_files),
        timestamp_ms,
    )
    for key, value in counts.class_histogram.items():
        _write_metric(metrics_dir, f"class_count_{key}", float(value), timestamp_ms)
    for path in (manifest_path, audit_csv_path, log_path):
        shutil.copy2(path, artifacts_dir / path.name)
    return run_id


def log_mlflow_run(
    *,
    run_name: str,
    dataset: str,
    workflow: str,
    command: str,
    manifest_path: Path,
    audit_csv_path: Path,
    log_path: Path,
    counts: DataAuditCounts,
    tracking_uri: str | None = None,
    experiment: str | None = None,
) -> str:
    for path in (manifest_path, audit_csv_path, log_path):
        if not path.exists():
            raise FileNotFoundError(f"Artefato ausente: {path}")
    try:
        import mlflow  # type: ignore
    except Exception:
        return _log_local_mlflow_run(
            run_name=run_name,
            dataset=dataset,
            workflow=workflow,
            command=command,
            manifest_path=manifest_path,
            audit_csv_path=audit_csv_path,
            log_path=log_path,
            counts=counts,
            tracking_uri=tracking_uri,
            experiment=experiment,
        )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment:
        mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "dataset": dataset,
                "workflow": workflow,
                "command": command,
                "manifest_path": str(manifest_path),
                "audit_csv_path": str(audit_csv_path),
                "log_path": str(log_path),
            }
        )
        mlflow.log_metrics(
            {
                "total_dicom_files": counts.total_dicom_files,
                "valid_dicom_files": counts.valid_dicom_files,
                "invalid_dicom_files": counts.invalid_dicom_files,
            }
        )
        for key, value in counts.class_histogram.items():
            metric_key = f"class_count_{key}"
            mlflow.log_metric(metric_key, value)
        mlflow.log_artifact(str(manifest_path))
        mlflow.log_artifact(str(audit_csv_path))
        mlflow.log_artifact(str(log_path))
        return run.info.run_id


def register_data_audit(
    *,
    manifest_path: Path,
    audit_csv_path: Path,
    log_path: Path,
    dataset: str,
    workflow: str,
    run_name: str,
    command: str,
    registry_csv: Path,
    registry_md: Path,
    tracking_uri: str | None = None,
    experiment: str | None = None,
) -> str:
    counts = load_manifest_counts(manifest_path)
    class_distribution = format_class_distribution(counts.class_histogram)
    run_id = log_mlflow_run(
        run_name=run_name,
        dataset=dataset,
        workflow=workflow,
        command=command,
        manifest_path=manifest_path,
        audit_csv_path=audit_csv_path,
        log_path=log_path,
        counts=counts,
        tracking_uri=tracking_uri,
        experiment=experiment,
    )
    row = build_registry_row(
        dataset=dataset,
        workflow=workflow,
        run_name=run_name,
        command=command,
        manifest_path=manifest_path,
        audit_csv_path=audit_csv_path,
        log_path=log_path,
        counts=counts,
        class_distribution=class_distribution,
        mlflow_run_id=run_id,
    )
    append_registry_csv(registry_csv, row)
    entry_lines = build_markdown_entry(
        dataset=dataset,
        workflow=workflow,
        run_name=run_name,
        command=command,
        outputs=[manifest_path, audit_csv_path, log_path],
        counts=counts,
        class_distribution=class_distribution,
    )
    append_registry_markdown(registry_md, entry_lines)
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Registra a auditoria de dados em MLflow e no registry local."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--dataset", default="archive")
    parser.add_argument("--workflow", default="data-audit")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("results/registry.csv"),
    )
    parser.add_argument("--registry-md", type=Path, default=Path("results/registry.md"))
    parser.add_argument("--tracking-uri", default="")
    parser.add_argument("--experiment", default="")
    args = parser.parse_args(argv)

    register_data_audit(
        manifest_path=args.manifest,
        audit_csv_path=args.audit_csv,
        log_path=args.log,
        dataset=args.dataset,
        workflow=args.workflow,
        run_name=args.run_name,
        command=args.command,
        registry_csv=args.registry_csv,
        registry_md=args.registry_md,
        tracking_uri=args.tracking_uri or None,
        experiment=args.experiment or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
