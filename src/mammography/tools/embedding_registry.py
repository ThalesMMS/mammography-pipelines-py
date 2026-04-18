#!/usr/bin/env python3
"""Register embedding extraction outputs in MLflow and local registry files."""

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
from urllib.parse import urlparse

from mammography.tools.data_audit_registry import (
    _resolve_tracking_root,
    _write_meta_yaml,
    _write_metric,
    _write_param,
    append_registry_csv,
    append_registry_markdown,
)

ARCH_ALIASES = {
    "efficientnet_b0": "effnet",
    "resnet50": "resnet50",
}
DATASET_ALIASES = {
    "patches_completo": "patches",
}


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "embed"


def default_run_name(dataset: str, arch: str) -> str:
    dataset_alias = DATASET_ALIASES.get(dataset, dataset)
    alias = ARCH_ALIASES.get(arch, arch)
    return f"{_sanitize_name(dataset_alias)}_embed_{_sanitize_name(alias)}"


@dataclass(frozen=True)
class EmbeddingArtifacts:
    features_path: Path
    metadata_path: Path
    joined_csv_path: Path | None
    session_info_path: Path | None
    example_embedding_path: Path | None
    reduction_outputs: list[Path]
    clustering_outputs: list[Path]
    preview_dir: Path | None


def _load_session_info(session_info_path: Path | None) -> dict:
    if session_info_path is None or not session_info_path.exists():
        return {}
    data = json.loads(session_info_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _collect_artifacts(outdir: Path) -> EmbeddingArtifacts:
    features_path = outdir / "features.npy"
    metadata_path = outdir / "metadata.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Artefato ausente: {features_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Artefato ausente: {metadata_path}")
    joined_csv_path = outdir / "joined.csv"
    if not joined_csv_path.exists():
        joined_csv_path = None
    session_info_path = outdir / "session_info.json"
    if not session_info_path.exists():
        session_info_path = None
    example_embedding_path = outdir / "example_embedding.json"
    if not example_embedding_path.exists():
        example_embedding_path = None
    reduction_outputs = [
        outdir / "pca_label.png",
        outdir / "tsne_label.png",
        outdir / "umap_label.png",
    ]
    reduction_outputs = [path for path in reduction_outputs if path.exists()]
    clustering_outputs = [
        outdir / "clustering_metrics.json",
        outdir / "clustering_metrics.png",
        outdir / "pca_cluster.png",
        outdir / "tsne_cluster.png",
    ]
    clustering_outputs = [path for path in clustering_outputs if path.exists()]
    preview_dir = outdir / "preview"
    if not preview_dir.exists():
        preview_dir = None
    return EmbeddingArtifacts(
        features_path=features_path,
        metadata_path=metadata_path,
        joined_csv_path=joined_csv_path,
        session_info_path=session_info_path,
        example_embedding_path=example_embedding_path,
        reduction_outputs=reduction_outputs,
        clustering_outputs=clustering_outputs,
        preview_dir=preview_dir,
    )


def _format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(str(path) for path in paths)


def _infer_args_from_session(session_info: Mapping[str, object]) -> dict[str, object]:
    args = session_info.get("args") if isinstance(session_info, dict) else None
    return args if isinstance(args, dict) else {}


def _collect_metrics(session_info: Mapping[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    num_samples = session_info.get("num_samples") if isinstance(session_info, dict) else None
    if isinstance(num_samples, (int, float)):
        metrics["num_samples"] = float(num_samples)
    features_shape = session_info.get("features_shape") if isinstance(session_info, dict) else None
    if (
        isinstance(features_shape, list)
        and len(features_shape) >= 2
        and all(isinstance(v, (int, float)) for v in features_shape[:2])
    ):
        metrics["embedding_dim"] = float(features_shape[1])
    return metrics


def _log_local_mlflow_run(
    *,
    run_name: str,
    params: Mapping[str, str],
    metrics: Mapping[str, float],
    artifacts: Sequence[Path],
    preview_dir: Path | None,
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
    if preview_dir and preview_dir.exists():
        shutil.copytree(preview_dir, artifacts_dir / "preview", dirs_exist_ok=True)
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
    preview_dir: Path | None,
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
            preview_dir=preview_dir,
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
            preview_dir=preview_dir,
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
        if preview_dir and preview_dir.exists():
            mlflow.log_artifacts(str(preview_dir), artifact_path="preview")
        return run.info.run_id


def build_registry_row(
    *,
    dataset: str,
    workflow: str,
    run_name: str,
    command: str,
    model: str,
    layer: str,
    batch_size: int,
    img_size: int,
    outdir: Path,
    features_path: Path,
    metadata_path: Path,
    joined_csv_path: Path | None,
    session_info_path: Path | None,
    reduction_outputs: Sequence[Path],
    clustering_outputs: Sequence[Path],
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
        "layer": layer,
        "batch_size": batch_size,
        "img_size": img_size,
        "outdir": str(outdir),
        "features_path": str(features_path),
        "metadata_path": str(metadata_path),
        "joined_csv_path": str(joined_csv_path) if joined_csv_path else "",
        "session_info_path": str(session_info_path) if session_info_path else "",
        "reduction_outputs": _format_paths(reduction_outputs),
        "clustering_outputs": _format_paths(clustering_outputs),
        "mlflow_run_id": mlflow_run_id or "",
    }


def build_markdown_entry(
    *,
    dataset: str,
    workflow: str,
    run_name: str,
    command: str,
    model: str,
    layer: str,
    batch_size: int,
    img_size: int,
    outdir: Path,
    outputs: Iterable[Path],
    reduction_outputs: Sequence[Path],
    clustering_outputs: Sequence[Path],
    timestamp: str | None = None,
) -> list[str]:
    stamp = timestamp or datetime.now().date().isoformat()
    outputs_str = ", ".join(f"`{path}`" for path in outputs)
    reduction_str = ", ".join(f"`{path}`" for path in reduction_outputs)
    clustering_str = ", ".join(f"`{path}`" for path in clustering_outputs)
    lines = [
        f"## {stamp} - {run_name}",
        f"- Dataset: {dataset}",
        f"- Workflow: {workflow}",
        f"- Run name: {run_name}",
        f"- Command: `{command}`",
        f"- Model: {model}",
        f"- Layer: {layer}",
        f"- Batch size: {batch_size}",
        f"- Image size: {img_size}",
        f"- Outdir: `{outdir}`",
        f"- Outputs: {outputs_str}",
    ]
    if reduction_outputs:
        lines.append(f"- Reduction outputs: {reduction_str}")
    if clustering_outputs:
        lines.append(f"- Clustering outputs: {clustering_str}")
    lines.append("")
    return lines


def register_embedding_run(
    *,
    outdir: Path,
    dataset: str,
    model: str,
    layer: str,
    batch_size: int,
    img_size: int,
    run_name: str,
    command: str,
    registry_csv: Path,
    registry_md: Path,
    tracking_uri: str | None = None,
    experiment: str | None = None,
    log_mlflow: bool = True,
) -> str | None:
    artifacts = _collect_artifacts(outdir)
    session_info = _load_session_info(artifacts.session_info_path)
    metrics = _collect_metrics(session_info)
    params = {
        "dataset": dataset,
        "workflow": "embed",
        "command": command,
        "model": model,
        "layer": layer,
        "batch_size": str(batch_size),
        "img_size": str(img_size),
        "outdir": str(outdir),
    }
    if artifacts.joined_csv_path:
        params["joined_csv_path"] = str(artifacts.joined_csv_path)
    if artifacts.session_info_path:
        params["session_info_path"] = str(artifacts.session_info_path)
    if artifacts.reduction_outputs:
        params["reduction_outputs"] = _format_paths(artifacts.reduction_outputs)
    if artifacts.clustering_outputs:
        params["clustering_outputs"] = _format_paths(artifacts.clustering_outputs)

    artifact_list: list[Path] = [artifacts.features_path, artifacts.metadata_path]
    if artifacts.joined_csv_path:
        artifact_list.append(artifacts.joined_csv_path)
    if artifacts.session_info_path:
        artifact_list.append(artifacts.session_info_path)
    if artifacts.example_embedding_path:
        artifact_list.append(artifacts.example_embedding_path)
    artifact_list.extend(artifacts.reduction_outputs)
    artifact_list.extend(artifacts.clustering_outputs)

    run_id: str | None = None
    if log_mlflow:
        run_id = log_mlflow_run(
            run_name=run_name,
            params=params,
            metrics=metrics,
            artifacts=artifact_list,
            preview_dir=artifacts.preview_dir,
            tracking_uri=tracking_uri,
            experiment=experiment,
        )

    row = build_registry_row(
        dataset=dataset,
        workflow="embed",
        run_name=run_name,
        command=command,
        model=model,
        layer=layer,
        batch_size=batch_size,
        img_size=img_size,
        outdir=outdir,
        features_path=artifacts.features_path,
        metadata_path=artifacts.metadata_path,
        joined_csv_path=artifacts.joined_csv_path,
        session_info_path=artifacts.session_info_path,
        reduction_outputs=artifacts.reduction_outputs,
        clustering_outputs=artifacts.clustering_outputs,
        mlflow_run_id=run_id,
    )
    append_registry_csv(registry_csv, row)
    outputs: list[Path] = [artifacts.features_path, artifacts.metadata_path]
    if artifacts.joined_csv_path:
        outputs.append(artifacts.joined_csv_path)
    if artifacts.session_info_path:
        outputs.append(artifacts.session_info_path)
    outputs.extend(artifacts.reduction_outputs)
    outputs.extend(artifacts.clustering_outputs)
    entry_lines = build_markdown_entry(
        dataset=dataset,
        workflow="embed",
        run_name=run_name,
        command=command,
        model=model,
        layer=layer,
        batch_size=batch_size,
        img_size=img_size,
        outdir=outdir,
        outputs=outputs,
        reduction_outputs=artifacts.reduction_outputs,
        clustering_outputs=artifacts.clustering_outputs,
    )
    append_registry_markdown(registry_md, entry_lines)
    return run_id


def _resolve_from_session(
    arg_value: str | None,
    session_args: Mapping[str, object],
    key: str,
) -> str | None:
    if arg_value:
        return arg_value
    value = session_args.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(int(value))
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Registra a extração de embeddings em MLflow e no registry local."
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--layer", default="")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--registry-csv", type=Path, default=Path("results/registry.csv"))
    parser.add_argument("--registry-md", type=Path, default=Path("results/registry.md"))
    parser.add_argument("--tracking-uri", default="")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    artifacts = _collect_artifacts(args.outdir)
    session_info = _load_session_info(artifacts.session_info_path)
    session_args = _infer_args_from_session(session_info)

    dataset = _resolve_from_session(args.dataset or None, session_args, "dataset") or "unknown"
    model = _resolve_from_session(args.model or None, session_args, "arch") or "unknown"
    layer = _resolve_from_session(args.layer or None, session_args, "layer_name") or "unknown"
    batch_size = args.batch_size or int(session_args.get("batch_size") or 0)
    img_size = args.img_size or int(session_args.get("img_size") or 0)

    if batch_size <= 0 or img_size <= 0:
        raise SystemExit("batch-size e img-size devem ser informados ou presentes em session_info.json")

    register_embedding_run(
        outdir=args.outdir,
        dataset=dataset,
        model=model,
        layer=layer,
        batch_size=batch_size,
        img_size=img_size,
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
