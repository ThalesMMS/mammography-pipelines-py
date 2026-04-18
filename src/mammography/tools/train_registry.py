# ruff: noqa
#!/usr/bin/env python3
"""Register training outputs in MLflow and local registry files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from mammography.tools.data_audit_registry import (
    append_registry_csv,
    append_registry_markdown,
)
from mammography.tools.train_registry_artifacts import (
    TrainingArtifacts,
    _collect_artifacts,
    _find_results_dir,
    _iter_summary_checkpoint_candidates,
    _load_json,
    _normalize_view,
    _pick_best_model,
    _resolve_checkpoint_candidate,
    _resolve_confusion_matrix,
    _resolve_metrics_fallback,
    _resolve_metrics_json,
)
from mammography.tools.train_registry_metrics import (
    _collect_baseline_metrics,
    _collect_metrics,
    _compute_metric_delta,
    _compute_sensitivity_specificity,
    _load_train_history,
    _load_val_metrics,
    _parse_optional_float,
)
from mammography.tools.train_registry_mlflow import (
    _collect_params,
    _log_local_mlflow_run,
    log_mlflow_run,
)
from mammography.tools.train_registry_rows import (
    _extract_tuned_params,
    _format_delta,
    _format_hparam_value,
    _format_metric,
    _format_tuned_params,
    _summarize_hyperparameters,
    build_markdown_entry,
    build_registry_row,
)


def _collect_ensemble_metrics(*args: Any, **kwargs: Any) -> Any:
    from mammography.tools.train_registry_ensemble import (
        _collect_ensemble_metrics as _impl,
    )

    return _impl(*args, **kwargs)


def _resolve_ensemble_confusion_matrix(*args: Any, **kwargs: Any) -> Any:
    from mammography.tools.train_registry_ensemble import (
        _resolve_ensemble_confusion_matrix as _impl,
    )

    return _impl(*args, **kwargs)


def build_ensemble_markdown_entry(*args: Any, **kwargs: Any) -> Any:
    from mammography.tools.train_registry_ensemble import (
        build_ensemble_markdown_entry as _impl,
    )

    return _impl(*args, **kwargs)


def build_ensemble_registry_row(*args: Any, **kwargs: Any) -> Any:
    from mammography.tools.train_registry_ensemble import (
        build_ensemble_registry_row as _impl,
    )

    return _impl(*args, **kwargs)


def register_ensemble_run(*args: Any, **kwargs: Any) -> Any:
    from mammography.tools.train_registry_ensemble import register_ensemble_run as _impl

    return _impl(*args, **kwargs)


def _build_artifact_list(
    artifacts: TrainingArtifacts, *, include_plot: bool
) -> list[Path]:
    paths = [
        artifacts.summary_path,
        artifacts.train_history_path,
        artifacts.val_metrics_path,
        artifacts.checkpoint_path,
    ]
    if artifacts.confusion_matrix_path:
        paths.append(artifacts.confusion_matrix_path)
    if artifacts.val_predictions_path:
        paths.append(artifacts.val_predictions_path)
    if include_plot and artifacts.train_history_plot_path:
        paths.append(artifacts.train_history_plot_path)
    return paths


def register_training_run(
    *,
    outdir: Path,
    dataset: str,
    workflow: str,
    run_name: str,
    command: str,
    registry_csv: Path,
    registry_md: Path,
    tracking_uri: str | None = None,
    experiment: str | None = None,
    baseline_outdir: Path | None = None,
    baseline_run_name: str | None = None,
    tuned_keys: Sequence[str] | None = None,
) -> str:
    """
    Register a training run: log artifacts and metrics to MLflow (or a local tracking store) and append the run to the CSV and Markdown registries.

    Parameters:
        outdir (Path): Output directory containing the run results (summary, train history, metrics, checkpoint).
        registry_csv (Path): Path to the CSV registry file to append a row for this run.
        registry_md (Path): Path to the Markdown registry file to append a human-readable entry.
        tracking_uri (str | None): MLflow tracking URI; if None or unavailable, a filesystem-only local tracking store is used.
        experiment (str | None): MLflow experiment name to use when logging; if None, the default experiment is used.
        baseline_outdir (Path | None): Optional output directory of a baseline run whose metrics will be used to compute deltas.
        baseline_run_name (str | None): Optional name of the baseline run for inclusion in registry entries.
        tuned_keys (Sequence[str] | None): Optional sequence of keys to extract from the run summary as tuned hyperparameters; ignored if None.

    Returns:
        str: The MLflow run ID (or the generated local run ID when using local tracking).
    """
    artifacts = _collect_artifacts(outdir)
    summary = _load_json(artifacts.summary_path)
    history_metrics = _load_train_history(artifacts.train_history_path)
    val_metrics = _load_val_metrics(artifacts.val_metrics_path)
    metrics = _collect_metrics(history_metrics, val_metrics)
    tuned_params = _extract_tuned_params(summary, tuned_keys)
    tuned_params_json = (
        json.dumps(tuned_params, ensure_ascii=True, sort_keys=True, default=str)
        if tuned_params
        else ""
    )

    baseline_metrics: dict[str, float | None] = {}
    baseline_deltas: dict[str, float | None] = {}
    if baseline_outdir:
        baseline_metrics = _collect_baseline_metrics(baseline_outdir)
        for key in ("val_loss", "val_acc", "val_f1", "val_auc"):
            baseline_deltas[key] = _compute_metric_delta(
                metrics.get(key), baseline_metrics.get(key)
            )

    params = _collect_params(summary)

    artifact_list = _build_artifact_list(artifacts, include_plot=True)

    metric_payload: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None or not math.isfinite(value):
            continue
        metric_payload[key] = float(value)

    run_id = log_mlflow_run(
        run_name=run_name,
        params=params,
        metrics=metric_payload,
        artifacts=artifact_list,
        tracking_uri=tracking_uri,
        experiment=experiment,
    )

    dataset_value = dataset or str(summary.get("dataset") or "unknown")
    model_value = str(summary.get("arch") or "unknown")
    batch_size = int(summary.get("batch_size") or 0)
    img_size = int(summary.get("img_size") or 0)
    hyperparameters = json.dumps(summary, ensure_ascii=True, default=str)

    row = build_registry_row(
        dataset=dataset_value,
        workflow=workflow,
        run_name=run_name,
        command=command,
        model=model_value,
        batch_size=batch_size,
        img_size=img_size,
        results_dir=artifacts.results_dir,
        summary_path=artifacts.summary_path,
        train_history_path=artifacts.train_history_path,
        val_metrics_path=artifacts.val_metrics_path,
        confusion_matrix_path=artifacts.confusion_matrix_path,
        checkpoint_path=artifacts.checkpoint_path,
        val_predictions_path=artifacts.val_predictions_path,
        hyperparameters=hyperparameters,
        metrics=metrics,
        baseline_run_name=baseline_run_name,
        baseline_metrics=baseline_metrics,
        baseline_deltas=baseline_deltas,
        tuned_hyperparameters=tuned_params_json,
        mlflow_run_id=run_id,
    )
    append_registry_csv(registry_csv, row)

    outputs = _build_artifact_list(artifacts, include_plot=False)

    entry_lines = build_markdown_entry(
        dataset=dataset_value,
        workflow=workflow,
        run_name=run_name,
        command=command,
        results_dir=artifacts.results_dir,
        outputs=outputs,
        metrics=metrics,
        hyperparameters_summary=_summarize_hyperparameters(summary),
        confusion_matrix_path=artifacts.confusion_matrix_path,
        baseline_run_name=baseline_run_name,
        baseline_metrics=baseline_metrics,
        baseline_deltas=baseline_deltas,
        tuned_params=tuned_params,
    )
    append_registry_markdown(registry_md, entry_lines)
    return run_id


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entrypoint that parses command-line arguments and registers either a single training run or an ensemble run into the local registry and MLflow.

    Parameters:
        argv (Sequence[str] | None): Optional list of command-line arguments to parse; if None, uses sys.argv.

    Returns:
        int: Exit status code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Registra resultados de treinamento em MLflow e no registry local."
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--workflow", default="train-density")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument(
        "--ensemble-metrics",
        type=Path,
        default=None,
        help="Path para ensemble_metrics.json (ativa registro de ensemble)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path para summary.json (usado no registro de ensemble)",
    )
    parser.add_argument(
        "--registry-csv", type=Path, default=Path("results/registry.csv")
    )
    parser.add_argument("--registry-md", type=Path, default=Path("results/registry.md"))
    parser.add_argument("--tracking-uri", default="")
    parser.add_argument("--experiment", default="")
    parser.add_argument(
        "--baseline-outdir",
        type=Path,
        default=None,
        help="Diretorio de resultados base para comparar desempenho (opcional).",
    )
    parser.add_argument(
        "--baseline-run-name",
        default="",
        help="Nome do run baseline para referencia no registry (opcional).",
    )
    parser.add_argument(
        "--tuned-keys",
        default="",
        help=(
            "Lista separada por virgulas de hiperparametros para registrar como tuned."
        ),
    )
    args = parser.parse_args(argv)

    if args.ensemble_metrics:
        register_ensemble_run(
            outdir=args.outdir,
            metrics_path=args.ensemble_metrics,
            summary_path=args.summary,
            dataset=args.dataset,
            workflow=args.workflow,
            run_name=args.run_name,
            command=args.command,
            registry_csv=args.registry_csv,
            registry_md=args.registry_md,
            tracking_uri=args.tracking_uri or None,
            experiment=args.experiment or None,
        )
    else:
        tuned_keys = [
            item.strip() for item in args.tuned_keys.split(",") if item.strip()
        ]
        register_training_run(
            outdir=args.outdir,
            dataset=args.dataset,
            workflow=args.workflow,
            run_name=args.run_name,
            command=args.command,
            registry_csv=args.registry_csv,
            registry_md=args.registry_md,
            tracking_uri=args.tracking_uri or None,
            experiment=args.experiment or None,
            baseline_outdir=args.baseline_outdir,
            baseline_run_name=args.baseline_run_name or None,
            tuned_keys=tuned_keys or None,
        )
    return 0
