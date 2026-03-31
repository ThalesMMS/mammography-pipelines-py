#!/usr/bin/env python3
#
# eval_export.py
# mammography-pipelines
#
# Export evaluation artifacts from density runs and register them in MLflow.
#
"""Export evaluation artifacts and register them in MLflow/registry."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from mammography.tools import eval_export_registry
from mammography.tools import train_registry

LOGGER = logging.getLogger("eval_export")
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class EvalExportResult:
    run_dir: Path
    export_dir: Path
    manifest_path: Path
    exported_paths: list[Path]
    missing: list[str]
    summary: dict[str, Any]
    artifacts: train_registry.TrainingArtifacts


def _resolve_export_root(output_dir: Path) -> Path:
    if output_dir.is_absolute():
        return output_dir
    return REPO_ROOT / output_dir


def _resolve_export_dir(run_dir: Path, output_root: Path) -> Path:
    try:
        rel = run_dir.relative_to(REPO_ROOT)
    except ValueError:
        return output_root / run_dir.name

    if rel.parts and rel.parts[0] == "outputs":
        rel = Path(*rel.parts[1:])
    return output_root / rel


def _build_export_items(
    artifacts: train_registry.TrainingArtifacts,
    run_dir: Path,
) -> tuple[list[Path], list[str]]:
    required = [
        artifacts.summary_path,
        artifacts.train_history_path,
        artifacts.val_metrics_path,
        artifacts.checkpoint_path,
    ]
    optional: list[Path] = []
    missing: list[str] = []

    if artifacts.train_history_plot_path:
        optional.append(artifacts.train_history_plot_path)
    else:
        missing.append("train_history.png")

    if artifacts.confusion_matrix_path:
        optional.append(artifacts.confusion_matrix_path)
    else:
        missing.append("confusion_matrix.png")
    explicit_confusion_matrix = run_dir / "metrics" / "confusion_matrix.png"
    if explicit_confusion_matrix.exists() and explicit_confusion_matrix not in optional:
        optional.append(explicit_confusion_matrix)

    if artifacts.val_predictions_path:
        optional.append(artifacts.val_predictions_path)
    else:
        missing.append("val_predictions.csv")

    test_metrics_path = run_dir / "metrics" / "test_metrics.json"
    if test_metrics_path.exists():
        optional.append(test_metrics_path)
    test_metrics_fig_path = run_dir / "metrics" / "test_metrics.png"
    if test_metrics_fig_path.exists():
        optional.append(test_metrics_fig_path)
    test_predictions_path = run_dir / "test_predictions.csv"
    if test_predictions_path.exists():
        optional.append(test_predictions_path)

    if (run_dir / "embeddings_val.csv").exists():
        optional.append(run_dir / "embeddings_val.csv")
    else:
        missing.append("embeddings_val.csv")

    if (run_dir / "embeddings_val.npy").exists():
        optional.append(run_dir / "embeddings_val.npy")
    elif (run_dir / "embeddings_val.npz").exists():
        optional.append(run_dir / "embeddings_val.npz")
    else:
        missing.append("embeddings_val.npy")

    if (run_dir / "run.log").exists():
        optional.append(run_dir / "run.log")
    else:
        missing.append("run.log")

    return required + optional, missing


def _copy_export_item(src: Path, run_dir: Path, export_dir: Path) -> Path:
    try:
        rel = src.relative_to(run_dir)
    except ValueError:
        rel = Path(src.name)
    dest = export_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def export_eval_run(run_dir: Path, output_root: Path) -> EvalExportResult:
    results_dir = train_registry._find_results_dir(run_dir)
    artifacts = train_registry._collect_artifacts(results_dir)
    summary = train_registry._load_json(artifacts.summary_path)

    export_dir = _resolve_export_dir(results_dir, output_root)
    export_dir.mkdir(parents=True, exist_ok=True)

    export_items, missing = _build_export_items(artifacts, results_dir)
    exported_paths = [_copy_export_item(path, results_dir, export_dir) for path in export_items]

    manifest = {
        "run_dir": str(results_dir),
        "export_dir": str(export_dir),
        "exported_files": [
            path.relative_to(export_dir).as_posix() for path in exported_paths
        ],
        "missing_files": missing,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    manifest_path = export_dir / "eval_export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    exported_paths.append(manifest_path)

    if missing:
        LOGGER.warning("Artefatos ausentes em %s: %s", results_dir, ", ".join(missing))

    return EvalExportResult(
        run_dir=results_dir,
        export_dir=export_dir,
        manifest_path=manifest_path,
        exported_paths=exported_paths,
        missing=missing,
        summary=summary,
        artifacts=artifacts,
    )


def _exported_copy(
    run_dir: Path,
    export_dir: Path,
    original: Path | None,
) -> Path | None:
    if original is None:
        return None
    try:
        rel = original.relative_to(run_dir)
    except ValueError:
        rel = Path(original.name)
    return export_dir / rel


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta artefatos de avaliacao e registra no MLflow.",
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        type=Path,
        required=True,
        help="Diretorio results_* ou pasta com results_* para exportacao.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/exports"),
        help="Diretorio base para exportacoes (default: outputs/exports).",
    )
    parser.add_argument("--run-name", default="", help="Nome do run no MLflow")
    parser.add_argument("--tracking-uri", default="", help="Tracking URI para MLflow")
    parser.add_argument("--experiment", default="", help="Experimento MLflow")
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("results/registry.csv"),
        help="Arquivo CSV do registry local",
    )
    parser.add_argument(
        "--registry-md",
        type=Path,
        default=Path("results/registry.md"),
        help="Arquivo Markdown do registry local",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Nao registrar no MLflow",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Nao atualizar registry local",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
    )

    output_root = _resolve_export_root(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    runs = args.runs or []
    for run in runs:
        run_dir = (REPO_ROOT / run) if not run.is_absolute() else run
        LOGGER.info("Exportando run: %s", run_dir)
        result = export_eval_run(run_dir, output_root)
        LOGGER.info("Exportacao concluida: %s", result.export_dir)

        if args.no_registry:
            continue

        dataset = eval_export_registry.infer_dataset_name(result.summary, result.run_dir)
        if args.run_name:
            run_name = args.run_name
            if len(runs) > 1:
                run_name = f"{run_name}_{result.run_dir.name}"
        else:
            run_name = eval_export_registry.default_run_name(dataset)
            if len(runs) > 1:
                run_name = f"{run_name}_{result.run_dir.name}"

        try:
            eval_export_registry.register_eval_export_run(
                dataset=dataset,
                run_name=run_name,
                command=shlex.join(sys.argv),
                export_dir=result.export_dir,
                export_paths=result.exported_paths,
                summary_path=_exported_copy(
                    result.run_dir,
                    result.export_dir,
                    result.artifacts.summary_path,
                ),
                train_history_path=_exported_copy(
                    result.run_dir,
                    result.export_dir,
                    result.artifacts.train_history_path,
                ),
                val_metrics_path=_exported_copy(
                    result.run_dir,
                    result.export_dir,
                    result.artifacts.val_metrics_path,
                ),
                confusion_matrix_path=_exported_copy(
                    result.run_dir,
                    result.export_dir,
                    result.artifacts.confusion_matrix_path,
                ),
                checkpoint_path=_exported_copy(
                    result.run_dir,
                    result.export_dir,
                    result.artifacts.checkpoint_path,
                ),
                val_predictions_path=_exported_copy(
                    result.run_dir,
                    result.export_dir,
                    result.artifacts.val_predictions_path,
                ),
                registry_csv=args.registry_csv,
                registry_md=args.registry_md,
                tracking_uri=args.tracking_uri or None,
                experiment=args.experiment or None,
                log_mlflow=not args.no_mlflow,
            )
            LOGGER.info("Registry atualizado (run_name=%s).", run_name)
        except Exception as exc:
            LOGGER.warning("Falha ao registrar exportacao: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
