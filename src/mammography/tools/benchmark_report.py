#!/usr/bin/env python3
"""Utilities to validate and consolidate the official rerun benchmark."""

from __future__ import annotations

import csv
import json
import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # pragma: no cover - torch is optional for report generation
    import torch
except Exception:  # pragma: no cover - fallback when torch is unavailable
    torch = None  # type: ignore[assignment]

from mammography.utils.class_modes import classes_mode_aliases, normalize_classes_mode

LOGGER = logging.getLogger("benchmark_report")
REPO_ROOT = Path(__file__).resolve().parents[3]

DATASET_ORDER = ("archive", "mamografias", "patches_completo")
TASK_ORDER = ("multiclass", "binary")
ARCH_ORDER = ("efficientnet_b0", "resnet50", "vit_b_16")

EXPECTED_SPLITS = {
    "archive": "patient",
    "mamografias": "random",
    "patches_completo": "random",
}

COMMON_CONFIG = {
    "seed": 42,
    "deterministic": True,
    "allow_tf32": True,
    "amp": True,
    "pretrained": True,
    "train_backbone": True,
    "unfreeze_last_block": True,
    "augment": True,
    "class_weights": "auto",
    "sampler_weighted": True,
    "test_frac": 0.1,
    "tracker": "local",
    "view_specific_training": False,
}

ARCH_CONFIG = {
    "efficientnet_b0": {
        "img_size": 512,
        "batch_size": 16,
        "epochs": 30,
        "lr": 1e-4,
        "backbone_lr": 1e-5,
        "warmup_epochs": 2,
        "early_stop_patience": 5,
    },
    "resnet50": {
        "img_size": 512,
        "batch_size": 16,
        "epochs": 30,
        "lr": 1e-4,
        "backbone_lr": 1e-5,
        "warmup_epochs": 2,
        "early_stop_patience": 5,
    },
    "vit_b_16": {
        "img_size": 224,
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-3,
        "backbone_lr": 1e-4,
        "warmup_epochs": 3,
        "early_stop_patience": 10,
    },
}

MASTER_COLUMNS = [
    "dataset",
    "task",
    "split_mode",
    "arch",
    "seed",
    "img_size",
    "batch_size",
    "epochs",
    "accuracy",
    "kappa",
    "macro_f1",
    "auc",
    "best_epoch",
    "run_path",
    "status",
]

ARTICLE_COLUMNS = [
    "dataset",
    "task",
    "split",
    "modelo",
    "accuracy",
    "kappa",
    "macro-F1",
    "AUC",
]


class BenchmarkValidationError(RuntimeError):
    """Raised when the official benchmark namespace is incomplete or invalid."""


@dataclass(frozen=True)
class ExpectedRun:
    """Expected benchmark combination for the official rerun battery."""

    dataset: str
    task: str
    arch: str
    split_mode: str
    seed: int = 42

    @property
    def seed_dir_name(self) -> str:
        return f"seed{self.seed}"

    @property
    def run_name(self) -> str:
        return f"{self.dataset}_{self.task}_{self.arch}_seed{self.seed}"

    @property
    def relative_seed_dir(self) -> Path:
        return Path(self.dataset) / self.task / self.arch / self.seed_dir_name

    @property
    def relative_seed_dirs(self) -> tuple[Path, ...]:
        return tuple(
            Path(self.dataset) / task / self.arch / self.seed_dir_name
            for task in classes_mode_aliases(self.task)
        )

    @property
    def run_name_aliases(self) -> tuple[str, ...]:
        return tuple(
            f"{self.dataset}_{task}_{self.arch}_seed{self.seed}"
            for task in classes_mode_aliases(self.task)
        )


@dataclass(frozen=True)
class CollectedRun:
    """Validated run plus resolved artifacts and normalized metrics."""

    expected: ExpectedRun
    results_dir: Path
    summary: dict[str, Any]
    metrics: dict[str, Any]
    accuracy: float
    kappa: float
    macro_f1: float
    auc: float
    best_epoch: int
    export_dir: Path | None
    export_manifest_path: Path | None

    def master_row(self) -> dict[str, Any]:
        return {
            "dataset": self.expected.dataset,
            "task": self.expected.task,
            "split_mode": self.expected.split_mode,
            "arch": self.expected.arch,
            "seed": self.expected.seed,
            "img_size": int(self.summary["img_size"]),
            "batch_size": int(self.summary["batch_size"]),
            "epochs": int(self.summary["epochs"]),
            "accuracy": self.accuracy,
            "kappa": self.kappa,
            "macro_f1": self.macro_f1,
            "auc": self.auc,
            "best_epoch": self.best_epoch,
            "run_path": str(self.results_dir),
            "status": "accepted",
        }

    def article_row(self) -> dict[str, Any]:
        return {
            "dataset": self.expected.dataset,
            "task": self.expected.task,
            "split": self.expected.split_mode,
            "modelo": self.expected.arch,
            "accuracy": self.accuracy,
            "kappa": self.kappa,
            "macro-F1": self.macro_f1,
            "AUC": self.auc,
        }


def expected_runs() -> list[ExpectedRun]:
    """Return the official 18-run matrix defined for the rerun battery."""
    runs: list[ExpectedRun] = []
    for dataset in DATASET_ORDER:
        split_mode = EXPECTED_SPLITS[dataset]
        for task in TASK_ORDER:
            for arch in ARCH_ORDER:
                runs.append(
                    ExpectedRun(
                        dataset=dataset,
                        task=task,
                        arch=arch,
                        split_mode=split_mode,
                    )
                )
    return runs


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _normalize_path_key(value: str | Path) -> str:
    text = str(value).replace("\\", "/").rstrip("/")
    return text.lower()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BenchmarkValidationError(f"JSON invalido em {path}: esperado objeto.")
    return payload


def _normalize_output_prefix(path: Path) -> Path:
    resolved = _resolve_path(path)
    if resolved.suffix:
        return resolved.with_suffix("")
    return resolved


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


def _markdown_escape(value: Any) -> str:
    return _format_cell(value).replace("|", "\\|")


def _tex_escape(value: Any) -> str:
    text = _format_cell(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
        "{": "\\{",
        "}": "\\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise BenchmarkValidationError(f"{description} ausente: {path}")
    return path


def _resolve_results_dir(seed_dir: Path) -> Path:
    if not seed_dir.exists():
        raise BenchmarkValidationError(f"Diretorio oficial ausente: {seed_dir}")

    if (seed_dir / "summary.json").exists():
        return seed_dir

    candidates = [
        path
        for path in seed_dir.iterdir()
        if path.is_dir()
        and path.name.startswith("results")
        and (path / "summary.json").exists()
    ]
    if len(candidates) != 1:
        raise BenchmarkValidationError(
            f"Esperado exatamente um results_* em {seed_dir}, encontrado {len(candidates)}."
        )
    return candidates[0]


def _resolve_seed_dir(namespace: Path, expected: ExpectedRun) -> Path:
    for relative_path in expected.relative_seed_dirs:
        candidate = namespace / relative_path
        if candidate.exists():
            return candidate
    return namespace / expected.relative_seed_dir


def _is_completed_results_dir(results_dir: Path) -> bool:
    summary_path = results_dir / "summary.json"
    train_history_path = results_dir / "train_history.csv"
    checkpoint_path = results_dir / "checkpoint.pt"
    metrics_path = results_dir / "metrics" / "val_metrics.json"
    if not summary_path.exists() or not train_history_path.exists() or not checkpoint_path.exists() or not metrics_path.exists():
        return False
    try:
        summary = _load_json(summary_path)
    except Exception:
        return False
    finished_at = str(summary.get("finished_at") or "").strip()
    if not finished_at:
        return False

    try:
        test_frac = float(summary.get("test_frac") or 0.0)
    except (TypeError, ValueError):
        test_frac = 0.0
    if test_frac > 0:
        metrics_dir = results_dir / "metrics"
        if not any(metrics_dir.glob("test_metrics*.json")):
            return False

    return True


def _resolve_run_results_dir(namespace: Path, expected: ExpectedRun) -> Path:
    seed_dir = _resolve_seed_dir(namespace, expected)
    try:
        expected_results_dir = _resolve_results_dir(seed_dir)
    except BenchmarkValidationError:
        expected_results_dir = seed_dir / "results"

    if _is_completed_results_dir(expected_results_dir):
        return expected_results_dir

    parent_dir = seed_dir.parent
    if not parent_dir.exists():
        return expected_results_dir

    seed_prefix = seed_dir.name
    candidates: list[tuple[float, Path]] = []
    for candidate_seed_dir in parent_dir.iterdir():
        if not candidate_seed_dir.is_dir() or not candidate_seed_dir.name.startswith(seed_prefix):
            continue
        try:
            candidate_results_dir = _resolve_results_dir(candidate_seed_dir)
        except BenchmarkValidationError:
            continue
        if not _is_completed_results_dir(candidate_results_dir):
            continue
        try:
            summary = _load_json(candidate_results_dir / "summary.json")
        except Exception:
            continue
        if summary.get("dataset") != expected.dataset:
            continue
        if summary.get("arch") != expected.arch:
            continue
        summary_classes = normalize_classes_mode(
            summary.get("classes"),
            default=expected.task,
            allow_unknown=True,
        )
        if summary_classes != expected.task:
            continue
        tracker_run_name = str(summary.get("tracker_run_name") or "").strip()
        if tracker_run_name and tracker_run_name not in expected.run_name_aliases:
            continue
        summary_mtime = (candidate_results_dir / "summary.json").stat().st_mtime
        candidates.append((summary_mtime, candidate_results_dir))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    return expected_results_dir


def _resolve_metrics_path(results_dir: Path, summary: Mapping[str, Any]) -> Path:
    metrics_dir = results_dir / "metrics"
    view = summary.get("view")
    try:
        test_frac = float(summary.get("test_frac") or 0.0)
    except (TypeError, ValueError):
        test_frac = 0.0

    view_name = view.strip().lower() if isinstance(view, str) and view.strip() else None
    if test_frac > 0:
        candidates: list[Path] = []
        if view_name:
            candidates.append(metrics_dir / f"test_metrics_{view_name}.json")
        candidates.append(metrics_dir / "test_metrics.json")
        for path in candidates:
            if path.exists():
                return path
        if isinstance(summary.get("test_metrics"), Mapping):
            fallback = metrics_dir / "summary_test_metrics.json"
            fallback.write_text(
                json.dumps(summary["test_metrics"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return fallback
        raise BenchmarkValidationError(
            f"Nenhuma metrica de teste consolidada encontrada em {metrics_dir}."
        )

    candidates: list[Path] = []
    if view_name:
        candidates.extend(
            [
                metrics_dir / f"best_metrics_{view_name}.json",
                metrics_dir / f"val_metrics_{view_name}.json",
            ]
        )
    candidates.extend([metrics_dir / "best_metrics.json", metrics_dir / "val_metrics.json"])

    for path in candidates:
        if path.exists():
            return path

    best_matches = sorted(metrics_dir.glob("best_metrics_*.json"))
    if len(best_matches) == 1:
        return best_matches[0]
    val_matches = sorted(metrics_dir.glob("val_metrics_*.json"))
    if len(val_matches) == 1:
        return val_matches[0]

    if isinstance(summary.get("val_metrics"), Mapping):
        fallback = metrics_dir / "summary_val_metrics.json"
        fallback.write_text(
            json.dumps(summary["val_metrics"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return fallback

    raise BenchmarkValidationError(
        f"Nenhuma metrica consolidada encontrada em {metrics_dir}."
    )


def _resolve_embeddings_paths(results_dir: Path) -> tuple[Path, ...]:
    candidates = [
        results_dir / "embeddings_val.csv",
        results_dir / "embeddings_val.npy",
        results_dir / "embeddings_val.npz",
    ]
    existing = tuple(path for path in candidates if path.exists())
    if not existing:
        raise BenchmarkValidationError(
            f"Exportacao de embeddings de validacao ausente em {results_dir}."
        )
    return existing


def _extract_metric(
    metrics: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    primary_keys: Iterable[str],
    summary_keys: Iterable[str],
    default_from_report: str | None = None,
) -> float:
    for key in primary_keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)

    if default_from_report:
        report = metrics.get("classification_report")
        if isinstance(report, Mapping):
            macro_avg = report.get("macro avg")
            if isinstance(macro_avg, Mapping):
                value = macro_avg.get(default_from_report)
                if value is not None:
                    return float(value)

    for key in summary_keys:
        source: Any = summary
        if "." in key:
            for part in key.split("."):
                if not isinstance(source, Mapping):
                    source = None
                    break
                source = source.get(part)
        else:
            source = summary.get(key)
        if source is not None:
            return float(source)

    raise BenchmarkValidationError(
        f"Metrica obrigatoria ausente: {', '.join(primary_keys)} / {', '.join(summary_keys)}"
    )


def _ensure_equal(actual: Any, expected: Any, field_name: str, run_name: str) -> None:
    if isinstance(expected, bool):
        if bool(actual) != expected:
            raise BenchmarkValidationError(
                f"{run_name}: valor invalido para {field_name}: {actual!r} != {expected!r}"
            )
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if actual is None or abs(float(actual) - float(expected)) > 1e-9:
            raise BenchmarkValidationError(
                f"{run_name}: valor invalido para {field_name}: {actual!r} != {expected!r}"
            )
        return
    if actual != expected:
        raise BenchmarkValidationError(
            f"{run_name}: valor invalido para {field_name}: {actual!r} != {expected!r}"
        )


def _validate_config(summary: Mapping[str, Any], expected: ExpectedRun) -> None:
    run_name = expected.run_name
    _ensure_equal(summary.get("dataset"), expected.dataset, "dataset", run_name)
    summary_classes = normalize_classes_mode(
        summary.get("classes"),
        default=expected.task,
        allow_unknown=True,
    )
    _ensure_equal(summary_classes, expected.task, "classes", run_name)
    _ensure_equal(summary.get("arch"), expected.arch, "arch", run_name)
    _ensure_equal(summary.get("split_mode"), expected.split_mode, "split_mode", run_name)

    for field_name, expected_value in COMMON_CONFIG.items():
        _ensure_equal(summary.get(field_name), expected_value, field_name, run_name)

    arch_requirements = ARCH_CONFIG[expected.arch]
    for field_name, expected_value in arch_requirements.items():
        _ensure_equal(summary.get(field_name), expected_value, field_name, run_name)

    subset = summary.get("subset")
    if subset not in (None, 0):
        raise BenchmarkValidationError(
            f"{run_name}: subset deve ser 0/None para o rerun oficial, recebido {subset!r}."
        )

    tracker_run_name = summary.get("tracker_run_name")
    if tracker_run_name not in expected.run_name_aliases:
        raise BenchmarkValidationError(
            f"{run_name}: tracker_run_name invalido: {tracker_run_name!r}"
        )

    if "save_val_preds" in summary:
        _ensure_equal(summary.get("save_val_preds"), True, "save_val_preds", run_name)
    if "export_val_embeddings" in summary:
        _ensure_equal(
            summary.get("export_val_embeddings"),
            True,
            "export_val_embeddings",
            run_name,
        )

    split_group_column = summary.get("split_group_column")
    if expected.split_mode == "patient" and split_group_column not in (
        "patient_id",
        "PatientID",
        None,
    ):
        raise BenchmarkValidationError(
            f"{run_name}: split_group_column invalido para split por paciente: {split_group_column!r}"
        )


def _validate_no_resume(summary: Mapping[str, Any], run_log_path: Path) -> None:
    resume_from = summary.get("resume_from")
    if resume_from not in (None, "", False):
        resume_results_dir = Path(str(resume_from)).parent
        resume_seed_dir = resume_results_dir.parent
        current_seed_dir = run_log_path.parent.parent
        same_parent = current_seed_dir.parent == resume_seed_dir.parent
        same_seed_family = current_seed_dir.name.startswith(resume_seed_dir.name)
        if not same_parent or not same_seed_family:
            raise BenchmarkValidationError(
                f"{run_log_path.parent.name}: summary.json indica resume_from={resume_from!r}."
            )
        return

    text = run_log_path.read_text(encoding="utf-8", errors="ignore")
    blocked_snippets = (
        "Checkpoint encontrado em",
        "Retomando automaticamente",
        "Retomando treino de",
    )
    for snippet in blocked_snippets:
        if snippet in text:
            raise BenchmarkValidationError(
                f"{run_log_path.parent.name}: run.log indica reaproveitamento de checkpoint ({snippet})."
            )


def _read_csv_values(path: Path, column: str) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise BenchmarkValidationError(
                f"Coluna {column!r} ausente em manifesto de split: {path}"
            )
        values = {
            row[column].strip()
            for row in reader
            if row.get(column) is not None and row[column].strip()
        }
    return values


def _validate_archive_leakage(results_dir: Path, run_log_path: Path) -> None:
    split_manifest_path = results_dir / "splits" / "split_manifest.json"
    train_split_path = results_dir / "splits" / "train.csv"
    val_split_path = results_dir / "splits" / "val.csv"
    test_split_path = results_dir / "splits" / "test.csv"

    if split_manifest_path.exists() and train_split_path.exists() and val_split_path.exists():
        manifest = _load_json(split_manifest_path)
        group_column = str(manifest.get("group_column") or "").strip()
        if group_column not in {"patient_id", "PatientID"}:
            raise BenchmarkValidationError(
                f"{results_dir.name}: split por paciente requer group_column patient_id/PatientID."
            )

        train_groups = _read_csv_values(train_split_path, group_column)
        val_groups = _read_csv_values(val_split_path, group_column)
        overlap = train_groups.intersection(val_groups)
        if overlap:
            sample = sorted(overlap)[:3]
            raise BenchmarkValidationError(
                f"{results_dir.name}: vazamento detectado entre train/val por {group_column}: {sample}"
            )

        if test_split_path.exists():
            test_groups = _read_csv_values(test_split_path, group_column)
            if train_groups.intersection(test_groups):
                raise BenchmarkValidationError(
                    f"{results_dir.name}: vazamento detectado entre train/test por {group_column}."
                )
            if val_groups.intersection(test_groups):
                raise BenchmarkValidationError(
                    f"{results_dir.name}: vazamento detectado entre val/test por {group_column}."
                )
        return

    log_text = run_log_path.read_text(encoding="utf-8", errors="ignore")
    required_snippets = (
        "Leakage check:",
        "PatientID",
        "Split criado:",
    )
    for snippet in required_snippets:
        if snippet not in log_text:
            raise BenchmarkValidationError(
                f"{results_dir.name}: evidencias de validacao de leakage ausentes no run.log."
            )
    if "CRITICO: vazamento de dados detectado" in log_text:
        raise BenchmarkValidationError(
            f"{results_dir.name}: run.log indica vazamento de dados entre treino e validacao."
        )


def _discover_export_index(search_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    index: dict[str, tuple[Path, dict[str, Any]]] = {}
    if not search_root.exists():
        return index

    for manifest_path in search_root.rglob("eval_export_manifest.json"):
        try:
            payload = _load_json(manifest_path)
        except Exception:
            continue
        run_dir = payload.get("run_dir")
        if not isinstance(run_dir, str) or not run_dir.strip():
            continue
        key = _normalize_path_key(run_dir)
        current = index.get(key)
        if current is None or manifest_path.stat().st_mtime > current[0].stat().st_mtime:
            index[key] = (manifest_path, payload)
    return index


def _collect_run(
    namespace: Path,
    expected: ExpectedRun,
    export_index: Mapping[str, tuple[Path, dict[str, Any]]],
) -> CollectedRun:
    results_dir = _resolve_run_results_dir(namespace, expected)

    summary_path = _require_file(results_dir / "summary.json", "summary.json")
    summary = _load_json(summary_path)
    _validate_config(summary, expected)

    metrics_path = _resolve_metrics_path(results_dir, summary)
    metrics = _load_json(metrics_path)
    _require_file(results_dir / "val_predictions.csv", "val_predictions.csv")
    _resolve_embeddings_paths(results_dir)
    run_log_path = _require_file(results_dir / "run.log", "run.log")
    _validate_no_resume(summary, run_log_path)
    if expected.dataset == "archive":
        _validate_archive_leakage(results_dir, run_log_path)

    accuracy = _extract_metric(
        metrics,
        summary,
        primary_keys=("acc", "accuracy"),
        summary_keys=("test_metrics.acc", "best_acc", "val_metrics.accuracy"),
    )
    kappa = _extract_metric(
        metrics,
        summary,
        primary_keys=("kappa_quadratic", "kappa"),
        summary_keys=("test_metrics.kappa_quadratic", "val_metrics.kappa_quadratic"),
    )
    macro_f1 = _extract_metric(
        metrics,
        summary,
        primary_keys=("macro_f1",),
        summary_keys=("test_metrics.macro_f1", "best_metric", "val_metrics.macro_f1"),
        default_from_report="f1-score",
    )
    auc = _extract_metric(
        metrics,
        summary,
        primary_keys=("auc_ovr", "auc", "roc_auc"),
        summary_keys=("test_metrics.auc_ovr", "val_metrics.auc_ovr"),
    )
    best_epoch = int(metrics.get("epoch") or summary.get("best_epoch") or 0)
    if best_epoch <= 0:
        raise BenchmarkValidationError(
            f"{expected.run_name}: best_epoch invalido em metrics/summary."
        )

    export_manifest_path: Path | None = None
    export_dir: Path | None = None
    export_entry = export_index.get(_normalize_path_key(results_dir))
    if export_entry is not None:
        export_manifest_path, payload = export_entry
        export_dir_text = payload.get("export_dir")
        if isinstance(export_dir_text, str) and export_dir_text.strip():
            export_dir = Path(export_dir_text)

    return CollectedRun(
        expected=expected,
        results_dir=results_dir,
        summary=summary,
        metrics=metrics,
        accuracy=accuracy,
        kappa=kappa,
        macro_f1=macro_f1,
        auc=auc,
        best_epoch=best_epoch,
        export_dir=export_dir,
        export_manifest_path=export_manifest_path,
    )


def _sort_runs(runs: list[CollectedRun]) -> list[CollectedRun]:
    dataset_index = {name: index for index, name in enumerate(DATASET_ORDER)}
    task_index = {name: index for index, name in enumerate(TASK_ORDER)}
    arch_index = {name: index for index, name in enumerate(ARCH_ORDER)}
    return sorted(
        runs,
        key=lambda run: (
            dataset_index[run.expected.dataset],
            task_index[run.expected.task],
            arch_index[run.expected.arch],
        ),
    )


def _write_csv_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _markdown_lines(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| "
        + " | ".join(_markdown_escape(row[column]) for column in columns)
        + " |"
        for row in rows
    ]
    return [header, divider, *body]


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_markdown_lines(rows, columns)) + "\n", encoding="utf-8")


def _write_master_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latex_lines = [
        "% Auto-generated by mammography benchmark-report",
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\scriptsize",
        "  \\caption{Tabela mestre do rerun oficial 2026Q1.}",
        "  \\label{tab:rerun-2026q1-master}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{llllrrrrrrrrrll}",
        "    \\toprule",
        "    dataset & task & split\\_mode & arch & seed & img\\_size & batch\\_size & epochs & accuracy & kappa & macro\\_f1 & auc & best\\_epoch & run\\_path & status \\\\",
        "    \\midrule",
    ]
    for row in rows:
        latex_lines.append(
            "    " + " & ".join(_tex_escape(row[column]) for column in MASTER_COLUMNS) + " \\\\"
        )
    latex_lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}%",
            "  }",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")


def _write_article_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latex_lines = [
        "% Auto-generated by mammography benchmark-report",
        "\\subsection{Tabela Consolidada do Rerun Oficial 2026Q1}",
        "\\label{sec:rerun-2026q1-table}",
        (
            "A tabela a seguir deriva automaticamente da tabela mestre do rerun oficial "
            "e resume um resultado por combinacao de dataset, tarefa e modelo."
        ),
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\scriptsize",
        "  \\caption{Rerun oficial 2026Q1 consolidado por dataset, tarefa e modelo.}",
        "  \\label{tab:rerun-2026q1-article}",
        "  \\begin{tabular}{llllrrrr}",
        "    \\toprule",
        "    dataset & task & split & modelo & accuracy & kappa & macro-F1 & AUC \\\\",
        "    \\midrule",
    ]
    for row in rows:
        latex_lines.append(
            "    " + " & ".join(_tex_escape(row[column]) for column in ARTICLE_COLUMNS) + " \\\\"
        )
    latex_lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")


def _current_environment() -> dict[str, str]:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": "unavailable",
        "cuda": "unavailable",
        "gpu": "unavailable",
    }
    if torch is None:
        return env

    env["torch"] = getattr(torch, "__version__", "unknown")
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    env["cuda"] = str(cuda_version or "cpu")
    try:
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        env["gpu"] = "unknown"
    return env


def _environment_from_runs(runs: list[CollectedRun]) -> dict[str, str]:
    env = _current_environment()

    def _value_from_summaries(summary_key: str, fallback_key: str) -> str:
        values = sorted(
            {
                str(run.summary.get("reproducibility", {}).get(summary_key)).strip()
                for run in runs
                if str(run.summary.get("reproducibility", {}).get(summary_key) or "").strip()
            }
        )
        if not values:
            return env[fallback_key]
        if len(values) == 1:
            return values[0]
        return ", ".join(values)

    return {
        "python": _value_from_summaries("python_version", "python"),
        "platform": _value_from_summaries("platform", "platform"),
        "torch": _value_from_summaries("torch_version", "torch"),
        "cuda": _value_from_summaries("cuda_version", "cuda"),
        "gpu": _value_from_summaries("gpu_name", "gpu"),
    }


def _render_docs_report(
    path: Path,
    namespace: Path,
    runs: list[CollectedRun],
    master_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    env = _environment_from_runs(runs)
    commits = sorted(
        {
            str(run.summary.get("reproducibility", {}).get("git_commit", "unknown"))
            for run in runs
        }
    )
    execution_timestamps = sorted(
        {
            str(
                run.summary.get("created_at")
                or run.summary.get("reproducibility", {}).get("timestamp")
                or "unknown"
            )
            for run in runs
        }
    )

    matrix_rows = [
        {
            "dataset": run.expected.dataset,
            "task": run.expected.task,
            "split_mode": run.expected.split_mode,
            "arch": run.expected.arch,
            "run_path": str(run.results_dir),
            "status": "accepted",
        }
        for run in runs
    ]
    matrix_columns = ["dataset", "task", "split_mode", "arch", "run_path", "status"]

    lines = [
        "# Technical Report: rerun_2026q1",
        "",
        "## Header",
        f"- Namespace: `{namespace}`",
        f"- Official runs validated: `{len(runs)}`",
        f"- Git commits seen in summaries: `{', '.join(commits)}`",
        f"- Execution timestamps seen in summaries: `{', '.join(execution_timestamps)}`",
        f"- Python: `{env['python']}`",
        f"- PyTorch: `{env['torch']}`",
        f"- CUDA: `{env['cuda']}`",
        f"- GPU: `{env['gpu']}`",
        "",
        "## Protocol",
        "- Official matrix: 3 datasets x 2 tasks x 3 models, one seed (42).",
        "- Official metrics come from a held-out test split; the best checkpoint is still selected on validation macro-F1.",
        "- Common settings enforced: deterministic=true, amp=true, allow_tf32=true, pretrained=true, train_backbone=true, unfreeze_last_block=true, augment=true, class_weights=auto, sampler_weighted=true, test_frac=0.1, tracker=local.",
        "- CNN profile: img_size=512, batch_size=16, epochs=30, lr=1e-4, backbone_lr=1e-5, warmup_epochs=2, early_stop_patience=5.",
        "- ViT profile: img_size=224, batch_size=8, epochs=30, lr=1e-3, backbone_lr=1e-4, warmup_epochs=3, early_stop_patience=10.",
        "- Explainability artifacts are explicitly out of scope for rerun acceptance.",
        "",
        "## Run Matrix",
        *_markdown_lines(matrix_rows, matrix_columns),
        "",
        "## Split Limitations",
        "- `archive` uses `split_mode=patient`, and leakage validation is checked against saved split manifests when available.",
        "- `mamografias` uses `split_mode=random` because the current loader does not expose a reliable patient grouping key for this benchmark.",
        "- `patches_completo` uses `split_mode=random` for the same limitation: no reliable patient grouping key is exposed in the current data format.",
        "",
        "## Master Table",
        *_markdown_lines(master_rows, MASTER_COLUMNS),
        "",
        "## Export References",
    ]
    for run in runs:
        export_dir_text = str(run.export_dir) if run.export_dir is not None else "not found"
        export_manifest_text = (
            str(run.export_manifest_path)
            if run.export_manifest_path is not None
            else "not found"
        )
        lines.append(
            f"- `{run.expected.run_name}`: export_dir=`{export_dir_text}` | manifest=`{export_manifest_text}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_benchmark_report(
    *,
    namespace: Path,
    output_prefix: Path,
    docs_report_path: Path,
    article_table_path: Path,
    exports_search_root: Path,
) -> list[CollectedRun]:
    """Validate the official rerun namespace and write consolidated artifacts."""
    namespace = _resolve_path(namespace)
    output_prefix = _normalize_output_prefix(output_prefix)
    docs_report_path = _resolve_path(docs_report_path)
    article_table_path = _resolve_path(article_table_path)
    exports_search_root = _resolve_path(exports_search_root)

    export_index = _discover_export_index(exports_search_root)
    collected_runs = _sort_runs(
        [_collect_run(namespace, expected, export_index) for expected in expected_runs()]
    )

    if len(collected_runs) != len(expected_runs()):
        raise BenchmarkValidationError(
            f"Esperado {len(expected_runs())} runs oficiais; encontrados {len(collected_runs)}."
        )

    master_rows = [run.master_row() for run in collected_runs]
    article_rows = [run.article_row() for run in collected_runs]

    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    json_path = output_prefix.with_suffix(".json")
    tex_path = output_prefix.with_suffix(".tex")

    _write_csv_table(csv_path, master_rows, MASTER_COLUMNS)
    _write_markdown_table(md_path, master_rows, MASTER_COLUMNS)
    _write_json_table(json_path, master_rows)
    _write_master_tex(tex_path, master_rows)
    _write_article_tex(article_table_path, article_rows)
    _render_docs_report(docs_report_path, namespace, collected_runs, master_rows)

    LOGGER.info("Tabela mestre CSV: %s", csv_path)
    LOGGER.info("Tabela mestre Markdown: %s", md_path)
    LOGGER.info("Tabela mestre JSON: %s", json_path)
    LOGGER.info("Tabela mestre TeX: %s", tex_path)
    LOGGER.info("Relatorio tecnico: %s", docs_report_path)
    LOGGER.info("Tabela do artigo: %s", article_table_path)

    return collected_runs
