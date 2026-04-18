"""Artifact discovery helpers for training registry entries."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TrainingArtifacts:
    results_dir: Path
    summary_path: Path
    train_history_path: Path
    val_metrics_path: Path
    confusion_matrix_path: Path | None
    checkpoint_path: Path
    val_predictions_path: Path | None
    train_history_plot_path: Path | None


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON invalido: esperado objeto. Recebido: {path}")
    return data


def _normalize_view(value: object) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned.lower()
    return None


def _find_results_dir(outdir: Path) -> Path:
    if (outdir / "summary.json").exists():
        return outdir
    candidates: list[tuple[int, Path]] = []
    for path in outdir.glob("results*"):
        if not path.is_dir():
            continue
        if not (path / "summary.json").exists():
            continue
        suffix = path.name.replace("results", "", 1)
        if suffix.startswith("_"):
            suffix = suffix[1:]
        try:
            index = int(suffix) if suffix else 0
        except ValueError:
            index = 0
        candidates.append((index, path))
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum results com summary.json encontrado em {outdir}"
        )
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _resolve_checkpoint_candidate(
    raw_path: object,
    *,
    results_dir: Path,
) -> Path | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = results_dir / path
    return path


def _iter_summary_checkpoint_candidates(
    results_dir: Path,
    summary: Mapping[str, Any],
) -> Sequence[Path]:
    candidates: list[Path] = []

    top_k = summary.get("top_k")
    if isinstance(top_k, Sequence) and not isinstance(top_k, (str, bytes)):
        for entry in top_k:
            if not isinstance(entry, Mapping):
                continue
            candidate = _resolve_checkpoint_candidate(
                entry.get("path"),
                results_dir=results_dir,
            )
            if candidate is not None:
                candidates.append(candidate)

    resume_from = _resolve_checkpoint_candidate(
        summary.get("resume_from"),
        results_dir=results_dir,
    )
    if resume_from is not None:
        candidates.append(resume_from)
        resume_results_dir = resume_from.parent
        view = _normalize_view(summary.get("view"))
        best_model_name = f"best_model_{view}.pt" if view else "best_model.pt"
        candidates.append(resume_results_dir / best_model_name)
        candidates.extend(sorted(resume_results_dir.glob("best_model*.pt")))

    return candidates


def _pick_best_model(
    results_dir: Path,
    summary: Mapping[str, Any] | None = None,
) -> Path:
    best_model = results_dir / "best_model.pt"
    if best_model.exists():
        return best_model
    matches = sorted(results_dir.glob("best_model*.pt"))
    if matches:
        return matches[0]
    if summary is not None:
        for candidate in _iter_summary_checkpoint_candidates(results_dir, summary):
            if candidate.exists():
                return candidate
    checkpoint = results_dir / "checkpoint.pt"
    if checkpoint.exists():
        return checkpoint
    raise FileNotFoundError(f"Checkpoint best_model*.pt ausente em {results_dir}")


def _resolve_metrics_json(metrics_dir: Path, view: str | None) -> Path:
    candidates = [metrics_dir / "val_metrics.json"]
    if view:
        candidates.append(metrics_dir / f"val_metrics_{view}.json")
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(metrics_dir.glob("val_metrics_*.json"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise FileNotFoundError(
            f"Multiplos val_metrics_*.json encontrados em {metrics_dir}; "
            "especifique a view no summary.json."
        )
    raise FileNotFoundError(f"val_metrics.json ausente: {metrics_dir}")


def _resolve_metrics_fallback(results_dir: Path, view: str | None) -> Path | None:
    if not view:
        return None
    suffix = f"_{view}"
    name = results_dir.name
    if name.lower().endswith(suffix):
        base_name = name[: -len(suffix)]
        base_dir = results_dir.parent / base_name
        fallback = base_dir / "metrics"
        if fallback.exists():
            return fallback
    return None


def _resolve_confusion_matrix(
    metrics_dir: Path,
    figures_dir: Path,
    view: str | None,
) -> Path | None:
    suffix = f"_{view}" if view else ""
    candidates = [
        metrics_dir / f"val_metrics{suffix}.png",
        metrics_dir / f"best_metrics{suffix}.png",
        figures_dir / f"val_metrics{suffix}.png",
        figures_dir / f"best_metrics{suffix}.png",
        metrics_dir / "val_metrics.png",
        metrics_dir / "best_metrics.png",
        figures_dir / "val_metrics.png",
        figures_dir / "best_metrics.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    for pattern in ("val_metrics_*.png", "best_metrics_*.png"):
        matches = sorted(metrics_dir.glob(pattern))
        if len(matches) == 1:
            return matches[0]
    return None


def _collect_artifacts(outdir: Path) -> TrainingArtifacts:
    results_dir = _find_results_dir(outdir)
    summary_path = results_dir / "summary.json"
    train_history_path = results_dir / "train_history.csv"
    metrics_dir = results_dir / "metrics"

    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json ausente: {summary_path}")
    if not train_history_path.exists():
        raise FileNotFoundError(f"train_history.csv ausente: {train_history_path}")

    summary = _load_json(summary_path)
    view = _normalize_view(summary.get("view"))
    metrics_root = metrics_dir
    try:
        val_metrics_path = _resolve_metrics_json(metrics_dir, view)
    except FileNotFoundError:
        fallback_dir = _resolve_metrics_fallback(results_dir, view)
        if fallback_dir is None:
            raise
        metrics_root = fallback_dir
        val_metrics_path = _resolve_metrics_json(metrics_root, view)

    checkpoint_path = _pick_best_model(results_dir, summary)
    confusion_matrix_path = _resolve_confusion_matrix(
        metrics_root,
        results_dir / "figures",
        view,
    )
    val_predictions_path = results_dir / "val_predictions.csv"
    if not val_predictions_path.exists():
        val_predictions_path = None
    train_history_plot_path = results_dir / "train_history.png"
    if not train_history_plot_path.exists():
        train_history_plot_path = None

    return TrainingArtifacts(
        results_dir=results_dir,
        summary_path=summary_path,
        train_history_path=train_history_path,
        val_metrics_path=val_metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        checkpoint_path=checkpoint_path,
        val_predictions_path=val_predictions_path,
        train_history_plot_path=train_history_plot_path,
    )
