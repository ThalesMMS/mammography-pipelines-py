"""Metric extraction helpers for training registry entries."""

from __future__ import annotations

import csv
import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from pathlib import Path

from mammography.tools.train_registry_artifacts import _collect_artifacts, _load_json


def _parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _compute_sensitivity_specificity(
    val_metrics: Mapping[str, Any],
) -> tuple[float | None, float | None]:
    cm = val_metrics.get("confusion_matrix")
    if isinstance(cm, Sequence) and len(cm) == 2:
        try:
            row0 = cm[0]
            row1 = cm[1]
            if (
                isinstance(row0, Sequence)
                and isinstance(row1, Sequence)
                and len(row0) == 2
                and len(row1) == 2
            ):
                tn = float(row0[0])
                fp = float(row0[1])
                fn = float(row1[0])
                tp = float(row1[1])
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
                specificity = tn / (tn + fp) if (tn + fp) > 0 else None
                return sensitivity, specificity
        except (TypeError, ValueError):
            pass

    report = val_metrics.get("classification_report")
    if isinstance(report, Mapping):
        pos = report.get("1") or report.get(1)
        neg = report.get("0") or report.get(0)
        sensitivity = _parse_optional_float(
            pos.get("recall") if isinstance(pos, Mapping) else None
        )
        specificity = _parse_optional_float(
            neg.get("recall") if isinstance(neg, Mapping) else None
        )
        return sensitivity, specificity

    return None, None


def _load_train_history(train_history_path: Path) -> dict[str, float | None]:
    with train_history_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("train_history.csv sem dados")
    last_row = rows[-1]
    return {
        "train_loss": _parse_optional_float(last_row.get("train_loss")),
        "train_acc": _parse_optional_float(last_row.get("train_acc")),
        "val_loss": _parse_optional_float(last_row.get("val_loss")),
        "val_acc": _parse_optional_float(last_row.get("val_acc")),
        "val_f1": _parse_optional_float(last_row.get("val_macro_f1")),
        "val_auc": _parse_optional_float(last_row.get("val_auc")),
    }


def _load_val_metrics(val_metrics_path: Path) -> dict[str, Any]:
    return _load_json(val_metrics_path)


def _collect_metrics(
    history_metrics: Mapping[str, float | None],
    val_metrics: Mapping[str, Any],
) -> dict[str, float | None]:
    report = val_metrics.get("classification_report")
    report_map = report if isinstance(report, Mapping) else {}

    val_acc = history_metrics.get("val_acc")
    if val_acc is None:
        val_acc = _parse_optional_float(val_metrics.get("acc"))
    if val_acc is None:
        val_acc = _parse_optional_float(val_metrics.get("accuracy"))
    if val_acc is None:
        val_acc = _parse_optional_float(report_map.get("accuracy"))

    val_loss = history_metrics.get("val_loss")
    if val_loss is None:
        val_loss = _parse_optional_float(val_metrics.get("loss"))

    val_f1 = history_metrics.get("val_f1")
    if val_f1 is None:
        val_f1 = _parse_optional_float(val_metrics.get("macro_f1"))
    if val_f1 is None:
        val_f1 = _parse_optional_float(val_metrics.get("f1"))
    if val_f1 is None:
        macro_avg = report_map.get("macro avg")
        if isinstance(macro_avg, Mapping):
            val_f1 = _parse_optional_float(
                macro_avg.get("f1-score") or macro_avg.get("f1")
            )

    val_auc = history_metrics.get("val_auc")
    if val_auc is None:
        val_auc = _parse_optional_float(val_metrics.get("auc_ovr"))
    if val_auc is None:
        val_auc = _parse_optional_float(val_metrics.get("auc"))
    if val_auc is None:
        val_auc = _parse_optional_float(val_metrics.get("roc_auc"))

    val_sensitivity, val_specificity = _compute_sensitivity_specificity(val_metrics)

    return {
        "train_loss": history_metrics.get("train_loss"),
        "train_acc": history_metrics.get("train_acc"),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_auc": val_auc,
        "val_sensitivity": val_sensitivity,
        "val_specificity": val_specificity,
    }


def _compute_metric_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    if not math.isfinite(value) or not math.isfinite(baseline):
        return None
    return value - baseline


def _collect_baseline_metrics(outdir: Path) -> dict[str, float | None]:
    artifacts = _collect_artifacts(outdir)
    history_metrics = _load_train_history(artifacts.train_history_path)
    val_metrics = _load_val_metrics(artifacts.val_metrics_path)
    return _collect_metrics(history_metrics, val_metrics)
