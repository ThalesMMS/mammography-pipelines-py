# ruff: noqa
#
# model_comparison.py
# mammography-pipelines
#
# Model comparison utilities for loading checkpoints and aggregating metrics across multiple models.
# DISCLAIMER: Educational project only - NOT for clinical or medical diagnostic purposes.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""
Model comparison utilities for comparing multiple trained models.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.

Provides utilities for loading multiple model checkpoints, extracting metrics,
and preparing data structures for comparison visualizations and statistical testing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as exc:  # pragma: no cover - optional dashboard deps
    px = None
    go = None
    make_subplots = None
    _PLOTLY_IMPORT_ERROR = exc
else:
    _PLOTLY_IMPORT_ERROR = None

from mammography.utils.statistics import (
    aggregate_cv_metrics,
    compute_confidence_interval,
    effect_size_cohen_d,
)

LOGGER = logging.getLogger("mammography")

def _resolve_metrics_json_path(run_path: Path, config: Dict[str, Any]) -> Path:
    """Resolve the best available metrics JSON for a training run.

    Resumed runs on Windows may finish in an incremented directory (for example
    ``seed42_1/results``) while the best-model artifacts remain in the original
    ``resume_from`` directory. Prefer ``best_metrics.json`` when available and
    fall back to ``val_metrics.json`` from the current or resumed results dir.
    """
    candidates: list[Path] = [
        run_path / "metrics" / "best_metrics.json",
        run_path / "metrics" / "val_metrics.json",
    ]

    resume_from = config.get("resume_from")
    if isinstance(resume_from, str) and resume_from.strip():
        resume_results_dir = Path(resume_from).parent
        candidates.extend(
            [
                resume_results_dir / "metrics" / "best_metrics.json",
                resume_results_dir / "metrics" / "val_metrics.json",
            ]
        )

    top_k = config.get("top_k")
    if isinstance(top_k, list):
        for entry in top_k:
            if not isinstance(entry, dict):
                continue
            raw_path = entry.get("path")
            if raw_path is None:
                continue
            top_k_path = Path(str(raw_path))
            metrics_dir = top_k_path.parent.parent / "metrics"
            candidates.extend(
                [
                    metrics_dir / "best_metrics.json",
                    metrics_dir / "val_metrics.json",
                ]
            )

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path

    LOGGER.error("Neither best_metrics.json nor val_metrics.json found in %s", run_path)
    raise FileNotFoundError(f"metrics/best_metrics.json not found in {run_path}")

def compute_mcnemar_test(
    y_true: NDArray[np.integer] | List[int],
    y_pred_a: NDArray[np.integer] | List[int],
    y_pred_b: NDArray[np.integer] | List[int],
    continuity_correction: bool = True,
) -> Tuple[float, float]:
    """
    Compute McNemar's chi-square statistic and p-value comparing two classifiers' predictions.
    
    This function compares two sets of predictions on the same ground-truth labels by converting each prediction to correct/incorrect per sample and evaluating disagreements between the classifiers. Optionally applies Yates' continuity correction.
    
    Parameters:
        y_true: Ground-truth labels (1D sequence of class indices).
        y_pred_a: Predictions from model A (1D sequence of class indices).
        y_pred_b: Predictions from model B (1D sequence of class indices).
        continuity_correction (bool): If True, apply Yates' continuity correction to the test statistic.
    
    Returns:
        tuple:
            statistic (float): McNemar chi-square statistic (df=1).
            p_value (float): Two-tailed p-value for the statistic.
    
    Raises:
        ValueError: If any input array is empty, arrays have different shapes, or arrays are not 1-dimensional.
    """
    # Convert to numpy arrays
    y_true_arr = np.asarray(y_true, dtype=np.int_)
    y_pred_a_arr = np.asarray(y_pred_a, dtype=np.int_)
    y_pred_b_arr = np.asarray(y_pred_b, dtype=np.int_)

    # Validate inputs
    if y_true_arr.size == 0:
        raise ValueError("y_true cannot be empty")

    if y_pred_a_arr.size == 0 or y_pred_b_arr.size == 0:
        raise ValueError("y_pred_a and y_pred_b cannot be empty")

    if not (y_true_arr.shape == y_pred_a_arr.shape == y_pred_b_arr.shape):
        raise ValueError(
            f"All arrays must have same shape. Got y_true: {y_true_arr.shape}, "
            f"y_pred_a: {y_pred_a_arr.shape}, y_pred_b: {y_pred_b_arr.shape}"
        )

    if y_true_arr.ndim != 1:
        raise ValueError(f"Arrays must be 1-dimensional, got {y_true_arr.ndim}D")

    # Compute correctness for each model
    correct_a = (y_pred_a_arr == y_true_arr)
    correct_b = (y_pred_b_arr == y_true_arr)

    # Build 2x2 contingency table
    # n00: both correct
    # n01: A correct, B wrong
    # n10: A wrong, B correct
    # n11: both wrong
    n00 = np.sum(correct_a & correct_b)
    n01 = np.sum(correct_a & ~correct_b)
    n10 = np.sum(~correct_a & correct_b)
    n11 = np.sum(~correct_a & ~correct_b)

    LOGGER.debug(
        "McNemar contingency table: [[%d, %d], [%d, %d]]",
        n00, n01, n10, n11,
    )

    # Handle edge case: no disagreements
    if n01 + n10 == 0:
        LOGGER.warning(
            "McNemar test: no disagreements between models (identical predictions). "
            "Returning statistic=0.0, p_value=1.0"
        )
        return (0.0, 1.0)

    # Compute McNemar's test statistic
    if continuity_correction:
        # With continuity correction (Yates' correction)
        numerator = max(abs(n01 - n10) - 1.0, 0.0) ** 2
    else:
        # Without continuity correction
        numerator = (n01 - n10) ** 2

    denominator = n01 + n10
    statistic = float(numerator / denominator)

    # Compute p-value using chi-square distribution (df=1)
    try:
        from scipy import stats
        p_value = float(stats.chi2.sf(statistic, df=1))
    except ImportError:
        # Fallback: approximate p-value using normal approximation
        import warnings
        warnings.warn(
            "scipy not available, using normal approximation for McNemar p-value. "
            "Install scipy for exact chi-square p-values.",
            UserWarning,
            stacklevel=2,
        )
        # For large samples, chi-square(1) ≈ Normal(0,1)²
        # P(χ² > x) ≈ P(|Z| > √x) for df=1
        z_stat = np.sqrt(statistic)
        # Two-tailed p-value using standard normal approximation
        # P(|Z| > z) = 2 * P(Z > z) = 2 * (1 - Φ(z))
        # Using complementary error function for tail probability
        from math import erfc
        p_value = float(erfc(z_stat / np.sqrt(2)))

    LOGGER.debug(
        "McNemar test: statistic=%.4f, p_value=%.4f (continuity_correction=%s)",
        statistic, p_value, continuity_correction,
    )

    return (statistic, p_value)

@dataclass
class ModelMetrics:
    """Metrics and metadata for a single trained model.

    Attributes:
        model_name: Human-readable model identifier
        run_path: Path to the training run directory
        arch: Model architecture (e.g., 'efficientnet_b0', 'resnet50', 'vit_base')
        dataset: Dataset name used for training
        accuracy: Validation accuracy
        kappa: Cohen's kappa score (quadratic weighted)
        macro_f1: Macro-averaged F1 score
        auc: AUC score (OVR for multiclass, or None if not available)
        balanced_accuracy: Balanced accuracy score
        confusion_matrix: Confusion matrix as 2D list
        classification_report: Per-class precision/recall/F1 dict
        val_loss: Validation loss
        num_epochs: Total number of training epochs
        best_epoch: Epoch with best validation performance
        num_classes: Number of output classes
        img_size: Input image size
        batch_size: Training batch size
        learning_rate: Learning rate used
        config: Full training configuration dict
        predictions: Optional validation predictions with true labels
    """

    model_name: str
    run_path: Path
    arch: str
    dataset: str
    accuracy: float
    kappa: float
    macro_f1: float
    auc: Optional[float]
    balanced_accuracy: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    val_loss: float
    num_epochs: int
    best_epoch: int
    num_classes: int
    img_size: int
    batch_size: int
    learning_rate: float
    config: Dict[str, Any] = field(default_factory=dict)
    predictions: Optional[List[Dict[str, Any]]] = None
