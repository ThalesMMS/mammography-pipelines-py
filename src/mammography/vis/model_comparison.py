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


def _require_plotly() -> None:
    """Raise ImportError if plotly is not available."""
    if px is None or go is None or make_subplots is None:
        raise ImportError(
            "Plotly is required for visualization features. "
            "Install with: pip install plotly"
        ) from _PLOTLY_IMPORT_ERROR


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
    raise FileNotFoundError(f"No metrics JSON found for {run_path}")


def compute_mcnemar_test(
    y_true: NDArray[np.integer] | List[int],
    y_pred_a: NDArray[np.integer] | List[int],
    y_pred_b: NDArray[np.integer] | List[int],
    continuity_correction: bool = True,
) -> Tuple[float, float]:
    """
    Compute McNemar's test for comparing two classification models.

    McNemar's test is a paired statistical test for assessing whether two models
    have significantly different error rates on the same dataset. It uses a 2x2
    contingency table based on correct/incorrect predictions:

        |           | Model B Correct | Model B Wrong |
        |-----------|-----------------|---------------|
        | Model A Correct |       n00       |      n01      |
        | Model A Wrong   |       n10       |      n11      |

    The test statistic focuses on disagreements (n01 and n10):
    - Without correction: chi² = (n01 - n10)² / (n01 + n10)
    - With correction: chi² = (|n01 - n10| - 1)² / (n01 + n10)

    A low p-value (typically < 0.05) indicates statistically significant difference.

    Args:
        y_true: Ground truth labels (1D array of class indices)
        y_pred_a: Predictions from model A (1D array of class indices)
        y_pred_b: Predictions from model B (1D array of class indices)
        continuity_correction: Apply continuity correction (default: True)
                               Recommended for small sample sizes

    Returns:
        Tuple of (statistic, p_value):
        - statistic: McNemar's test chi-square statistic
        - p_value: Two-tailed p-value from chi-square distribution (df=1)

    Raises:
        ValueError: If arrays have different lengths or are empty

    Example:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
        >>> y_pred_a = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
        >>> y_pred_b = np.array([0, 1, 0, 0, 1, 1, 0, 0, 1, 0])
        >>> stat, p_value = compute_mcnemar_test(y_true, y_pred_a, y_pred_b)
        >>> print(f"McNemar statistic: {stat:.3f}, p-value: {p_value:.3f}")
        McNemar statistic: 0.000, p-value: 1.000

    Note:
        - For binary classification, both models must predict same two classes
        - For multiclass, predictions are converted to correct/incorrect
        - Requires at least one disagreement (n01 + n10 > 0) for valid test
        - If both models make identical predictions, returns (0.0, 1.0)
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
        numerator = (abs(n01 - n10) - 1.0) ** 2
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


class ModelComparisonEngine:
    """Engine for loading and comparing multiple trained models.

    Loads model checkpoints and metrics from training run directories,
    extracts relevant metrics, and prepares data structures for visualization
    and statistical comparison.

    Args:
        run_paths: List of paths to training run directories (containing summary.json, metrics/best_metrics.json)
        model_names: Optional list of custom model names (defaults to directory names)

    Example:
        >>> engine = ModelComparisonEngine([
        ...     "outputs/archive_density_effnet/results",
        ...     "outputs/archive_density_resnet/results",
        ... ])
        >>> metrics = engine.load_all_metrics()
        >>> df = engine.get_metrics_dataframe(metrics)
        >>> print(df[['model_name', 'accuracy', 'macro_f1']])
    """

    def __init__(
        self,
        run_paths: List[Union[str, Path]],
        model_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the engine with paths to model run directories and optional display names.
        
        Parameters:
            run_paths (List[Union[str, Path]]): Iterable of filesystem paths to model run directories; must be non-empty and each path must exist.
            model_names (Optional[List[str]]): Optional list of names corresponding to run_paths. If provided, its length must match run_paths; otherwise names are derived from each run path's parent directory.
        
        Raises:
            ValueError: If run_paths is empty or if model_names is provided with a different length than run_paths.
            FileNotFoundError: If any path in run_paths does not exist.
        
        Side effects:
            Sets self.run_paths to a list of Path objects and self.model_names to either the provided names or the derived defaults.
        """
        if not run_paths:
            raise ValueError("run_paths cannot be empty")

        self.run_paths = [Path(p) for p in run_paths]

        # Validate all paths exist
        for path in self.run_paths:
            if not path.exists():
                raise FileNotFoundError(f"Run path does not exist: {path}")

        # Use custom names or derive from directory names
        if model_names is not None:
            if len(model_names) != len(run_paths):
                raise ValueError(
                    f"Number of model_names ({len(model_names)}) must match "
                    f"number of run_paths ({len(run_paths)})"
                )
            self.model_names = model_names
        else:
            # Use parent directory name as default (e.g., "archive_density_effnet")
            self.model_names = [p.parent.name for p in self.run_paths]

        LOGGER.info(
            "ModelComparisonEngine initialized with %d models: %s",
            len(self.run_paths),
            self.model_names,
        )

    def load_model_metrics(self, run_path: Path, model_name: str) -> Optional[ModelMetrics]:
        """
        Load model training configuration and validation metrics from a single run directory and return a populated ModelMetrics instance.
        
        Parameters:
            run_path (Path): Path to the training run directory containing summary.json and metrics/best_metrics.json.
            model_name (str): Human-readable identifier to assign to the returned ModelMetrics.
        
        Returns:
            ModelMetrics: An instance populated from the run's configuration and best_metrics.
        
        Raises:
            FileNotFoundError: If summary.json or metrics/best_metrics.json is missing.
            json.JSONDecodeError: If either JSON file is malformed.
            KeyError: If required metric keys are absent from best_metrics.json.
        """
        LOGGER.info("Loading metrics for model '%s' from %s", model_name, run_path)

        # Load summary.json (training configuration)
        summary_path = run_path / "summary.json"
        if not summary_path.exists():
            LOGGER.error("summary.json not found in %s", run_path)
            raise FileNotFoundError(f"summary.json not found in {run_path}")

        with summary_path.open("r") as f:
            config = json.load(f)

        metrics_path = _resolve_metrics_json_path(run_path, config)

        with metrics_path.open("r") as f:
            metrics_data = json.load(f)

        # Extract metrics
        try:
            model_metrics = ModelMetrics(
                model_name=model_name,
                run_path=run_path,
                arch=config.get("arch", "unknown"),
                dataset=config.get("dataset", "unknown"),
                accuracy=metrics_data["acc"],
                kappa=metrics_data.get("kappa_quadratic", 0.0),
                macro_f1=metrics_data.get("macro_f1", 0.0),
                auc=metrics_data.get("auc_ovr"),
                balanced_accuracy=metrics_data.get("bal_acc", 0.0),
                confusion_matrix=metrics_data.get("confusion_matrix", []),
                classification_report=metrics_data.get("classification_report", {}),
                val_loss=metrics_data.get("loss", 0.0),
                num_epochs=config.get("epochs", 0),
                best_epoch=metrics_data.get("epoch", 0) if "epoch" in metrics_data else 0,
                num_classes=len(metrics_data.get("confusion_matrix", [])),
                img_size=config.get("img_size", 0),
                batch_size=config.get("batch_size", 0),
                learning_rate=config.get("lr", 0.0),
                config=config,
                predictions=metrics_data.get("val_rows"),
            )

            LOGGER.info(
                "Loaded metrics for '%s': acc=%.4f, kappa=%.4f, macro_f1=%.4f",
                model_name,
                model_metrics.accuracy,
                model_metrics.kappa,
                model_metrics.macro_f1,
            )

            return model_metrics

        except KeyError as e:
            LOGGER.error("Missing required metric in %s: %s", metrics_path, e)
            raise

    def load_all_metrics(self) -> List[ModelMetrics]:
        """
        Load metrics for every configured run and return those that load successfully.
        
        Attempts to load each model's metrics and skips models that fail to load; if no models are loaded successfully this method raises an error.
        
        Returns:
            List[ModelMetrics]: Metrics objects for each model that loaded successfully.
        
        Raises:
            RuntimeError: If no models could be loaded successfully.
        """
        all_metrics = []

        for run_path, model_name in zip(self.run_paths, self.model_names):
            try:
                metrics = self.load_model_metrics(run_path, model_name)
                if metrics is not None:
                    all_metrics.append(metrics)
            except Exception as e:
                LOGGER.warning(
                    "Failed to load metrics for model '%s' from %s: %s",
                    model_name,
                    run_path,
                    e,
                )
                # Continue loading other models even if one fails

        if not all_metrics:
            raise RuntimeError(
                "Failed to load metrics for any of the configured models. "
                "Ensure run directories contain summary.json and metrics/best_metrics.json"
            )

        LOGGER.info("Successfully loaded metrics for %d/%d models", len(all_metrics), len(self.run_paths))

        return all_metrics

    def get_metrics_dataframe(self, metrics_list: List[ModelMetrics]) -> pd.DataFrame:
        """
        Create a DataFrame summarizing key metrics and metadata for each ModelMetrics in metrics_list.
        
        Columns include: model_name, arch, dataset, accuracy, kappa, macro_f1, auc, balanced_accuracy, val_loss, num_epochs, best_epoch, num_classes, img_size, batch_size, and learning_rate.
        
        Parameters:
            metrics_list (List[ModelMetrics]): Sequence of ModelMetrics objects to convert.
        
        Returns:
            pd.DataFrame: One row per model containing the columns listed above, sorted by `macro_f1` descending.
        """
        data = []
        for m in metrics_list:
            row = {
                "model_name": m.model_name,
                "arch": m.arch,
                "dataset": m.dataset,
                "accuracy": m.accuracy,
                "kappa": m.kappa,
                "macro_f1": m.macro_f1,
                "auc": m.auc,
                "balanced_accuracy": m.balanced_accuracy,
                "val_loss": m.val_loss,
                "num_epochs": m.num_epochs,
                "best_epoch": m.best_epoch,
                "num_classes": m.num_classes,
                "img_size": m.img_size,
                "batch_size": m.batch_size,
                "learning_rate": m.learning_rate,
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Sort by macro_f1 descending (best models first)
        df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)

        return df

    @staticmethod
    def get_per_class_metrics(metrics_list: List[ModelMetrics]) -> pd.DataFrame:
        """Extract per-class precision, recall, F1 scores for all models.

        Args:
            metrics_list: List of ModelMetrics objects

        Returns:
            DataFrame with per-class metrics in long format (one row per model-class pair)

        Example:
            >>> df = engine.get_per_class_metrics(metrics)
            >>> print(df.head())
               model_name  class  precision  recall  f1-score  support
            0  model_a       1       0.85    0.82      0.83       50
            1  model_a       2       0.88    0.90      0.89       75
        """
        data = []

        for m in metrics_list:
            report = m.classification_report
            if not report:
                continue

            # Extract per-class metrics (skip aggregated keys like 'accuracy', 'macro avg')
            for class_label, class_metrics in report.items():
                if isinstance(class_metrics, dict) and "f1-score" in class_metrics:
                    # Skip aggregate metrics
                    if class_label in ["accuracy", "macro avg", "weighted avg"]:
                        continue

                    row = {
                        "model_name": m.model_name,
                        "class": class_label,
                        "precision": class_metrics.get("precision", 0.0),
                        "recall": class_metrics.get("recall", 0.0),
                        "f1-score": class_metrics.get("f1-score", 0.0),
                        "support": int(class_metrics.get("support", 0)),
                    }
                    data.append(row)

        df = pd.DataFrame(data)

        return df

    def get_confusion_matrices(self, metrics_list: List[ModelMetrics]) -> Dict[str, NDArray[np.int_]]:
        """
        Collect confusion matrices from the provided ModelMetrics objects, keyed by model name.
        
        Args:
            metrics_list: Iterable of ModelMetrics; any model without a confusion_matrix entry is skipped.
        
        Returns:
            matrices (dict): Mapping from model_name to a NumPy integer array of shape (n_classes, n_classes) representing that model's confusion matrix.
        """
        matrices = {}

        for m in metrics_list:
            if m.confusion_matrix:
                matrices[m.model_name] = np.array(m.confusion_matrix, dtype=np.int_)

        return matrices

    def aggregate_metrics_across_models(self, metrics_list: List[ModelMetrics]) -> Dict[str, Any]:
        """
        Aggregate core performance metrics across a collection of models and produce cross-validation-style summary statistics.
        
        Computes per-metric statistics including mean, standard deviation, minimum, maximum, and 95% confidence interval bounds. Includes `accuracy`, `kappa`, `macro_f1`, and `balanced_accuracy` for every model; includes `auc` when present on a model.
        
        Parameters:
            metrics_list (List[ModelMetrics]): List of ModelMetrics instances to aggregate.
        
        Returns:
            Dict[str, Any]: Mapping from metric name to an aggregated-statistics dictionary (e.g., `{"mean": ..., "std": ..., "min": ..., "max": ..., "ci_lower": ..., "ci_upper": ..., "n": ...}`).
        """
        # Collect metrics into dictionaries suitable for aggregate_cv_metrics
        fold_metrics = []

        for m in metrics_list:
            fold = {
                "accuracy": m.accuracy,
                "kappa": m.kappa,
                "macro_f1": m.macro_f1,
                "balanced_accuracy": m.balanced_accuracy,
            }
            if m.auc is not None:
                fold["auc"] = m.auc

            fold_metrics.append(fold)

        # Use existing cross-validation aggregation logic
        aggregated = aggregate_cv_metrics(fold_metrics)

        return aggregated

    def get_side_by_side_comparison(
        self,
        metrics_list: List[ModelMetrics],
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Builds a table comparing multiple models across a selected set of metrics.
        
        Parameters:
            metrics_list (List[ModelMetrics]): Models to include in the comparison.
            metrics (Optional[List[str]]): Metric names to include as columns. If omitted, uses a default set including "accuracy", "kappa", "macro_f1", "auc", "balanced_accuracy", and "val_loss".
        
        Returns:
            pd.DataFrame: A DataFrame with one row per model containing a "rank" column, "model_name", "arch", and the requested metric columns. When present, rows are sorted by `macro_f1` in descending order.
        """
        if metrics is None:
            metrics = [
                "accuracy",
                "kappa",
                "macro_f1",
                "auc",
                "balanced_accuracy",
                "val_loss",
            ]

        data = []
        for m in metrics_list:
            row = {"model_name": m.model_name, "arch": m.arch}

            # Add requested metrics
            for metric_name in metrics:
                value = getattr(m, metric_name, None)
                row[metric_name] = value

            data.append(row)

        df = pd.DataFrame(data)

        # Sort by macro_f1 descending (best models first)
        if "macro_f1" in df.columns:
            df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)

        # Add rank column
        df.insert(0, "rank", range(1, len(df) + 1))

        return df

    def rank_models_by_metric(
        self,
        metrics_list: List[ModelMetrics],
        metric: str = "macro_f1",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Produce a DataFrame ranking models by a specified metric.
        
        Parameters:
            metrics_list (List[ModelMetrics]): List of ModelMetrics to evaluate.
            metric (str): Name of the metric attribute on ModelMetrics to rank by (default: "macro_f1").
            ascending (bool): If True, sort from lowest to highest; if False, sort from highest to lowest.
        
        Returns:
            pd.DataFrame: DataFrame containing columns `rank`, `model_name`, `arch`, and the requested metric value,
            sorted according to `ascending`. Models missing the metric are skipped (a warning is emitted).
        """
        data = []
        for m in metrics_list:
            value = getattr(m, metric, None)
            if value is None:
                LOGGER.warning("Model '%s' does not have metric '%s'", m.model_name, metric)
                continue

            row = {
                "model_name": m.model_name,
                "arch": m.arch,
                metric: value,
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Sort by metric
        df = df.sort_values(metric, ascending=ascending).reset_index(drop=True)

        # Add rank column
        df.insert(0, "rank", range(1, len(df) + 1))

        return df

    def get_best_model(
        self,
        metrics_list: List[ModelMetrics],
        metric: str = "macro_f1",
    ) -> ModelMetrics:
        """
        Selects the ModelMetrics with the highest value for a specified metric.
        
        If multiple models tie for the top value, the first encountered in metrics_list is returned.
        
        Parameters:
            metrics_list (List[ModelMetrics]): Sequence of ModelMetrics to evaluate.
            metric (str): Attribute name on ModelMetrics to use for ranking (e.g., "macro_f1", "accuracy").
        
        Returns:
            ModelMetrics: The model whose specified metric has the highest numeric value.
        
        Raises:
            ValueError: If metrics_list is empty or if no model in metrics_list has a valid value for the specified metric.
        """
        if not metrics_list:
            raise ValueError("metrics_list cannot be empty")

        # Find model with highest metric value
        best_model = None
        best_value = -float("inf")

        for m in metrics_list:
            value = getattr(m, metric, None)
            if value is None:
                LOGGER.warning("Model '%s' does not have metric '%s'", m.model_name, metric)
                continue

            if value > best_value:
                best_value = value
                best_model = m

        if best_model is None:
            raise ValueError(f"No models found with valid '{metric}' metric")

        LOGGER.info(
            "Best model by %s: '%s' (%.4f)",
            metric,
            best_model.model_name,
            best_value,
        )

        return best_model

    def compare_models_pairwise(
        self,
        model_a: ModelMetrics,
        model_b: ModelMetrics,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare two ModelMetrics instances across a set of numeric metrics and report per-metric values, differences, percent changes, and the winner.
        
        Parameters:
            model_a: First model to compare.
            model_b: Second model to compare.
            metrics: Optional list of metric attribute names to compare (defaults to ["accuracy", "kappa", "macro_f1", "balanced_accuracy"]). Only metrics that exist for both models are considered; missing metrics are skipped with a warning.
        
        Returns:
            A dictionary keyed by metric name. Each value is a dictionary with:
              - "model_a": numeric value for model_a,
              - "model_b": numeric value for model_b,
              - "difference": value_a - value_b,
              - "percent_change": (difference / value_b) * 100, or 0.0 if model_b's value is zero,
              - "winner": the model_name of the model with the higher value (ties resolved in favor of model_b).
        """
        if metrics is None:
            metrics = ["accuracy", "kappa", "macro_f1", "balanced_accuracy"]

        comparison = {}

        for metric_name in metrics:
            value_a = getattr(model_a, metric_name, None)
            value_b = getattr(model_b, metric_name, None)

            if value_a is None or value_b is None:
                LOGGER.warning(
                    "Skipping metric '%s': not available for both models",
                    metric_name,
                )
                continue

            difference = value_a - value_b
            percent_change = (difference / value_b * 100) if value_b != 0 else 0.0

            comparison[metric_name] = {
                "model_a": value_a,
                "model_b": value_b,
                "difference": difference,
                "percent_change": percent_change,
                "winner": model_a.model_name if value_a > value_b else model_b.model_name,
            }

        return comparison

    def get_metrics_summary_table(
        self,
        metrics_list: List[ModelMetrics],
    ) -> pd.DataFrame:
        """
        Produce a table summarizing models' performance metrics, training hyperparameters, and dataset metadata.
        
        Parameters:
            metrics_list (List[ModelMetrics]): Sequence of ModelMetrics objects to include in the summary.
        
        Returns:
            pd.DataFrame: A DataFrame with one row per model containing columns such as
            model_name, arch, dataset, accuracy, kappa, macro_f1, auc, balanced_accuracy,
            val_loss, num_epochs, best_epoch, batch_size, learning_rate, img_size, and
            num_classes. The rows are sorted by `macro_f1` in descending order.
        """
        data = []

        for m in metrics_list:
            row = {
                # Model identification
                "model_name": m.model_name,
                "arch": m.arch,
                "dataset": m.dataset,
                # Performance metrics
                "accuracy": m.accuracy,
                "kappa": m.kappa,
                "macro_f1": m.macro_f1,
                "auc": m.auc,
                "balanced_accuracy": m.balanced_accuracy,
                "val_loss": m.val_loss,
                # Training info
                "num_epochs": m.num_epochs,
                "best_epoch": m.best_epoch,
                "batch_size": m.batch_size,
                "learning_rate": m.learning_rate,
                "img_size": m.img_size,
                "num_classes": m.num_classes,
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Sort by macro_f1 descending
        df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)

        return df

    def export_comparison_report(
        self,
        metrics_list: List[ModelMetrics],
        output_path: Union[str, Path],
    ) -> None:
        """
        Export a comprehensive JSON report summarizing and comparing multiple models.
        
        The report file includes:
        - summary: overall counts and the best model by `macro_f1`
        - aggregated_metrics: aggregated statistics across models
        - model_comparison: side-by-side comparison table (list of records)
        - per_class_metrics: per-class performance records
        - individual_models: selected fields for each model (arch, accuracy, kappa, macro_f1, auc, balanced_accuracy, confusion_matrix)
        
        Parameters:
            metrics_list (List[ModelMetrics]): Models to include in the report.
            output_path (Union[str, Path]): File path where the JSON report will be written.
        
        Example:
            >>> engine.export_comparison_report(metrics, "comparison_report.json")
        """
        output_path = Path(output_path)

        # Prepare comparison data
        comparison_df = self.get_side_by_side_comparison(metrics_list)
        per_class_df = self.get_per_class_metrics(metrics_list)
        aggregated = self.aggregate_metrics_across_models(metrics_list)
        best_model = self.get_best_model(metrics_list, metric="macro_f1")

        report = {
            "summary": {
                "num_models": len(metrics_list),
                "model_names": [m.model_name for m in metrics_list],
                "best_model": best_model.model_name,
                "best_macro_f1": best_model.macro_f1,
            },
            "aggregated_metrics": aggregated,
            "model_comparison": comparison_df.to_dict(orient="records"),
            "per_class_metrics": per_class_df.to_dict(orient="records"),
            "individual_models": [
                {
                    "model_name": m.model_name,
                    "arch": m.arch,
                    "accuracy": m.accuracy,
                    "kappa": m.kappa,
                    "macro_f1": m.macro_f1,
                    "auc": m.auc,
                    "balanced_accuracy": m.balanced_accuracy,
                    "confusion_matrix": m.confusion_matrix,
                }
                for m in metrics_list
            ],
        }

        # Save to JSON
        with output_path.open("w") as f:
            json.dump(report, f, indent=2)

        LOGGER.info("Exported comparison report to %s", output_path)

    def load_predictions(
        self,
        run_path: Path,
    ) -> Tuple[NDArray[np.integer], NDArray[np.integer]]:
        """
        Load validation predictions and ground truth labels from checkpoint.

        Args:
            run_path: Path to model run directory (should contain val_predictions.csv)

        Returns:
            Tuple of (y_true, y_pred) as numpy arrays

        Raises:
            FileNotFoundError: If val_predictions.csv doesn't exist
            ValueError: If CSV is malformed or missing required columns

        Example:
            >>> engine = ModelComparisonEngine([Path("outputs/run1")])
            >>> y_true, y_pred = engine.load_predictions(Path("outputs/run1"))
            >>> print(f"Loaded {len(y_true)} predictions")
            Loaded 200 predictions
        """
        pred_path = run_path / "val_predictions.csv"

        if not pred_path.exists():
            raise FileNotFoundError(
                f"Prediction file not found: {pred_path}\n"
                f"Model checkpoints must save val_predictions.csv for statistical tests.\n"
                f"Update training pipeline to save predictions."
            )

        try:
            df = pd.read_csv(pred_path)
        except Exception as e:
            raise ValueError(f"Failed to read {pred_path}: {e}") from e

        # Validate required columns
        required_cols = ["true_label", "prediction"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"val_predictions.csv missing required columns: {missing}\n"
                f"Found columns: {list(df.columns)}"
            )

        y_true = df["true_label"].values.astype(np.int_)
        y_pred = df["prediction"].values.astype(np.int_)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Mismatched lengths: y_true={len(y_true)}, y_pred={len(y_pred)}"
            )

        if len(y_true) == 0:
            raise ValueError("Empty predictions file")

        LOGGER.debug(
            "Loaded predictions from %s: %d samples",
            pred_path,
            len(y_true),
        )

        return y_true, y_pred

    def __repr__(self) -> str:
        """
        Return a concise string representation of the engine including the number of configured runs and their model names.
        
        Returns:
            repr_str (str): String formatted as "ModelComparisonEngine(n_models=<n>, models=<model_names>)".
        """
        return (
            f"ModelComparisonEngine(n_models={len(self.run_paths)}, "
            f"models={self.model_names})"
        )


def format_statistical_summary(
    aggregated: Dict[str, Dict[str, float]],
    decimal_places: int = 4,
    title: str = "Model Comparison Statistical Summary",
) -> str:
    """
    Produce a human-readable multi-line summary of aggregated metrics including 95% confidence intervals.
    
    Formats an aggregated-metrics mapping into lines of the form:
        metric_name:  mean ± std  [ci_lower, ci_upper]
    Lines are aligned by metric name and the header includes the provided title with "(95% CI)".
    
    Parameters:
        aggregated (dict): Mapping from metric name to statistics dict with keys
            "mean", "std", "ci_lower", and "ci_upper". Example:
            {
                "accuracy": {"mean": 0.82, "std": 0.015, "ci_lower": 0.805, "ci_upper": 0.835},
                ...
            }
        decimal_places (int): Number of decimal places to display for numeric values (default: 4).
        title (str): Header title for the summary (default: "Model Comparison Statistical Summary").
    
    Returns:
        formatted_summary (str): Multi-line string containing the titled header and one aligned line per metric.
    """
    if not aggregated:
        return f"{title}:\n{'=' * len(title)}\n(No metrics available)"

    # Build header
    header = f"{title} (95% CI):"
    separator = "=" * len(header)
    lines = [header, separator]

    # Find longest metric name for alignment
    max_name_len = max((len(name) for name in aggregated.keys()), default=0)

    # Format each metric with alignment
    for metric_name in sorted(aggregated.keys()):
        stats = aggregated[metric_name]

        mean = stats["mean"]
        std = stats["std"]
        ci_lower = stats["ci_lower"]
        ci_upper = stats["ci_upper"]

        # Format with proper alignment
        name_padded = metric_name.ljust(max_name_len)
        mean_str = f"{mean:.{decimal_places}f}"
        std_str = f"{std:.{decimal_places}f}"
        ci_str = f"[{ci_lower:.{decimal_places}f}, {ci_upper:.{decimal_places}f}]"

        line = f"{name_padded}:  {mean_str} ± {std_str}  {ci_str}"
        lines.append(line)

    return "\n".join(lines)


def create_metrics_comparison_table(
    metrics_list: List[ModelMetrics],
    metrics: Optional[List[str]] = None,
    show_rank: bool = True,
    title: str = "Model Metrics Comparison",
) -> go.Figure:
    """
    Create an interactive Plotly table that compares selected performance metrics for multiple models side-by-side.
    
    Parameters:
        metrics_list (List[ModelMetrics]): Models to include in the comparison.
        metrics (Optional[List[str]]): Metric names to display. Default: ['accuracy', 'kappa', 'macro_f1', 'auc', 'balanced_accuracy', 'val_loss'].
        show_rank (bool): If True, include a rank column (sorted by macro_f1). Default: True.
        title (str): Title displayed above the table. Default: "Model Metrics Comparison".
    
    Returns:
        go.Figure: A Plotly Figure containing a styled, interactive table of the requested metrics.
    
    Raises:
        ImportError: If Plotly is not available.
        ValueError: If metrics_list is empty.
    """
    _require_plotly()

    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")

    if metrics is None:
        metrics = [
            "accuracy",
            "kappa",
            "macro_f1",
            "auc",
            "balanced_accuracy",
            "val_loss",
        ]

    # Prepare data
    data = []
    for m in metrics_list:
        row = {"model_name": m.model_name, "arch": m.arch}
        for metric_name in metrics:
            value = getattr(m, metric_name, None)
            row[metric_name] = value
        data.append(row)

    df = pd.DataFrame(data)

    # Sort by macro_f1 descending (best models first)
    if "macro_f1" in df.columns:
        df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    # Add rank column if requested
    if show_rank:
        df.insert(0, "rank", range(1, len(df) + 1))

    # Prepare table data
    header_values = [col.replace("_", " ").title() for col in df.columns]
    cell_values = []

    for col in df.columns:
        if col in ["model_name", "arch", "rank"]:
            # String columns - no formatting
            cell_values.append(df[col].tolist())
        else:
            # Numeric columns - format to 4 decimal places
            formatted = []
            for val in df[col]:
                if val is None:
                    formatted.append("N/A")
                elif isinstance(val, (int, float)):
                    formatted.append(f"{val:.4f}")
                else:
                    formatted.append(str(val))
            cell_values.append(formatted)

    # Color coding for metrics (normalize to [0, 1] range)
    cell_colors = []
    for col in df.columns:
        if col in metrics and col != "val_loss":
            # Higher is better - green gradient
            values = df[col].fillna(0).values
            if values.max() > values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
            else:
                normalized = np.ones_like(values)
            # Green color gradient
            colors = [f"rgba(144, 238, 144, {0.3 + 0.7 * v})" for v in normalized]
            cell_colors.append(colors)
        elif col == "val_loss":
            # Lower is better - red gradient (inverted)
            values = df[col].fillna(0).values
            if values.max() > values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
            else:
                normalized = np.ones_like(values)
            # Red color gradient (inverted)
            colors = [f"rgba(255, 182, 193, {0.3 + 0.7 * (1 - v)})" for v in normalized]
            cell_colors.append(colors)
        else:
            # No color coding for non-metric columns
            cell_colors.append(["white"] * len(df))

    # Create table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header_values,
                    fill_color="paleturquoise",
                    align="center",
                    font=dict(size=12, color="black", family="Arial Black"),
                    height=30,
                ),
                cells=dict(
                    values=cell_values,
                    fill_color=cell_colors,
                    align=["center"] * len(df.columns),
                    font=dict(size=11, color="black"),
                    height=25,
                ),
            )
        ]
    )

    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "family": "Arial Black"},
        },
        margin=dict(l=20, r=20, t=60, b=20),
        height=min(400, 100 + 30 * len(df)),
    )

    LOGGER.info("Created metrics comparison table with %d models", len(metrics_list))

    return fig


def create_metrics_comparison_chart(
    metrics_list: List[ModelMetrics],
    metrics: Optional[List[str]] = None,
    chart_type: str = "grouped",
    title: str = "Model Metrics Comparison",
    show_values: bool = True,
) -> go.Figure:
    """
    Render an interactive Plotly bar chart comparing selected performance metrics across multiple models.
    
    Supports "grouped" (side-by-side) and "stacked" bar layouts. By default compares core metrics ['accuracy', 'kappa', 'macro_f1', 'balanced_accuracy'] (excludes inverse metrics such as val_loss). Charts display metric values and sort models by macro_f1 when available.
    
    Parameters:
        metrics_list: List of ModelMetrics objects to include in the chart.
        metrics: Optional list of metric names to plot. Defaults to ['accuracy', 'kappa', 'macro_f1', 'balanced_accuracy'].
        chart_type: Either "grouped" or "stacked" to select bar layout. Default is "grouped".
        title: Chart title.
        show_values: If True, annotate bars with numeric values.
    
    Returns:
        Plotly Figure containing the interactive bar chart.
    
    Raises:
        ImportError: If plotly is not installed.
        ValueError: If metrics_list is empty or chart_type is not one of "grouped" or "stacked".
    """
    _require_plotly()

    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")

    if chart_type not in ["grouped", "stacked"]:
        raise ValueError(f"chart_type must be 'grouped' or 'stacked', got '{chart_type}'")

    if metrics is None:
        # Default metrics (exclude val_loss as it's inverse)
        metrics = ["accuracy", "kappa", "macro_f1", "balanced_accuracy"]

    # Prepare data
    data = []
    for m in metrics_list:
        row = {"model_name": m.model_name}
        for metric_name in metrics:
            value = getattr(m, metric_name, None)
            row[metric_name] = value if value is not None else 0.0
        data.append(row)

    df = pd.DataFrame(data)

    # Sort by macro_f1 if available
    if "macro_f1" in df.columns:
        df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    # Create bar chart
    fig = go.Figure()

    # Color palette for metrics
    colors = px.colors.qualitative.Set2

    if chart_type == "grouped":
        # Grouped bar chart (side-by-side bars for each metric)
        for idx, metric_name in enumerate(metrics):
            if metric_name not in df.columns:
                LOGGER.warning("Metric '%s' not found in data, skipping", metric_name)
                continue

            values = df[metric_name].values
            color = colors[idx % len(colors)]

            fig.add_trace(
                go.Bar(
                    name=metric_name.replace("_", " ").title(),
                    x=df["model_name"],
                    y=values,
                    marker_color=color,
                    text=[f"{v:.3f}" for v in values] if show_values else None,
                    textposition="outside" if show_values else None,
                    textfont=dict(size=10),
                )
            )

        fig.update_layout(
            barmode="group",
            bargap=0.15,
            bargroupgap=0.1,
        )

    else:  # stacked
        # Stacked bar chart
        for idx, metric_name in enumerate(metrics):
            if metric_name not in df.columns:
                LOGGER.warning("Metric '%s' not found in data, skipping", metric_name)
                continue

            values = df[metric_name].values
            color = colors[idx % len(colors)]

            fig.add_trace(
                go.Bar(
                    name=metric_name.replace("_", " ").title(),
                    x=df["model_name"],
                    y=values,
                    marker_color=color,
                    text=[f"{v:.3f}" for v in values] if show_values else None,
                    textposition="inside" if show_values else None,
                    textfont=dict(size=9),
                )
            )

        fig.update_layout(
            barmode="stack",
        )

    # Update layout
    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "family": "Arial Black"},
        },
        xaxis_title="Model",
        yaxis_title="Metric Value",
        yaxis=dict(range=[0, 1.05]),  # Metrics are typically in [0, 1] range
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=60, r=150, t=80, b=60),
        height=500,
        hovermode="x unified",
    )

    # Update axes
    fig.update_xaxes(
        tickangle=-45 if len(metrics_list) > 5 else 0,
        tickfont=dict(size=10),
    )

    LOGGER.info(
        "Created %s bar chart with %d models and %d metrics",
        chart_type,
        len(metrics_list),
        len(metrics),
    )

    return fig


def create_per_class_comparison(
    metrics_list: List[ModelMetrics],
    metric: str = "f1-score",
    chart_type: str = "grouped",
    title: Optional[str] = None,
    show_values: bool = True,
) -> go.Figure:
    """
    Create an interactive Plotly bar chart comparing a per-class metric across multiple models.
    
    Generates either grouped (side-by-side) or stacked bars showing per-class values for `precision`, `recall`, or `f1-score` extracted from each ModelMetrics object's classification report. Missing class/model combinations are filled with 0.0 to preserve alignment; classes are sorted numerically. An automatic title is created when `title` is None.
    
    Parameters:
        metrics_list (List[ModelMetrics]): Models to include in the comparison.
        metric (str): Per-class metric to plot. One of "f1-score", "precision", "recall".
        chart_type (str): "grouped" for side-by-side bars or "stacked" for stacked bars.
        title (Optional[str]): Custom chart title; auto-generated if None.
        show_values (bool): If True, display numeric values on the bars.
    
    Returns:
        go.Figure: A Plotly Figure containing the per-class comparison chart.
    
    Raises:
        ImportError: If Plotly is not available.
        ValueError: If `metrics_list` is empty, `metric` is invalid, or no per-class metrics are present.
    """
    _require_plotly()

    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")

    valid_metrics = ["f1-score", "precision", "recall"]
    if metric not in valid_metrics:
        raise ValueError(
            f"Invalid metric '{metric}'. Must be one of: {valid_metrics}"
        )

    # Extract per-class metrics using static method
    per_class_df = ModelComparisonEngine.get_per_class_metrics(metrics_list)

    if per_class_df.empty:
        raise ValueError(
            "No per-class metrics available. Ensure models have classification_report data."
        )

    # Filter to selected metric
    metric_df = per_class_df[["model_name", "class", metric]].copy()

    # Sort classes for consistent ordering
    metric_df["class"] = pd.to_numeric(metric_df["class"], errors="coerce")
    metric_df = metric_df.dropna(subset=["class"])
    metric_df = metric_df.sort_values(["class", "model_name"]).reset_index(drop=True)

    # Get unique classes and models
    unique_classes = sorted(metric_df["class"].unique())
    unique_models = metric_df["model_name"].unique().tolist()

    # Auto-generate title if not provided
    if title is None:
        title = f"Per-Class {metric.replace('-', ' ').title()} Comparison"

    # Create figure
    fig = go.Figure()

    # Color palette for models
    colors = px.colors.qualitative.Set2

    if chart_type == "grouped":
        # Grouped bar chart (side-by-side bars for each model)
        for idx, model_name in enumerate(unique_models):
            model_data = metric_df[metric_df["model_name"] == model_name]

            # Ensure all classes are represented (fill missing with 0)
            class_values = []
            for class_label in unique_classes:
                class_row = model_data[model_data["class"] == class_label]
                if not class_row.empty:
                    class_values.append(class_row[metric].iloc[0])
                else:
                    class_values.append(0.0)

            color = colors[idx % len(colors)]

            fig.add_trace(
                go.Bar(
                    name=model_name,
                    x=[f"Class {int(c)}" for c in unique_classes],
                    y=class_values,
                    marker_color=color,
                    text=[f"{v:.3f}" for v in class_values] if show_values else None,
                    textposition="outside" if show_values else None,
                    textfont=dict(size=10),
                    hovertemplate=(
                        f"<b>{model_name}</b><br>"
                        f"Class: %{{x}}<br>"
                        f"{metric.title()}: %{{y:.4f}}<br>"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            barmode="group",
            bargap=0.15,
            bargroupgap=0.1,
        )

    else:  # stacked
        # Stacked bar chart
        for idx, model_name in enumerate(unique_models):
            model_data = metric_df[metric_df["model_name"] == model_name]

            # Ensure all classes are represented (fill missing with 0)
            class_values = []
            for class_label in unique_classes:
                class_row = model_data[model_data["class"] == class_label]
                if not class_row.empty:
                    class_values.append(class_row[metric].iloc[0])
                else:
                    class_values.append(0.0)

            color = colors[idx % len(colors)]

            fig.add_trace(
                go.Bar(
                    name=model_name,
                    x=[f"Class {int(c)}" for c in unique_classes],
                    y=class_values,
                    marker_color=color,
                    text=[f"{v:.3f}" for v in class_values] if show_values else None,
                    textposition="inside" if show_values else None,
                    textfont=dict(size=9),
                    hovertemplate=(
                        f"<b>{model_name}</b><br>"
                        f"Class: %{{x}}<br>"
                        f"{metric.title()}: %{{y:.4f}}<br>"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            barmode="stack",
        )

    # Update layout
    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "family": "Arial Black"},
        },
        xaxis_title="Density Class (BI-RADS)",
        yaxis_title=metric.replace("-", " ").title(),
        yaxis=dict(range=[0, 1.05]),  # Metrics are typically in [0, 1] range
        legend=dict(
            title="Model",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=60, r=150, t=80, b=60),
        height=500,
        hovermode="x unified",
    )

    # Update axes
    fig.update_xaxes(
        tickfont=dict(size=11),
    )

    LOGGER.info(
        "Created per-class %s comparison with %d models and %d classes",
        metric,
        len(unique_models),
        len(unique_classes),
    )

    return fig


def create_confusion_matrix_comparison(
    metrics_list: List[ModelMetrics],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix Comparison",
    colorscale: str = "Blues",
) -> go.Figure:
    """
    Create a subplot grid of confusion matrix heatmaps to compare multiple models' prediction patterns.
    
    Each subplot shows a model's confusion matrix (optionally row-normalized to per-class recall) with class labels on both axes and interactive hover text. All models must provide a confusion_matrix of the same square shape.
    
    Parameters:
        metrics_list (List[ModelMetrics]): Models to include; each must have a square confusion_matrix.
        class_names (Optional[List[str]]): Labels for classes in display order. If None, numeric labels "0", "1", ... are used.
        normalize (bool): If True, normalize each confusion matrix by true-label row sums (showing per-class recall); if False, show raw counts.
        title (str): Overall figure title.
        colorscale (str): Plotly colorscale name for heatmaps.
    
    Returns:
        go.Figure: A Plotly Figure containing a grid of confusion matrix heatmaps, one subplot per model.
    
    Raises:
        ImportError: If Plotly is not available.
        ValueError: If metrics_list is empty, a model lacks a confusion_matrix, or confusion matrices differ in size.
    """
    _require_plotly()

    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")

    # Extract confusion matrices and validate
    matrices = []
    model_names = []
    num_classes = None

    for m in metrics_list:
        if not m.confusion_matrix:
            raise ValueError(
                f"Model '{m.model_name}' has no confusion matrix. "
                "Ensure models were trained with metrics tracking enabled."
            )

        cm = np.array(m.confusion_matrix, dtype=np.float64)

        # Validate squareness
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            raise ValueError(
                f"Confusion matrix for model '{m.model_name}' is not square: "
                f"got shape {cm.shape}"
            )

        # Validate shape consistency
        if num_classes is None:
            num_classes = cm.shape[0]
        elif cm.shape[0] != num_classes:
            raise ValueError(
                f"Inconsistent confusion matrix sizes: expected {num_classes}x{num_classes}, "
                f"got {cm.shape[0]}x{cm.shape[0]} for model '{m.model_name}'"
            )

        # Normalize if requested (by rows = true labels)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm = cm / row_sums

        matrices.append(cm)
        model_names.append(m.model_name)

    # Prepare class labels
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    elif len(class_names) != num_classes:
        raise ValueError(
            f"Number of class_names ({len(class_names)}) must match "
            f"number of classes ({num_classes})"
        )

    # Determine subplot grid dimensions (try to keep roughly square)
    n_models = len(metrics_list)
    if n_models <= 2:
        n_cols = n_models
        n_rows = 1
    elif n_models <= 4:
        n_cols = 2
        n_rows = (n_models + 1) // 2
    elif n_models <= 6:
        n_cols = 3
        n_rows = (n_models + 2) // 3
    else:
        n_cols = 3
        n_rows = (n_models + 2) // 3

    # Create subplot grid
    subplot_titles = [f"{name}" for name in model_names]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15 / n_rows if n_rows > 1 else 0.1,
        horizontal_spacing=0.12 / n_cols if n_cols > 1 else 0.1,
    )

    # Add heatmap for each model
    for idx, (cm, model_name) in enumerate(zip(matrices, model_names)):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Format hover text with appropriate precision
        if normalize:
            hover_text = [
                [
                    f"True: {class_names[i]}<br>"
                    f"Pred: {class_names[j]}<br>"
                    f"Rate: {cm[i, j]:.3f}"
                    for j in range(num_classes)
                ]
                for i in range(num_classes)
            ]
            text_values = [[f"{val:.2f}" for val in row] for row in cm]
        else:
            hover_text = [
                [
                    f"True: {class_names[i]}<br>"
                    f"Pred: {class_names[j]}<br>"
                    f"Count: {int(cm[i, j])}"
                    for j in range(num_classes)
                ]
                for i in range(num_classes)
            ]
            text_values = [[f"{int(val)}" for val in row] for row in cm]

        # Create heatmap trace
        heatmap = go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            text=text_values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale=colorscale,
            showscale=(idx == 0),  # Only show colorbar for first subplot
            hovertext=hover_text,
            hoverinfo="text",
            colorbar=dict(
                title="Proportion" if normalize else "Count",
                len=0.7,
                x=1.02,
            ) if idx == 0 else None,
        )

        fig.add_trace(heatmap, row=row, col=col)

        # Update axes for this subplot
        fig.update_xaxes(
            title_text="Predicted Label" if row == n_rows else "",
            tickmode="array",
            tickvals=list(range(num_classes)),
            ticktext=class_names,
            side="bottom",
            row=row,
            col=col,
        )

        fig.update_yaxes(
            title_text="True Label" if col == 1 else "",
            tickmode="array",
            tickvals=list(range(num_classes)),
            ticktext=class_names,
            autorange="reversed",  # Standard confusion matrix orientation
            row=row,
            col=col,
        )

    # Update overall layout
    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "family": "Arial Black"},
        },
        height=max(400, 300 * n_rows),
        width=max(600, 400 * n_cols),
        showlegend=False,
        margin=dict(l=80, r=120, t=100, b=60),
    )

    LOGGER.info(
        "Created confusion matrix comparison for %d models (%dx%d grid, normalize=%s)",
        n_models,
        n_rows,
        n_cols,
        normalize,
    )

    return fig


def export_comparison_table(
    df: pd.DataFrame,
    base_path: Union[str, Path],
    formats: Optional[List[str]] = None,
    index: bool = True,
    float_format: str = "%.4f",
) -> List[Path]:
    """Export comparison table to multiple publication-ready formats.

    Exports pandas DataFrame containing model comparison metrics to various
    table formats suitable for publications, reports, and documentation.

    Args:
        df: Pandas DataFrame to export (typically from get_metrics_dataframe or
            get_side_by_side_comparison)
        base_path: Base path without extension (e.g., "outputs/comparison_table")
        formats: List of formats to export. Defaults to ['csv', 'xlsx', 'json', 'md']
                 Supported: 'csv', 'xlsx', 'json', 'md' (markdown), 'tex' (LaTeX)
        index: Whether to include DataFrame index in exported files (default: True)
        float_format: Format string for floating point numbers (default: "%.4f")

    Returns:
        List of Path objects for successfully exported files

    Raises:
        ValueError: If df is empty or formats list is empty

    Example:
        >>> metrics_df = engine.get_metrics_dataframe(metrics_list)
        >>> paths = export_comparison_table(
        ...     metrics_df,
        ...     "outputs/model_comparison",
        ...     formats=['csv', 'xlsx', 'md']
        ... )
        >>> print(f"Exported to: {[str(p) for p in paths]}")
        Exported to: ['outputs/model_comparison.csv', ...]

    Note:
        - CSV: Plain text, widely compatible
        - XLSX: Excel format, requires openpyxl
        - JSON: Structured data, machine-readable
        - MD: Markdown table for documentation
        - TEX: LaTeX table for academic papers
    """
    if df.empty:
        raise ValueError("Cannot export empty DataFrame")

    if formats is None:
        formats = ['csv', 'xlsx', 'json', 'md']

    if not formats:
        raise ValueError("formats list cannot be empty")

    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    exported_paths: List[Path] = []

    for fmt in formats:
        fmt_lower = fmt.lower()
        out_path = base_path.with_suffix(f".{fmt_lower}")

        try:
            if fmt_lower == 'csv':
                df.to_csv(out_path, index=index, float_format=float_format)
                exported_paths.append(out_path)
                LOGGER.info(f"Exported table to CSV: {out_path}")

            elif fmt_lower == 'xlsx':
                try:
                    df.to_excel(out_path, index=index, float_format=float_format)
                    exported_paths.append(out_path)
                    LOGGER.info(f"Exported table to Excel: {out_path}")
                except ImportError:
                    LOGGER.warning(
                        "openpyxl not available, skipping XLSX export. "
                        "Install with: pip install openpyxl"
                    )

            elif fmt_lower == 'json':
                # Export as records format for better readability
                df.to_json(out_path, orient='records', indent=2)
                exported_paths.append(out_path)
                LOGGER.info(f"Exported table to JSON: {out_path}")

            elif fmt_lower == 'md' or fmt_lower == 'markdown':
                # Use .md extension for markdown
                if fmt_lower == 'markdown':
                    out_path = base_path.with_suffix('.md')

                md_content = df.to_markdown(index=index, floatfmt=float_format)
                out_path.write_text(md_content, encoding='utf-8')
                exported_paths.append(out_path)
                LOGGER.info(f"Exported table to Markdown: {out_path}")

            elif fmt_lower == 'tex' or fmt_lower == 'latex':
                # Use .tex extension for LaTeX
                if fmt_lower == 'latex':
                    out_path = base_path.with_suffix('.tex')

                latex_content = df.to_latex(
                    index=index,
                    float_format=float_format,
                    caption="Model Comparison Table",
                    label="tab:model_comparison",
                )
                out_path.write_text(latex_content, encoding='utf-8')
                exported_paths.append(out_path)
                LOGGER.info(f"Exported table to LaTeX: {out_path}")

            else:
                LOGGER.warning(f"Unsupported format '{fmt}', skipping. "
                             f"Supported: csv, xlsx, json, md, tex")

        except Exception as e:
            LOGGER.error(f"Failed to export table as {fmt}: {e}")

    if not exported_paths:
        LOGGER.warning("No tables were successfully exported")

    return exported_paths
