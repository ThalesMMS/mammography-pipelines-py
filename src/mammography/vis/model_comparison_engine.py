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

from mammography.vis.model_comparison_stats import ModelMetrics, _resolve_metrics_json_path

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
        Produce a tabular summary of key metrics and metadata for each model.
        
        Parameters:
            metrics_list (List[ModelMetrics]): Sequence of ModelMetrics objects to convert.
        
        Returns:
            pd.DataFrame: One row per model containing columns
            `model_name, arch, dataset, accuracy, kappa, macro_f1, auc, balanced_accuracy, val_loss, num_epochs, best_epoch, num_classes, img_size, batch_size, learning_rate`, sorted by `macro_f1` descending.
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

        if not data:
            return pd.DataFrame(columns=["rank", "model_name", "arch", metric])

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
        Create a one-row-per-model summary table of performance metrics, training hyperparameters, and dataset metadata.
        
        The resulting DataFrame contains columns including: model_name, arch, dataset, accuracy, kappa, macro_f1, auc, balanced_accuracy, val_loss, num_epochs, best_epoch, batch_size, learning_rate, img_size, and num_classes. Rows are sorted by `macro_f1` in descending order.
         
        Returns:
            pd.DataFrame: Summary table with one row per ModelMetrics entry, sorted by `macro_f1` (highest first).
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
        Create a JSON file that summarizes and compares a list of trained models.
        
        The JSON report contains:
        - summary: count of models, list of model names, the best model by `macro_f1`, and its `macro_f1` value
        - aggregated_metrics: cross-model aggregated statistics
        - model_comparison: side-by-side comparison rows for requested metrics
        - per_class_metrics: per-class precision/recall/f1/support rows
        - individual_models: selected fields for each model including confusion matrices
        
        Parameters:
            metrics_list (List[ModelMetrics]): Models to include in the report.
            output_path (Union[str, Path]): Destination file path for the JSON report.
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
