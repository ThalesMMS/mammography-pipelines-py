# DISCLAIMER: Educational project only - NOT for clinical or medical diagnostic purposes.
"""
Unit tests for model comparison utilities.

These tests validate model comparison functionality including checkpoint loading,
metrics aggregation, statistical tests, and visualization generation.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit

from mammography.vis.model_comparison import (
    ModelComparisonEngine,
    ModelMetrics,
    compute_mcnemar_test,
    export_comparison_table,
    format_statistical_summary,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_model_metrics() -> ModelMetrics:
    """Create a sample ModelMetrics object for testing."""
    return ModelMetrics(
        model_name="test_model_efficientnet",
        run_path=Path("outputs/test_run"),
        arch="efficientnet_b0",
        dataset="mamografias",
        accuracy=0.8500,
        kappa=0.8200,
        macro_f1=0.8320,
        auc=0.8900,
        balanced_accuracy=0.8480,
        confusion_matrix=[[45, 5, 0, 0], [3, 42, 5, 0], [0, 4, 38, 8], [0, 0, 6, 44]],
        classification_report={
            "0": {"precision": 0.94, "recall": 0.90, "f1-score": 0.92, "support": 50},
            "1": {"precision": 0.82, "recall": 0.84, "f1-score": 0.83, "support": 50},
            "2": {"precision": 0.77, "recall": 0.76, "f1-score": 0.77, "support": 50},
            "3": {"precision": 0.85, "recall": 0.88, "f1-score": 0.86, "support": 50},
        },
        val_loss=0.4500,
        num_epochs=20,
        best_epoch=15,
        num_classes=4,
        img_size=224,
        batch_size=32,
        learning_rate=0.001,
    )


@pytest.fixture
def sample_model_metrics_list() -> List[ModelMetrics]:
    """
    Create a list of sample ModelMetrics instances representing different model runs for comparison tests.
    
    The returned list contains three representative models (efficientnet_b0, resnet50, vit_base) populated with classification metrics (accuracy, kappa, macro_f1, auc, balanced_accuracy), confusion matrices, per-class classification reports, and training/configuration fields useful for unit tests.
    
    Returns:
        List[ModelMetrics]: A list of three ModelMetrics objects prepared for comparison and aggregation tests.
    """
    metrics = [
        ModelMetrics(
            model_name="efficientnet_b0",
            run_path=Path("outputs/run_effnet"),
            arch="efficientnet_b0",
            dataset="mamografias",
            accuracy=0.8500,
            kappa=0.8200,
            macro_f1=0.8320,
            auc=0.8900,
            balanced_accuracy=0.8480,
            confusion_matrix=[[45, 5, 0, 0], [3, 42, 5, 0], [0, 4, 38, 8], [0, 0, 6, 44]],
            classification_report={
                "0": {"precision": 0.94, "recall": 0.90, "f1-score": 0.92, "support": 50},
                "1": {"precision": 0.82, "recall": 0.84, "f1-score": 0.83, "support": 50},
                "2": {"precision": 0.77, "recall": 0.76, "f1-score": 0.77, "support": 50},
                "3": {"precision": 0.85, "recall": 0.88, "f1-score": 0.86, "support": 50},
            },
            val_loss=0.4500,
            num_epochs=20,
            best_epoch=15,
            num_classes=4,
            img_size=224,
            batch_size=32,
            learning_rate=0.001,
        ),
        ModelMetrics(
            model_name="resnet50",
            run_path=Path("outputs/run_resnet"),
            arch="resnet50",
            dataset="mamografias",
            accuracy=0.8400,
            kappa=0.8100,
            macro_f1=0.8250,
            auc=0.8800,
            balanced_accuracy=0.8390,
            confusion_matrix=[[44, 6, 0, 0], [4, 41, 5, 0], [0, 5, 37, 8], [0, 0, 7, 43]],
            classification_report={
                "0": {"precision": 0.92, "recall": 0.88, "f1-score": 0.90, "support": 50},
                "1": {"precision": 0.79, "recall": 0.82, "f1-score": 0.80, "support": 50},
                "2": {"precision": 0.76, "recall": 0.74, "f1-score": 0.75, "support": 50},
                "3": {"precision": 0.84, "recall": 0.86, "f1-score": 0.85, "support": 50},
            },
            val_loss=0.4700,
            num_epochs=20,
            best_epoch=14,
            num_classes=4,
            img_size=224,
            batch_size=32,
            learning_rate=0.001,
        ),
        ModelMetrics(
            model_name="vit_base",
            run_path=Path("outputs/run_vit"),
            arch="vit_base_patch16_224",
            dataset="mamografias",
            accuracy=0.8300,
            kappa=0.8000,
            macro_f1=0.8150,
            auc=0.8700,
            balanced_accuracy=0.8280,
            confusion_matrix=[[43, 7, 0, 0], [5, 40, 5, 0], [0, 6, 36, 8], [0, 0, 8, 42]],
            classification_report={
                "0": {"precision": 0.90, "recall": 0.86, "f1-score": 0.88, "support": 50},
                "1": {"precision": 0.75, "recall": 0.80, "f1-score": 0.77, "support": 50},
                "2": {"precision": 0.73, "recall": 0.72, "f1-score": 0.73, "support": 50},
                "3": {"precision": 0.82, "recall": 0.84, "f1-score": 0.83, "support": 50},
            },
            val_loss=0.5000,
            num_epochs=20,
            best_epoch=13,
            num_classes=4,
            img_size=224,
            batch_size=16,
            learning_rate=0.0005,
        ),
    ]
    return metrics


@pytest.fixture
def mock_model_checkpoint_dir(tmp_path: Path) -> Path:
    """
    Create a temporary directory simulating a model checkpoint run with sample summary and metrics files.
    
    Creates the following structure under the provided tmp_path:
    - test_run/results/summary.json (contains basic run configuration like arch, dataset, epochs, img_size, batch_size, lr)
    - test_run/results/metrics/best_metrics.json (contains sample metrics including accuracy, kappa_quadratic, macro_f1, auc_ovr, bal_acc, loss, epoch, confusion_matrix, and classification_report)
    
    Parameters:
        tmp_path (Path): Temporary base directory provided by pytest.
    
    Returns:
        Path: Path to the created run results directory (tmp_path / "test_run" / "results").
    """
    run_dir = tmp_path / "test_run" / "results"
    run_dir.mkdir(parents=True)

    # Create summary.json
    summary_data = {
        "arch": "efficientnet_b0",
        "dataset": "mamografias",
        "epochs": 20,
        "img_size": 224,
        "batch_size": 32,
        "lr": 0.001,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_data, indent=2))

    # Create metrics/best_metrics.json
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir()
    metrics_data = {
        "acc": 0.8500,
        "kappa_quadratic": 0.8200,
        "macro_f1": 0.8320,
        "auc_ovr": 0.8900,
        "bal_acc": 0.8480,
        "loss": 0.4500,
        "epoch": 15,
        "confusion_matrix": [[45, 5, 0, 0], [3, 42, 5, 0], [0, 4, 38, 8], [0, 0, 6, 44]],
        "classification_report": {
            "0": {"precision": 0.94, "recall": 0.90, "f1-score": 0.92, "support": 50},
            "1": {"precision": 0.82, "recall": 0.84, "f1-score": 0.83, "support": 50},
            "2": {"precision": 0.77, "recall": 0.76, "f1-score": 0.77, "support": 50},
            "3": {"precision": 0.85, "recall": 0.88, "f1-score": 0.86, "support": 50},
        },
    }
    (metrics_dir / "best_metrics.json").write_text(json.dumps(metrics_data, indent=2))

    return run_dir


# ============================================================================
# Tests for compute_mcnemar_test
# ============================================================================


class TestComputeMcNemarTest:
    """Unit tests for McNemar's test function."""

    def test_mcnemar_basic_calculation(self) -> None:
        """Test basic McNemar's test calculation."""
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
        y_pred_a = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
        y_pred_b = np.array([0, 1, 0, 0, 1, 1, 0, 0, 1, 0])

        stat, p_value = compute_mcnemar_test(y_true, y_pred_a, y_pred_b)

        # Validate outputs
        assert isinstance(stat, float)
        assert isinstance(p_value, float)
        assert stat >= 0
        assert 0 <= p_value <= 1
        assert not np.isnan(stat)
        assert not np.isnan(p_value)

    def test_mcnemar_identical_predictions(self) -> None:
        """Test McNemar's test with identical predictions (no disagreements)."""
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0])
        y_pred_a = np.array([0, 1, 1, 0, 0, 1, 0, 0])
        y_pred_b = np.array([0, 1, 1, 0, 0, 1, 0, 0])  # Identical to y_pred_a

        stat, p_value = compute_mcnemar_test(y_true, y_pred_a, y_pred_b)

        # Should return (0.0, 1.0) for identical predictions
        assert stat == 0.0
        assert p_value == 1.0

    def test_mcnemar_continuity_correction(self) -> None:
        """Test McNemar's test with and without continuity correction."""
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
        y_pred_a = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
        y_pred_b = np.array([0, 1, 0, 0, 1, 1, 0, 0, 1, 0])

        stat_with, p_with = compute_mcnemar_test(
            y_true, y_pred_a, y_pred_b, continuity_correction=True
        )
        stat_without, p_without = compute_mcnemar_test(
            y_true, y_pred_a, y_pred_b, continuity_correction=False
        )

        # With correction should have lower statistic (more conservative)
        assert stat_with <= stat_without
        assert p_with >= p_without

    def test_mcnemar_list_inputs(self) -> None:
        """Test McNemar's test with list inputs (not numpy arrays)."""
        y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
        y_pred_a = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1]
        y_pred_b = [0, 1, 0, 0, 1, 1, 0, 0, 1, 0]

        stat, p_value = compute_mcnemar_test(y_true, y_pred_a, y_pred_b)

        # Should handle lists correctly
        assert isinstance(stat, float)
        assert isinstance(p_value, float)
        assert stat >= 0
        assert 0 <= p_value <= 1

    def test_mcnemar_empty_arrays_error(self) -> None:
        """Test McNemar's test raises error for empty arrays."""
        with pytest.raises(ValueError, match="y_true cannot be empty"):
            compute_mcnemar_test(np.array([]), np.array([1, 2]), np.array([1, 2]))

        with pytest.raises(ValueError, match="y_pred_a and y_pred_b cannot be empty"):
            compute_mcnemar_test(np.array([1, 2]), np.array([]), np.array([1, 2]))

    def test_mcnemar_mismatched_shapes_error(self) -> None:
        """Test McNemar's test raises error for mismatched array shapes."""
        y_true = np.array([0, 1, 1, 0])
        y_pred_a = np.array([0, 1, 1])  # Different length
        y_pred_b = np.array([0, 1, 1, 0])

        with pytest.raises(ValueError, match="All arrays must have same shape"):
            compute_mcnemar_test(y_true, y_pred_a, y_pred_b)

    def test_mcnemar_multidimensional_array_error(self) -> None:
        """Test McNemar's test raises error for multi-dimensional arrays."""
        y_true = np.array([[0, 1], [1, 0]])  # 2D array
        y_pred_a = np.array([[0, 1], [1, 0]])
        y_pred_b = np.array([[0, 1], [1, 0]])

        with pytest.raises(ValueError, match="Arrays must be 1-dimensional"):
            compute_mcnemar_test(y_true, y_pred_a, y_pred_b)


# ============================================================================
# Tests for ModelMetrics
# ============================================================================


class TestModelMetrics:
    """Unit tests for ModelMetrics dataclass."""

    def test_model_metrics_initialization(self, sample_model_metrics: ModelMetrics) -> None:
        """Test ModelMetrics initialization."""
        assert sample_model_metrics.model_name == "test_model_efficientnet"
        assert sample_model_metrics.arch == "efficientnet_b0"
        assert sample_model_metrics.accuracy == 0.8500
        assert sample_model_metrics.kappa == 0.8200
        assert sample_model_metrics.macro_f1 == 0.8320
        assert sample_model_metrics.num_classes == 4

    def test_model_metrics_optional_fields(self) -> None:
        """Test ModelMetrics with optional fields."""
        metrics = ModelMetrics(
            model_name="test_model",
            run_path=Path("outputs/test"),
            arch="resnet50",
            dataset="test_dataset",
            accuracy=0.80,
            kappa=0.75,
            macro_f1=0.78,
            auc=None,  # Optional field
            balanced_accuracy=0.79,
            confusion_matrix=[],
            classification_report={},
            val_loss=0.50,
            num_epochs=10,
            best_epoch=8,
            num_classes=4,
            img_size=224,
            batch_size=32,
            learning_rate=0.001,
        )

        assert metrics.auc is None
        assert metrics.predictions is None
        assert metrics.config == {}


# ============================================================================
# Tests for ModelComparisonEngine
# ============================================================================


class TestModelComparisonEngine:
    """Unit tests for ModelComparisonEngine class."""

    def test_engine_initialization_valid(self, mock_model_checkpoint_dir: Path) -> None:
        """Test ModelComparisonEngine initialization with valid paths."""
        engine = ModelComparisonEngine([mock_model_checkpoint_dir])

        assert len(engine.run_paths) == 1
        assert len(engine.model_names) == 1
        assert engine.model_names[0] == "test_run"

    def test_engine_initialization_custom_names(self, mock_model_checkpoint_dir: Path) -> None:
        """Test ModelComparisonEngine initialization with custom model names."""
        engine = ModelComparisonEngine(
            [mock_model_checkpoint_dir],
            model_names=["custom_model_name"]
        )

        assert engine.model_names[0] == "custom_model_name"

    def test_engine_initialization_empty_paths_error(self) -> None:
        """Test ModelComparisonEngine raises error for empty run_paths."""
        with pytest.raises(ValueError, match="run_paths cannot be empty"):
            ModelComparisonEngine([])

    def test_engine_initialization_nonexistent_path_error(self, tmp_path: Path) -> None:
        """Test ModelComparisonEngine raises error for non-existent path."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="Run path does not exist"):
            ModelComparisonEngine([nonexistent])

    def test_engine_initialization_mismatched_names_error(
        self, mock_model_checkpoint_dir: Path
    ) -> None:
        """Test ModelComparisonEngine raises error for mismatched model names count."""
        with pytest.raises(ValueError, match="Number of model_names"):
            ModelComparisonEngine(
                [mock_model_checkpoint_dir],
                model_names=["name1", "name2"]  # Too many names
            )

    def test_load_model_metrics_success(
        self, mock_model_checkpoint_dir: Path
    ) -> None:
        """Test successful loading of model metrics."""
        engine = ModelComparisonEngine([mock_model_checkpoint_dir])
        metrics = engine.load_model_metrics(mock_model_checkpoint_dir, "test_model")

        assert metrics is not None
        assert metrics.model_name == "test_model"
        assert metrics.arch == "efficientnet_b0"
        assert metrics.accuracy == 0.8500
        assert metrics.macro_f1 == 0.8320
        assert len(metrics.confusion_matrix) == 4

    def test_load_model_metrics_missing_summary_error(self, tmp_path: Path) -> None:
        """Test load_model_metrics raises error when summary.json is missing."""
        run_dir = tmp_path / "incomplete_run"
        run_dir.mkdir()

        engine = ModelComparisonEngine.__new__(ModelComparisonEngine)
        with pytest.raises(FileNotFoundError, match="summary.json not found"):
            engine.load_model_metrics(run_dir, "test")

    def test_load_model_metrics_missing_best_metrics_error(self, tmp_path: Path) -> None:
        """Test load_model_metrics raises error when best_metrics.json is missing."""
        run_dir = tmp_path / "incomplete_run"
        run_dir.mkdir()

        # Create only summary.json
        summary_data = {"arch": "resnet50", "dataset": "test"}
        (run_dir / "summary.json").write_text(json.dumps(summary_data))

        engine = ModelComparisonEngine.__new__(ModelComparisonEngine)
        with pytest.raises(FileNotFoundError, match="metrics/best_metrics.json not found"):
            engine.load_model_metrics(run_dir, "test")

    def test_load_all_metrics_success(
        self, mock_model_checkpoint_dir: Path
    ) -> None:
        """Test successful loading of all model metrics."""
        engine = ModelComparisonEngine([mock_model_checkpoint_dir])
        all_metrics = engine.load_all_metrics()

        assert len(all_metrics) == 1
        assert all_metrics[0].model_name == "test_run"
        assert all_metrics[0].accuracy == 0.8500

    def test_load_all_metrics_partial_failure(
        self, tmp_path: Path, mock_model_checkpoint_dir: Path
    ) -> None:
        """Test load_all_metrics continues when some models fail to load."""
        # Create an invalid run directory
        invalid_run = tmp_path / "invalid_run"
        invalid_run.mkdir()

        engine = ModelComparisonEngine([mock_model_checkpoint_dir, invalid_run])
        all_metrics = engine.load_all_metrics()

        # Should successfully load the valid one and skip the invalid one
        assert len(all_metrics) == 1
        assert all_metrics[0].model_name == "test_run"

    def test_load_all_metrics_all_failures_error(self, tmp_path: Path) -> None:
        """Test load_all_metrics raises error when all models fail to load."""
        invalid_run = tmp_path / "invalid_run"
        invalid_run.mkdir()

        engine = ModelComparisonEngine([invalid_run])
        with pytest.raises(RuntimeError, match="Failed to load metrics for any"):
            engine.load_all_metrics()

    def test_get_metrics_dataframe(
        self, sample_model_metrics_list: List[ModelMetrics]
    ) -> None:
        """Test conversion of metrics list to DataFrame."""
        engine = ModelComparisonEngine.__new__(ModelComparisonEngine)
        df = engine.get_metrics_dataframe(sample_model_metrics_list)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "model_name" in df.columns
        assert "accuracy" in df.columns
        assert "macro_f1" in df.columns

        # Should be sorted by macro_f1 descending
        assert df.iloc[0]["model_name"] == "efficientnet_b0"  # Highest macro_f1
        assert df.iloc[2]["model_name"] == "vit_base"  # Lowest macro_f1

    def test_get_per_class_metrics(
        self, sample_model_metrics_list: List[ModelMetrics]
    ) -> None:
        """Test extraction of per-class metrics."""
        engine = ModelComparisonEngine.__new__(ModelComparisonEngine)
        df = engine.get_per_class_metrics(sample_model_metrics_list)

        assert isinstance(df, pd.DataFrame)
        assert "model_name" in df.columns
        assert "class" in df.columns
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1-score" in df.columns
        assert "support" in df.columns

        # Should have metrics for all classes and models
        # 3 models * 4 classes = 12 rows
        assert len(df) == 12

    def test_get_confusion_matrices(
        self, sample_model_metrics_list: List[ModelMetrics]
    ) -> None:
        """Test extraction of confusion matrices."""
        engine = ModelComparisonEngine.__new__(ModelComparisonEngine)
        matrices = engine.get_confusion_matrices(sample_model_metrics_list)

        assert isinstance(matrices, dict)
        assert len(matrices) == 3
        assert "efficientnet_b0" in matrices
        assert isinstance(matrices["efficientnet_b0"], np.ndarray)
        assert matrices["efficientnet_b0"].shape == (4, 4)

    def test_aggregate_metrics_across_models(
        self, sample_model_metrics_list: List[ModelMetrics]
    ) -> None:
        """Test aggregation of metrics across models."""
        engine = ModelComparisonEngine.__new__(ModelComparisonEngine)
        aggregated = engine.aggregate_metrics_across_models(sample_model_metrics_list)

        assert isinstance(aggregated, dict)
        assert "accuracy" in aggregated
        assert "kappa" in aggregated
        assert "macro_f1" in aggregated

        # Check structure of aggregated metrics
        assert "mean" in aggregated["accuracy"]
        assert "std" in aggregated["accuracy"]
        assert "ci_lower" in aggregated["accuracy"]
        assert "ci_upper" in aggregated["accuracy"]

        # Validate mean is reasonable
        mean_acc = aggregated["accuracy"]["mean"]
        assert 0.83 <= mean_acc <= 0.86  # Average of 0.85, 0.84, 0.83

    def test_get_side_by_side_comparison(
        self, sample_model_metrics_list: List[ModelMetrics]
    ) -> None:
        """Test side-by-side comparison table generation."""
        engine = ModelComparisonEngine.__new__(ModelComparisonEngine)
        comparison = engine.get_side_by_side_comparison(sample_model_metrics_list)

        assert isinstance(comparison, pd.DataFrame)
        assert "rank" in comparison.columns
        assert "model_name" in comparison.columns
        assert "accuracy" in comparison.columns
        assert "macro_f1" in comparison.columns

        # Should have rank column starting from 1
        assert comparison["rank"].tolist() == [1, 2, 3]

        # Best model should be ranked first
        assert comparison.iloc[0]["model_name"] == "efficientnet_b0"

    def test_rank_models_by_metric(
        self, sample_model_metrics_list: List[ModelMetrics]
    ) -> None:
        """Test ranking models by specific metric."""
        engine = ModelComparisonEngine.__new__(ModelComparisonEngine)
        ranked = engine.rank_models_by_metric(
            sample_model_metrics_list,
            metric="accuracy"
        )

        assert isinstance(ranked, pd.DataFrame)
        assert "rank" in ranked.columns
        assert "accuracy" in ranked.columns

        # Should be ranked by accuracy descending
        assert ranked.iloc[0]["model_name"] == "efficientnet_b0"  # acc=0.85
        assert ranked.iloc[1]["model_name"] == "resnet50"  # acc=0.84
        assert ranked.iloc[2]["model_name"] == "vit_base"  # acc=0.83

    def test_rank_models_by_metric_ascending(
        self, sample_model_metrics_list: List[ModelMetrics]
    ) -> None:
        """Test ranking models by metric in ascending order."""
        engine = ModelComparisonEngine.__new__(ModelComparisonEngine)
        ranked = engine.rank_models_by_metric(
            sample_model_metrics_list,
            metric="val_loss",
            ascending=True  # Lower loss is better
        )

        # Should be ranked by val_loss ascending
        assert ranked.iloc[0]["model_name"] == "efficientnet_b0"  # loss=0.45
        assert ranked.iloc[2]["model_name"] == "vit_base"  # loss=0.50

    def test_get_best_model(
        self, sample_model_metrics_list: List[ModelMetrics]
    ) -> None:
        """Test getting the best performing model."""
        engine = ModelComparisonEngine.__new__(ModelComparisonEngine)
        best = engine.get_best_model(sample_model_metrics_list, metric="macro_f1")

        assert isinstance(best, ModelMetrics)
        assert best.model_name == "efficientnet_b0"
        assert best.macro_f1 == 0.8320

    def test_get_best_model_empty_list_error(self) -> None:
        """Test get_best_model raises error for empty list."""
        engine = ModelComparisonEngine.__new__(ModelComparisonEngine)
        with pytest.raises(ValueError, match="metrics_list cannot be empty"):
            engine.get_best_model([], metric="accuracy")

    def test_get_best_model_invalid_metric_error(self) -> None:
        """Test get_best_model raises error for invalid metric."""
        # Create metrics with missing auc field
        metrics_no_auc = [
            ModelMetrics(
                model_name="test",
                run_path=Path("test"),
                arch="test",
                dataset="test",
                accuracy=0.8,
                kappa=0.7,
                macro_f1=0.75,
                auc=None,
                balanced_accuracy=0.78,
                confusion_matrix=[],
                classification_report={},
                val_loss=0.5,
                num_epochs=10,
                best_epoch=5,
                num_classes=4,
                img_size=224,
                batch_size=32,
                learning_rate=0.001,
            )
        ]

        engine = ModelComparisonEngine.__new__(ModelComparisonEngine)
        with pytest.raises(ValueError, match="No models found with valid"):
            engine.get_best_model(metrics_no_auc, metric="auc")

    def test_compare_models_pairwise(
        self, sample_model_metrics_list: List[ModelMetrics]
    ) -> None:
        """Test pairwise comparison of two models."""
        engine = ModelComparisonEngine.__new__(ModelComparisonEngine)
        model_a = sample_model_metrics_list[0]  # efficientnet
        model_b = sample_model_metrics_list[1]  # resnet

        comparison = engine.compare_models_pairwise(model_a, model_b)

        assert isinstance(comparison, dict)
        assert "accuracy" in comparison
        assert "macro_f1" in comparison

        # Check structure of comparison
        acc_comp = comparison["accuracy"]
        assert "model_a" in acc_comp
        assert "model_b" in acc_comp
        assert "difference" in acc_comp
        assert "percent_change" in acc_comp
        assert "winner" in acc_comp

        # efficientnet should be the winner for accuracy
        assert acc_comp["winner"] == "efficientnet_b0"
        assert acc_comp["model_a"] > acc_comp["model_b"]

    def test_get_metrics_summary_table(
        self, sample_model_metrics_list: List[ModelMetrics]
    ) -> None:
        """Test generation of comprehensive metrics summary table."""
        engine = ModelComparisonEngine.__new__(ModelComparisonEngine)
        summary = engine.get_metrics_summary_table(sample_model_metrics_list)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 3

        # Check for key columns
        expected_cols = [
            "model_name", "arch", "dataset", "accuracy", "kappa", "macro_f1",
            "num_epochs", "batch_size", "learning_rate"
        ]
        for col in expected_cols:
            assert col in summary.columns

        # Should be sorted by macro_f1
        assert summary.iloc[0]["model_name"] == "efficientnet_b0"

    def test_export_comparison_report(
        self, tmp_path: Path, sample_model_metrics_list: List[ModelMetrics]
    ) -> None:
        """Test exporting comparison report to JSON."""
        engine = ModelComparisonEngine.__new__(ModelComparisonEngine)
        output_path = tmp_path / "comparison_report.json"

        engine.export_comparison_report(sample_model_metrics_list, output_path)

        assert output_path.exists()

        # Validate JSON structure
        with output_path.open("r") as f:
            report = json.load(f)

        assert "summary" in report
        assert "aggregated_metrics" in report
        assert "model_comparison" in report
        assert "per_class_metrics" in report
        assert "individual_models" in report

        # Check summary
        assert report["summary"]["num_models"] == 3
        assert report["summary"]["best_model"] == "efficientnet_b0"

    def test_engine_repr(self, mock_model_checkpoint_dir: Path) -> None:
        """Test string representation of ModelComparisonEngine."""
        engine = ModelComparisonEngine([mock_model_checkpoint_dir])
        repr_str = repr(engine)

        assert "ModelComparisonEngine" in repr_str
        assert "n_models=1" in repr_str
        assert "test_run" in repr_str

    def test_load_predictions_success(
        self, mock_model_checkpoint_dir: Path
    ) -> None:
        """Test loading predictions from checkpoint."""
        run_path = mock_model_checkpoint_dir / "run1" / "results"
        run_path.mkdir(parents=True, exist_ok=True)

        # Create val_predictions.csv
        pred_csv = run_path / "val_predictions.csv"
        pred_df = pd.DataFrame({
            "true_label": [0, 1, 1, 0, 1],
            "prediction": [0, 1, 0, 0, 1],
        })
        pred_df.to_csv(pred_csv, index=False)

        # Load predictions
        engine = ModelComparisonEngine([run_path])
        y_true, y_pred = engine.load_predictions(run_path)

        # Verify
        assert len(y_true) == 5
        assert len(y_pred) == 5
        np.testing.assert_array_equal(y_true, [0, 1, 1, 0, 1])
        np.testing.assert_array_equal(y_pred, [0, 1, 0, 0, 1])

    def test_load_predictions_missing_file_error(self, tmp_path: Path) -> None:
        """Test that loading predictions raises error if file missing."""
        run_path = tmp_path / "run1"
        run_path.mkdir()

        engine = ModelComparisonEngine([run_path])

        with pytest.raises(FileNotFoundError, match="Prediction file not found"):
            engine.load_predictions(run_path)

    def test_load_predictions_missing_columns_error(self, tmp_path: Path) -> None:
        """Test that loading predictions raises error if CSV missing required columns."""
        run_path = tmp_path / "run1"
        run_path.mkdir()

        # Create CSV with wrong columns
        pred_csv = run_path / "val_predictions.csv"
        pred_df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        pred_df.to_csv(pred_csv, index=False)

        engine = ModelComparisonEngine([run_path])

        with pytest.raises(ValueError, match="missing required columns"):
            engine.load_predictions(run_path)


# ============================================================================
# Tests for format_statistical_summary
# ============================================================================


class TestFormatStatisticalSummary:
    """Unit tests for format_statistical_summary function."""

    def test_format_statistical_summary_basic(self) -> None:
        """Test basic formatting of statistical summary."""
        aggregated = {
            "accuracy": {"mean": 0.8470, "std": 0.0150, "ci_lower": 0.8320, "ci_upper": 0.8620},
            "macro_f1": {"mean": 0.8280, "std": 0.0160, "ci_lower": 0.8120, "ci_upper": 0.8440},
        }

        summary = format_statistical_summary(aggregated)

        assert isinstance(summary, str)
        assert "Model Comparison Statistical Summary" in summary
        assert "accuracy" in summary
        assert "macro_f1" in summary
        assert "0.8470" in summary
        assert "0.0150" in summary
        assert "[0.8320, 0.8620]" in summary

    def test_format_statistical_summary_custom_title(self) -> None:
        """Test formatting with custom title."""
        aggregated = {
            "accuracy": {"mean": 0.85, "std": 0.02, "ci_lower": 0.83, "ci_upper": 0.87},
        }

        summary = format_statistical_summary(aggregated, title="Custom Title")

        assert "Custom Title" in summary
        assert "Custom Title (95% CI):" in summary

    def test_format_statistical_summary_custom_decimal_places(self) -> None:
        """Test formatting with custom decimal places."""
        aggregated = {
            "accuracy": {"mean": 0.8471234, "std": 0.0152345, "ci_lower": 0.8321, "ci_upper": 0.8621},
        }

        summary = format_statistical_summary(aggregated, decimal_places=2)

        assert "0.85" in summary  # 2 decimal places
        assert "0.85 ± 0.02" in summary

    def test_format_statistical_summary_empty_dict(self) -> None:
        """Test formatting with empty aggregated dict."""
        summary = format_statistical_summary({})

        assert "(No metrics available)" in summary

    def test_format_statistical_summary_sorted_metrics(self) -> None:
        """Test that metrics are sorted alphabetically."""
        aggregated = {
            "z_metric": {"mean": 0.9, "std": 0.01, "ci_lower": 0.89, "ci_upper": 0.91},
            "a_metric": {"mean": 0.8, "std": 0.02, "ci_lower": 0.78, "ci_upper": 0.82},
            "m_metric": {"mean": 0.85, "std": 0.015, "ci_lower": 0.835, "ci_upper": 0.865},
        }

        summary = format_statistical_summary(aggregated)
        lines = summary.split("\n")

        # Find metric lines (skip header and separator)
        metric_lines = [line for line in lines if ":" in line and "Summary" not in line]

        # Should be sorted alphabetically
        assert "a_metric" in metric_lines[0]
        assert "m_metric" in metric_lines[1]
        assert "z_metric" in metric_lines[2]


# ============================================================================
# Tests for export_comparison_table
# ============================================================================


class TestExportComparisonTable:
    """Unit tests for export_comparison_table function."""

    def test_export_comparison_table_csv(self, tmp_path: Path) -> None:
        """Test exporting comparison table to CSV."""
        df = pd.DataFrame({
            "model_name": ["model_a", "model_b"],
            "accuracy": [0.85, 0.82],
            "macro_f1": [0.83, 0.80],
        })

        base_path = tmp_path / "comparison"
        paths = export_comparison_table(df, base_path, formats=["csv"])

        assert len(paths) == 1
        assert paths[0].suffix == ".csv"
        assert paths[0].exists()

        # Verify CSV content
        df_loaded = pd.read_csv(paths[0])
        assert len(df_loaded) == 2
        assert "accuracy" in df_loaded.columns

    def test_export_comparison_table_json(self, tmp_path: Path) -> None:
        """Test exporting comparison table to JSON."""
        df = pd.DataFrame({
            "model_name": ["model_a", "model_b"],
            "accuracy": [0.85, 0.82],
        })

        base_path = tmp_path / "comparison"
        paths = export_comparison_table(df, base_path, formats=["json"])

        assert len(paths) == 1
        assert paths[0].suffix == ".json"
        assert paths[0].exists()

        # Verify JSON content
        with paths[0].open("r") as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["model_name"] == "model_a"

    def test_export_comparison_table_markdown(self, tmp_path: Path) -> None:
        """Test exporting comparison table to Markdown."""
        df = pd.DataFrame({
            "model_name": ["model_a"],
            "accuracy": [0.85],
        })

        base_path = tmp_path / "comparison"
        paths = export_comparison_table(df, base_path, formats=["md"])

        assert len(paths) == 1
        assert paths[0].suffix == ".md"
        assert paths[0].exists()

        # Verify Markdown content
        content = paths[0].read_text()
        assert "|" in content  # Markdown table syntax
        assert "model_name" in content

    def test_export_comparison_table_latex(self, tmp_path: Path) -> None:
        """Test exporting comparison table to LaTeX."""
        df = pd.DataFrame({
            "model_name": ["model_a"],
            "accuracy": [0.85],
        })

        base_path = tmp_path / "comparison"
        paths = export_comparison_table(df, base_path, formats=["tex"])

        assert len(paths) == 1
        assert paths[0].suffix == ".tex"
        assert paths[0].exists()

        # Verify LaTeX content
        content = paths[0].read_text()
        assert "\\begin{tabular}" in content
        assert "\\end{tabular}" in content

    def test_export_comparison_table_multiple_formats(self, tmp_path: Path) -> None:
        """Test exporting comparison table to multiple formats."""
        df = pd.DataFrame({
            "model_name": ["model_a"],
            "accuracy": [0.85],
        })

        base_path = tmp_path / "comparison"
        paths = export_comparison_table(
            df, base_path, formats=["csv", "json", "md", "tex"]
        )

        assert len(paths) == 4
        assert any(p.suffix == ".csv" for p in paths)
        assert any(p.suffix == ".json" for p in paths)
        assert any(p.suffix == ".md" for p in paths)
        assert any(p.suffix == ".tex" for p in paths)

    def test_export_comparison_table_default_formats(self, tmp_path: Path) -> None:
        """Test exporting comparison table with default formats."""
        df = pd.DataFrame({
            "model_name": ["model_a"],
            "accuracy": [0.85],
        })

        base_path = tmp_path / "comparison"
        paths = export_comparison_table(df, base_path)  # Use defaults

        # Default formats: csv, xlsx, json, md
        # xlsx may fail if openpyxl not installed, so check for at least csv, json, md
        assert len(paths) >= 3
        assert any(p.suffix == ".csv" for p in paths)
        assert any(p.suffix == ".json" for p in paths)
        assert any(p.suffix == ".md" for p in paths)

    def test_export_comparison_table_empty_dataframe_error(self, tmp_path: Path) -> None:
        """Test export raises error for empty DataFrame."""
        df = pd.DataFrame()
        base_path = tmp_path / "comparison"

        with pytest.raises(ValueError, match="Cannot export empty DataFrame"):
            export_comparison_table(df, base_path)

    def test_export_comparison_table_empty_formats_error(self, tmp_path: Path) -> None:
        """Test export raises error for empty formats list."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        base_path = tmp_path / "comparison"

        with pytest.raises(ValueError, match="formats list cannot be empty"):
            export_comparison_table(df, base_path, formats=[])

    def test_export_comparison_table_unsupported_format_warning(
        self, tmp_path: Path
    ) -> None:
        """Test export logs warning for unsupported format."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        base_path = tmp_path / "comparison"

        paths = export_comparison_table(df, base_path, formats=["csv", "unsupported"])

        # Should export CSV successfully and skip unsupported
        assert len(paths) == 1
        assert paths[0].suffix == ".csv"

    def test_export_comparison_table_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test export creates parent directories if they don't exist."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        base_path = tmp_path / "nested" / "dir" / "comparison"

        paths = export_comparison_table(df, base_path, formats=["csv"])

        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].parent.exists()

    def test_export_comparison_table_no_index(self, tmp_path: Path) -> None:
        """Test exporting without DataFrame index."""
        df = pd.DataFrame({
            "model_name": ["model_a", "model_b"],
            "accuracy": [0.85, 0.82],
        })

        base_path = tmp_path / "comparison"
        paths = export_comparison_table(df, base_path, formats=["csv"], index=False)

        # Verify CSV has no index column
        df_loaded = pd.read_csv(paths[0])
        assert "Unnamed: 0" not in df_loaded.columns


# ============================================================================
# Tests for Plotly visualization functions (optional - requires plotly)
# ============================================================================


@pytest.mark.skipif(
    not pytest.importorskip("plotly", minversion=None),
    reason="plotly not installed"
)
class TestPlotlyVisualizations:
    """Unit tests for Plotly visualization functions (optional)."""

    def test_create_metrics_comparison_table_import(self) -> None:
        """Test that create_metrics_comparison_table can be imported."""
        from mammography.vis.model_comparison import create_metrics_comparison_table
        assert callable(create_metrics_comparison_table)

    def test_create_metrics_comparison_chart_import(self) -> None:
        """Test that create_metrics_comparison_chart can be imported."""
        from mammography.vis.model_comparison import create_metrics_comparison_chart
        assert callable(create_metrics_comparison_chart)

    def test_create_per_class_comparison_import(self) -> None:
        """Test that create_per_class_comparison can be imported."""
        from mammography.vis.model_comparison import create_per_class_comparison
        assert callable(create_per_class_comparison)

    def test_create_confusion_matrix_comparison_import(self) -> None:
        """Test that create_confusion_matrix_comparison can be imported."""
        from mammography.vis.model_comparison import create_confusion_matrix_comparison
        assert callable(create_confusion_matrix_comparison)

    def test_visualization_requires_plotly(
        self, sample_model_metrics_list: List[ModelMetrics], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that visualizations raise ImportError without plotly."""
        # Mock plotly as not available
        from mammography.vis import model_comparison
        monkeypatch.setattr(model_comparison, "px", None)
        monkeypatch.setattr(model_comparison, "go", None)

        from mammography.vis.model_comparison import create_metrics_comparison_table

        with pytest.raises(ImportError, match="Plotly is required"):
            create_metrics_comparison_table(sample_model_metrics_list)