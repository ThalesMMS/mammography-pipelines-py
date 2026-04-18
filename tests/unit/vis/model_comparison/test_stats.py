# ruff: noqa
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
