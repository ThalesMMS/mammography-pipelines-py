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
