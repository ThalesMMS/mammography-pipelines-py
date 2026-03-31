from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from mammography.config import TrainConfig
from mammography.training.cv_engine import CrossValidationEngine, FoldResult


def _create_sample_dataframe(num_samples: int = 100) -> pd.DataFrame:
    """Create a sample DataFrame for testing CV engine."""
    data = {
        "image_path": [f"/path/to/image_{i}.png" for i in range(num_samples)],
        "professional_label": [(i % 4) + 1 for i in range(num_samples)],
        "accession": [f"ACC{i // 3:03d}" for i in range(num_samples)],
    }
    return pd.DataFrame(data)


def _create_sample_fold_result(fold_idx: int, val_acc: float = 0.8) -> FoldResult:
    """Create a sample FoldResult for testing."""
    return FoldResult(
        fold_idx=fold_idx,
        train_size=80,
        val_size=20,
        best_epoch=5,
        best_val_acc=val_acc,
        best_val_kappa=0.75,
        best_val_macro_f1=0.78,
        best_val_auc=0.85,
        final_train_loss=0.5,
        final_train_acc=0.82,
        checkpoint_path=Path(f"/tmp/fold_{fold_idx}/checkpoint.pt"),
        metrics_path=Path(f"/tmp/fold_{fold_idx}/metrics.json"),
    )


def test_cv_engine_initialization(tmp_path: Path) -> None:
    """Test basic CrossValidationEngine initialization."""
    config = TrainConfig(
        csv="dummy.csv",
        outdir=str(tmp_path / "cv_output"),
        epochs=1,
        batch_size=4,
    )

    engine = CrossValidationEngine(config, n_folds=5, cv_seed=42)

    assert engine.config == config
    assert engine.n_folds == 5
    assert engine.cv_seed == 42
    assert engine.save_all_folds is False
    assert engine.output_root.exists()


def test_cv_engine_invalid_n_folds(tmp_path: Path) -> None:
    """Test that n_folds < 2 raises ValueError."""
    config = TrainConfig(
        csv="dummy.csv",
        outdir=str(tmp_path / "cv_output"),
        epochs=1,
    )

    with pytest.raises(ValueError, match="n_folds deve ser >= 2"):
        CrossValidationEngine(config, n_folds=1)

    with pytest.raises(ValueError, match="n_folds deve ser >= 2"):
        CrossValidationEngine(config, n_folds=0)

    with pytest.raises(ValueError, match="n_folds deve ser >= 2"):
        CrossValidationEngine(config, n_folds=-1)


def test_cv_engine_save_all_folds_flag(tmp_path: Path) -> None:
    """Test save_all_folds flag is stored correctly."""
    config = TrainConfig(csv="dummy.csv", outdir=str(tmp_path / "cv_output"))

    engine_default = CrossValidationEngine(config, n_folds=3)
    assert engine_default.save_all_folds is False

    engine_save_all = CrossValidationEngine(config, n_folds=3, save_all_folds=True)
    assert engine_save_all.save_all_folds is True


def test_cv_engine_output_directory_creation(tmp_path: Path) -> None:
    """Test that output directory is created on initialization."""
    output_dir = tmp_path / "nested" / "cv_output"
    config = TrainConfig(csv="dummy.csv", outdir=str(output_dir))

    assert not output_dir.exists()

    engine = CrossValidationEngine(config, n_folds=3)

    assert output_dir.exists()
    assert engine.output_root == output_dir


def test_aggregate_results_basic() -> None:
    """Test basic aggregation of fold results."""
    config = TrainConfig(csv="dummy.csv", outdir="/tmp/cv")
    engine = CrossValidationEngine(config, n_folds=3)

    # Create sample fold results
    fold_results = [
        _create_sample_fold_result(0, val_acc=0.80),
        _create_sample_fold_result(1, val_acc=0.82),
        _create_sample_fold_result(2, val_acc=0.78),
    ]

    aggregated = engine._aggregate_results(fold_results)

    # Check required keys
    assert "n_folds" in aggregated
    assert "mean_val_acc" in aggregated
    assert "std_val_acc" in aggregated
    assert "mean_val_kappa" in aggregated
    assert "mean_val_macro_f1" in aggregated
    assert "fold_results" in aggregated

    # Check values
    assert aggregated["n_folds"] == 3
    assert abs(aggregated["mean_val_acc"] - 0.80) < 0.01  # (0.80 + 0.82 + 0.78) / 3
    assert aggregated["std_val_acc"] > 0  # Should have non-zero std

    # Check fold_results list
    assert len(aggregated["fold_results"]) == 3
    assert aggregated["fold_results"][0]["fold_idx"] == 0
    assert aggregated["fold_results"][0]["best_val_acc"] == 0.80


def test_aggregate_results_confidence_intervals() -> None:
    """Test that confidence intervals are computed in aggregation."""
    config = TrainConfig(csv="dummy.csv", outdir="/tmp/cv")
    engine = CrossValidationEngine(config, n_folds=5)

    fold_results = [
        _create_sample_fold_result(i, val_acc=0.80 + i * 0.01)
        for i in range(5)
    ]

    aggregated = engine._aggregate_results(fold_results)

    # Check CI keys exist
    assert "ci_lower_val_acc" in aggregated
    assert "ci_upper_val_acc" in aggregated
    assert "ci_lower_val_kappa" in aggregated
    assert "ci_upper_val_kappa" in aggregated
    assert "ci_lower_val_macro_f1" in aggregated
    assert "ci_upper_val_macro_f1" in aggregated

    # CI bounds should be reasonable
    mean_acc = aggregated["mean_val_acc"]
    ci_lower = aggregated["ci_lower_val_acc"]
    ci_upper = aggregated["ci_upper_val_acc"]

    assert ci_lower <= mean_acc
    assert ci_upper >= mean_acc
    assert ci_lower < ci_upper


def test_aggregate_results_min_max() -> None:
    """Test that min and max values are computed correctly."""
    config = TrainConfig(csv="dummy.csv", outdir="/tmp/cv")
    engine = CrossValidationEngine(config, n_folds=4)

    fold_results = [
        _create_sample_fold_result(0, val_acc=0.75),
        _create_sample_fold_result(1, val_acc=0.90),
        _create_sample_fold_result(2, val_acc=0.80),
        _create_sample_fold_result(3, val_acc=0.85),
    ]

    aggregated = engine._aggregate_results(fold_results)

    assert aggregated["min_val_acc"] == 0.75
    assert aggregated["max_val_acc"] == 0.90
    assert "min_val_kappa" in aggregated
    assert "max_val_kappa" in aggregated


def test_aggregate_results_with_auc() -> None:
    """Test aggregation when AUC is present."""
    config = TrainConfig(csv="dummy.csv", outdir="/tmp/cv")
    engine = CrossValidationEngine(config, n_folds=3)

    fold_results = [
        _create_sample_fold_result(0, val_acc=0.80),
        _create_sample_fold_result(1, val_acc=0.82),
        _create_sample_fold_result(2, val_acc=0.78),
    ]
    # All have AUC values (set in _create_sample_fold_result)

    aggregated = engine._aggregate_results(fold_results)

    # AUC statistics should be present
    assert aggregated["mean_val_auc"] is not None
    assert aggregated["std_val_auc"] is not None
    assert aggregated["ci_lower_val_auc"] is not None
    assert aggregated["ci_upper_val_auc"] is not None


def test_aggregate_results_without_auc() -> None:
    """Test aggregation when AUC is None (binary classification)."""
    config = TrainConfig(csv="dummy.csv", outdir="/tmp/cv")
    engine = CrossValidationEngine(config, n_folds=3)

    # Create fold results without AUC
    fold_results = []
    for i in range(3):
        fr = _create_sample_fold_result(i, val_acc=0.80 + i * 0.01)
        fr = FoldResult(
            fold_idx=fr.fold_idx,
            train_size=fr.train_size,
            val_size=fr.val_size,
            best_epoch=fr.best_epoch,
            best_val_acc=fr.best_val_acc,
            best_val_kappa=fr.best_val_kappa,
            best_val_macro_f1=fr.best_val_macro_f1,
            best_val_auc=None,  # No AUC
            final_train_loss=fr.final_train_loss,
            final_train_acc=fr.final_train_acc,
            checkpoint_path=fr.checkpoint_path,
            metrics_path=fr.metrics_path,
        )
        fold_results.append(fr)

    aggregated = engine._aggregate_results(fold_results)

    # AUC statistics should be None
    assert aggregated["mean_val_auc"] is None
    assert aggregated["std_val_auc"] is None
    assert aggregated["ci_lower_val_auc"] is None
    assert aggregated["ci_upper_val_auc"] is None


def test_aggregate_results_empty_raises_error() -> None:
    """Test that empty fold_results raises ValueError."""
    config = TrainConfig(csv="dummy.csv", outdir="/tmp/cv")
    engine = CrossValidationEngine(config, n_folds=3)

    with pytest.raises(ValueError, match="fold_results nao pode estar vazio"):
        engine._aggregate_results([])


def test_aggregate_results_detailed_stats() -> None:
    """Test that detailed_stats structure is included."""
    config = TrainConfig(csv="dummy.csv", outdir="/tmp/cv")
    engine = CrossValidationEngine(config, n_folds=3)

    fold_results = [
        _create_sample_fold_result(i, val_acc=0.80 + i * 0.01)
        for i in range(3)
    ]

    aggregated = engine._aggregate_results(fold_results)

    # Should include detailed_stats for downstream use
    assert "detailed_stats" in aggregated
    assert isinstance(aggregated["detailed_stats"], dict)
    assert "val_acc" in aggregated["detailed_stats"]
    assert "val_kappa" in aggregated["detailed_stats"]


def test_save_aggregated_results(tmp_path: Path) -> None:
    """Test saving aggregated results to JSON."""
    config = TrainConfig(csv="dummy.csv", outdir=str(tmp_path / "cv_output"))
    engine = CrossValidationEngine(config, n_folds=3, cv_seed=42)

    fold_results = [
        _create_sample_fold_result(i, val_acc=0.80 + i * 0.01)
        for i in range(3)
    ]

    aggregated = engine._aggregate_results(fold_results)
    engine._save_aggregated_results(aggregated, fold_results)

    # Check that cv_summary.json was created
    summary_path = engine.output_root / "cv_summary.json"
    assert summary_path.exists()

    # Read and validate content
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert "n_folds" in summary
    assert "cv_seed" in summary
    assert "config" in summary
    assert "results" in summary

    assert summary["n_folds"] == 3
    assert summary["cv_seed"] == 42
    assert "mean_val_acc" in summary["results"]


def test_load_dataset_with_subset(tmp_path: Path) -> None:
    """Test _load_dataset applies subset correctly."""
    # Create a mock CSV file
    csv_path = tmp_path / "data.csv"
    df = _create_sample_dataframe(num_samples=100)
    df.to_csv(csv_path, index=False)

    config = TrainConfig(
        csv=str(csv_path),
        outdir=str(tmp_path / "cv_output"),
        subset=20,
    )
    engine = CrossValidationEngine(config, n_folds=3)

    loaded_df = engine._load_dataset()

    assert loaded_df is not None
    assert len(loaded_df) == 20  # Subset applied


def test_load_dataset_no_subset(tmp_path: Path) -> None:
    """Test _load_dataset without subset returns full dataset."""
    # Create a mock CSV file
    csv_path = tmp_path / "data.csv"
    df = _create_sample_dataframe(num_samples=50)
    df.to_csv(csv_path, index=False)

    config = TrainConfig(
        csv=str(csv_path),
        outdir=str(tmp_path / "cv_output"),
        subset=0,  # No subset
    )
    engine = CrossValidationEngine(config, n_folds=3)

    loaded_df = engine._load_dataset()

    assert loaded_df is not None
    assert len(loaded_df) == 50


def test_load_dataset_handles_errors(tmp_path: Path) -> None:
    """Test _load_dataset returns None on errors."""
    config = TrainConfig(
        csv="nonexistent.csv",  # File doesn't exist
        outdir=str(tmp_path / "cv_output"),
    )
    engine = CrossValidationEngine(config, n_folds=3)

    loaded_df = engine._load_dataset()

    assert loaded_df is None


@mock.patch("mammography.training.cv_engine.create_kfold_splits")
@mock.patch("mammography.training.cv_engine.load_dataset_dataframe")
def test_run_empty_dataset_raises_error(
    mock_load_dataset: mock.MagicMock,
    mock_create_splits: mock.MagicMock,
    tmp_path: Path,
) -> None:
    """Test that run() raises RuntimeError for empty dataset."""
    # Mock empty dataset
    mock_load_dataset.return_value = pd.DataFrame()

    config = TrainConfig(csv="dummy.csv", outdir=str(tmp_path / "cv_output"))
    engine = CrossValidationEngine(config, n_folds=3)

    with pytest.raises(RuntimeError, match="Dataset vazio ou invalido"):
        engine.run()


@mock.patch("mammography.training.cv_engine.create_kfold_splits")
@mock.patch("mammography.training.cv_engine.load_dataset_dataframe")
def test_run_insufficient_folds_raises_error(
    mock_load_dataset: mock.MagicMock,
    mock_create_splits: mock.MagicMock,
    tmp_path: Path,
) -> None:
    """Test that run() raises RuntimeError when insufficient folds are created."""
    # Mock dataset
    mock_load_dataset.return_value = _create_sample_dataframe(100)

    # Mock create_kfold_splits to return fewer folds than expected
    mock_create_splits.return_value = [
        (_create_sample_dataframe(80), _create_sample_dataframe(20)),
        (_create_sample_dataframe(80), _create_sample_dataframe(20)),
    ]  # Only 2 folds instead of 5

    config = TrainConfig(csv="dummy.csv", outdir=str(tmp_path / "cv_output"), epochs=1)
    engine = CrossValidationEngine(config, n_folds=5)

    with pytest.raises(RuntimeError, match="Esperado 5 folds, obtido 2"):
        engine.run()


def test_fold_result_dataclass_creation() -> None:
    """Test FoldResult dataclass can be created and accessed."""
    fold_result = FoldResult(
        fold_idx=0,
        train_size=80,
        val_size=20,
        best_epoch=10,
        best_val_acc=0.85,
        best_val_kappa=0.80,
        best_val_macro_f1=0.83,
        best_val_auc=0.90,
        final_train_loss=0.3,
        final_train_acc=0.88,
        checkpoint_path=Path("/tmp/checkpoint.pt"),
        metrics_path=Path("/tmp/metrics.json"),
    )

    assert fold_result.fold_idx == 0
    assert fold_result.train_size == 80
    assert fold_result.val_size == 20
    assert fold_result.best_epoch == 10
    assert fold_result.best_val_acc == 0.85
    assert fold_result.best_val_kappa == 0.80
    assert fold_result.best_val_macro_f1 == 0.83
    assert fold_result.best_val_auc == 0.90
    assert fold_result.final_train_loss == 0.3
    assert fold_result.final_train_acc == 0.88
    assert isinstance(fold_result.checkpoint_path, Path)
    assert isinstance(fold_result.metrics_path, Path)


def test_fold_result_with_none_auc() -> None:
    """Test FoldResult with None AUC value."""
    fold_result = FoldResult(
        fold_idx=1,
        train_size=100,
        val_size=25,
        best_epoch=5,
        best_val_acc=0.75,
        best_val_kappa=0.70,
        best_val_macro_f1=0.72,
        best_val_auc=None,  # Can be None
        final_train_loss=0.5,
        final_train_acc=0.78,
        checkpoint_path=Path("/tmp/checkpoint.pt"),
        metrics_path=Path("/tmp/metrics.json"),
    )

    assert fold_result.best_val_auc is None


def test_cv_engine_reproducibility_seed(tmp_path: Path) -> None:
    """Test that cv_seed is used for fold splitting."""
    config = TrainConfig(csv="dummy.csv", outdir=str(tmp_path / "cv_output"))

    engine1 = CrossValidationEngine(config, n_folds=5, cv_seed=42)
    engine2 = CrossValidationEngine(config, n_folds=5, cv_seed=42)
    engine3 = CrossValidationEngine(config, n_folds=5, cv_seed=123)

    assert engine1.cv_seed == engine2.cv_seed
    assert engine1.cv_seed != engine3.cv_seed


def test_cv_engine_custom_n_folds(tmp_path: Path) -> None:
    """Test CrossValidationEngine with different n_folds values."""
    config = TrainConfig(csv="dummy.csv", outdir=str(tmp_path / "cv_output"))

    for n_folds in [2, 3, 5, 10]:
        engine = CrossValidationEngine(config, n_folds=n_folds)
        assert engine.n_folds == n_folds


def test_aggregate_results_fold_details_structure() -> None:
    """Test that fold_results in aggregated output has correct structure."""
    config = TrainConfig(csv="dummy.csv", outdir="/tmp/cv")
    engine = CrossValidationEngine(config, n_folds=3)

    fold_results = [
        _create_sample_fold_result(i, val_acc=0.80 + i * 0.01)
        for i in range(3)
    ]

    aggregated = engine._aggregate_results(fold_results)

    fold_details = aggregated["fold_results"]

    # Check each fold entry has required fields
    for i, fold_entry in enumerate(fold_details):
        assert fold_entry["fold_idx"] == i
        assert "best_val_acc" in fold_entry
        assert "best_val_kappa" in fold_entry
        assert "best_val_macro_f1" in fold_entry
        assert "best_val_auc" in fold_entry
        assert "best_epoch" in fold_entry
        assert "train_size" in fold_entry
        assert "val_size" in fold_entry


def test_aggregate_results_numerical_stability() -> None:
    """Test aggregation handles numerical edge cases."""
    config = TrainConfig(csv="dummy.csv", outdir="/tmp/cv")
    engine = CrossValidationEngine(config, n_folds=3)

    # Create fold results with identical values (std should be 0)
    fold_results = [
        _create_sample_fold_result(i, val_acc=0.85)
        for i in range(3)
    ]

    aggregated = engine._aggregate_results(fold_results)

    assert aggregated["mean_val_acc"] == 0.85
    assert aggregated["std_val_acc"] == 0.0
    assert aggregated["min_val_acc"] == 0.85
    assert aggregated["max_val_acc"] == 0.85
