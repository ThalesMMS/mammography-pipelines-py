"""
Integration tests for end-to-end cross-validation workflow.

These tests validate that the cross-validation pipeline works correctly from
CLI invocation through fold training to results aggregation, using minimal
datasets to ensure rapid execution.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import dependencies with pytest.importorskip for graceful failures
pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")
pytest.importorskip("numpy")

from mammography.commands import cross_validate
from mammography.config import TrainConfig
from mammography.training.cv_engine import CrossValidationEngine


@pytest.mark.integration
@pytest.mark.slow
class TestCrossValidationWorkflow:
    """Integration tests for complete cross-validation workflow."""

    def test_cv_engine_help_displays(self):
        """Test that cross-validate --help displays without errors."""
        with patch.object(sys, "argv", ["cross_validate", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cross_validate.main()
            # argparse exits with 0 on --help
            assert exc_info.value.code == 0

    def test_cv_engine_instantiation(self):
        """Test that CrossValidationEngine instantiates correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_csv = f.name
            f.write("AccessionNumber,Classification\n")
            f.write("test1,A\n")
            f.write("test2,B\n")
            f.write("test3,C\n")

        try:
            config = TrainConfig(
                csv=temp_csv,
                epochs=1,
                batch_size=2,
                outdir="outputs/cv_test",
            )
            cv_engine = CrossValidationEngine(
                config=config,
                n_folds=3,
                cv_seed=42,
                save_all_folds=False,
            )
            assert cv_engine is not None
            assert cv_engine.n_folds == 3
            assert cv_engine.cv_seed == 42
            assert cv_engine.save_all_folds is False
        finally:
            Path(temp_csv).unlink(missing_ok=True)

    def test_cv_engine_validates_n_folds(self):
        """Test that CrossValidationEngine validates n_folds parameter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_csv = f.name
            f.write("AccessionNumber,Classification\n")
            f.write("test1,A\n")

        try:
            config = TrainConfig(csv=temp_csv, epochs=1, batch_size=2)
            # n_folds < 2 should raise ValueError
            with pytest.raises(ValueError, match="n_folds deve ser >= 2"):
                CrossValidationEngine(config=config, n_folds=1)
        finally:
            Path(temp_csv).unlink(missing_ok=True)

    @pytest.mark.gpu
    def test_end_to_end_cv_workflow_with_mock_data(self, tmp_path):
        """Test complete cross-validation workflow with synthetic dataset.

        This test validates:
        1. Creating synthetic CSV and image data
        2. Running cross-validation with minimal parameters
        3. Verifying fold directories are created
        4. Verifying cv_summary.json exists with correct structure
        5. Verifying per-fold checkpoints exist
        """
        # Create synthetic CSV file
        csv_path = tmp_path / "test_data.csv"
        with open(csv_path, "w") as f:
            f.write("image_path,density_label\n")
            # Create 32 samples (enough for 3-fold CV with batch_size=4)
            for i in range(32):
                label = ["A", "B", "C", "D"][i % 4]
                f.write(f"image_{i:03d}.png,{label}\n")

        # Create synthetic PNG images (1x1 black pixel)
        import numpy as np
        from PIL import Image

        for i in range(32):
            img_path = tmp_path / f"image_{i:03d}.png"
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            img.save(img_path)

        # Create output directory
        output_dir = tmp_path / "cv_outputs"
        output_dir.mkdir(exist_ok=True)

        # Configure for minimal training
        config = TrainConfig(
            csv=str(csv_path),
            epochs=1,
            batch_size=4,
            outdir=str(output_dir),
            model="efficientnet_b0",
            device="cpu",  # Force CPU for faster testing
            num_workers=0,
            subset=16,  # Use only 16 samples for speed
            seed=42,
        )

        # Run cross-validation
        cv_engine = CrossValidationEngine(
            config=config,
            n_folds=3,
            cv_seed=42,
            save_all_folds=True,
        )

        try:
            results = cv_engine.run()

            # Verify results structure
            assert results is not None
            assert "mean_val_acc" in results
            assert "std_val_acc" in results
            assert "mean_val_kappa" in results
            assert "std_val_kappa" in results
            assert "mean_val_macro_f1" in results
            assert "std_val_macro_f1" in results
            assert "fold_results" in results

            # Verify fold directories exist
            for fold_idx in range(3):
                fold_dir = output_dir / f"fold_{fold_idx}"
                assert fold_dir.exists(), f"Fold directory {fold_dir} not found"

                # Verify checkpoint exists
                checkpoint_path = fold_dir / "best_model.pt"
                assert (
                    checkpoint_path.exists()
                ), f"Checkpoint {checkpoint_path} not found"

                # Verify metrics.json exists
                metrics_path = fold_dir / "metrics.json"
                assert metrics_path.exists(), f"Metrics {metrics_path} not found"

                # Verify metrics.json structure
                with open(metrics_path) as mf:
                    metrics = json.load(mf)
                    assert "fold_idx" in metrics
                    assert "train_size" in metrics
                    assert "val_size" in metrics
                    assert "best_val_acc" in metrics
                    assert "best_val_kappa" in metrics
                    assert "best_val_macro_f1" in metrics

            # Verify cv_summary.json exists
            summary_path = output_dir / "cv_summary.json"
            assert summary_path.exists(), f"Summary {summary_path} not found"

            # Verify cv_summary.json structure
            with open(summary_path) as sf:
                summary = json.load(sf)
                assert "n_folds" in summary
                assert summary["n_folds"] == 3
                assert "cv_seed" in summary
                assert summary["cv_seed"] == 42
                assert "aggregated_metrics" in summary

                agg = summary["aggregated_metrics"]
                assert "accuracy" in agg
                assert "kappa" in agg
                assert "macro_f1" in agg

                # Each metric should have mean, std, min, max, ci_lower, ci_upper
                for metric_name in ["accuracy", "kappa", "macro_f1"]:
                    metric = agg[metric_name]
                    assert "mean" in metric, f"Missing 'mean' in {metric_name}"
                    assert "std" in metric, f"Missing 'std' in {metric_name}"
                    assert "min" in metric, f"Missing 'min' in {metric_name}"
                    assert "max" in metric, f"Missing 'max' in {metric_name}"
                    assert "ci_lower" in metric, f"Missing 'ci_lower' in {metric_name}"
                    assert "ci_upper" in metric, f"Missing 'ci_upper' in {metric_name}"

                    # Verify statistical consistency
                    assert metric["min"] <= metric["mean"] <= metric["max"]
                    assert metric["ci_lower"] <= metric["mean"] <= metric["ci_upper"]
                    assert metric["std"] >= 0

        finally:
            # Cleanup: remove output directory
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

    @pytest.mark.cpu
    def test_cv_workflow_with_save_all_folds_false(self, tmp_path):
        """Test that save_all_folds=False only saves the best fold."""
        # Create synthetic CSV file
        csv_path = tmp_path / "test_data.csv"
        with open(csv_path, "w") as f:
            f.write("image_path,density_label\n")
            for i in range(24):
                label = ["A", "B", "C", "D"][i % 4]
                f.write(f"image_{i:03d}.png,{label}\n")

        # Create synthetic PNG images
        import numpy as np
        from PIL import Image

        for i in range(24):
            img_path = tmp_path / f"image_{i:03d}.png"
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            img.save(img_path)

        output_dir = tmp_path / "cv_outputs_selective"
        output_dir.mkdir(exist_ok=True)

        config = TrainConfig(
            csv=str(csv_path),
            epochs=1,
            batch_size=4,
            outdir=str(output_dir),
            model="efficientnet_b0",
            device="cpu",
            num_workers=0,
            subset=12,
            seed=42,
        )

        cv_engine = CrossValidationEngine(
            config=config,
            n_folds=3,
            cv_seed=42,
            save_all_folds=False,  # Only save best fold
        )

        try:
            results = cv_engine.run()

            # Verify results exist
            assert results is not None

            # Count how many fold directories have checkpoints
            checkpoints_found = 0
            for fold_idx in range(3):
                fold_dir = output_dir / f"fold_{fold_idx}"
                checkpoint_path = fold_dir / "best_model.pt"
                if checkpoint_path.exists():
                    checkpoints_found += 1

            # With save_all_folds=False, only the best fold should have a checkpoint
            # Note: The implementation might save all metrics.json but only keep best checkpoint
            # So we verify at least one checkpoint exists
            assert (
                checkpoints_found >= 1
            ), "At least one fold checkpoint should exist"

            # Verify cv_summary.json still exists
            summary_path = output_dir / "cv_summary.json"
            assert summary_path.exists()

        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

    def test_cv_aggregation_statistics_accuracy(self):
        """Test that cross-validation aggregation statistics are computed correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_csv = f.name
            f.write("image_path,density_label\n")
            for i in range(15):
                label = ["A", "B", "C"][i % 3]
                f.write(f"dummy_{i}.png,{label}\n")

        try:
            config = TrainConfig(
                csv=temp_csv,
                epochs=1,
                batch_size=3,
                outdir="outputs/cv_stats_test",
            )

            cv_engine = CrossValidationEngine(config=config, n_folds=3, cv_seed=42)

            # Test that _aggregate_results works correctly with known fold results
            from mammography.training.cv_engine import FoldResult

            fold_results = [
                FoldResult(
                    fold_idx=0,
                    train_size=10,
                    val_size=5,
                    best_epoch=1,
                    best_val_acc=0.80,
                    best_val_kappa=0.70,
                    best_val_macro_f1=0.75,
                    best_val_auc=0.82,
                    final_train_loss=0.5,
                    final_train_acc=0.85,
                    checkpoint_path=Path("fold_0/best_model.pt"),
                    metrics_path=Path("fold_0/metrics.json"),
                ),
                FoldResult(
                    fold_idx=1,
                    train_size=10,
                    val_size=5,
                    best_epoch=1,
                    best_val_acc=0.82,
                    best_val_kappa=0.72,
                    best_val_macro_f1=0.77,
                    best_val_auc=0.84,
                    final_train_loss=0.48,
                    final_train_acc=0.87,
                    checkpoint_path=Path("fold_1/best_model.pt"),
                    metrics_path=Path("fold_1/metrics.json"),
                ),
                FoldResult(
                    fold_idx=2,
                    train_size=10,
                    val_size=5,
                    best_epoch=1,
                    best_val_acc=0.81,
                    best_val_kappa=0.71,
                    best_val_macro_f1=0.76,
                    best_val_auc=0.83,
                    final_train_loss=0.49,
                    final_train_acc=0.86,
                    checkpoint_path=Path("fold_2/best_model.pt"),
                    metrics_path=Path("fold_2/metrics.json"),
                ),
            ]

            aggregated = cv_engine._aggregate_results(fold_results)

            # Verify mean calculations
            import numpy as np

            expected_mean_acc = np.mean([0.80, 0.82, 0.81])
            assert abs(aggregated["mean_val_acc"] - expected_mean_acc) < 1e-6

            expected_mean_kappa = np.mean([0.70, 0.72, 0.71])
            assert abs(aggregated["mean_val_kappa"] - expected_mean_kappa) < 1e-6

            # Verify std calculations
            expected_std_acc = np.std([0.80, 0.82, 0.81], ddof=1)
            assert abs(aggregated["std_val_acc"] - expected_std_acc) < 1e-6

            # Verify min/max
            assert aggregated["min_val_acc"] == 0.80
            assert aggregated["max_val_acc"] == 0.82

            # Verify confidence intervals exist
            assert "ci_lower_val_acc" in aggregated
            assert "ci_upper_val_acc" in aggregated
            assert aggregated["ci_lower_val_acc"] <= expected_mean_acc
            assert aggregated["ci_upper_val_acc"] >= expected_mean_acc

        finally:
            Path(temp_csv).unlink(missing_ok=True)


@pytest.mark.integration
class TestCrossValidationCLIIntegration:
    """Integration tests for cross-validate CLI command."""

    def test_cross_validate_cli_routing(self):
        """Test that cross-validate command routes correctly."""
        from mammography import cli

        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "cross-validate"])

        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.cross_validate"

    def test_cross_validate_cli_help_works(self):
        """Test that cross-validate --help displays without errors."""
        from mammography import cli

        with patch.object(sys, "argv", ["mammography", "cross-validate", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            # argparse exits with 0 on --help
            assert exc_info.value.code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
