# DISCLAIMER: Educational project only - NOT for clinical or medical diagnostic purposes.
"""
Integration tests for end-to-end model comparison workflow.

These tests validate that the model comparison pipeline works correctly from
CLI invocation through metrics loading to report generation, using minimal
synthetic datasets to ensure rapid execution.

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
pd = pytest.importorskip("pandas")
pytest.importorskip("numpy")

from mammography.commands import compare_models
from mammography.vis.model_comparison import ModelComparisonEngine, ModelMetrics


@pytest.mark.integration
@pytest.mark.slow
class TestCompareModelsWorkflow:
    """Integration tests for complete model comparison workflow."""

    def test_compare_models_help_displays(self) -> None:
        """Test that compare-models --help displays without errors."""
        with patch.object(sys, "argv", ["compare_models", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                compare_models.main()
            # argparse exits with 0 on --help
            assert exc_info.value.code == 0

    def test_comparison_engine_instantiation(self, tmp_path: Path) -> None:
        """Test that ModelComparisonEngine instantiates correctly."""
        # Create minimal run directories
        run1 = tmp_path / "run1" / "results"
        run2 = tmp_path / "run2" / "results"
        run1.mkdir(parents=True)
        run2.mkdir(parents=True)

        # Create minimal summary.json files
        for run_dir in [run1, run2]:
            summary = {
                "arch": "efficientnet_b0",
                "dataset": "test_dataset",
                "epochs": 1,
                "batch_size": 4,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))

            # Create metrics directory with best_metrics.json
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            metrics_data = {
                "acc": 0.85,
                "kappa_quadratic": 0.82,
                "macro_f1": 0.83,
                "auc_ovr": 0.89,
                "balanced_acc": 0.84,
                "confusion_matrix": [[10, 2], [1, 12]],
                "classification_report": {
                    "0": {"precision": 0.91, "recall": 0.83, "f1-score": 0.87, "support": 12},
                    "1": {"precision": 0.86, "recall": 0.92, "f1-score": 0.89, "support": 13},
                },
                "val_loss": 0.45,
                "epoch": 10,
            }
            (metrics_dir / "best_metrics.json").write_text(json.dumps(metrics_data))

        # Instantiate engine
        engine = ModelComparisonEngine([run1, run2])
        assert engine is not None
        assert len(engine.run_paths) == 2
        assert len(engine.model_names) == 2

    def test_comparison_engine_validates_empty_paths(self) -> None:
        """Test that ModelComparisonEngine validates empty run_paths."""
        with pytest.raises(ValueError, match="run_paths cannot be empty"):
            ModelComparisonEngine([])

    def test_comparison_engine_validates_nonexistent_paths(self) -> None:
        """Test that ModelComparisonEngine validates nonexistent paths."""
        nonexistent = Path("/nonexistent/path/to/model")
        with pytest.raises(FileNotFoundError, match="Run path does not exist"):
            ModelComparisonEngine([nonexistent])

    @pytest.mark.cpu
    def test_end_to_end_comparison_workflow(self, tmp_path: Path) -> None:
        """Test complete comparison workflow with synthetic model runs.

        This test validates:
        1. Creating synthetic model run directories with metrics
        2. Loading metrics via ModelComparisonEngine
        3. Generating side-by-side comparison
        4. Exporting comparison tables
        5. Ranking models by metric
        6. Verifying output files exist
        """
        # Create two synthetic model runs
        run1_dir = tmp_path / "model_run_1" / "results"
        run2_dir = tmp_path / "model_run_2" / "results"
        run1_dir.mkdir(parents=True)
        run2_dir.mkdir(parents=True)

        # Model 1: EfficientNet with higher accuracy
        summary1 = {
            "arch": "efficientnet_b0",
            "dataset": "mamografias",
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 0.001,
            "img_size": 224,
        }
        (run1_dir / "summary.json").write_text(json.dumps(summary1))

        metrics1_dir = run1_dir / "metrics"
        metrics1_dir.mkdir()
        metrics1 = {
            "acc": 0.8700,
            "kappa_quadratic": 0.8400,
            "macro_f1": 0.8550,
            "auc_ovr": 0.9100,
            "balanced_acc": 0.8650,
            "confusion_matrix": [[45, 3, 2, 0], [2, 43, 4, 1], [1, 3, 41, 5], [0, 1, 3, 46]],
            "classification_report": {
                "0": {"precision": 0.94, "recall": 0.90, "f1-score": 0.92, "support": 50},
                "1": {"precision": 0.86, "recall": 0.86, "f1-score": 0.86, "support": 50},
                "2": {"precision": 0.82, "recall": 0.82, "f1-score": 0.82, "support": 50},
                "3": {"precision": 0.88, "recall": 0.92, "f1-score": 0.90, "support": 50},
            },
            "val_loss": 0.3800,
            "epoch": 15,
        }
        (metrics1_dir / "best_metrics.json").write_text(json.dumps(metrics1))

        # Model 2: ResNet50 with lower accuracy
        summary2 = {
            "arch": "resnet50",
            "dataset": "mamografias",
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 0.001,
            "img_size": 224,
        }
        (run2_dir / "summary.json").write_text(json.dumps(summary2))

        metrics2_dir = run2_dir / "metrics"
        metrics2_dir.mkdir()
        metrics2 = {
            "acc": 0.8400,
            "kappa_quadratic": 0.8100,
            "macro_f1": 0.8250,
            "auc_ovr": 0.8850,
            "balanced_acc": 0.8350,
            "confusion_matrix": [[43, 4, 3, 0], [3, 41, 5, 1], [2, 4, 39, 5], [0, 1, 4, 45]],
            "classification_report": {
                "0": {"precision": 0.89, "recall": 0.86, "f1-score": 0.88, "support": 50},
                "1": {"precision": 0.82, "recall": 0.82, "f1-score": 0.82, "support": 50},
                "2": {"precision": 0.76, "recall": 0.78, "f1-score": 0.77, "support": 50},
                "3": {"precision": 0.88, "recall": 0.90, "f1-score": 0.89, "support": 50},
            },
            "val_loss": 0.4200,
            "epoch": 17,
        }
        (metrics2_dir / "best_metrics.json").write_text(json.dumps(metrics2))

        # Create output directory
        output_dir = tmp_path / "comparison_outputs"
        output_dir.mkdir(exist_ok=True)

        # Initialize comparison engine
        engine = ModelComparisonEngine([run1_dir, run2_dir])

        try:
            # Load all metrics
            model1_metrics = engine.load_model_metrics(run1_dir, "efficientnet_b0")
            model2_metrics = engine.load_model_metrics(run2_dir, "resnet50")

            # Verify metrics loaded correctly
            assert model1_metrics is not None
            assert model2_metrics is not None
            assert model1_metrics.accuracy == 0.8700
            assert model2_metrics.accuracy == 0.8400

            # Build metrics list for comparison
            metrics_list = [model1_metrics, model2_metrics]

            # Generate side-by-side comparison
            comparison_df = engine.get_side_by_side_comparison(metrics_list)
            assert comparison_df is not None
            assert len(comparison_df) == 2  # Two models

            # Verify comparison includes key metrics
            assert "model_name" in comparison_df.columns
            assert "accuracy" in comparison_df.columns
            assert "kappa" in comparison_df.columns
            assert "macro_f1" in comparison_df.columns

            # Verify metric values match
            effnet_row = comparison_df[comparison_df["model_name"] == "efficientnet_b0"]
            assert len(effnet_row) == 1
            assert abs(effnet_row["accuracy"].values[0] - 0.8700) < 1e-4

            resnet_row = comparison_df[comparison_df["model_name"] == "resnet50"]
            assert len(resnet_row) == 1
            assert abs(resnet_row["accuracy"].values[0] - 0.8400) < 1e-4

            # Test ranking
            ranked_df = engine.rank_models_by_metric(metrics_list, metric="accuracy")
            assert ranked_df is not None
            assert len(ranked_df) == 2

            # Verify ranking order (descending by accuracy)
            assert ranked_df.iloc[0]["model_name"] == "efficientnet_b0"  # Higher accuracy first
            assert ranked_df.iloc[1]["model_name"] == "resnet50"

            # Test export functionality
            from mammography.vis.model_comparison import export_comparison_table

            exported_paths = export_comparison_table(
                comparison_df,
                output_dir / "comparison_table",
                formats=["csv", "json"],
            )

            # Verify exported files exist
            assert len(exported_paths) >= 2
            for path in exported_paths:
                assert path.exists(), f"Exported file not found: {path}"

            # Verify CSV content
            csv_path = output_dir / "comparison_table.csv"
            if csv_path.exists():
                import pandas as pd
                df_loaded = pd.read_csv(csv_path)
                assert len(df_loaded) == 2
                assert "model_name" in df_loaded.columns
                assert "accuracy" in df_loaded.columns

        finally:
            # Cleanup
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

    @pytest.mark.cpu
    def test_comparison_cli_with_synthetic_runs(self, tmp_path: Path) -> None:
        """Test compare-models CLI command with synthetic model runs."""
        # Create two synthetic runs
        run1 = tmp_path / "run1" / "results"
        run2 = tmp_path / "run2" / "results"
        run1.mkdir(parents=True)
        run2.mkdir(parents=True)

        # Create minimal required files
        for idx, run_dir in enumerate([run1, run2], start=1):
            summary = {
                "arch": f"model_{idx}",
                "dataset": "test",
                "epochs": 5,
                "batch_size": 16,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))

            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir()
            metrics = {
                "acc": 0.80 + idx * 0.02,
                "kappa_quadratic": 0.75 + idx * 0.02,
                "macro_f1": 0.78 + idx * 0.02,
                "auc_ovr": 0.85 + idx * 0.02,
                "balanced_acc": 0.79 + idx * 0.02,
                "confusion_matrix": [[20, 5], [3, 22]],
                "classification_report": {
                    "0": {"precision": 0.87, "recall": 0.80, "f1-score": 0.83, "support": 25},
                    "1": {"precision": 0.81, "recall": 0.88, "f1-score": 0.85, "support": 25},
                },
                "val_loss": 0.50 - idx * 0.02,
                "epoch": idx * 3,
            }
            (metrics_dir / "best_metrics.json").write_text(json.dumps(metrics))

        output_dir = tmp_path / "cli_comparison_output"

        try:
            # Run CLI command
            exit_code = compare_models.main([
                "--run", str(run1),
                "--run", str(run2),
                "--outdir", str(output_dir),
                "--export", "csv,json",
                "--log-level", "warning",
            ])

            # Verify command succeeded
            assert exit_code == 0

            # Verify output directory created
            assert output_dir.exists()

            # Verify comparison files exist
            comparison_json = output_dir / "comparison_metrics.json"
            assert comparison_json.exists(), "comparison_metrics.json not found"

            # Verify comparison table exported
            comparison_csv = output_dir / "comparison_table.csv"
            assert comparison_csv.exists(), "comparison_table.csv not found"

            # Verify JSON structure
            with open(comparison_json) as f:
                comparison_data = json.load(f)
                assert isinstance(comparison_data, list)
                assert len(comparison_data) == 2

        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

    @pytest.mark.cpu
    def test_comparison_with_cv_results(self, tmp_path: Path) -> None:
        """Test comparison workflow with cross-validation results."""
        # Create synthetic CV run directory
        cv_run = tmp_path / "cv_run"
        cv_run.mkdir()

        # Create cv_summary.json
        cv_summary = {
            "n_folds": 3,
            "cv_seed": 42,
            "aggregated_metrics": {
                "accuracy": {
                    "mean": 0.8500,
                    "std": 0.0200,
                    "min": 0.8300,
                    "max": 0.8700,
                    "ci_lower": 0.8000,
                    "ci_upper": 0.9000,
                },
                "kappa": {
                    "mean": 0.8200,
                    "std": 0.0250,
                    "min": 0.7950,
                    "max": 0.8450,
                    "ci_lower": 0.7700,
                    "ci_upper": 0.8700,
                },
                "macro_f1": {
                    "mean": 0.8350,
                    "std": 0.0220,
                    "min": 0.8130,
                    "max": 0.8570,
                    "ci_lower": 0.7900,
                    "ci_upper": 0.8800,
                },
            },
            "fold_results": [
                {"fold_idx": 0, "val_acc": 0.8500, "val_kappa": 0.8200, "val_macro_f1": 0.8350},
                {"fold_idx": 1, "val_acc": 0.8300, "val_kappa": 0.7950, "val_macro_f1": 0.8130},
                {"fold_idx": 2, "val_acc": 0.8700, "val_kappa": 0.8450, "val_macro_f1": 0.8570},
            ],
        }
        (cv_run / "cv_summary.json").write_text(json.dumps(cv_summary))

        # Create a regular run for comparison
        regular_run = tmp_path / "regular_run" / "results"
        regular_run.mkdir(parents=True)

        summary = {
            "arch": "resnet50",
            "dataset": "test",
            "epochs": 10,
            "batch_size": 32,
        }
        (regular_run / "summary.json").write_text(json.dumps(summary))

        metrics_dir = regular_run / "metrics"
        metrics_dir.mkdir()
        metrics = {
            "acc": 0.8400,
            "kappa_quadratic": 0.8100,
            "macro_f1": 0.8250,
            "auc_ovr": 0.8850,
            "balanced_acc": 0.8350,
            "confusion_matrix": [[20, 5], [3, 22]],
            "classification_report": {
                "0": {"precision": 0.87, "recall": 0.80, "f1-score": 0.83, "support": 25},
                "1": {"precision": 0.81, "recall": 0.88, "f1-score": 0.85, "support": 25},
            },
            "val_loss": 0.42,
            "epoch": 8,
        }
        (metrics_dir / "best_metrics.json").write_text(json.dumps(metrics))

        # Test discovery function
        from mammography.commands.compare_models import discover_run_directories

        discovered = discover_run_directories([cv_run, regular_run])

        # Should discover CV run and regular run
        assert len(discovered) >= 1  # At least regular_run should be discovered
        assert regular_run in discovered

    @pytest.mark.cpu
    def test_statistical_tests_workflow(self, tmp_path: Path) -> None:
        """
        Exercise the end-to-end statistical test workflow using two synthetic runs and their predictions.
        
        Creates two model run directories with summary and best_metrics, writes per-run validation prediction CSVs with slightly different accuracies, invokes the compare_models CLI with --statistical-tests, and verifies that the command succeeds and produces:
        - a statistical_tests.csv file containing one pairwise comparison with columns `statistic`, `p_value`, and `significant`
        - a statistical_tests.tex LaTeX export
        """
        # Create two model runs with predictions
        run1 = tmp_path / "run1" / "results"
        run2 = tmp_path / "run2" / "results"

        for run_dir in [run1, run2]:
            run_dir.mkdir(parents=True)

            # Create summary.json
            summary = {
                "arch": "efficientnet_b0",
                "dataset": "test_dataset",
                "epochs": 1,
                "batch_size": 4,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))

            # Create metrics
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            metrics_data = {
                "acc": 0.85,
                "kappa_quadratic": 0.82,
                "macro_f1": 0.83,
            }
            (metrics_dir / "best_metrics.json").write_text(json.dumps(metrics_data))

        # Create predictions (slightly different for each model)
        y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1]

        pred1_df = pd.DataFrame({
            "true_label": y_true,
            "prediction": [0, 1, 1, 0, 0, 1, 0, 1, 1, 1],  # 8/10 correct
        })
        (run1 / "val_predictions.csv").write_text(pred1_df.to_csv(index=False))

        pred2_df = pd.DataFrame({
            "true_label": y_true,
            "prediction": [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],  # 7/10 correct
        })
        (run2 / "val_predictions.csv").write_text(pred2_df.to_csv(index=False))

        # Run comparison with statistical tests
        outdir = tmp_path / "comparison_output"
        outdir.mkdir()

        argv = [
            "--run", str(run1),
            "--run", str(run2),
            "--outdir", str(outdir),
            "--statistical-tests",
        ]

        with patch.object(sys, "argv", ["compare_models"] + argv):
            exit_code = compare_models.main()

        # Verify success
        assert exit_code == 0

        # Verify statistical test results file created
        test_csv = outdir / "statistical_tests.csv"
        assert test_csv.exists(), "statistical_tests.csv should be created"

        # Verify results content
        test_df = pd.read_csv(test_csv)
        assert len(test_df) == 1, "Should have 1 pairwise comparison"
        assert "statistic" in test_df.columns
        assert "p_value" in test_df.columns
        assert "significant" in test_df.columns

        # Verify LaTeX export
        test_tex = outdir / "statistical_tests.tex"
        assert test_tex.exists(), "statistical_tests.tex should be created"


@pytest.mark.integration
class TestCompareModelsCLIIntegration:
    """Integration tests for compare-models CLI command."""

    def test_compare_models_cli_routing(self) -> None:
        """Test that compare-models command routes correctly."""
        from mammography import cli

        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "compare-models"])

        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.compare_models"

    def test_compare_models_cli_help_works(self) -> None:
        """Test that compare-models --help displays without errors."""
        from mammography import cli

        with patch.object(sys, "argv", ["mammography", "compare-models", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            # argparse exits with 0 on --help
            assert exc_info.value.code == 0

    def test_compare_models_requires_multiple_runs(self, tmp_path: Path) -> None:
        """Test that compare-models requires at least 2 runs."""
        # Create single run
        run1 = tmp_path / "run1" / "results"
        run1.mkdir(parents=True)

        summary = {"arch": "test", "dataset": "test"}
        (run1 / "summary.json").write_text(json.dumps(summary))

        metrics_dir = run1 / "metrics"
        metrics_dir.mkdir()
        metrics = {
            "acc": 0.85,
            "kappa_quadratic": 0.82,
            "macro_f1": 0.83,
            "auc_ovr": 0.89,
            "balanced_acc": 0.84,
            "confusion_matrix": [[20, 5], [3, 22]],
            "classification_report": {
                "0": {"precision": 0.87, "recall": 0.80, "f1-score": 0.83, "support": 25},
                "1": {"precision": 0.81, "recall": 0.88, "f1-score": 0.85, "support": 25},
            },
            "val_loss": 0.45,
            "epoch": 10,
        }
        (metrics_dir / "best_metrics.json").write_text(json.dumps(metrics))

        # Try to run with single run
        exit_code = compare_models.main([
            "--run", str(run1),
            "--log-level", "critical",
        ])

        # Should fail with exit code 1
        assert exit_code == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])