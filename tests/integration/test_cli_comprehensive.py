"""
Comprehensive CLI integration tests for all mammography commands.

Tests all CLI subcommands with various argument combinations to ensure
proper routing, argument parsing, and error handling.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography import cli


class TestEmbedCommand:
    """Tests for the embed (extract_features) command."""

    def test_embed_basic_dry_run(self):
        """Test embed with minimal arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "embed"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.extract_features"

    def test_embed_with_config(self):
        """Test embed with config file argument."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "embed",
                "--config", "configs/paths.yaml"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()

    def test_embed_with_forwarded_args(self):
        """Test embed with forwarded arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "embed",
                "--dataset", "mamografias",
                "--subset", "10"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()
        # Verify forwarded args are passed
        cmd_args = mock_run.call_args[0][2]
        assert "--dataset" in cmd_args
        assert "mamografias" in cmd_args


class TestTrainDensityCommand:
    """Tests for the train-density command."""

    def test_train_density_basic_dry_run(self):
        """Test train-density with minimal arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "train-density"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.train"

    def test_train_density_with_epochs(self):
        """Test train-density with epochs argument."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "train-density",
                "--epochs", "1",
                "--subset", "32"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()

    def test_train_density_with_config(self):
        """Test train-density with config file."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "train-density",
                "--config", "configs/density.yaml"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()


class TestVisualizeCommand:
    """Tests for the visualize command."""

    def test_visualize_basic_dry_run(self):
        """Test visualize with minimal required arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "visualize",
                "--input", "embeddings.npy",
                "--outdir", "outputs/vis"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.visualize"

    def test_visualize_with_labels(self):
        """Test visualize with labels file."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "visualize",
                "--input", "embeddings.npy",
                "--labels", "labels.csv",
                "--outdir", "outputs/vis"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()

    def test_visualize_with_tsne_options(self):
        """Test visualize with t-SNE specific options."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "visualize",
                "--input", "embeddings.npy",
                "--tsne",
                "--perplexity", "50",
                "--tsne-iter", "2000",
                "--outdir", "outputs/vis"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()

    def test_visualize_multiple_plot_types(self):
        """Test visualize with multiple visualization types."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "visualize",
                "--input", "embeddings.npy",
                "--tsne",
                "--pca",
                "--umap",
                "--heatmap",
                "--outdir", "outputs/vis"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()

    def test_visualize_from_run(self):
        """Test visualize with --from-run flag."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "visualize",
                "--input", "outputs/run_dir",
                "--from-run",
                "--outdir", "outputs/vis"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()


class TestExplainCommand:
    """Tests for the explain command."""

    def test_explain_basic_dry_run(self):
        """Test explain with minimal arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "explain"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.explain"

    def test_explain_with_config(self):
        """Test explain with config file."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "explain",
                "--config", "configs/explain.yaml"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()


class TestInferenceCommand:
    """Tests for the inference command."""

    def test_inference_basic_dry_run(self):
        """Test inference with minimal arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "inference"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.inference"

    def test_inference_with_forwarded_args(self):
        """Test inference with forwarded arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "inference",
                "--checkpoint", "model.pt",
                "--input", "image.dcm"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()


class TestEmbeddingsBaselinesCommand:
    """Tests for the embeddings-baselines command."""

    def test_embeddings_baselines_basic_dry_run(self):
        """Test embeddings-baselines with minimal arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "embeddings-baselines"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.embeddings_baselines"

    def test_embeddings_baselines_with_forwarded_args(self):
        """Test embeddings-baselines with forwarded arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "embeddings-baselines",
                "--embeddings", "embeddings.npy"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()


class TestDataAuditCommand:
    """Tests for the data-audit command."""

    def test_data_audit_basic_dry_run(self):
        """Test data-audit with minimal arguments."""
        with patch.object(cli, "_run_data_audit") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "data-audit"])
        assert exit_code == 0
        mock_run.assert_called_once()

    def test_data_audit_with_archive(self):
        """Test data-audit with archive path."""
        with patch.object(cli, "_run_data_audit") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "data-audit",
                "--archive", "archive",
                "--csv", "classificacao.csv"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()

    def test_data_audit_with_manifest(self):
        """Test data-audit with manifest output."""
        with patch.object(cli, "_run_data_audit") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "data-audit",
                "--manifest", "manifest.json"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()


class TestEvalExportCommand:
    """Tests for the eval-export command."""

    def test_eval_export_basic_dry_run(self):
        """Test eval-export with minimal arguments."""
        with patch.object(cli, "_print_eval_guidance") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "eval-export"])
        assert exit_code == 0
        mock_run.assert_called_once()

    def test_eval_export_with_runs(self):
        """Test eval-export with run directories."""
        with patch.object(cli, "_print_eval_guidance") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "eval-export",
                "--run", "outputs/results_1",
                "--run", "outputs/results_2"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()


class TestReportPackCommand:
    """Tests for the report-pack command."""

    def test_report_pack_missing_runs(self):
        """Test report-pack without required --run arguments."""
        with pytest.raises(SystemExit):
            cli.main(["--dry-run", "report-pack"])

    def test_report_pack_with_runs(self):
        """Test report-pack with run directories."""
        with patch.object(cli, "_run_report_pack") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "report-pack",
                "--run", "outputs/results_1"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()

    def test_report_pack_with_assets_dir(self):
        """Test report-pack with custom assets directory."""
        with patch.object(cli, "_run_report_pack") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "report-pack",
                "--run", "outputs/results_1",
                "--assets-dir", "custom/assets"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()


class TestTuneCommand:
    """Tests for the tune command."""

    def test_tune_basic_dry_run(self):
        """Test tune with minimal arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "tune"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.tune"

    def test_tune_with_forwarded_args(self):
        """Test tune with forwarded arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "tune",
                "--trials", "10"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()


class TestAugmentCommand:
    """Tests for the augment command."""

    def test_augment_basic_dry_run(self):
        """Test augment with minimal arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "augment"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.augment"

    def test_augment_with_forwarded_args(self):
        """Test augment with forwarded arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "augment",
                "--input", "images/",
                "--output", "augmented/"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()


class TestLabelDensityCommand:
    """Tests for the label-density command."""

    def test_label_density_basic_dry_run(self):
        """Test label-density with minimal arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "label-density"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.label_density"


class TestLabelPatchesCommand:
    """Tests for the label-patches command."""

    def test_label_patches_basic_dry_run(self):
        """Test label-patches with minimal arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "label-patches"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.label_patches"


class TestWebCommand:
    """Tests for the web command."""

    def test_web_basic_dry_run(self):
        """Test web with minimal arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "web"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.web"


class TestEdaCancerCommand:
    """Tests for the eda-cancer command."""

    def test_eda_cancer_basic_dry_run(self):
        """Test eda-cancer with minimal arguments."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "eda-cancer"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.eda_cancer"

    def test_eda_cancer_entrypoint(self):
        """Test eda-cancer uses correct entrypoint."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "eda-cancer"])
        assert exit_code == 0
        # Verify correct entrypoint is specified
        if len(mock_run.call_args[0]) > 3:
            assert "run_density_classifier_cli" in str(mock_run.call_args)


class TestWizardCommand:
    """Tests for the wizard command."""

    def test_wizard_basic_dry_run(self):
        """Test wizard with dry-run."""
        with patch("mammography.wizard.run_wizard") as mock_wizard:
            mock_wizard.return_value = 0
            exit_code = cli.main(["--dry-run", "wizard"])
        assert exit_code == 0
        mock_wizard.assert_called_once_with(dry_run=True)

    def test_wizard_without_dry_run(self):
        """Test wizard without dry-run."""
        with patch("mammography.wizard.run_wizard") as mock_wizard:
            mock_wizard.return_value = 0
            exit_code = cli.main(["wizard"])
        assert exit_code == 0
        mock_wizard.assert_called_once_with(dry_run=False)


class TestCLIGlobalOptions:
    """Tests for global CLI options."""

    def test_help_flag(self):
        """Test --help displays without errors."""
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["--help"])
        assert exc_info.value.code == 0

    def test_log_level_option(self):
        """Test --log-level option."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--log-level", "DEBUG",
                "--dry-run", "embed"
            ])
        assert exit_code == 0

    def test_all_log_levels(self):
        """Test all valid log levels."""
        log_levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
        for level in log_levels:
            with patch.object(cli, "_run_module_passthrough") as mock_run:
                mock_run.return_value = 0
                exit_code = cli.main([
                    "--log-level", level,
                    "--dry-run", "embed"
                ])
            assert exit_code == 0

    def test_no_command_shows_help(self):
        """Test that no command shows help and exits cleanly."""
        exit_code = cli.main([])
        assert exit_code == 0


class TestCommandHelp:
    """Tests for command-specific help."""

    def test_embed_help(self):
        """Test embed --help."""
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["embed", "--help"])
        assert exc_info.value.code == 0

    def test_train_density_help(self):
        """Test train-density --help."""
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["train-density", "--help"])
        assert exc_info.value.code == 0

    def test_visualize_help(self):
        """Test visualize --help."""
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["visualize", "--help"])
        assert exc_info.value.code == 0

    def test_inference_help(self):
        """Test inference --help."""
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["inference", "--help"])
        assert exc_info.value.code == 0

    def test_explain_help(self):
        """Test explain --help."""
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["explain", "--help"])
        assert exc_info.value.code == 0

    def test_tune_help(self):
        """Test tune --help."""
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["tune", "--help"])
        assert exc_info.value.code == 0


class TestErrorHandling:
    """Tests for CLI error handling."""

    def test_unknown_command(self):
        """Test unknown command raises error."""
        with pytest.raises(SystemExit):
            cli.main(["unknown-command"])

    def test_command_failure_propagates(self):
        """Test that command failures propagate correctly."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 1
            exit_code = cli.main(["--dry-run", "embed"])
        assert exit_code == 1

    def test_system_exit_with_string(self):
        """Test SystemExit with string code."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.side_effect = SystemExit("Error message")
            exit_code = cli.main(["--dry-run", "embed"])
        assert exit_code == 1


class TestConfigHandling:
    """Tests for config file handling."""

    def test_config_with_nonexistent_file(self):
        """Test config with nonexistent file logs warning but continues."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "embed",
                "--config", "nonexistent.yaml"
            ])
        # Should continue despite missing config
        assert exit_code == 0

    def test_multiple_commands_with_configs(self):
        """Test that different commands can use different configs."""
        commands = [
            (["--dry-run", "embed", "--config", "configs/paths.yaml"], "_run_module_passthrough"),
            (["--dry-run", "train-density", "--config", "configs/density.yaml"], "_run_module_passthrough"),
        ]
        for args, mock_target in commands:
            with patch.object(cli, mock_target) as mock_run:
                mock_run.return_value = 0
                exit_code = cli.main(args)
            assert exit_code == 0


class TestArgumentForwarding:
    """Tests for argument forwarding to subcommands."""

    def test_forwarded_args_preserved(self):
        """Test that forwarded arguments are preserved."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            cli.main([
                "--dry-run", "embed",
                "--custom-arg", "value",
                "--flag"
            ])
        # Verify args were forwarded
        cmd_args = mock_run.call_args[0][2]
        assert "--custom-arg" in cmd_args
        assert "value" in cmd_args
        assert "--flag" in cmd_args

    def test_forwarded_args_order(self):
        """Test that forwarded arguments maintain order."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            cli.main([
                "--dry-run", "train-density",
                "--arg1", "val1",
                "--arg2", "val2",
                "--arg3", "val3"
            ])
        cmd_args = mock_run.call_args[0][2]
        # Find indices to verify order
        idx1 = cmd_args.index("--arg1")
        idx2 = cmd_args.index("--arg2")
        idx3 = cmd_args.index("--arg3")
        assert idx1 < idx2 < idx3


class TestIntegrationScenarios:
    """Tests for realistic usage scenarios."""

    def test_complete_embedding_workflow(self):
        """Test a complete embedding extraction workflow."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--log-level", "INFO",
                "--dry-run", "embed",
                "--dataset", "mamografias",
                "--subset", "100",
                "--model", "resnet50",
                "--batch-size", "32"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()

    def test_complete_training_workflow(self):
        """Test a complete training workflow."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--log-level", "DEBUG",
                "--dry-run", "train-density",
                "--epochs", "10",
                "--batch-size", "16",
                "--lr", "0.001"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()

    def test_complete_visualization_workflow(self):
        """Test a complete visualization workflow."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "visualize",
                "--input", "embeddings.npy",
                "--labels", "labels.csv",
                "--tsne",
                "--pca",
                "--umap",
                "--report",
                "--outdir", "outputs/visualizations"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()
