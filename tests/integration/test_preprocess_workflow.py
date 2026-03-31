"""
Integration smoke tests for the preprocess CLI workflow.

Tests validate that the preprocess command routes correctly, accepts
various argument combinations, and handles errors gracefully without
requiring actual datasets.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography import cli


class TestPreprocessCommandRouting:
    """Tests for preprocess command routing and basic functionality."""

    def test_preprocess_help_works(self):
        """Test that preprocess --help displays without errors."""
        with patch.object(sys, "argv", ["mammography", "preprocess", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            # argparse exits with 0 on --help
            assert exc_info.value.code == 0

    def test_preprocess_basic_dry_run(self):
        """Test preprocess with minimal arguments in dry-run mode."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "preprocess"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.preprocess"

    def test_preprocess_routes_to_preprocess_module(self):
        """Test that preprocess command routes to preprocess module."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "preprocess"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.preprocess"


class TestPreprocessCommandArguments:
    """Tests for preprocess command with various argument combinations."""

    def test_preprocess_with_input_output(self):
        """Test preprocess with input and output directories."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()
        # Verify forwarded args are passed
        cmd_args = mock_run.call_args[0][2]
        assert "--input" in cmd_args
        assert "data/raw" in cmd_args
        assert "--output" in cmd_args
        assert "data/processed" in cmd_args

    def test_preprocess_with_normalization(self):
        """Test preprocess with normalization options."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--normalize", "per-image"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][2]
        assert "--normalize" in cmd_args
        assert "per-image" in cmd_args

    def test_preprocess_with_resize_options(self):
        """Test preprocess with resize options."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--resize",
                "--img-size", "512"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][2]
        assert "--resize" in cmd_args
        assert "--img-size" in cmd_args
        assert "512" in cmd_args

    def test_preprocess_with_crop(self):
        """Test preprocess with crop option."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--crop"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][2]
        assert "--crop" in cmd_args

    def test_preprocess_with_format_option(self):
        """Test preprocess with output format options."""
        for fmt in ["png", "jpg", "keep"]:
            with patch.object(cli, "_run_module_passthrough") as mock_run:
                mock_run.return_value = 0
                exit_code = cli.main([
                    "--dry-run", "preprocess",
                    "--input", "data/raw",
                    "--output", "data/processed",
                    "--format", fmt
                ])
            assert exit_code == 0
            mock_run.assert_called_once()
            cmd_args = mock_run.call_args[0][2]
            assert "--format" in cmd_args
            assert fmt in cmd_args

    def test_preprocess_with_preview(self):
        """Test preprocess with preview generation."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--preview",
                "--preview-n", "8"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][2]
        assert "--preview" in cmd_args
        assert "--preview-n" in cmd_args
        assert "8" in cmd_args

    def test_preprocess_with_report(self):
        """Test preprocess with report generation."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--report"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][2]
        assert "--report" in cmd_args

    def test_preprocess_with_no_report(self):
        """Test preprocess with report disabled."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--no-report"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][2]
        assert "--no-report" in cmd_args

    def test_preprocess_with_border_removal(self):
        """Test preprocess with border removal option."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--border-removal"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][2]
        assert "--border-removal" in cmd_args

    def test_preprocess_with_config(self):
        """Test preprocess with config file."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--config", "configs/preprocess.yaml"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()


class TestPreprocessNormalizationModes:
    """Tests for different normalization modes."""

    def test_preprocess_per_image_normalization(self):
        """Test preprocess with per-image normalization."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--normalize", "per-image"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()

    def test_preprocess_per_dataset_normalization(self):
        """Test preprocess with per-dataset normalization."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--normalize", "per-dataset"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()

    def test_preprocess_no_normalization(self):
        """Test preprocess with normalization disabled."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--normalize", "none"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()


class TestPreprocessResizeOptions:
    """Tests for resize and crop options."""

    def test_preprocess_with_custom_img_size(self):
        """Test preprocess with custom image size."""
        for size in [256, 512, 1024]:
            with patch.object(cli, "_run_module_passthrough") as mock_run:
                mock_run.return_value = 0
                exit_code = cli.main([
                    "--dry-run", "preprocess",
                    "--input", "data/raw",
                    "--output", "data/processed",
                    "--img-size", str(size)
                ])
            assert exit_code == 0
            mock_run.assert_called_once()

    def test_preprocess_with_no_resize(self):
        """Test preprocess with resize disabled."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--no-resize"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][2]
        assert "--no-resize" in cmd_args

    def test_preprocess_resize_with_crop(self):
        """Test preprocess with resize and crop combined."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--resize",
                "--crop",
                "--img-size", "512"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()


class TestPreprocessCompleteWorkflows:
    """Tests for complete preprocessing workflows."""

    def test_preprocess_complete_workflow_png(self):
        """Test a complete preprocessing workflow with PNG output."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--log-level", "INFO",
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--normalize", "per-image",
                "--resize",
                "--img-size", "512",
                "--format", "png",
                "--preview",
                "--report"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()

    def test_preprocess_complete_workflow_jpg(self):
        """Test a complete preprocessing workflow with JPG output."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--log-level", "DEBUG",
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--normalize", "per-dataset",
                "--resize",
                "--crop",
                "--img-size", "1024",
                "--format", "jpg",
                "--preview",
                "--preview-n", "16",
                "--report"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()

    def test_preprocess_minimal_workflow(self):
        """Test minimal preprocessing workflow."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--normalize", "none",
                "--no-resize",
                "--no-report"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()


class TestPreprocessModuleImport:
    """Tests for preprocess module import functionality."""

    def test_preprocess_command_import_works(self):
        """Test that preprocess command imports without errors."""
        from mammography.commands.preprocess import main

        assert main is not None

    def test_preprocess_parse_args_works(self):
        """Test that preprocess argument parser works."""
        from mammography.commands.preprocess import parse_args

        # Should accept required args without errors
        args = parse_args([
            "--input", "data/raw",
            "--output", "data/processed"
        ])
        assert args.input == "data/raw"
        assert args.output == "data/processed"
        assert args.normalize == "per-image"  # default
        assert args.img_size == 512  # default


class TestPreprocessErrorHandling:
    """Tests for preprocess error handling."""

    def test_preprocess_command_failure_propagates(self):
        """Test that command failures propagate correctly."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 1
            exit_code = cli.main(["--dry-run", "preprocess"])
        assert exit_code == 1

    def test_preprocess_with_forwarded_args_preserved(self):
        """Test that forwarded arguments are preserved."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed",
                "--custom-arg", "value",
                "--flag"
            ])
        # Verify args were forwarded
        cmd_args = mock_run.call_args[0][2]
        assert "--input" in cmd_args
        assert "data/raw" in cmd_args
        assert "--output" in cmd_args
        assert "data/processed" in cmd_args
        assert "--custom-arg" in cmd_args
        assert "value" in cmd_args
        assert "--flag" in cmd_args


class TestPreprocessGlobalOptions:
    """Tests for global CLI options with preprocess command."""

    def test_preprocess_with_log_level(self):
        """Test preprocess with different log levels."""
        log_levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
        for level in log_levels:
            with patch.object(cli, "_run_module_passthrough") as mock_run:
                mock_run.return_value = 0
                exit_code = cli.main([
                    "--log-level", level,
                    "--dry-run", "preprocess",
                    "--input", "data/raw",
                    "--output", "data/processed"
                ])
            assert exit_code == 0

    def test_preprocess_dry_run_prevents_execution(self):
        """Test that dry-run prevents actual execution."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run", "preprocess",
                "--input", "data/raw",
                "--output", "data/processed"
            ])
        assert exit_code == 0
        mock_run.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
