#!/usr/bin/env python3
#
# test_cli.py
# mammography-pipelines
#
# Unit tests for CLI argument parsing and command routing.
#
# Medical Disclaimer: This is an educational research project.
# Not intended for clinical use or medical diagnosis.
#
"""Unit tests for the CLI module."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mammography import cli


class TestBuildParser:
    """Test _build_parser function."""

    def test_build_parser_creates_parser(self):
        """Test that _build_parser creates an ArgumentParser."""
        parser = cli._build_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.prog == "mammography"

    def test_build_parser_has_dry_run_flag(self):
        """Test that parser has --dry-run flag."""
        parser = cli._build_parser()
        args = parser.parse_args(["--dry-run", "embed"])
        assert args.dry_run is True

    def test_build_parser_has_log_level_flag(self):
        """Test that parser has --log-level flag."""
        parser = cli._build_parser()
        args = parser.parse_args(["--log-level", "DEBUG", "embed"])
        assert args.log_level == "DEBUG"

    def test_build_parser_log_level_choices(self):
        """Test that --log-level validates choices."""
        parser = cli._build_parser()

        # Valid choices
        for level in ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]:
            args = parser.parse_args(["--log-level", level, "embed"])
            assert args.log_level == level

        # Invalid choice
        with pytest.raises(SystemExit):
            parser.parse_args(["--log-level", "INVALID", "embed"])

    def test_build_parser_has_subcommands(self):
        """Test that parser has expected subcommands."""
        parser = cli._build_parser()

        # Test each subcommand parses successfully
        subcommands = [
            "embed",
            "train-density",
            "eval-export",
            "report-pack",
            "data-audit",
            "benchmark-report",
        ]

        for subcmd in subcommands:
            args = parser.parse_args([subcmd])
            assert args.command == subcmd

    def test_build_parser_embed_subcommand(self):
        """Test embed subcommand configuration."""
        parser = cli._build_parser()
        args = parser.parse_args(["embed", "--config", "configs/test.yaml"])
        assert args.command == "embed"
        assert args.config == Path("configs/test.yaml")

    def test_build_parser_train_density_subcommand(self):
        """Test train-density subcommand configuration."""
        parser = cli._build_parser()
        args = parser.parse_args(["train-density", "--config", "configs/density.yaml"])
        assert args.command == "train-density"
        assert args.config == Path("configs/density.yaml")

    def test_build_parser_eval_export_options(self):
        """Test eval-export subcommand with options."""
        parser = cli._build_parser()
        args = parser.parse_args([
            "eval-export",
            "--run", "outputs/run1",
            "--run", "outputs/run2",
            "--output-dir", "exports",
            "--run-name", "test_run",
        ])

        assert args.command == "eval-export"
        assert len(args.runs) == 2
        assert args.output_dir == Path("exports")
        assert args.run_name == "test_run"

    def test_build_parser_eval_export_mlflow_flags(self):
        """Test eval-export MLflow flags."""
        parser = cli._build_parser()
        args = parser.parse_args([
            "eval-export",
            "--no-mlflow",
            "--no-registry",
        ])

        assert args.no_mlflow is True
        assert args.no_registry is True

    def test_build_parser_report_pack_options(self):
        """Test report-pack subcommand with options."""
        parser = cli._build_parser()
        args = parser.parse_args([
            "report-pack",
            "--run", "outputs/run1",
            "--assets-dir", "Article/assets",
            "--tex", "Article/sections/density.tex",
            "--gradcam-limit", "8",
        ])

        assert args.command == "report-pack"
        assert len(args.runs) == 1
        assert args.assets_dir == Path("Article/assets")
        assert args.tex_path == Path("Article/sections/density.tex")
        assert args.gradcam_limit == 8

    def test_build_parser_data_audit_options(self):
        """Test data-audit subcommand with options."""
        parser = cli._build_parser()
        args = parser.parse_args([
            "data-audit",
            "--archive", "data/archive",
            "--csv", "data/labels.csv",
            "--manifest", "manifest.json",
        ])

        assert args.command == "data-audit"
        assert args.archive == Path("data/archive")
        assert args.csv == Path("data/labels.csv")
        assert args.manifest == Path("manifest.json")

    def test_build_parser_benchmark_report_options(self):
        """Test benchmark-report subcommand with options."""
        parser = cli._build_parser()
        args = parser.parse_args([
            "benchmark-report",
            "--namespace", "outputs/rerun_2026q1",
            "--output-prefix", "results/custom_master",
            "--docs-report", "docs/reports/custom.md",
            "--article-table", "Article/sections/custom.tex",
        ])

        assert args.command == "benchmark-report"
        assert args.namespace == Path("outputs/rerun_2026q1")
        assert args.output_prefix == Path("results/custom_master")
        assert args.docs_report == Path("docs/reports/custom.md")
        assert args.article_table == Path("Article/sections/custom.tex")


class TestDefaultConfigs:
    """Test DEFAULT_CONFIGS dictionary."""

    def test_default_configs_exists(self):
        """Test that DEFAULT_CONFIGS is defined."""
        assert hasattr(cli, "DEFAULT_CONFIGS")
        assert isinstance(cli.DEFAULT_CONFIGS, dict)

    def test_default_configs_has_expected_keys(self):
        """Test that DEFAULT_CONFIGS has expected command keys."""
        expected_keys = {
            "embed",
            "train-density",
            "eval-export",
            "visualize",
            "explain",
            "embeddings-baselines",
            "data-audit",
            "tune",
            "preprocess",
            "cross-validate",
            "batch-inference",
            "compare-models",
            "benchmark-report",
            "automl",
        }

        assert set(cli.DEFAULT_CONFIGS.keys()) == expected_keys

    def test_default_configs_embed_path(self):
        """Test that embed has correct default config path."""
        embed_config = cli.DEFAULT_CONFIGS["embed"]
        assert embed_config is not None
        assert "paths.yaml" in str(embed_config)

    def test_default_configs_train_density_path(self):
        """Test that train-density has correct default config path."""
        density_config = cli.DEFAULT_CONFIGS["train-density"]
        assert density_config is not None
        assert "density.yaml" in str(density_config)

    def test_default_configs_none_values(self):
        """Test that some commands have None as default config."""
        none_configs = [
            "eval-export",
            "visualize",
            "explain",
            "embeddings-baselines",
            "data-audit",
            "tune",
            "preprocess",
            "cross-validate",
            "batch-inference",
            "compare-models",
            "benchmark-report",
            "automl",
        ]

        for cmd in none_configs:
            assert cli.DEFAULT_CONFIGS[cmd] is None


class TestCliEdgeCases:
    """Test edge cases and boundary conditions for CLI."""

    def test_parser_no_subcommand_fails(self):
        """Test that parser leaves command unset when no subcommand is provided."""
        parser = cli._build_parser()
        args = parser.parse_args([])
        assert getattr(args, "command", None) is None

    def test_parser_unknown_subcommand_fails(self):
        """Test that parser fails with unknown subcommand."""
        parser = cli._build_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["unknown-command"])

    def test_parser_multiple_runs_for_eval_export(self):
        """Test that eval-export accepts multiple --run arguments."""
        parser = cli._build_parser()
        args = parser.parse_args([
            "eval-export",
            "--run", "run1",
            "--run", "run2",
            "--run", "run3",
        ])

        assert len(args.runs) == 3
        assert args.runs == [Path("run1"), Path("run2"), Path("run3")]

    def test_parser_multiple_runs_for_report_pack(self):
        """Test that report-pack accepts multiple --run arguments."""
        parser = cli._build_parser()
        args = parser.parse_args([
            "report-pack",
            "--run", "run1",
            "--run", "run2",
        ])

        assert len(args.runs) == 2

    def test_parser_default_values(self):
        """Test that parser uses correct default values."""
        parser = cli._build_parser()

        # Test eval-export defaults
        args = parser.parse_args(["eval-export"])
        assert args.output_dir == Path("outputs/exports")
        assert args.registry_csv == Path("results/registry.csv")
        assert args.registry_md == Path("results/registry.md")
        assert args.no_mlflow is False
        assert args.no_registry is False

        # Test data-audit defaults
        args = parser.parse_args(["data-audit"])
        assert args.archive == Path("archive")
        assert args.csv == Path("classificacao.csv")
        assert args.manifest == Path("data_manifest.json")

    def test_parser_config_flag_accepts_path(self):
        """Test that --config flag accepts Path objects."""
        parser = cli._build_parser()

        for cmd in ["embed", "train-density"]:
            args = parser.parse_args([cmd, "--config", "path/to/config.yaml"])
            assert args.config == Path("path/to/config.yaml")

    def test_dry_run_flag_default_false(self):
        """Test that --dry-run defaults to False."""
        parser = cli._build_parser()
        args = parser.parse_args(["embed"])
        assert args.dry_run is False

    def test_log_level_default_info(self):
        """Test that --log-level defaults to INFO."""
        parser = cli._build_parser()
        args = parser.parse_args(["embed"])
        assert args.log_level == "INFO"


class TestRepoRoot:
    """Test REPO_ROOT constant."""

    def test_repo_root_exists(self):
        """Test that REPO_ROOT is defined."""
        assert hasattr(cli, "REPO_ROOT")

    def test_repo_root_is_path(self):
        """Test that REPO_ROOT is a Path object."""
        assert isinstance(cli.REPO_ROOT, Path)

    def test_repo_root_points_to_valid_directory(self):
        """Test that REPO_ROOT points to an existing directory."""
        # REPO_ROOT should be 2 levels up from cli.py
        assert cli.REPO_ROOT.exists()
        assert cli.REPO_ROOT.is_dir()


class TestParserIntegration:
    """Integration tests for argument parser."""

    def test_parse_embed_with_all_options(self):
        """Test parsing embed command with multiple options."""
        parser = cli._build_parser()
        args = parser.parse_args([
            "--dry-run",
            "--log-level", "DEBUG",
            "embed",
            "--config", "configs/paths.yaml",
        ])

        assert args.dry_run is True
        assert args.log_level == "DEBUG"
        assert args.command == "embed"
        assert args.config == Path("configs/paths.yaml")

    def test_parse_report_pack_with_all_options(self):
        """Test parsing report-pack with all options."""
        parser = cli._build_parser()
        args = parser.parse_args([
            "report-pack",
            "--run", "outputs/run1",
            "--run", "outputs/run2",
            "--assets-dir", "Article/assets",
            "--tex", "Article/sections/density.tex",
            "--gradcam-limit", "6",
            "--run-name", "test_pack",
            "--tracking-uri", "mlruns/",
            "--experiment", "density_exp",
            "--no-mlflow",
        ])

        assert args.command == "report-pack"
        assert len(args.runs) == 2
        assert args.assets_dir == Path("Article/assets")
        assert args.tex_path == Path("Article/sections/density.tex")
        assert args.gradcam_limit == 6
        assert args.run_name == "test_pack"
        assert args.tracking_uri == "mlruns/"
        assert args.experiment == "density_exp"
        assert args.no_mlflow is True

    def test_parse_eval_export_with_all_options(self):
        """Test parsing eval-export with all options."""
        parser = cli._build_parser()
        args = parser.parse_args([
            "eval-export",
            "--run", "outputs/run1",
            "--output-dir", "exports/",
            "--run-name", "test_eval",
            "--tracking-uri", "file:///mlruns",
            "--experiment", "eval_exp",
            "--registry-csv", "results/custom.csv",
            "--registry-md", "results/custom.md",
            "--no-registry",
        ])

        assert args.command == "eval-export"
        assert args.runs == [Path("outputs/run1")]
        assert args.output_dir == Path("exports/")
        assert args.run_name == "test_eval"
        assert args.tracking_uri == "file:///mlruns"
        assert args.experiment == "eval_exp"
        assert args.registry_csv == Path("results/custom.csv")
        assert args.registry_md == Path("results/custom.md")
        assert args.no_registry is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
