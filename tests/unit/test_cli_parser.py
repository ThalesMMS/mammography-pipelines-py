#!/usr/bin/env python3
#
# test_cli_parser.py
# mammography-pipelines
#
# Unit tests for _build_parser as defined in mammography.cli_parser.
#
"""Unit tests for mammography.cli_parser module."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from mammography.cli_parser import _build_parser


class TestBuildParserModule:
    """Tests for _build_parser defined in mammography.cli_parser."""

    def test_returns_argument_parser(self):
        """Returns an ArgumentParser instance."""
        parser = _build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_prog_name_is_mammography(self):
        """Parser prog is set to 'mammography'."""
        parser = _build_parser()
        assert parser.prog == "mammography"

    def test_has_dry_run_flag(self):
        """Parser includes --dry-run flag."""
        parser = _build_parser()
        args = parser.parse_args(["--dry-run", "embed"])
        assert args.dry_run is True

    def test_dry_run_defaults_to_false(self):
        """--dry-run defaults to False."""
        parser = _build_parser()
        args = parser.parse_args(["embed"])
        assert args.dry_run is False

    def test_has_log_level_flag(self):
        """Parser includes --log-level flag."""
        parser = _build_parser()
        args = parser.parse_args(["--log-level", "DEBUG", "embed"])
        assert args.log_level == "DEBUG"

    def test_log_level_defaults_to_info(self):
        """--log-level defaults to INFO."""
        parser = _build_parser()
        args = parser.parse_args(["embed"])
        assert args.log_level == "INFO"

    def test_log_level_rejects_invalid_values(self):
        """--log-level rejects invalid level names."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--log-level", "NOTALEVEL", "embed"])


class TestSubcommandRegistration:
    """Tests that all expected subcommands are registered."""

    EXPECTED_SUBCOMMANDS = [
        "embed",
        "train-density",
        "eval-export",
        "report-pack",
        "data-audit",
        "visualize",
        "explain",
        "wizard",
        "inference",
        "augment",
        "preprocess",
        "label-density",
        "label-patches",
        "web",
        "eda-cancer",
        "embeddings-baselines",
        "tune",
        "cross-validate",
        "batch-inference",
        "compare-models",
        "benchmark-report",
        "automl",
    ]

    @pytest.mark.parametrize("subcommand", EXPECTED_SUBCOMMANDS)
    def test_subcommand_is_registered(self, subcommand):
        """Each expected subcommand is parseable."""
        parser = _build_parser()
        args = parser.parse_args([subcommand])
        assert args.command == subcommand

    def test_no_subcommand_leaves_command_as_none(self):
        """When no subcommand is given, command is None."""
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_unknown_subcommand_raises_system_exit(self):
        """Parser raises SystemExit for unknown subcommand."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["unknown-xyz"])


class TestEvalExportSubcommand:
    """Tests for eval-export subcommand arguments in cli_parser._build_parser."""

    def test_run_argument_appends_paths(self):
        """--run flag accumulates multiple paths."""
        parser = _build_parser()
        args = parser.parse_args([
            "eval-export",
            "--run", "outputs/run1",
            "--run", "outputs/run2",
        ])
        assert len(args.runs) == 2
        assert Path("outputs/run1") in args.runs

    def test_output_dir_default(self):
        """--output-dir defaults to outputs/exports."""
        parser = _build_parser()
        args = parser.parse_args(["eval-export"])
        assert args.output_dir == Path("outputs/exports")

    def test_output_dir_custom(self):
        """--output-dir accepts a custom path."""
        parser = _build_parser()
        args = parser.parse_args(["eval-export", "--output-dir", "custom/exports"])
        assert args.output_dir == Path("custom/exports")


class TestReportPackSubcommand:
    """Tests for report-pack subcommand arguments in cli_parser._build_parser."""

    def test_assets_dir_default(self):
        """--assets-dir defaults to Article/assets."""
        parser = _build_parser()
        args = parser.parse_args(["report-pack"])
        assert args.assets_dir == Path("Article") / "assets"

    def test_gradcam_limit_default(self):
        """--gradcam-limit defaults to 4."""
        parser = _build_parser()
        args = parser.parse_args(["report-pack"])
        assert args.gradcam_limit == 4

    def test_gradcam_limit_custom(self):
        """--gradcam-limit accepts custom integer values."""
        parser = _build_parser()
        args = parser.parse_args(["report-pack", "--gradcam-limit", "8"])
        assert args.gradcam_limit == 8

    def test_tex_path_dest(self):
        """--tex stores to tex_path attribute."""
        parser = _build_parser()
        args = parser.parse_args(["report-pack", "--tex", "custom.tex"])
        assert args.tex_path == Path("custom.tex")


class TestDataAuditSubcommand:
    """Tests for data-audit subcommand arguments in cli_parser._build_parser."""

    def test_archive_default(self):
        """--archive defaults to 'archive'."""
        parser = _build_parser()
        args = parser.parse_args(["data-audit"])
        assert args.archive == Path("archive")

    def test_csv_default(self):
        """--csv defaults to 'classificacao.csv'."""
        parser = _build_parser()
        args = parser.parse_args(["data-audit"])
        assert args.csv == Path("classificacao.csv")

    def test_manifest_default(self):
        """--manifest defaults to 'data_manifest.json'."""
        parser = _build_parser()
        args = parser.parse_args(["data-audit"])
        assert args.manifest == Path("data_manifest.json")

    def test_audit_csv_dest(self):
        """--audit-csv stores to audit_csv attribute."""
        parser = _build_parser()
        args = parser.parse_args(["data-audit", "--audit-csv", "custom_audit.csv"])
        assert args.audit_csv == Path("custom_audit.csv")


class TestVisualizeSubcommand:
    """Tests for visualize subcommand arguments in cli_parser._build_parser."""

    def test_seed_default(self):
        """--seed defaults to 42."""
        parser = _build_parser()
        args = parser.parse_args(["visualize"])
        assert args.seed == 42

    def test_perplexity_default(self):
        """--perplexity defaults to 30.0."""
        parser = _build_parser()
        args = parser.parse_args(["visualize"])
        assert args.perplexity == 30.0

    def test_tsne_iter_default(self):
        """--tsne-iter defaults to 1000."""
        parser = _build_parser()
        args = parser.parse_args(["visualize"])
        assert args.tsne_iter == 1000

    def test_pca_svd_solver_default(self):
        """--pca-svd-solver defaults to 'auto'."""
        parser = _build_parser()
        args = parser.parse_args(["visualize"])
        assert args.pca_svd_solver == "auto"

    def test_pca_svd_solver_choices(self):
        """--pca-svd-solver validates against allowed choices."""
        parser = _build_parser()
        for choice in ["auto", "full", "randomized", "arpack"]:
            args = parser.parse_args(["visualize", "--pca-svd-solver", choice])
            assert args.pca_svd_solver == choice

    def test_pca_svd_solver_rejects_invalid(self):
        """--pca-svd-solver rejects invalid choice."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["visualize", "--pca-svd-solver", "invalid"])

    def test_boolean_flags_default_false(self):
        """Visualization boolean flags default to False."""
        parser = _build_parser()
        args = parser.parse_args(["visualize"])
        for flag in ["tsne", "pca", "umap", "report", "binary", "from_run"]:
            assert getattr(args, flag) is False

    def test_input_shorthand(self):
        """--input accepts -i shorthand."""
        parser = _build_parser()
        args = parser.parse_args(["visualize", "-i", "features.npy"])
        assert args.input == Path("features.npy")

    def test_output_shorthand(self):
        """--output accepts -o shorthand."""
        parser = _build_parser()
        args = parser.parse_args(["visualize", "-o", "my_outputs"])
        assert args.output == Path("my_outputs")


class TestBenchmarkReportSubcommand:
    """Tests for benchmark-report subcommand arguments in cli_parser._build_parser."""

    def test_namespace_default(self):
        """--namespace defaults to outputs/rerun_2026q1."""
        parser = _build_parser()
        args = parser.parse_args(["benchmark-report"])
        assert args.namespace == Path("outputs/rerun_2026q1")

    def test_output_prefix_default(self):
        """--output-prefix defaults to results/rerun_2026q1_master."""
        parser = _build_parser()
        args = parser.parse_args(["benchmark-report"])
        assert args.output_prefix == Path("results/rerun_2026q1_master")

    def test_exports_search_root_default(self):
        """--exports-search-root defaults to outputs."""
        parser = _build_parser()
        args = parser.parse_args(["benchmark-report"])
        assert args.exports_search_root == Path("outputs")

    def test_docs_report_path_default(self):
        """--docs-report has expected default path."""
        parser = _build_parser()
        args = parser.parse_args(["benchmark-report"])
        assert "rerun_2026q1" in str(args.docs_report)


class TestConfigFlagForSubcommands:
    """Tests that subcommands with --config flag work correctly."""

    @pytest.mark.parametrize("subcommand", [
        "embed", "train-density", "eval-export", "visualize", "explain", "wizard",
    ])
    def test_subcommand_accepts_config_flag(self, subcommand):
        """Subcommands with --config flag accept a path value."""
        parser = _build_parser()
        args = parser.parse_args([subcommand, "--config", "my_config.yaml"])
        assert args.config == Path("my_config.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])