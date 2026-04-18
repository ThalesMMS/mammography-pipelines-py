#!/usr/bin/env python3
#
# test_cli_config.py
# mammography-pipelines
#
# Unit tests for CLI configuration helpers extracted to cli_config.py.
#
"""Unit tests for mammography.cli_config module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mammography.cli_config import (
    _coerce_cli_args,
    _default_config,
    _dict_to_cli_args,
    _filter_embed_config_args,
    _forwarded_has_flag,
    _strip_flags_with_values,
)


class TestForwardedHasFlag:
    """Tests for _forwarded_has_flag."""

    def test_exact_flag_match(self):
        """Returns True when forwarded contains exact flag token."""
        assert _forwarded_has_flag(["--dataset", "archive"], "--dataset") is True

    def test_flag_equals_form(self):
        """Returns True when forwarded contains flag=value form."""
        assert _forwarded_has_flag(["--dataset=archive"], "--dataset") is True

    def test_flag_not_present(self):
        """Returns False when flag is absent from forwarded."""
        assert _forwarded_has_flag(["--csv", "labels.csv"], "--dataset") is False

    def test_empty_forwarded(self):
        """Returns False for empty forwarded sequence."""
        assert _forwarded_has_flag([], "--dataset") is False

    def test_partial_match_not_treated_as_flag(self):
        """Returns False when token only contains flag as substring (not prefix form)."""
        assert _forwarded_has_flag(["--dataset-extra", "val"], "--dataset") is False

    def test_multiple_flags_one_matches(self):
        """Returns True when at least one token matches."""
        forwarded = ["--csv", "labels.csv", "--dataset", "mamografias"]
        assert _forwarded_has_flag(forwarded, "--dataset") is True

    def test_flag_equals_form_with_complex_value(self):
        """Returns True for flag=value form with path-like value."""
        assert _forwarded_has_flag(["--data_dir=/some/path"], "--data_dir") is True

    def test_data_dir_flag_standalone(self):
        """Returns True when --data_dir appears as standalone token."""
        assert _forwarded_has_flag(["--data_dir", "./archive"], "--data_dir") is True


class TestStripFlagsWithValues:
    """Tests for _strip_flags_with_values."""

    def test_empty_flags_returns_copy(self):
        """Returns a list copy of args when flags set is empty."""
        args = ["--csv", "labels.csv", "--epochs", "10"]
        result = _strip_flags_with_values(args, set())
        assert result == args
        assert result is not args  # must be a copy

    def test_strips_standalone_flag_and_value(self):
        """Removes a flag and its following value token."""
        args = ["--csv", "labels.csv", "--epochs", "10"]
        result = _strip_flags_with_values(args, {"--csv"})
        assert result == ["--epochs", "10"]

    def test_strips_equals_form(self):
        """Removes a flag=value token."""
        args = ["--csv=labels.csv", "--epochs", "10"]
        result = _strip_flags_with_values(args, {"--csv"})
        assert result == ["--epochs", "10"]

    def test_strips_multiple_flags(self):
        """Removes multiple specified flags and their values."""
        args = ["--csv", "a.csv", "--dicom-root", "/data", "--epochs", "5"]
        result = _strip_flags_with_values(args, {"--csv", "--dicom-root"})
        assert result == ["--epochs", "5"]

    def test_empty_args_returns_empty(self):
        """Returns empty list when args is empty."""
        result = _strip_flags_with_values([], {"--csv"})
        assert result == []

    def test_flag_at_end_no_following_value(self):
        """Handles trailing flag gracefully, skipping 'next' (which is absent)."""
        args = ["--epochs", "10", "--csv"]
        result = _strip_flags_with_values(args, {"--csv"})
        # --csv is at the end, skip_next set but nothing to skip
        assert result == ["--epochs", "10"]

    def test_preserves_unrelated_tokens(self):
        """Keeps tokens that are not in the flags set."""
        args = ["--batch-size", "32", "--csv", "labels.csv", "--lr", "0.001"]
        result = _strip_flags_with_values(args, {"--csv"})
        assert result == ["--batch-size", "32", "--lr", "0.001"]

    def test_strips_only_exact_flag_match(self):
        """Does not strip tokens that only partially match a flag."""
        args = ["--csv-path", "labels.csv", "--csv", "other.csv"]
        result = _strip_flags_with_values(args, {"--csv"})
        # --csv-path should not be stripped; only --csv and its value
        assert "--csv-path" in result
        assert "labels.csv" in result
        assert "other.csv" not in result


class TestFilterEmbedConfigArgs:
    """Tests for _filter_embed_config_args."""

    def test_empty_config_args_returns_empty_list(self):
        """Returns empty list when config_args is empty."""
        result = _filter_embed_config_args([], ["--dataset", "archive"])
        assert result == []

    def test_no_dataset_or_data_dir_in_forwarded_returns_unchanged(self):
        """Returns config_args unchanged when user did not specify dataset or data_dir."""
        config_args = ["--csv", "labels.csv", "--dicom-root", "/data"]
        forwarded = ["--epochs", "10"]
        result = _filter_embed_config_args(config_args, forwarded)
        assert result == config_args

    def test_dataset_flag_strips_dataset_csv_dicom_root(self):
        """Strips --dataset, --csv, and --dicom-root from config_args when --dataset forwarded."""
        config_args = [
            "--dataset", "archive",
            "--csv", "labels.csv",
            "--dicom-root", "/data",
            "--epochs", "10",
        ]
        forwarded = ["--dataset", "mamografias"]
        result = _filter_embed_config_args(config_args, forwarded)
        assert "--dataset" not in result
        assert "archive" not in result
        assert "--csv" not in result
        assert "labels.csv" not in result
        assert "--dicom-root" not in result
        assert "/data" not in result
        assert "--epochs" in result
        assert "10" in result

    def test_data_dir_flag_strips_csv_dicom_root(self):
        """Strips --csv and --dicom-root when user provides --data_dir."""
        config_args = ["--csv", "labels.csv", "--dicom-root", "/data", "--batch-size", "32"]
        forwarded = ["--data_dir", "./archive"]
        result = _filter_embed_config_args(config_args, forwarded)
        assert "--csv" not in result
        assert "--dicom-root" not in result
        assert "--batch-size" in result
        assert "32" in result

    def test_dataset_flag_equals_form_triggers_filtering(self):
        """Strips config flags when --dataset=value appears in forwarded."""
        config_args = ["--dataset", "old_dataset", "--csv", "old.csv"]
        forwarded = ["--dataset=new_dataset"]
        result = _filter_embed_config_args(config_args, forwarded)
        assert "--dataset" not in result
        assert "--csv" not in result

    def test_user_csv_in_forwarded_preserves_dataset_stripping_with_csv_retained(self):
        """When user specifies both --dataset and --csv in forwarded, --csv config is still stripped."""
        config_args = [
            "--dataset", "archive",
            "--csv", "config_labels.csv",
            "--dicom-root", "/data",
            "--epochs", "5",
        ]
        forwarded = ["--dataset", "mamografias", "--csv", "user_labels.csv"]
        result = _filter_embed_config_args(config_args, forwarded)
        # Config --csv should be stripped because the user has forwarded their own
        assert "--dataset" not in result
        # The value "config_labels.csv" should not appear (stripped from config)
        assert "config_labels.csv" not in result
        # --epochs should be preserved
        assert "--epochs" in result


class TestDefaultConfig:
    """Tests for _default_config."""

    def test_returns_none_for_unknown_command(self):
        """Returns None for commands not in DEFAULT_CONFIGS."""
        result = _default_config("nonexistent-command-xyz")
        assert result is None

    def test_returns_none_when_config_file_absent(self):
        """Returns None when a configured path does not exist on disk."""
        with patch("mammography.cli_config.DEFAULT_CONFIGS", {"test-cmd": Path("/nonexistent/path/config.yaml")}):
            result = _default_config("test-cmd")
            assert result is None

    def test_returns_path_when_config_exists(self, tmp_path):
        """Returns the Path when the configured file exists."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("key: value", encoding="utf-8")
        with patch("mammography.cli_config.DEFAULT_CONFIGS", {"test-cmd": config_file}):
            result = _default_config("test-cmd")
            assert result == config_file

    def test_returns_none_for_command_with_none_config(self):
        """Returns None when the DEFAULT_CONFIGS entry is None."""
        with patch("mammography.cli_config.DEFAULT_CONFIGS", {"eval-export": None}):
            result = _default_config("eval-export")
            assert result is None


class TestDictToCliArgs:
    """Tests for _dict_to_cli_args delegation."""

    def test_delegates_to_dict_to_cli_args(self):
        """Verifies that _dict_to_cli_args delegates to the underlying utility."""
        with patch("mammography.cli_config.dict_to_cli_args", return_value=["--key", "val"]) as mock_fn:
            result = _dict_to_cli_args({"key": "val"})
            mock_fn.assert_called_once_with({"key": "val"})
            assert result == ["--key", "val"]


class TestCoerceCliArgs:
    """Tests for _coerce_cli_args delegation."""

    def test_delegates_to_coerce_cli_args(self):
        """Verifies that _coerce_cli_args delegates to the underlying utility."""
        with patch("mammography.cli_config.coerce_cli_args", return_value=["--flag"]) as mock_fn:
            result = _coerce_cli_args({"flag": True})
            mock_fn.assert_called_once_with({"flag": True})
            assert result == ["--flag"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])