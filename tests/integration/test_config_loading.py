"""
Integration tests for YAML/JSON config file loading and CLI argument merging.

Tests config loading functionality including YAML parsing, CLI overrides,
config merging, and error handling for invalid config files.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Check if yaml is available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from mammography import cli


class TestYAMLConfigLoading:
    """Tests for YAML config file loading."""

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_load_valid_yaml_config(self, tmp_path: Path):
        """Test loading a valid YAML config file."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
            embed:
              arch: resnet50
              batch_size: 32
              device: cpu
            """,
            encoding="utf-8"
        )

        result = cli._read_config(config_file)
        assert isinstance(result, dict)
        assert "embed" in result
        assert result["embed"]["arch"] == "resnet50"
        assert result["embed"]["batch_size"] == 32

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_load_yaml_with_global_section(self, tmp_path: Path):
        """Test loading YAML config with global section."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
            global:
              seed: 42
              device: cpu
            embed:
              arch: efficientnet_b0
            train-density:
              epochs: 10
            """,
            encoding="utf-8"
        )

        result = cli._read_config(config_file)
        assert "global" in result
        assert result["global"]["seed"] == 42
        assert result["embed"]["arch"] == "efficientnet_b0"
        assert result["train-density"]["epochs"] == 10

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_load_yaml_with_boolean_values(self, tmp_path: Path):
        """Test loading YAML config with boolean values."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
            train-density:
              amp: true
              augment: false
              pretrained: true
            """,
            encoding="utf-8"
        )

        result = cli._read_config(config_file)
        assert result["train-density"]["amp"] is True
        assert result["train-density"]["augment"] is False
        assert result["train-density"]["pretrained"] is True

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_load_yaml_with_null_values(self, tmp_path: Path):
        """Test loading YAML config with null values."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
            embed:
              csv: null
              mean: null
              std: null
            """,
            encoding="utf-8"
        )

        result = cli._read_config(config_file)
        assert result["embed"]["csv"] is None
        assert result["embed"]["mean"] is None

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_load_yaml_with_lists(self, tmp_path: Path):
        """Test loading YAML config with list values."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
            train-density:
              views_to_train:
                - CC
                - MLO
            """,
            encoding="utf-8"
        )

        result = cli._read_config(config_file)
        assert isinstance(result["train-density"]["views_to_train"], list)
        assert result["train-density"]["views_to_train"] == ["CC", "MLO"]

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_empty_yaml_config(self, tmp_path: Path):
        """Test loading empty YAML config file."""
        config_file = tmp_path / "empty_config.yaml"
        config_file.write_text("", encoding="utf-8")

        result = cli._read_config(config_file)
        # Empty YAML should return None or empty dict
        assert result is None or result == {}


class TestJSONConfigLoading:
    """Tests for JSON config file loading (fallback when yaml unavailable)."""

    def test_load_valid_json_config(self, tmp_path: Path):
        """Test loading a valid JSON config file."""
        config_file = tmp_path / "test_config.json"
        config_data = {
            "embed": {
                "arch": "resnet50",
                "batch_size": 32,
                "device": "cpu"
            }
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        # Temporarily disable yaml to test JSON fallback
        with patch.object(cli, "yaml", None):
            result = cli._read_config(config_file)

        assert isinstance(result, dict)
        assert "embed" in result
        assert result["embed"]["arch"] == "resnet50"

    def test_load_json_with_nested_values(self, tmp_path: Path):
        """Test loading JSON config with nested values."""
        config_file = tmp_path / "test_config.json"
        config_data = {
            "global": {"seed": 42, "device": "cpu"},
            "train-density": {"epochs": 10, "batch_size": 16}
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        with patch.object(cli, "yaml", None):
            result = cli._read_config(config_file)

        assert result["global"]["seed"] == 42
        assert result["train-density"]["epochs"] == 10


class TestConfigToArgsConversion:
    """Tests for converting config dicts to CLI arguments."""

    def test_dict_to_cli_args_simple(self):
        """Test converting simple dict to CLI args."""
        payload = {
            "arch": "resnet50",
            "batch_size": 32,
            "device": "cpu"
        }

        args = cli._dict_to_cli_args(payload)

        assert "--arch" in args
        assert "resnet50" in args
        assert "--batch-size" in args
        assert "32" in args
        assert "--device" in args
        assert "cpu" in args

    def test_dict_to_cli_args_boolean_true(self):
        """Test converting boolean True to CLI flag."""
        payload = {"amp": True, "augment": True}

        args = cli._dict_to_cli_args(payload)

        assert "--amp" in args
        assert "--augment" in args
        # True values should just be flags without values
        assert args.count("True") == 0

    def test_dict_to_cli_args_boolean_false(self):
        """Test converting boolean False (should be omitted)."""
        payload = {"amp": False, "augment": False}

        args = cli._dict_to_cli_args(payload)

        # False values should not appear in args
        assert "--amp" not in args
        assert "--augment" not in args

    def test_dict_to_cli_args_null_values(self):
        """Test that null values are omitted from CLI args."""
        payload = {"csv": None, "mean": None, "arch": "resnet50"}

        args = cli._dict_to_cli_args(payload)

        # None values should be skipped
        assert "--csv" not in args
        assert "--mean" not in args
        # But non-None values should be included
        assert "--arch" in args
        assert "resnet50" in args

    def test_dict_to_cli_args_list_values(self):
        """Test converting list values to repeated CLI flags."""
        payload = {"views_to_train": ["CC", "MLO"]}

        args = cli._dict_to_cli_args(payload)

        # List values should produce repeated flags
        assert args.count("--views-to-train") == 2
        assert "CC" in args
        assert "MLO" in args

    def test_dict_to_cli_args_underscore_to_dash(self):
        """Test that underscores in keys are converted to dashes."""
        payload = {
            "batch_size": 32,
            "num_workers": 4,
            "early_stop_patience": 5
        }

        args = cli._dict_to_cli_args(payload)

        assert "--batch-size" in args
        assert "--num-workers" in args
        assert "--early-stop-patience" in args


class TestLoadConfigArgs:
    """Tests for loading and converting config files to CLI args."""

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_load_config_args_for_embed(self, tmp_path: Path):
        """Test loading config args for embed command."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
            embed:
              arch: resnet50
              batch_size: 32
              device: cpu
            """,
            encoding="utf-8"
        )

        args = cli._load_config_args(config_file, "embed")

        assert "--arch" in args
        assert "resnet50" in args
        assert "--batch-size" in args
        assert "32" in args

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_load_config_args_with_global_section(self, tmp_path: Path):
        """Test that global section is merged with command-specific config."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
            global:
              seed: 42
              device: cpu
            embed:
              arch: resnet50
            """,
            encoding="utf-8"
        )

        args = cli._load_config_args(config_file, "embed")

        # Should include both global and embed-specific args
        assert "--seed" in args
        assert "42" in args
        assert "--device" in args
        assert "cpu" in args
        assert "--arch" in args
        assert "resnet50" in args

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_load_config_args_command_with_dash(self, tmp_path: Path):
        """Test loading config for command with dash (train-density)."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
            train-density:
              epochs: 10
              batch_size: 16
            """,
            encoding="utf-8"
        )

        args = cli._load_config_args(config_file, "train-density")

        assert "--epochs" in args
        assert "10" in args

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_load_config_args_command_with_underscore(self, tmp_path: Path):
        """Test loading config with underscore variant of command name."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
            train_density:
              epochs: 10
            """,
            encoding="utf-8"
        )

        # Should normalize train-density to also check train_density
        args = cli._load_config_args(config_file, "train-density")

        assert "--epochs" in args
        assert "10" in args

    def test_load_config_args_missing_file(self, tmp_path: Path):
        """Test loading config from non-existent file."""
        config_file = tmp_path / "nonexistent.yaml"

        args = cli._load_config_args(config_file, "embed")

        # Should return empty list and log warning
        assert args == []

    def test_load_config_args_no_config_provided(self):
        """Test loading config when no config file is provided."""
        args = cli._load_config_args(None, "embed")

        # Without a config file, should try default config
        # If no default exists, should return empty list
        assert isinstance(args, list)


class TestInvalidConfigHandling:
    """Tests for error handling with invalid config files."""

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_malformed_yaml(self, tmp_path: Path):
        """Test handling of malformed YAML."""
        config_file = tmp_path / "bad_config.yaml"
        config_file.write_text(
            """
            embed:
              arch: resnet50
                invalid_indentation: true
            broken: [unclosed list
            """,
            encoding="utf-8"
        )

        args = cli._load_config_args(config_file, "embed")

        # Should handle error gracefully and return empty list
        assert args == []

    def test_malformed_json(self, tmp_path: Path):
        """Test handling of malformed JSON."""
        config_file = tmp_path / "bad_config.json"
        config_file.write_text(
            '{"embed": {"arch": "resnet50", invalid}}',
            encoding="utf-8"
        )

        with patch.object(cli, "yaml", None):
            args = cli._load_config_args(config_file, "embed")

        # Should handle error gracefully and return empty list
        assert args == []

    def test_config_with_invalid_encoding(self, tmp_path: Path):
        """Test handling of config file with encoding issues."""
        config_file = tmp_path / "bad_encoding.yaml"
        # Write binary data that's not valid UTF-8
        config_file.write_bytes(b"\xff\xfe\x00\x00invalid")

        args = cli._load_config_args(config_file, "embed")

        # Should handle error gracefully
        assert args == []


class TestDefaultConfigHandling:
    """Tests for default config file handling."""

    def test_default_config_for_embed(self):
        """Test getting default config path for embed command."""
        config_path = cli._default_config("embed")

        # Should point to configs/paths.yaml
        if config_path is not None:
            assert config_path.name == "paths.yaml"
            assert "configs" in str(config_path)

    def test_default_config_for_train_density(self):
        """Test getting default config path for train-density command."""
        config_path = cli._default_config("train-density")

        # Should point to configs/density.yaml
        if config_path is not None:
            assert config_path.name == "density.yaml"

    def test_default_config_for_command_without_default(self):
        """Test getting default config for command without default."""
        config_path = cli._default_config("visualize")

        # visualize has None in DEFAULT_CONFIGS
        assert config_path is None


class TestCLIConfigIntegration:
    """Integration tests for config loading in full CLI context."""

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_cli_with_config_file(self, tmp_path: Path):
        """Test full CLI execution with config file."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
            embed:
              arch: efficientnet_b0
              batch_size: 16
            """,
            encoding="utf-8"
        )

        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run",
                "embed",
                "--config", str(config_file)
            ])

        assert exit_code == 0
        mock_run.assert_called_once()

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_cli_args_override_config(self, tmp_path: Path):
        """Test that CLI arguments override config file values."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
            embed:
              arch: resnet50
              batch_size: 32
            """,
            encoding="utf-8"
        )

        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run",
                "embed",
                "--config", str(config_file),
                "--arch", "efficientnet_b0",  # Override config
                "--batch-size", "16"  # Override config
            ])

        assert exit_code == 0
        # Verify that CLI args are passed (they come after config args)
        args_passed = mock_run.call_args[0][2]
        assert "--arch" in args_passed
        assert "efficientnet_b0" in args_passed
        assert "--batch-size" in args_passed
        assert "16" in args_passed

    def test_cli_with_nonexistent_config(self, tmp_path: Path):
        """Test CLI execution with non-existent config file."""
        config_file = tmp_path / "nonexistent.yaml"

        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main([
                "--dry-run",
                "embed",
                "--config", str(config_file)
            ])

        # Should still work, just without config args
        assert exit_code == 0
        mock_run.assert_called_once()


class TestCoerceCliArgs:
    """Tests for the _coerce_cli_args utility function."""

    def test_coerce_none(self):
        """Test coercing None to empty list."""
        result = cli._coerce_cli_args(None)
        assert result == []

    def test_coerce_string(self):
        """Test coercing string to list (with shell parsing)."""
        result = cli._coerce_cli_args("--arch resnet50 --batch-size 32")
        assert result == ["--arch", "resnet50", "--batch-size", "32"]

    def test_coerce_string_with_quotes(self):
        """Test coercing string with quoted values."""
        result = cli._coerce_cli_args('--outdir "path with spaces"')
        assert result == ["--outdir", "path with spaces"]

    def test_coerce_dict(self):
        """Test coercing dict to CLI args."""
        result = cli._coerce_cli_args({"arch": "resnet50", "batch_size": 32})
        assert "--arch" in result
        assert "resnet50" in result

    def test_coerce_list(self):
        """Test coercing list to string list."""
        result = cli._coerce_cli_args(["--arch", "resnet50", 32])
        assert result == ["--arch", "resnet50", "32"]

    def test_coerce_single_value(self):
        """Test coercing single value to list."""
        result = cli._coerce_cli_args(42)
        assert result == ["42"]
