# ruff: noqa
#
# test_config.py
# mammography-pipelines
#
# Tests for Pydantic config validation in config.py
#
# Medical Disclaimer: This is an educational research project.
# Not intended for clinical use or medical diagnosis.
#
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest
from pydantic import ValidationError

from mammography.config import (
    HP,
    BaseConfig,
    BatchInferenceConfig,
    ExtractConfig,
    InferenceConfig,
    PreprocessConfig,
    TrainConfig,
)

class TestPreprocessConfig:
    """Test PreprocessConfig validation."""

    def test_preprocess_config_requires_input_and_output(self):
        """Test that input and output are required."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input"
            input_path.mkdir()
            output_path = Path(tmpdir) / "output"

            # Valid config
            config = PreprocessConfig(input=input_path, output=output_path)
            assert config.input == input_path
            assert config.output == output_path

    def test_preprocess_config_input_validation(self):
        """Test input path validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output"

            # Nonexistent input
            with pytest.raises(ValueError, match="input nao encontrado"):
                PreprocessConfig(input=Path("/nonexistent"), output=output_path)

    def test_preprocess_config_output_validation(self):
        """Test output path validation and parent directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input"
            input_path.mkdir()

            # Output with non-existent parent (should create parent)
            output_path = Path(tmpdir) / "new_dir" / "output"
            config = PreprocessConfig(input=input_path, output=output_path)
            assert config.output == output_path
            assert output_path.parent.exists()

    def test_preprocess_config_defaults(self):
        """Test default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input"
            input_path.mkdir()
            output_path = Path(tmpdir) / "output"

            config = PreprocessConfig(input=input_path, output=output_path)
            assert config.normalize == "per-image"
            assert config.img_size == HP.IMG_SIZE
            assert config.resize is True
            assert config.crop is False
            assert config.format == "png"
            assert config.preview is False
            assert config.preview_n == 8
            assert config.report is True
            assert config.border_removal is False
            assert config.log_level == HP.LOG_LEVEL

    def test_preprocess_config_normalize_validation(self):
        """Test normalize parameter validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input"
            input_path.mkdir()
            output_path = Path(tmpdir) / "output"

            # Valid normalize options
            for normalize in ["per-image", "per-dataset", "none"]:
                config = PreprocessConfig(
                    input=input_path, output=output_path, normalize=normalize
                )
                assert config.normalize == normalize

            # Invalid normalize option
            with pytest.raises(ValueError, match="normalize deve ser um de"):
                PreprocessConfig(input=input_path, output=output_path, normalize="invalid")

    def test_preprocess_config_format_validation(self):
        """Test format parameter validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input"
            input_path.mkdir()
            output_path = Path(tmpdir) / "output"

            # Valid format options
            for fmt in ["png", "jpg", "keep"]:
                config = PreprocessConfig(input=input_path, output=output_path, format=fmt)
                assert config.format == fmt

            # Invalid format option
            with pytest.raises(ValueError, match="format deve ser um de"):
                PreprocessConfig(input=input_path, output=output_path, format="invalid")

    def test_preprocess_config_numeric_constraints(self):
        """Test numeric field constraints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input"
            input_path.mkdir()
            output_path = Path(tmpdir) / "output"

            # Valid config
            config = PreprocessConfig(
                input=input_path, output=output_path, img_size=256, preview_n=16
            )
            assert config.img_size == 256
            assert config.preview_n == 16

            # Invalid img_size (< 1)
            with pytest.raises(ValidationError):
                PreprocessConfig(input=input_path, output=output_path, img_size=0)

            # Invalid preview_n (< 1)
            with pytest.raises(ValidationError):
                PreprocessConfig(input=input_path, output=output_path, preview_n=0)

    def test_preprocess_config_boolean_flags(self):
        """Test boolean configuration flags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input"
            input_path.mkdir()
            output_path = Path(tmpdir) / "output"

            config = PreprocessConfig(
                input=input_path,
                output=output_path,
                resize=False,
                crop=True,
                preview=True,
                report=False,
                border_removal=True,
            )
            assert config.resize is False
            assert config.crop is True
            assert config.preview is True
            assert config.report is False
            assert config.border_removal is True

    def test_preprocess_config_from_args(self):
        """Test PreprocessConfig.from_args integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input"
            input_path.mkdir()
            output_path = Path(tmpdir) / "output"

            args = Namespace(
                input=input_path,
                output=output_path,
                normalize="per-dataset",
                img_size=256,
                format="jpg",
                preview=True,
            )

            config = PreprocessConfig.from_args(args)
            assert config.input == input_path
            assert config.output == output_path
            assert config.normalize == "per-dataset"
            assert config.img_size == 256
            assert config.format == "jpg"
            assert config.preview is True
