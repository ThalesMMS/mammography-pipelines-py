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

class TestBatchInferenceConfig:
    """Test BatchInferenceConfig validation."""

    def test_batch_inference_config_requires_checkpoint_and_input(self):
        """Test that checkpoint and input are required."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir) / "images"
            input_path.mkdir()

            # Valid config
            config = BatchInferenceConfig(checkpoint=checkpoint, input=input_path)
            assert config.checkpoint == checkpoint
            assert config.input == input_path

    def test_batch_inference_config_checkpoint_validation(self):
        """Test checkpoint path validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="checkpoint nao encontrado"):
                BatchInferenceConfig(
                    checkpoint=Path("/nonexistent.pt"),
                    input=Path(tmpdir),
                )

    def test_batch_inference_config_checkpoint_must_be_file(self):
        """Test that checkpoint must be a file, not directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            checkpoint_dir.mkdir()

            with pytest.raises(ValueError, match="checkpoint invalido"):
                BatchInferenceConfig(checkpoint=checkpoint_dir, input=Path(tmpdir))

    def test_batch_inference_config_input_validation(self):
        """Test input path validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")

            # Nonexistent input
            with pytest.raises(ValueError, match="input nao encontrado"):
                BatchInferenceConfig(checkpoint=checkpoint, input=Path("/nonexistent"))

    def test_batch_inference_config_defaults(self):
        """Test default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            config = BatchInferenceConfig(checkpoint=checkpoint, input=input_path)
            assert config.arch == "resnet50"
            assert config.batch_size == 16
            assert config.img_size == HP.IMG_SIZE
            assert config.device == HP.DEVICE
            assert config.amp is False
            assert config.output_format == "csv"
            assert config.resume is False
            assert config.checkpoint_interval == 100

    def test_batch_inference_config_output_format_validation(self):
        """Test output format validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            # Valid formats
            for fmt in ["csv", "json", "jsonl"]:
                config = BatchInferenceConfig(
                    checkpoint=checkpoint, input=input_path, output_format=fmt
                )
                assert config.output_format == fmt

            # Invalid format
            with pytest.raises(ValueError, match="output_format deve ser um de"):
                BatchInferenceConfig(
                    checkpoint=checkpoint, input=input_path, output_format="invalid"
                )

    def test_batch_inference_config_checkpoint_file_validation(self):
        """Test checkpoint_file validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            # Valid checkpoint_file (file doesn't need to exist yet)
            checkpoint_file = Path(tmpdir) / "checkpoint.json"
            config = BatchInferenceConfig(
                checkpoint=checkpoint,
                input=input_path,
                checkpoint_file=checkpoint_file,
            )
            assert config.checkpoint_file == checkpoint_file

            # Invalid checkpoint_file (parent directory doesn't exist)
            with pytest.raises(ValueError, match="checkpoint_file parent directory nao encontrado"):
                BatchInferenceConfig(
                    checkpoint=checkpoint,
                    input=input_path,
                    checkpoint_file=Path("/nonexistent/dir/checkpoint.json"),
                )

    def test_batch_inference_config_numeric_constraints(self):
        """Test numeric field constraints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            # Valid config
            config = BatchInferenceConfig(
                checkpoint=checkpoint,
                input=input_path,
                batch_size=32,
                img_size=256,
                checkpoint_interval=50,
            )
            assert config.batch_size == 32
            assert config.img_size == 256
            assert config.checkpoint_interval == 50

            # Invalid batch_size (< 1)
            with pytest.raises(ValidationError):
                BatchInferenceConfig(checkpoint=checkpoint, input=input_path, batch_size=0)

            # Invalid img_size (< 1)
            with pytest.raises(ValidationError):
                BatchInferenceConfig(checkpoint=checkpoint, input=input_path, img_size=0)

            # Invalid checkpoint_interval (< 1)
            with pytest.raises(ValidationError):
                BatchInferenceConfig(
                    checkpoint=checkpoint, input=input_path, checkpoint_interval=0
                )

    def test_batch_inference_config_architecture(self):
        """Test architecture configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            for arch in ["resnet50", "efficientnet_b0"]:
                config = BatchInferenceConfig(
                    checkpoint=checkpoint, input=input_path, arch=arch
                )
                assert config.arch == arch

    def test_batch_inference_rejects_transformer_with_non_224_img_size(self):
        """Transformer inference configs should validate image size upfront."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            with pytest.raises(ValueError, match="requires img_size=224"):
                BatchInferenceConfig(
                    checkpoint=checkpoint,
                    input=input_path,
                    arch="vit_b_16",
                    img_size=64,
                )

    def test_batch_inference_config_classes_parameter(self):
        """Test classes parameter configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            multiclass_config = BatchInferenceConfig(
                checkpoint=checkpoint, input=input_path, classes="multiclass"
            )
            assert multiclass_config.classes == "multiclass"

            with pytest.warns(FutureWarning, match="density"):
                legacy_config = BatchInferenceConfig(
                    checkpoint=checkpoint, input=input_path, classes="density"
                )
            assert legacy_config.classes == "multiclass"

            binary_config = BatchInferenceConfig(
                checkpoint=checkpoint, input=input_path, classes="binary"
            )
            assert binary_config.classes == "binary"

    def test_batch_inference_config_dataloader_parameters(self):
        """Test DataLoader configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            config = BatchInferenceConfig(
                checkpoint=checkpoint,
                input=input_path,
                num_workers=8,
                prefetch_factor=8,
                persistent_workers=False,
            )
            assert config.num_workers == 8
            assert config.prefetch_factor == 8
            assert config.persistent_workers is False

            # Invalid num_workers (< 0)
            with pytest.raises(ValidationError):
                BatchInferenceConfig(checkpoint=checkpoint, input=input_path, num_workers=-1)

            # Invalid prefetch_factor (< 0)
            with pytest.raises(ValidationError):
                BatchInferenceConfig(
                    checkpoint=checkpoint, input=input_path, prefetch_factor=-1
                )

    def test_batch_inference_config_from_args(self):
        """Test BatchInferenceConfig.from_args integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            args = Namespace(
                checkpoint=checkpoint,
                input=input_path,
                arch="efficientnet_b0",
                batch_size=32,
                output_format="json",
                resume=True,
            )

            config = BatchInferenceConfig.from_args(args)
            assert config.checkpoint == checkpoint
            assert config.input == input_path
            assert config.arch == "efficientnet_b0"
            assert config.batch_size == 32
            assert config.output_format == "json"
            assert config.resume is True
