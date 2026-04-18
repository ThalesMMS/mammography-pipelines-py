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

class TestInferenceConfig:
    """Test InferenceConfig validation."""

    def test_inference_config_requires_checkpoint_and_input(self):
        """Test that checkpoint and input are required."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir) / "images"
            input_path.mkdir()

            # Valid config
            config = InferenceConfig(checkpoint=checkpoint, input=input_path)
            assert config.checkpoint == checkpoint
            assert config.input == input_path

    def test_inference_config_checkpoint_validation(self):
        """Test checkpoint path validation."""
        # Nonexistent checkpoint
        with pytest.raises(ValueError, match="checkpoint nao encontrado"):
            InferenceConfig(checkpoint=Path("/nonexistent.pt"), input=Path("/tmp"))

    def test_inference_config_checkpoint_must_be_file(self):
        """Test that checkpoint must be a file, not directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            checkpoint_dir.mkdir()

            with pytest.raises(ValueError, match="checkpoint invalido"):
                InferenceConfig(checkpoint=checkpoint_dir, input=Path(tmpdir))

    def test_inference_config_input_validation(self):
        """Test input path validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")

            # Nonexistent input
            with pytest.raises(ValueError, match="input nao encontrado"):
                InferenceConfig(checkpoint=checkpoint, input=Path("/nonexistent"))

    def test_inference_config_validates_csv_and_dicom_root_sources(self):
        """Test alternate inference sources are validated and accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            csv_path = Path(tmpdir) / "inputs.csv"
            csv_path.write_text("image_path\nimage.png\n")
            dicom_root = Path(tmpdir) / "dicoms"
            dicom_root.mkdir()

            csv_config = InferenceConfig(checkpoint=checkpoint, csv=csv_path)
            assert csv_config.csv == csv_path

            dicom_config = InferenceConfig(
                checkpoint=checkpoint,
                dicom_root=dicom_root,
            )
            assert dicom_config.dicom_root == dicom_root

            with pytest.raises(ValueError, match="csv nao encontrado"):
                InferenceConfig(checkpoint=checkpoint, csv=Path("/nonexistent.csv"))
            with pytest.raises(ValueError, match="dicom_root nao encontrado"):
                InferenceConfig(checkpoint=checkpoint, dicom_root=Path("/missing"))

    def test_inference_config_requires_at_least_one_input_source(self):
        """Test that inference config needs input, csv, or dicom_root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")

            with pytest.raises(ValueError, match="--input, --csv ou --dicom-root"):
                InferenceConfig(checkpoint=checkpoint)

    def test_inference_config_defaults(self):
        """Test default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            config = InferenceConfig(checkpoint=checkpoint, input=input_path)
            assert config.arch == "resnet50"
            assert config.batch_size == 16
            assert config.img_size == HP.IMG_SIZE
            assert config.device == HP.DEVICE
            assert config.amp is False

    def test_inference_config_architecture(self):
        """Test architecture configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            for arch in ["resnet50", "efficientnet_b0", "vit_base_patch16_224"]:
                config = InferenceConfig(
                    checkpoint=checkpoint, input=input_path, arch=arch
                )
                assert config.arch == arch

    def test_inference_config_classes_parameter(self):
        """Test classes parameter configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            config = InferenceConfig(
                checkpoint=checkpoint, input=input_path, classes="multiclass"
            )
            assert config.classes == "multiclass"

            with pytest.warns(FutureWarning, match="density"):
                legacy_config = InferenceConfig(
                    checkpoint=checkpoint, input=input_path, classes="density"
                )
            assert legacy_config.classes == "multiclass"

            cancer_config = InferenceConfig(
                checkpoint=checkpoint, input=input_path, classes="cancer"
            )
            assert cancer_config.classes == "cancer"

    def test_inference_config_output_parameter(self):
        """Test output path configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)
            output = str(Path(tmpdir) / "predictions.csv")

            config = InferenceConfig(
                checkpoint=checkpoint, input=input_path, output=output
            )
            assert config.output == output

    def test_inference_config_numeric_constraints(self):
        """Test numeric field constraints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            # Valid config
            config = InferenceConfig(
                checkpoint=checkpoint, input=input_path, batch_size=32, img_size=256
            )
            assert config.batch_size == 32
            assert config.img_size == 256

            # Invalid batch_size (< 1)
            with pytest.raises(Exception):
                InferenceConfig(checkpoint=checkpoint, input=input_path, batch_size=0)

            # Invalid img_size (< 1)
            with pytest.raises(Exception):
                InferenceConfig(checkpoint=checkpoint, input=input_path, img_size=0)

    def test_inference_config_normalization_parameters(self):
        """Test normalization configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            config = InferenceConfig(
                checkpoint=checkpoint,
                input=input_path,
                mean="0.485,0.456,0.406",
                std="0.229,0.224,0.225",
            )
            assert config.mean == "0.485,0.456,0.406"
            assert config.std == "0.229,0.224,0.225"

    def test_inference_config_performance_flags(self):
        """Test performance configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            config = InferenceConfig(
                checkpoint=checkpoint, input=input_path, amp=True, device="cuda"
            )
            assert config.amp is True
            assert config.device == "cuda"
