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

class TestConfigIntegration:
    """Integration tests for config classes."""

    def test_train_config_from_args(self):
        """Test TrainConfig.from_args integration."""
        args = Namespace(
            dataset="mamografias",
            epochs=50,
            batch_size=32,
            lr=1e-3,
            arch="resnet50",
            device="cpu",
        )

        config = TrainConfig.from_args(args)
        assert config.dataset == "mamografias"
        assert config.epochs == 50
        assert config.batch_size == 32
        assert config.lr == 1e-3
        assert config.arch == "resnet50"

    def test_extract_config_from_args(self):
        """Test ExtractConfig.from_args integration."""
        args = Namespace(
            dataset="patches_completo",
            arch="efficientnet_b0",
            batch_size=16,
            device="cuda",
        )

        config = ExtractConfig.from_args(args)
        assert config.dataset == "patches_completo"
        assert config.arch == "efficientnet_b0"
        assert config.batch_size == 16
        assert config.device == "cuda"

    def test_config_model_dump(self):
        """Test that Pydantic v2 model_dump works."""
        config = TrainConfig(dataset="mamografias", epochs=10)
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert config_dict["dataset"] == "mamografias"
        assert config_dict["epochs"] == 10

    def test_config_model_validate(self):
        """Test that Pydantic v2 model_validate works."""
        data = {"dataset": "mamografias", "epochs": 20, "batch_size": 16}
        config = TrainConfig.model_validate(data)

        assert config.dataset == "mamografias"
        assert config.epochs == 20
        assert config.batch_size == 16

    def test_inference_config_from_args(self):
        """Test InferenceConfig.from_args integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            args = Namespace(
                checkpoint=checkpoint,
                input=input_path,
                arch="efficientnet_b0",
                batch_size=32,
            )

            config = InferenceConfig.from_args(args)
            assert config.checkpoint == checkpoint
            assert config.input == input_path
            assert config.arch == "efficientnet_b0"
            assert config.batch_size == 32

class TestConfigEdgeCases:
    """Test edge cases and special scenarios."""

    def test_train_config_with_both_csv_and_dataset(self):
        """Test that both csv and dataset can be provided together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("id,label\n1,A\n")

            # Both provided should work
            config = TrainConfig(
                dataset="mamografias", csv=csv_path, dicom_root=Path(tmpdir)
            )
            assert config.dataset == "mamografias"
            assert config.csv == csv_path

    def test_extract_config_with_both_csv_and_dataset(self):
        """Test that both csv and dataset can be provided together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("id,label\n1,A\n")

            # Both provided should work
            config = ExtractConfig(
                dataset="patches_completo", csv=csv_path, dicom_root=Path(tmpdir)
            )
            assert config.dataset == "patches_completo"
            assert config.csv == csv_path

    def test_train_config_pretrained_flag(self):
        """Test pretrained flag configuration."""
        # Pretrained enabled (default)
        config = TrainConfig(dataset="mamografias", pretrained=True)
        assert config.pretrained is True

        # Pretrained disabled
        config = TrainConfig(dataset="mamografias", pretrained=False)
        assert config.pretrained is False

    def test_extract_config_pretrained_flag(self):
        """Test pretrained flag for feature extraction."""
        # Pretrained enabled (default)
        config = ExtractConfig(dataset="mamografias", pretrained=True)
        assert config.pretrained is True

        # Pretrained disabled
        config = ExtractConfig(dataset="mamografias", pretrained=False)
        assert config.pretrained is False

    def test_train_config_include_class_5(self):
        """Test include_class_5 flag."""
        config = TrainConfig(dataset="mamografias", include_class_5=True)
        assert config.include_class_5 is True

        config = TrainConfig(dataset="mamografias", include_class_5=False)
        assert config.include_class_5 is False

    def test_extract_config_include_class_5(self):
        """Test include_class_5 flag for extraction."""
        config = ExtractConfig(dataset="mamografias", include_class_5=True)
        assert config.include_class_5 is True

        config = ExtractConfig(dataset="mamografias", include_class_5=False)
        assert config.include_class_5 is False

    def test_train_config_auto_normalize(self):
        """Test auto-normalization configuration."""
        config = TrainConfig(
            dataset="mamografias", auto_normalize=True, auto_normalize_samples=2000
        )
        assert config.auto_normalize is True
        assert config.auto_normalize_samples == 2000

        # Invalid auto_normalize_samples (< 1)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", auto_normalize_samples=0)

    def test_train_config_log_level(self):
        """Test log level configuration."""
        for log_level in ["debug", "info", "warning", "error"]:
            config = TrainConfig(dataset="mamografias", log_level=log_level)
            assert config.log_level == log_level

    def test_extract_config_log_level(self):
        """Test log level configuration for extraction."""
        for log_level in ["debug", "info", "warning", "error"]:
            config = ExtractConfig(dataset="mamografias", log_level=log_level)
            assert config.log_level == log_level

    def test_train_config_classes_parameter(self):
        """Test classes parameter configuration."""
        config = TrainConfig(dataset="mamografias", classes="multiclass")
        assert config.classes == "multiclass"

        with pytest.warns(FutureWarning, match="density"):
            legacy_config = TrainConfig(dataset="mamografias", classes="density")
        assert legacy_config.classes == "multiclass"

        cancer_config = TrainConfig(dataset="mamografias", classes="cancer")
        assert cancer_config.classes == "cancer"

    def test_extract_config_classes_parameter(self):
        """Test classes parameter for extraction."""
        config = ExtractConfig(dataset="mamografias", classes="multiclass")
        assert config.classes == "multiclass"

        with pytest.warns(FutureWarning, match="density"):
            legacy_config = ExtractConfig(dataset="mamografias", classes="density")
        assert legacy_config.classes == "multiclass"

        cancer_config = ExtractConfig(dataset="mamografias", classes="cancer")
        assert cancer_config.classes == "cancer"

    def test_base_config_from_args_missing_fields(self):
        """Test from_args with args missing some fields."""

        class TestConfig(BaseConfig):
            field1: str = "default1"
            field2: int = 42
            field3: bool = True

        # Args missing field3
        args = Namespace(field1="custom", field2=100)
        config = TestConfig.from_args(args)

        assert config.field1 == "custom"
        assert config.field2 == 100
        assert config.field3 is True  # Should use default

    def test_config_immutability_after_creation(self):
        """Test that config values can be read after creation."""
        config = TrainConfig(dataset="mamografias", epochs=50)

        # Should be able to read values
        assert config.epochs == 50
        assert config.dataset == "mamografias"
        assert config.arch == "efficientnet_b0"

    def test_train_config_scheduler_constraints(self):
        """Test scheduler numeric constraints."""
        # Valid scheduler parameters
        config = TrainConfig(
            dataset="mamografias",
            scheduler_min_lr=1e-8,
            scheduler_step_size=1,
            scheduler_gamma=0.1,
        )
        assert config.scheduler_min_lr == 1e-8

        # Invalid scheduler_min_lr (< 0)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", scheduler_min_lr=-1e-5)

        # Invalid scheduler_step_size (< 1)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", scheduler_step_size=0)

        # Invalid scheduler_gamma (<= 0)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", scheduler_gamma=0)

    def test_extract_config_outdir_default(self):
        """Test ExtractConfig outdir default value."""
        config = ExtractConfig(dataset="mamografias")
        assert config.outdir == "outputs/features"

    def test_train_config_outdir_default(self):
        """Test TrainConfig outdir default value."""
        config = TrainConfig(dataset="mamografias")
        assert config.outdir == "outputs/run"

    def test_extract_config_dicom_root_with_hint(self):
        """Test that _normalize_dir_hint works for dicom_root."""
        # This would test the typo correction from 'archieve' to 'archive'
        # but we can't create that scenario easily in tests without actual filesystem
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_dir = Path(tmpdir) / "archive"
            archive_dir.mkdir()
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("id,label\n1,A\n")

            # Valid path should work
            config = ExtractConfig(csv=csv_path, dicom_root=archive_dir)
            assert config.dicom_root == archive_dir
