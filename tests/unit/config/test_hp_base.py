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

class TestHP:
    """Test the HP dataclass for default hyperparameters."""

    def test_hp_defaults(self):
        """Test that HP dataclass has correct default values."""
        assert HP.IMG_SIZE == 512
        assert HP.EPOCHS == 100
        assert HP.BATCH_SIZE == 16
        assert HP.LR == 1e-4
        assert HP.SEED == 42
        assert HP.DEVICE == "auto"
        assert HP.CACHE_MODE == "auto"

    def test_hp_types(self):
        """Test that HP values have correct types."""
        assert isinstance(HP.IMG_SIZE, int)
        assert isinstance(HP.EPOCHS, int)
        assert isinstance(HP.LR, float)
        assert isinstance(HP.DETERMINISTIC, bool)
        assert isinstance(HP.CACHE_MODE, str)

    def test_hp_additional_defaults(self):
        """Test additional HP default values for comprehensive coverage."""
        assert HP.WINDOW_P_LOW == 0.5
        assert HP.WINDOW_P_HIGH == 99.5
        assert HP.NUM_WORKERS == 4
        assert HP.BACKBONE_LR == 1e-5
        assert HP.VAL_FRAC == 0.20
        assert HP.UNFREEZE_LAST_BLOCK is True
        assert HP.TRAIN_BACKBONE is False
        assert HP.CLASS_WEIGHTS == "none"
        assert HP.SAMPLER_WEIGHTED is False
        assert HP.WARMUP_EPOCHS == 0
        assert HP.ALLOW_TF32 is True
        assert HP.PREFETCH_FACTOR == 4
        assert HP.PERSISTENT_WORKERS is True
        assert HP.LOG_LEVEL == "info"
        assert HP.TRAIN_AUGMENT is True
        assert HP.LOADER_HEURISTICS is True
        assert HP.FUSED_OPTIM is False
        assert HP.TORCH_COMPILE is False

    def test_hp_early_stopping_defaults(self):
        """Test early stopping and LR reduction defaults."""
        assert HP.EARLY_STOP_PATIENCE == 0
        assert HP.EARLY_STOP_MIN_DELTA == 0.0
        assert HP.LR_REDUCE_PATIENCE == 0
        assert HP.LR_REDUCE_FACTOR == 0.5
        assert HP.LR_REDUCE_MIN_LR == 1e-7
        assert HP.LR_REDUCE_COOLDOWN == 0

class TestBaseConfig:
    """Test the BaseConfig base class."""

    def test_base_config_extra_ignore(self):
        """Test that extra fields are ignored."""

        class TestConfig(BaseConfig):
            field1: str = "default"

        # Should not raise error even with extra fields
        config = TestConfig(field1="test", unknown_field="value")
        assert config.field1 == "test"

    def test_from_args(self):
        """Test from_args class method."""

        class TestConfig(BaseConfig):
            field1: str = "default"
            field2: int = 42

        args = Namespace(field1="custom", field2=100)
        config = TestConfig.from_args(args)

        assert config.field1 == "custom"
        assert config.field2 == 100

    def test_from_args_with_overrides(self):
        """Test from_args with overrides parameter."""

        class TestConfig(BaseConfig):
            field1: str = "default"
            field2: int = 42

        args = Namespace(field1="custom", field2=100)
        config = TestConfig.from_args(args, field2=200)

        assert config.field1 == "custom"
        assert config.field2 == 200  # Override takes precedence
