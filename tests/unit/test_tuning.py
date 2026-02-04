"""Unit tests for hyperparameter tuning module."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography.tuning.search_space import (
    SearchSpace,
    CategoricalParam,
    IntParam,
    FloatParam,
)


class TestSearchSpace:
    """Test SearchSpace YAML loading and validation."""

    def test_from_yaml_valid(self):
        """Test loading valid search space from YAML."""
        space = SearchSpace.from_yaml("configs/tune.yaml")
        assert len(space.parameters) == 6
        assert "lr" in space.parameters
        assert "backbone_lr" in space.parameters
        assert "batch_size" in space.parameters
        assert "warmup_epochs" in space.parameters
        assert "early_stop_patience" in space.parameters
        assert "unfreeze_last_block" in space.parameters

    def test_from_yaml_missing_file(self):
        """Test error handling for missing YAML file."""
        with pytest.raises(FileNotFoundError):
            SearchSpace.from_yaml("nonexistent.yaml")

    def test_from_yaml_invalid_format(self, tmp_path):
        """Test error handling for invalid YAML format."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("not a dict: [1, 2, 3]")
        with pytest.raises(ValueError, match="expected dict"):
            SearchSpace.from_yaml(invalid_yaml)

    def test_to_dict(self):
        """Test converting search space to dictionary."""
        space = SearchSpace.from_yaml("configs/tune.yaml")
        result = space.to_dict()
        assert "parameters" in result
        assert "description" in result
        assert len(result["parameters"]) == 6


class TestCategoricalParam:
    """Test CategoricalParam validation."""

    def test_valid_choices(self):
        """Test creating categorical parameter with valid choices."""
        param = CategoricalParam(choices=[1, 2, 3])
        assert param.type == "categorical"
        assert param.choices == [1, 2, 3]

    def test_empty_choices(self):
        """Test validation fails for empty choices."""
        with pytest.raises(ValueError):
            CategoricalParam(choices=[])

    def test_mixed_type_choices(self):
        """Test choices can contain mixed types."""
        param = CategoricalParam(choices=[1, "two", 3.0, True])
        assert len(param.choices) == 4


class TestIntParam:
    """Test IntParam validation."""

    def test_valid_range(self):
        """Test creating int parameter with valid range."""
        param = IntParam(type="int", low=1, high=10, step=2)
        assert param.low == 1
        assert param.high == 10
        assert param.step == 2
        assert param.log is False

    def test_invalid_bounds(self):
        """Test validation fails when low >= high."""
        with pytest.raises(ValueError, match="low .* must be < high"):
            IntParam(type="int", low=10, high=5)

    def test_log_scale(self):
        """Test log scale parameter."""
        param = IntParam(type="int", low=1, high=100, log=True)
        assert param.log is True


class TestFloatParam:
    """Test FloatParam validation."""

    def test_valid_range(self):
        """Test creating float parameter with valid range."""
        param = FloatParam(type="float", low=0.0001, high=0.01, log=True)
        assert param.low == 0.0001
        assert param.high == 0.01
        assert param.log is True

    def test_invalid_bounds(self):
        """Test validation fails when low >= high."""
        with pytest.raises(ValueError, match="low .* must be < high"):
            FloatParam(type="float", low=0.01, high=0.0001)

    def test_invalid_step(self):
        """Test validation fails when step > range."""
        with pytest.raises(ValueError, match="step .* must be <="):
            FloatParam(type="float", low=0.0, high=1.0, step=2.0)

    def test_with_step(self):
        """Test float parameter with discrete step."""
        param = FloatParam(type="float", low=0.0, high=1.0, step=0.1)
        assert param.step == 0.1
