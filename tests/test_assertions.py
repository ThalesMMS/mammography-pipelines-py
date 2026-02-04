"""
Tests for custom test assertions in tests/utils/assertions.py

This file verifies that all custom assertion utilities work correctly.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pydicom = pytest.importorskip("pydicom")

from tests.utils.assertions import (
    assert_array_dtype,
    assert_array_range,
    assert_array_shape,
    assert_checkpoint_epoch,
    assert_config_field_value,
    assert_dir_exists,
    assert_dicom_metadata,
    assert_file_exists,
    assert_model_output_shape,
    assert_model_trainable_params,
    assert_tensor_dtype,
    assert_tensor_range,
    assert_tensor_shape,
    assert_valid_checkpoint,
    assert_valid_config,
    assert_valid_dicom,
)


class TestTensorAssertions:
    """Test tensor assertion utilities."""

    def test_assert_tensor_shape_valid(self):
        """Test assert_tensor_shape with valid shape."""
        x = torch.randn(2, 3, 224, 224)
        assert_tensor_shape(x, (2, 3, 224, 224))

    def test_assert_tensor_shape_invalid(self):
        """Test assert_tensor_shape with invalid shape."""
        x = torch.randn(2, 3, 224, 224)
        with pytest.raises(AssertionError, match="Tensor shape mismatch"):
            assert_tensor_shape(x, (2, 3, 128, 128))

    def test_assert_tensor_dtype_valid(self):
        """Test assert_tensor_dtype with valid dtype."""
        x = torch.randn(10, 10, dtype=torch.float32)
        assert_tensor_dtype(x, torch.float32)

    def test_assert_tensor_dtype_invalid(self):
        """Test assert_tensor_dtype with invalid dtype."""
        x = torch.randn(10, 10, dtype=torch.float32)
        with pytest.raises(AssertionError, match="Tensor dtype mismatch"):
            assert_tensor_dtype(x, torch.float64)

    def test_assert_tensor_range_valid(self):
        """Test assert_tensor_range with valid range."""
        x = torch.randn(100, 100)
        x = x.clamp(0, 1)
        assert_tensor_range(x, min_val=0.0, max_val=1.0)

    def test_assert_tensor_range_invalid_min(self):
        """Test assert_tensor_range with value below min."""
        x = torch.tensor([-1.0, 0.0, 1.0])
        with pytest.raises(AssertionError, match="min value"):
            assert_tensor_range(x, min_val=0.0)

    def test_assert_tensor_range_invalid_max(self):
        """Test assert_tensor_range with value above max."""
        x = torch.tensor([0.0, 0.5, 2.0])
        with pytest.raises(AssertionError, match="max value"):
            assert_tensor_range(x, min_val=0.0, max_val=1.0)


class TestCheckpointAssertions:
    """Test checkpoint assertion utilities."""

    def test_assert_valid_checkpoint_with_dict(self):
        """Test assert_valid_checkpoint with valid checkpoint dict."""
        checkpoint = {
            "model_state": {"weight": torch.randn(10, 10)},
            "optimizer_state": {},
            "epoch": 10,
            "best_acc": 0.85,
        }
        assert_valid_checkpoint(checkpoint)

    def test_assert_valid_checkpoint_missing_model_state(self):
        """Test assert_valid_checkpoint with missing model_state."""
        checkpoint = {"epoch": 10}
        with pytest.raises(AssertionError, match="missing required keys"):
            assert_valid_checkpoint(checkpoint)

    def test_assert_valid_checkpoint_with_optimizer(self):
        """Test assert_valid_checkpoint requiring optimizer_state."""
        checkpoint = {
            "model_state": {},
            "optimizer_state": {},
        }
        assert_valid_checkpoint(checkpoint, require_optimizer_state=True)

        checkpoint_no_opt = {"model_state": {}}
        with pytest.raises(AssertionError, match="optimizer_state"):
            assert_valid_checkpoint(checkpoint_no_opt, require_optimizer_state=True)

    def test_assert_valid_checkpoint_from_file(self):
        """Test assert_valid_checkpoint loading from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            checkpoint = {
                "model_state": {"weight": torch.randn(10, 10)},
                "epoch": 5,
            }
            torch.save(checkpoint, checkpoint_path)

            assert_valid_checkpoint(checkpoint_path)

    def test_assert_checkpoint_epoch_valid(self):
        """Test assert_checkpoint_epoch with correct epoch."""
        checkpoint = {"model_state": {}, "epoch": 42}
        assert_checkpoint_epoch(checkpoint, 42)

    def test_assert_checkpoint_epoch_invalid(self):
        """Test assert_checkpoint_epoch with wrong epoch."""
        checkpoint = {"model_state": {}, "epoch": 42}
        with pytest.raises(AssertionError, match="epoch mismatch"):
            assert_checkpoint_epoch(checkpoint, 10)


class TestConfigAssertions:
    """Test config assertion utilities."""

    def test_assert_valid_config(self):
        """Test assert_valid_config with Pydantic model."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            field1: str = "default"
            field2: int = 42

        config = TestConfig()
        assert_valid_config(config)
        assert_valid_config(config, config_class=TestConfig)

    def test_assert_valid_config_invalid_type(self):
        """Test assert_valid_config with non-Pydantic object."""
        config = {"field1": "value"}
        with pytest.raises(AssertionError, match="Pydantic BaseModel"):
            assert_valid_config(config)

    def test_assert_config_field_value_valid(self):
        """Test assert_config_field_value with correct value."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            epochs: int = 100
            lr: float = 1e-4

        config = TestConfig()
        assert_config_field_value(config, "epochs", 100)
        assert_config_field_value(config, "lr", 1e-4)

    def test_assert_config_field_value_invalid(self):
        """Test assert_config_field_value with wrong value."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            epochs: int = 100

        config = TestConfig()
        with pytest.raises(AssertionError, match="mismatch"):
            assert_config_field_value(config, "epochs", 50)


class TestDICOMAssertions:
    """Test DICOM assertion utilities."""

    def test_assert_valid_dicom(self):
        """Test assert_valid_dicom with valid DICOM."""
        ds = pydicom.Dataset()
        ds.PatientID = "TEST001"
        ds.StudyDate = "20231201"

        assert_valid_dicom(ds)

    def test_assert_valid_dicom_with_metadata(self):
        """Test assert_valid_dicom requiring metadata."""
        ds = pydicom.Dataset()
        ds.PatientID = "TEST001"
        ds.StudyDate = "20231201"

        assert_valid_dicom(ds, require_metadata=["PatientID", "StudyDate"])

    def test_assert_valid_dicom_missing_metadata(self):
        """Test assert_valid_dicom with missing required metadata."""
        ds = pydicom.Dataset()
        ds.PatientID = "TEST001"

        with pytest.raises(AssertionError, match="missing required tags"):
            assert_valid_dicom(ds, require_metadata=["PatientID", "StudyDate"])

    def test_assert_dicom_metadata_valid(self):
        """Test assert_dicom_metadata with correct values."""
        ds = pydicom.Dataset()
        ds.PatientID = "TEST001"
        ds.ViewPosition = "CC"
        ds.Laterality = "L"

        assert_dicom_metadata(
            ds,
            {"PatientID": "TEST001", "ViewPosition": "CC", "Laterality": "L"},
        )

    def test_assert_dicom_metadata_invalid(self):
        """Test assert_dicom_metadata with wrong values."""
        ds = pydicom.Dataset()
        ds.PatientID = "TEST001"
        ds.ViewPosition = "CC"

        with pytest.raises(AssertionError, match="metadata mismatches"):
            assert_dicom_metadata(ds, {"ViewPosition": "MLO"})


class TestArrayAssertions:
    """Test NumPy array assertion utilities."""

    def test_assert_array_shape_valid(self):
        """Test assert_array_shape with valid shape."""
        arr = np.zeros((10, 20, 30))
        assert_array_shape(arr, (10, 20, 30))

    def test_assert_array_shape_invalid(self):
        """Test assert_array_shape with invalid shape."""
        arr = np.zeros((10, 20, 30))
        with pytest.raises(AssertionError, match="Array shape mismatch"):
            assert_array_shape(arr, (10, 20))

    def test_assert_array_dtype_valid(self):
        """Test assert_array_dtype with valid dtype."""
        arr = np.zeros((10, 10), dtype=np.float32)
        assert_array_dtype(arr, np.float32)

    def test_assert_array_dtype_invalid(self):
        """Test assert_array_dtype with invalid dtype."""
        arr = np.zeros((10, 10), dtype=np.float32)
        with pytest.raises(AssertionError, match="Array dtype mismatch"):
            assert_array_dtype(arr, np.float64)

    def test_assert_array_range_valid(self):
        """Test assert_array_range with valid range."""
        arr = np.random.rand(100, 100)  # values in [0, 1)
        assert_array_range(arr, min_val=0.0, max_val=1.0)

    def test_assert_array_range_invalid(self):
        """Test assert_array_range with invalid range."""
        arr = np.array([-1.0, 0.0, 1.0])
        with pytest.raises(AssertionError, match="min value"):
            assert_array_range(arr, min_val=0.0)


class TestFileAssertions:
    """Test file and path assertion utilities."""

    def test_assert_file_exists_valid(self):
        """Test assert_file_exists with existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_path = Path(f.name)
            try:
                assert_file_exists(file_path)
            finally:
                file_path.unlink()

    def test_assert_file_exists_invalid(self):
        """Test assert_file_exists with non-existent file."""
        with pytest.raises(AssertionError, match="does not exist"):
            assert_file_exists("/nonexistent/file.txt")

    def test_assert_dir_exists_valid(self):
        """Test assert_dir_exists with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert_dir_exists(tmpdir)

    def test_assert_dir_exists_invalid(self):
        """Test assert_dir_exists with non-existent directory."""
        with pytest.raises(AssertionError, match="does not exist"):
            assert_dir_exists("/nonexistent/directory")


class TestModelAssertions:
    """Test model assertion utilities."""

    def test_assert_model_output_shape(self):
        """Test assert_model_output_shape with valid output."""
        model = torch.nn.Linear(10, 5)
        x = torch.randn(2, 10)
        assert_model_output_shape(model, x, (2, 5))

    def test_assert_model_output_shape_invalid(self):
        """Test assert_model_output_shape with wrong output shape."""
        model = torch.nn.Linear(10, 5)
        x = torch.randn(2, 10)
        with pytest.raises(AssertionError, match="shape mismatch"):
            assert_model_output_shape(model, x, (2, 10))

    def test_assert_model_trainable_params_valid(self):
        """Test assert_model_trainable_params with trainable model."""
        model = torch.nn.Linear(10, 5)
        assert_model_trainable_params(model, require_trainable=True)

    def test_assert_model_trainable_params_frozen(self):
        """Test assert_model_trainable_params with frozen model."""
        model = torch.nn.Linear(10, 5)
        for param in model.parameters():
            param.requires_grad = False

        with pytest.raises(AssertionError, match="no trainable parameters"):
            assert_model_trainable_params(model, require_trainable=True)
