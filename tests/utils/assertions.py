"""
Custom test assertions for mammography pipelines testing.

Provides reusable assertion utilities for validating tensors, checkpoints,
configs, DICOM metadata, and other common test scenarios. These assertions
provide more descriptive error messages than basic assert statements.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


# =============================================================================
# Tensor Assertions
# =============================================================================

def assert_tensor_shape(
    tensor: "torch.Tensor",
    expected_shape: Union[Tuple[int, ...], List[int]],
    message: Optional[str] = None,
) -> None:
    """
    Assert that a tensor has the expected shape.

    Args:
        tensor: PyTorch tensor to validate
        expected_shape: Expected shape as tuple or list (e.g., (3, 224, 224))
        message: Optional custom error message

    Raises:
        AssertionError: If tensor shape doesn't match expected shape
        ImportError: If PyTorch is not available

    Example:
        >>> import torch
        >>> x = torch.randn(2, 3, 224, 224)
        >>> assert_tensor_shape(x, (2, 3, 224, 224))
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for tensor assertions")

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")

    actual_shape = tuple(tensor.shape)
    expected_shape = tuple(expected_shape)

    if actual_shape != expected_shape:
        error_msg = (
            f"Tensor shape mismatch: expected {expected_shape}, got {actual_shape}"
        )
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


def assert_tensor_dtype(
    tensor: "torch.Tensor",
    expected_dtype: "torch.dtype",
    message: Optional[str] = None,
) -> None:
    """
    Assert that a tensor has the expected dtype.

    Args:
        tensor: PyTorch tensor to validate
        expected_dtype: Expected dtype (e.g., torch.float32, torch.int64)
        message: Optional custom error message

    Raises:
        AssertionError: If tensor dtype doesn't match expected dtype
        ImportError: If PyTorch is not available
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for tensor assertions")

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")

    if tensor.dtype != expected_dtype:
        error_msg = f"Tensor dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


def assert_tensor_range(
    tensor: "torch.Tensor",
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    message: Optional[str] = None,
) -> None:
    """
    Assert that tensor values are within expected range.

    Args:
        tensor: PyTorch tensor to validate
        min_val: Minimum expected value (inclusive)
        max_val: Maximum expected value (inclusive)
        message: Optional custom error message

    Raises:
        AssertionError: If tensor values are outside expected range
        ImportError: If PyTorch is not available
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for tensor assertions")

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")

    actual_min = tensor.min().item()
    actual_max = tensor.max().item()

    if min_val is not None and actual_min < min_val:
        error_msg = f"Tensor min value {actual_min} is less than expected {min_val}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)

    if max_val is not None and actual_max > max_val:
        error_msg = f"Tensor max value {actual_max} is greater than expected {max_val}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


# =============================================================================
# Checkpoint Assertions
# =============================================================================

def assert_valid_checkpoint(
    checkpoint: Union[Path, Dict[str, Any]],
    require_model_state: bool = True,
    require_optimizer_state: bool = False,
    require_epoch: bool = False,
    additional_keys: Optional[List[str]] = None,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a checkpoint is valid and contains expected keys.

    Args:
        checkpoint: Path to checkpoint file or loaded checkpoint dict
        require_model_state: Whether to require 'model_state' or 'model_state_dict' key
        require_optimizer_state: Whether to require 'optimizer_state' key
        require_epoch: Whether to require 'epoch' key
        additional_keys: Additional keys that must be present
        message: Optional custom error message

    Raises:
        AssertionError: If checkpoint is invalid or missing required keys
        ImportError: If PyTorch is not available
        FileNotFoundError: If checkpoint path doesn't exist

    Example:
        >>> checkpoint = {
        ...     "model_state": {...},
        ...     "optimizer_state": {...},
        ...     "epoch": 10,
        ...     "best_acc": 0.85
        ... }
        >>> assert_valid_checkpoint(checkpoint, require_optimizer_state=True)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for checkpoint assertions")

    # Load checkpoint if path provided
    if isinstance(checkpoint, (Path, str)):
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        error_msg = f"Checkpoint must be a dict, got {type(checkpoint).__name__}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)

    # Check required keys
    missing_keys = []

    if require_model_state:
        if "model_state" not in checkpoint and "model_state_dict" not in checkpoint and "state_dict" not in checkpoint:
            missing_keys.append("model_state (or model_state_dict/state_dict)")

    if require_optimizer_state:
        if "optimizer_state" not in checkpoint:
            missing_keys.append("optimizer_state")

    if require_epoch:
        if "epoch" not in checkpoint:
            missing_keys.append("epoch")

    if additional_keys:
        for key in additional_keys:
            if key not in checkpoint:
                missing_keys.append(key)

    if missing_keys:
        error_msg = f"Checkpoint missing required keys: {', '.join(missing_keys)}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


def assert_checkpoint_epoch(
    checkpoint: Union[Path, Dict[str, Any]],
    expected_epoch: int,
    message: Optional[str] = None,
) -> None:
    """
    Assert that checkpoint is from expected epoch.

    Args:
        checkpoint: Path to checkpoint file or loaded checkpoint dict
        expected_epoch: Expected epoch number
        message: Optional custom error message

    Raises:
        AssertionError: If checkpoint epoch doesn't match
        ImportError: If PyTorch is not available
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for checkpoint assertions")

    if isinstance(checkpoint, (Path, str)):
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "epoch" not in checkpoint:
        raise AssertionError("Checkpoint does not contain 'epoch' key")

    actual_epoch = checkpoint["epoch"]
    if actual_epoch != expected_epoch:
        error_msg = f"Checkpoint epoch mismatch: expected {expected_epoch}, got {actual_epoch}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


# =============================================================================
# Config Assertions
# =============================================================================

def assert_valid_config(
    config: Any,
    config_class: Optional[type] = None,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a config object is valid (Pydantic model).

    Args:
        config: Config object to validate
        config_class: Expected Pydantic model class
        message: Optional custom error message

    Raises:
        AssertionError: If config is invalid
        ImportError: If Pydantic is not available
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic is required for config assertions")

    if config_class is not None and not isinstance(config, config_class):
        error_msg = f"Config type mismatch: expected {config_class.__name__}, got {type(config).__name__}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)

    # Verify it's a Pydantic model
    if not isinstance(config, BaseModel):
        error_msg = f"Config must be a Pydantic BaseModel, got {type(config).__name__}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


def assert_config_field_value(
    config: Any,
    field_name: str,
    expected_value: Any,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a config field has expected value.

    Args:
        config: Config object
        field_name: Name of field to check
        expected_value: Expected value for the field
        message: Optional custom error message

    Raises:
        AssertionError: If field value doesn't match
    """
    if not hasattr(config, field_name):
        error_msg = f"Config missing field: {field_name}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)

    actual_value = getattr(config, field_name)
    if actual_value != expected_value:
        error_msg = (
            f"Config field '{field_name}' mismatch: "
            f"expected {expected_value}, got {actual_value}"
        )
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


# =============================================================================
# DICOM Assertions
# =============================================================================

def assert_valid_dicom(
    dicom: Any,
    require_pixel_array: bool = False,
    require_metadata: Optional[List[str]] = None,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a DICOM dataset is valid.

    Args:
        dicom: pydicom Dataset object
        require_pixel_array: Whether to require pixel_array attribute
        require_metadata: List of required metadata tags (e.g., ['PatientID', 'StudyDate'])
        message: Optional custom error message

    Raises:
        AssertionError: If DICOM is invalid or missing required data
        ImportError: If pydicom is not available
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required for DICOM assertions")

    if not isinstance(dicom, pydicom.Dataset):
        error_msg = f"Expected pydicom.Dataset, got {type(dicom).__name__}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)

    if require_pixel_array:
        if not hasattr(dicom, "pixel_array"):
            error_msg = "DICOM dataset missing pixel_array"
            if message:
                error_msg = f"{message}: {error_msg}"
            raise AssertionError(error_msg)

    if require_metadata:
        missing_tags = []
        for tag in require_metadata:
            if not hasattr(dicom, tag):
                missing_tags.append(tag)

        if missing_tags:
            error_msg = f"DICOM dataset missing required tags: {', '.join(missing_tags)}"
            if message:
                error_msg = f"{message}: {error_msg}"
            raise AssertionError(error_msg)


def assert_dicom_metadata(
    dicom: Any,
    expected_metadata: Dict[str, Any],
    message: Optional[str] = None,
) -> None:
    """
    Assert that DICOM metadata matches expected values.

    Args:
        dicom: pydicom Dataset object
        expected_metadata: Dict of tag names and expected values
        message: Optional custom error message

    Raises:
        AssertionError: If metadata doesn't match
        ImportError: If pydicom is not available
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required for DICOM assertions")

    if not isinstance(dicom, pydicom.Dataset):
        error_msg = f"Expected pydicom.Dataset, got {type(dicom).__name__}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)

    mismatches = []
    for tag, expected_value in expected_metadata.items():
        if not hasattr(dicom, tag):
            mismatches.append(f"{tag}: missing")
        else:
            actual_value = getattr(dicom, tag)
            if actual_value != expected_value:
                mismatches.append(f"{tag}: expected {expected_value}, got {actual_value}")

    if mismatches:
        error_msg = f"DICOM metadata mismatches: {'; '.join(mismatches)}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


# =============================================================================
# NumPy Assertions
# =============================================================================

def assert_array_shape(
    array: np.ndarray,
    expected_shape: Union[Tuple[int, ...], List[int]],
    message: Optional[str] = None,
) -> None:
    """
    Assert that a NumPy array has the expected shape.

    Args:
        array: NumPy array to validate
        expected_shape: Expected shape as tuple or list
        message: Optional custom error message

    Raises:
        AssertionError: If array shape doesn't match
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(array).__name__}")

    actual_shape = array.shape
    expected_shape = tuple(expected_shape)

    if actual_shape != expected_shape:
        error_msg = f"Array shape mismatch: expected {expected_shape}, got {actual_shape}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


def assert_array_dtype(
    array: np.ndarray,
    expected_dtype: np.dtype,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a NumPy array has the expected dtype.

    Args:
        array: NumPy array to validate
        expected_dtype: Expected dtype
        message: Optional custom error message

    Raises:
        AssertionError: If array dtype doesn't match
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(array).__name__}")

    if array.dtype != expected_dtype:
        error_msg = f"Array dtype mismatch: expected {expected_dtype}, got {array.dtype}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


def assert_array_range(
    array: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    message: Optional[str] = None,
) -> None:
    """
    Assert that array values are within expected range.

    Args:
        array: NumPy array to validate
        min_val: Minimum expected value (inclusive)
        max_val: Maximum expected value (inclusive)
        message: Optional custom error message

    Raises:
        AssertionError: If array values are outside expected range
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(array).__name__}")

    actual_min = array.min()
    actual_max = array.max()

    if min_val is not None and actual_min < min_val:
        error_msg = f"Array min value {actual_min} is less than expected {min_val}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)

    if max_val is not None and actual_max > max_val:
        error_msg = f"Array max value {actual_max} is greater than expected {max_val}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


# =============================================================================
# File and Path Assertions
# =============================================================================

def assert_file_exists(
    file_path: Union[Path, str],
    message: Optional[str] = None,
) -> None:
    """
    Assert that a file exists.

    Args:
        file_path: Path to file
        message: Optional custom error message

    Raises:
        AssertionError: If file doesn't exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        error_msg = f"File does not exist: {file_path}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)

    if not file_path.is_file():
        error_msg = f"Path is not a file: {file_path}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


def assert_dir_exists(
    dir_path: Union[Path, str],
    message: Optional[str] = None,
) -> None:
    """
    Assert that a directory exists.

    Args:
        dir_path: Path to directory
        message: Optional custom error message

    Raises:
        AssertionError: If directory doesn't exist
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        error_msg = f"Directory does not exist: {dir_path}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)

    if not dir_path.is_dir():
        error_msg = f"Path is not a directory: {dir_path}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


# =============================================================================
# Model Assertions
# =============================================================================

def assert_model_output_shape(
    model: Any,
    input_tensor: "torch.Tensor",
    expected_output_shape: Union[Tuple[int, ...], List[int]],
    message: Optional[str] = None,
) -> None:
    """
    Assert that model produces output with expected shape.

    Args:
        model: PyTorch model
        input_tensor: Input tensor for model
        expected_output_shape: Expected output shape
        message: Optional custom error message

    Raises:
        AssertionError: If output shape doesn't match
        ImportError: If PyTorch is not available
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for model assertions")

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    assert_tensor_shape(output, expected_output_shape, message)


def assert_model_trainable_params(
    model: Any,
    require_trainable: bool = True,
    message: Optional[str] = None,
) -> None:
    """
    Assert that model has trainable parameters.

    Args:
        model: PyTorch model
        require_trainable: Whether to require at least one trainable parameter
        message: Optional custom error message

    Raises:
        AssertionError: If model doesn't have trainable params when required
        ImportError: If PyTorch is not available
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for model assertions")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if require_trainable and trainable_params == 0:
        error_msg = "Model has no trainable parameters"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


# =============================================================================
# Utility Functions
# =============================================================================

__all__ = [
    # Tensor assertions
    "assert_tensor_shape",
    "assert_tensor_dtype",
    "assert_tensor_range",
    # Checkpoint assertions
    "assert_valid_checkpoint",
    "assert_checkpoint_epoch",
    # Config assertions
    "assert_valid_config",
    "assert_config_field_value",
    # DICOM assertions
    "assert_valid_dicom",
    "assert_dicom_metadata",
    # NumPy assertions
    "assert_array_shape",
    "assert_array_dtype",
    "assert_array_range",
    # File assertions
    "assert_file_exists",
    "assert_dir_exists",
    # Model assertions
    "assert_model_output_shape",
    "assert_model_trainable_params",
]
