"""
PreprocessedTensor model for standardized image data representation.

This module defines the data structure for representing preprocessed
mammography images as PyTorch tensors, including preprocessing metadata
and configuration tracking.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- This model represents the second stage in our unsupervised learning pipeline
- It standardizes images for consistent ResNet-50 feature extraction
- Preprocessing configuration tracking enables reproducibility
- Tensor validation ensures proper input format for deep learning models

Author: Research Team
Version: 1.0.0
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# Configure logging for educational purposes
logger = logging.getLogger(__name__)


class PreprocessedTensor:
    """
    Represents standardized image data after preprocessing operations.

    This class encapsulates preprocessed mammography images as PyTorch tensors
    along with all preprocessing metadata needed for reproducibility and
    analysis. It serves as the bridge between raw DICOM data and deep learning
    feature extraction.

    Educational Notes:
    - Tensor format: (C, H, W) where C=3 channels (RGB), H=height, W=width
    - Preprocessing steps: resizing, normalization, input adapter application
    - Configuration tracking enables exact reproduction of preprocessing
    - Validation ensures compatibility with ResNet-50 input requirements

    Attributes:
        image_id (str): References MammographyImage.instance_id
        tensor_data (torch.Tensor): Preprocessed image tensor (C, H, W)
        preprocessing_config (dict): Configuration used for preprocessing
        normalization_method (str): Normalization method applied
        target_size (tuple[int, int]): Final image dimensions (H, W)
        input_adapter (str): Channel handling method used
        border_removed (bool): Whether borders were removed
        created_at (datetime): Timestamp of preprocessing
    """

    # Define valid normalization methods
    VALID_NORMALIZATION_METHODS = [
        "z_score_per_image",
        "fixed_window",
        "min_max_scaling",
        "percentile_scaling",
    ]

    # Define valid input adapters for grayscale to RGB conversion
    VALID_INPUT_ADAPTERS = ["1to3_replication", "conv1_adapted"]

    # Define valid padding strategies
    VALID_PADDING_STRATEGIES = ["reflect", "constant", "edge"]

    def __init__(
        self,
        image_id: str,
        tensor_data: torch.Tensor,
        preprocessing_config: Dict[str, Any],
        normalization_method: str,
        target_size: Tuple[int, int],
        input_adapter: str,
        border_removed: bool = False,
        original_shape: Optional[Tuple[int, int]] = None,
        processing_time: Optional[float] = None,
        created_at: Optional[datetime] = None,
    ):
        """
        Initialize a PreprocessedTensor instance.

        Args:
            image_id: References MammographyImage.instance_id
            tensor_data: Preprocessed image tensor (C, H, W)
            preprocessing_config: Configuration used for preprocessing
            normalization_method: Normalization method applied
            target_size: Final image dimensions (H, W)
            input_adapter: Channel handling method used
            border_removed: Whether borders were removed
            original_shape: Original image dimensions before preprocessing
            processing_time: Time taken for preprocessing (seconds)
            created_at: Timestamp of preprocessing (default: now)

        Raises:
            ValueError: If validation rules are violated
            TypeError: If data types are incorrect
        """
        # Initialize core attributes with validation
        self.image_id = self._validate_image_id(image_id)
        self.tensor_data = self._validate_tensor_data(tensor_data)
        self.preprocessing_config = self._validate_preprocessing_config(
            preprocessing_config
        )
        self.normalization_method = self._validate_normalization_method(
            normalization_method
        )
        self.target_size = self._validate_target_size(target_size)
        self.input_adapter = self._validate_input_adapter(input_adapter)
        self.border_removed = self._validate_border_removed(border_removed)
        self.original_shape = original_shape or self._extract_original_shape()
        self.processing_time = processing_time or 0.0
        self.created_at = created_at or datetime.now()

        # Initialize tracking attributes
        self.validation_errors: List[str] = []
        self.updated_at = datetime.now()

        # Validate tensor matches target size
        self._validate_tensor_target_size_match()

        # Log creation for educational purposes
        logger.info(
            f"Created PreprocessedTensor: {self.image_id} with shape {self.tensor_data.shape}"
        )

    def _validate_image_id(self, image_id: str) -> str:
        """
        Validate image ID.

        Educational Note: Image ID links this preprocessed tensor back to
        the original MammographyImage for traceability.

        Args:
            image_id: Image identifier to validate

        Returns:
            str: Validated image ID

        Raises:
            ValueError: If image ID is invalid
            TypeError: If image ID is not a string
        """
        if not isinstance(image_id, str):
            raise TypeError(f"image_id must be a string, got {type(image_id)}")

        if not image_id.strip():
            raise ValueError("image_id cannot be empty or whitespace")

        return image_id.strip()

    def _validate_tensor_data(self, tensor_data: torch.Tensor) -> torch.Tensor:
        """
        Validate tensor data.

        Educational Note: Tensor validation ensures proper format for
        ResNet-50 feature extraction: 3 channels (RGB), float32 dtype,
        and reasonable value ranges.

        Args:
            tensor_data: Tensor to validate

        Returns:
            torch.Tensor: Validated tensor

        Raises:
            ValueError: If tensor is invalid
            TypeError: If tensor is not a PyTorch tensor
        """
        if not isinstance(tensor_data, torch.Tensor):
            raise TypeError(
                f"tensor_data must be a torch.Tensor, got {type(tensor_data)}"
            )

        # Check tensor dimensions (should be 3D: C, H, W)
        if tensor_data.ndim != 3:
            raise ValueError(
                f"tensor_data must be 3D (C, H, W), got {tensor_data.ndim}D"
            )

        # Check number of channels (should be 3 for RGB)
        if tensor_data.shape[0] != 3:
            raise ValueError(
                f"tensor_data must have 3 channels, got {tensor_data.shape[0]}"
            )

        # Check height and width are positive
        if tensor_data.shape[1] <= 0 or tensor_data.shape[2] <= 0:
            raise ValueError(
                f"tensor_data height and width must be positive, got {tensor_data.shape[1:3]}"
            )

        # Check data type (should be float32 for ResNet-50)
        if tensor_data.dtype != torch.float32:
            logger.warning(f"Converting tensor from {tensor_data.dtype} to float32")
            tensor_data = tensor_data.float()

        # Check for NaN or infinite values
        if torch.any(torch.isnan(tensor_data)):
            raise ValueError("tensor_data contains NaN values")

        if torch.any(torch.isinf(tensor_data)):
            raise ValueError("tensor_data contains infinite values")

        return tensor_data

    def _validate_preprocessing_config(
        self, preprocessing_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate preprocessing configuration.

        Educational Note: Configuration validation ensures all necessary
        preprocessing parameters are present and valid.

        Args:
            preprocessing_config: Configuration dictionary to validate

        Returns:
            Dict[str, Any]: Validated configuration

        Raises:
            ValueError: If configuration is invalid
            TypeError: If configuration is not a dictionary
        """
        if not isinstance(preprocessing_config, dict):
            raise TypeError(
                f"preprocessing_config must be a dictionary, got {type(preprocessing_config)}"
            )

        # Check for required configuration keys
        required_keys = ["target_size", "normalization_method", "input_adapter"]
        for key in required_keys:
            if key not in preprocessing_config:
                raise ValueError(f"Missing required configuration key: {key}")

        return preprocessing_config

    def _validate_normalization_method(self, normalization_method: str) -> str:
        """
        Validate normalization method.

        Educational Note: Normalization is crucial for consistent feature
        extraction. Different methods have different effects on the data.

        Args:
            normalization_method: Normalization method to validate

        Returns:
            str: Validated normalization method

        Raises:
            ValueError: If normalization method is invalid
            TypeError: If normalization method is not a string
        """
        if not isinstance(normalization_method, str):
            raise TypeError(
                f"normalization_method must be a string, got {type(normalization_method)}"
            )

        if normalization_method not in self.VALID_NORMALIZATION_METHODS:
            raise ValueError(
                f"normalization_method must be one of {self.VALID_NORMALIZATION_METHODS}, got {normalization_method}"
            )

        return normalization_method

    def _validate_target_size(self, target_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Validate target size.

        Educational Note: Target size determines the final dimensions
        of the preprocessed image, typically powers of 2 for efficiency.

        Args:
            target_size: Target size to validate (H, W)

        Returns:
            Tuple[int, int]: Validated target size

        Raises:
            ValueError: If target size is invalid
            TypeError: If target size is not a tuple
        """
        if not isinstance(target_size, tuple):
            raise TypeError(f"target_size must be a tuple, got {type(target_size)}")

        if len(target_size) != 2:
            raise ValueError(
                f"target_size must have exactly 2 elements, got {len(target_size)}"
            )

        for i, size in enumerate(target_size):
            if not isinstance(size, int):
                raise TypeError(
                    f"target_size[{i}] must be an integer, got {type(size)}"
                )

            if size <= 0:
                raise ValueError(f"target_size[{i}] must be positive, got {size}")

            # Check for reasonable size limits
            if size < 64 or size > 2048:
                logger.warning(
                    f"target_size[{i}] = {size} is outside typical range [64, 2048]"
                )

        return target_size

    def _validate_input_adapter(self, input_adapter: str) -> str:
        """
        Validate input adapter.

        Educational Note: Input adapters handle the conversion from
        grayscale mammography images to 3-channel RGB format required
        by ResNet-50.

        Args:
            input_adapter: Input adapter to validate

        Returns:
            str: Validated input adapter

        Raises:
            ValueError: If input adapter is invalid
            TypeError: If input adapter is not a string
        """
        if not isinstance(input_adapter, str):
            raise TypeError(
                f"input_adapter must be a string, got {type(input_adapter)}"
            )

        if input_adapter not in self.VALID_INPUT_ADAPTERS:
            raise ValueError(
                f"input_adapter must be one of {self.VALID_INPUT_ADAPTERS}, got {input_adapter}"
            )

        return input_adapter

    def _validate_border_removed(self, border_removed: bool) -> bool:
        """
        Validate border removed flag.

        Educational Note: Border removal is an optional preprocessing step
        that can improve clustering quality by removing non-breast tissue.

        Args:
            border_removed: Border removed flag to validate

        Returns:
            bool: Validated border removed flag

        Raises:
            TypeError: If border removed is not a boolean
        """
        if not isinstance(border_removed, bool):
            raise TypeError(
                f"border_removed must be a boolean, got {type(border_removed)}"
            )

        return border_removed

    def _extract_original_shape(self) -> Tuple[int, int]:
        """
        Extract original shape from tensor data.

        Educational Note: Original shape is used for tracking the
        preprocessing transformation applied to the image.

        Returns:
            Tuple[int, int]: Original image dimensions (H, W)
        """
        # For now, we'll use the current tensor shape as original
        # In a real implementation, this would be stored during preprocessing
        return (self.tensor_data.shape[1], self.tensor_data.shape[2])

    def _validate_tensor_target_size_match(self) -> None:
        """
        Validate that tensor dimensions match target size.

        Educational Note: This validation ensures consistency between
        the preprocessing configuration and the actual tensor data.

        Raises:
            ValueError: If tensor dimensions don't match target size
        """
        tensor_h, tensor_w = self.tensor_data.shape[1], self.tensor_data.shape[2]
        target_h, target_w = self.target_size

        if tensor_h != target_h or tensor_w != target_w:
            error_msg = f"Tensor dimensions ({tensor_h}, {tensor_w}) don't match target size ({target_h}, {target_w})"
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)

    def get_tensor_stats(self) -> Dict[str, float]:
        """
        Get statistical information about the tensor.

        Educational Note: Tensor statistics are useful for understanding
        the preprocessing effects and ensuring proper normalization.

        Returns:
            Dict[str, float]: Dictionary containing tensor statistics
        """
        return {
            "mean": float(torch.mean(self.tensor_data).item()),
            "std": float(torch.std(self.tensor_data).item()),
            "min": float(torch.min(self.tensor_data).item()),
            "max": float(torch.max(self.tensor_data).item()),
            "shape": list(self.tensor_data.shape),
            "dtype": str(self.tensor_data.dtype),
        }

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the preprocessing applied.

        Educational Note: This summary provides a complete record of
        all preprocessing steps for reproducibility and analysis.

        Returns:
            Dict[str, Any]: Dictionary containing preprocessing summary
        """
        return {
            "image_id": self.image_id,
            "original_shape": self.original_shape,
            "target_size": self.target_size,
            "normalization_method": self.normalization_method,
            "input_adapter": self.input_adapter,
            "border_removed": self.border_removed,
            "processing_time": self.processing_time,
            "tensor_stats": self.get_tensor_stats(),
            "preprocessing_config": self.preprocessing_config,
            "created_at": self.created_at.isoformat(),
            "validation_errors": self.validation_errors,
        }

    def save_tensor(self, file_path: Union[str, Path]) -> bool:
        """
        Save tensor data to file.

        Educational Note: Tensor saving enables caching of preprocessed
        data to avoid reprocessing during experiments.

        Args:
            file_path: Path where to save the tensor

        Returns:
            bool: True if saving successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save tensor data
            torch.save(
                {
                    "tensor_data": self.tensor_data,
                    "image_id": self.image_id,
                    "preprocessing_config": self.preprocessing_config,
                    "normalization_method": self.normalization_method,
                    "target_size": self.target_size,
                    "input_adapter": self.input_adapter,
                    "border_removed": self.border_removed,
                    "original_shape": self.original_shape,
                    "processing_time": self.processing_time,
                    "created_at": self.created_at.isoformat(),
                    "validation_errors": self.validation_errors,
                },
                file_path,
            )

            logger.info(f"Saved PreprocessedTensor to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving PreprocessedTensor to {file_path}: {e!s}")
            return False

    @classmethod
    def load_tensor(cls, file_path: Union[str, Path]) -> "PreprocessedTensor":
        """
        Load tensor data from file.

        Educational Note: This class method enables loading of previously
        saved preprocessed tensors for analysis and experimentation.

        Args:
            file_path: Path to the saved tensor file

        Returns:
            PreprocessedTensor: Loaded instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is corrupted or invalid
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Tensor file not found: {file_path}")

            # Load tensor data
            data = torch.load(file_path, map_location="cpu")

            # Parse creation timestamp
            created_at = (
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else None
            )

            # Create PreprocessedTensor instance
            preprocessed_tensor = cls(
                image_id=data["image_id"],
                tensor_data=data["tensor_data"],
                preprocessing_config=data["preprocessing_config"],
                normalization_method=data["normalization_method"],
                target_size=tuple(data["target_size"]),
                input_adapter=data["input_adapter"],
                border_removed=data["border_removed"],
                original_shape=(
                    tuple(data["original_shape"])
                    if data.get("original_shape")
                    else None
                ),
                processing_time=data.get("processing_time", 0.0),
                created_at=created_at,
            )

            # Restore validation errors if any
            if data.get("validation_errors"):
                preprocessed_tensor.validation_errors = data["validation_errors"]

            logger.info(f"Loaded PreprocessedTensor from {file_path}")
            return preprocessed_tensor

        except Exception as e:
            raise ValueError(
                f"Error loading PreprocessedTensor from {file_path}: {e!s}"
            )

    def __repr__(self) -> str:
        """String representation for debugging and logging."""
        return (
            f"PreprocessedTensor("
            f"image_id='{self.image_id}', "
            f"shape={list(self.tensor_data.shape)}, "
            f"normalization='{self.normalization_method}', "
            f"adapter='{self.input_adapter}')"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Preprocessed Tensor: {self.image_id}\n"
            f"Shape: {list(self.tensor_data.shape)}\n"
            f"Target Size: {self.target_size}\n"
            f"Normalization: {self.normalization_method}\n"
            f"Input Adapter: {self.input_adapter}\n"
            f"Border Removed: {self.border_removed}\n"
            f"Processing Time: {self.processing_time:.3f}s"
        )


def create_preprocessed_tensor_from_config(
    image_id: str, tensor_data: torch.Tensor, config: Dict[str, Any]
) -> PreprocessedTensor:
    """
    Create a PreprocessedTensor instance from configuration.

    Educational Note: This factory function demonstrates how to create
    PreprocessedTensor instances from preprocessing configurations,
    enabling standardized tensor creation.

    Args:
        image_id: Image identifier
        tensor_data: Preprocessed tensor data
        config: Preprocessing configuration dictionary

    Returns:
        PreprocessedTensor: Created instance

    Raises:
        ValueError: If configuration is invalid
    """
    # Extract configuration parameters
    normalization_method = config.get("normalization_method", "z_score_per_image")
    target_size = tuple(config.get("target_size", [512, 512]))
    input_adapter = config.get("input_adapter", "1to3_replication")
    border_removed = config.get("border_removed", False)
    processing_time = config.get("processing_time", 0.0)

    # Create PreprocessedTensor instance
    preprocessed_tensor = PreprocessedTensor(
        image_id=image_id,
        tensor_data=tensor_data,
        preprocessing_config=config,
        normalization_method=normalization_method,
        target_size=target_size,
        input_adapter=input_adapter,
        border_removed=border_removed,
        processing_time=processing_time,
    )

    return preprocessed_tensor
