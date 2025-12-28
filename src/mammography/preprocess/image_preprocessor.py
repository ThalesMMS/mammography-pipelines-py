"""
Image preprocessing module for mammography DICOM data.

This module provides comprehensive image preprocessing capabilities including
border removal, normalization, resizing, and input adapter application for
ResNet-50 feature extraction in the breast density exploration pipeline.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- Image preprocessing is crucial for consistent feature extraction
- Different normalization methods affect clustering quality
- Input adapters handle grayscale to RGB conversion for ResNet-50
- Border removal improves clustering by focusing on breast tissue

Author: Research Team
Version: 1.0.0
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pydicom
from scipy import ndimage
from skimage import morphology
import torch

from ..io.dicom import MammographyImage
from .preprocessed_tensor import (
    PreprocessedTensor,
    create_preprocessed_tensor_from_config,
)

# Configure logging for educational purposes
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessor for mammography DICOM data with comprehensive preprocessing pipeline.

    This class provides methods for preprocessing mammography images including
    border removal, normalization, resizing, and input adapter application.
    It ensures consistent preprocessing for ResNet-50 feature extraction.

    Educational Notes:
    - Preprocessing standardizes images for consistent feature extraction
    - Different normalization methods have different effects on clustering
    - Input adapters handle grayscale to RGB conversion for ResNet-50
    - Border removal focuses analysis on breast tissue

    Attributes:
        config: Preprocessing configuration dictionary
        supported_normalization_methods: List of supported normalization methods
        supported_input_adapters: List of supported input adapters
    """

    # Supported normalization methods
    SUPPORTED_NORMALIZATION_METHODS = [
        "z_score_per_image",
        "fixed_window",
        "min_max_scaling",
        "percentile_scaling",
    ]

    # Supported input adapters
    SUPPORTED_INPUT_ADAPTERS = ["1to3_replication", "conv1_adapted"]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize image preprocessor with configuration.

        Args:
            config: Preprocessing configuration dictionary

        Educational Note: Configuration validation ensures all required
        parameters are present and valid for preprocessing.

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = self._validate_config(config)

        # Set random seed for reproducibility
        if "seed" in self.config:
            torch.manual_seed(self.config["seed"])
            np.random.seed(self.config["seed"])

        logger.info(f"Initialized ImagePreprocessor with config: {self.config}")

    def preprocess_image(
        self, mammography_image: MammographyImage
    ) -> Optional[PreprocessedTensor]:
        """
        Preprocess a single mammography image.

        Educational Note: This method demonstrates the complete preprocessing
        pipeline from DICOM pixel data to ResNet-50-ready tensors.

        Args:
            mammography_image: MammographyImage instance to preprocess

        Returns:
            PreprocessedTensor: Preprocessed tensor if successful, None otherwise

        Raises:
            ValueError: If mammography image is invalid
        """
        start_time = datetime.now()

        try:
            # Read pixel data from DICOM
            pixel_array = self._read_pixel_data(mammography_image)
            if pixel_array is None:
                return None

            # Apply preprocessing pipeline
            processed_array = self._apply_preprocessing_pipeline(
                pixel_array, mammography_image
            )
            if processed_array is None:
                return None

            # Convert to tensor
            tensor_data = self._array_to_tensor(processed_array)

            # Apply input adapter
            tensor_data = self._apply_input_adapter(tensor_data)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Create PreprocessedTensor instance
            preprocessed_tensor = create_preprocessed_tensor_from_config(
                image_id=mammography_image.instance_id,
                tensor_data=tensor_data,
                config=self.config,
            )

            # Set processing time
            preprocessed_tensor.processing_time = processing_time

            logger.info(
                f"Successfully preprocessed image {mammography_image.instance_id} in {processing_time:.3f}s"
            )
            return preprocessed_tensor

        except Exception as e:
            logger.error(
                f"Error preprocessing image {mammography_image.instance_id}: {e!s}"
            )
            return None

    def preprocess_batch(
        self, mammography_images: List[MammographyImage]
    ) -> List[Optional[PreprocessedTensor]]:
        """
        Preprocess a batch of mammography images.

        Educational Note: Batch processing improves efficiency when
        preprocessing large numbers of images.

        Args:
            mammography_images: List of MammographyImage instances

        Returns:
            List[Optional[PreprocessedTensor]]: List of preprocessed tensors
        """
        preprocessed_tensors = []

        for mammography_image in mammography_images:
            preprocessed_tensor = self.preprocess_image(mammography_image)
            preprocessed_tensors.append(preprocessed_tensor)

        successful_count = sum(
            1 for tensor in preprocessed_tensors if tensor is not None
        )
        logger.info(
            f"Successfully preprocessed {successful_count}/{len(mammography_images)} images"
        )

        return preprocessed_tensors

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate preprocessing configuration.

        Educational Note: Configuration validation ensures all required
        parameters are present and within valid ranges.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Dict[str, Any]: Validated configuration

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required parameters
        required_params = ["target_size", "normalization_method", "input_adapter"]
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required configuration parameter: {param}")

        # Validate target size
        target_size = config["target_size"]
        if not isinstance(target_size, (list, tuple)) or len(target_size) != 2:
            raise ValueError("target_size must be a tuple/list of 2 integers")

        if not all(isinstance(size, int) and size > 0 for size in target_size):
            raise ValueError("target_size values must be positive integers")

        # Validate normalization method
        normalization_method = config["normalization_method"]
        if normalization_method not in self.SUPPORTED_NORMALIZATION_METHODS:
            raise ValueError(
                f"Unsupported normalization method: {normalization_method}"
            )

        # Validate input adapter
        input_adapter = config["input_adapter"]
        if input_adapter not in self.SUPPORTED_INPUT_ADAPTERS:
            raise ValueError(f"Unsupported input adapter: {input_adapter}")

        # Set default values for optional parameters
        config.setdefault("border_removal", True)
        config.setdefault("padding_strategy", "reflect")
        config.setdefault("keep_aspect_ratio", True)
        config.setdefault("border_threshold", 0.1)
        config.setdefault("min_breast_area", 0.05)
        config.setdefault("seed", 42)

        return config

    def _read_pixel_data(
        self, mammography_image: MammographyImage
    ) -> Optional[np.ndarray]:
        """
        Read pixel data from DICOM file.

        Educational Note: This method handles DICOM pixel data extraction
        with proper error handling and data type conversion.

        Args:
            mammography_image: MammographyImage instance

        Returns:
            np.ndarray: Pixel array if successful, None otherwise
        """
        try:
            # Read DICOM file
            dataset = pydicom.dcmread(mammography_image.file_path)

            # Extract pixel array
            pixel_array = dataset.pixel_array

            if pixel_array is None or pixel_array.size == 0:
                logger.error(f"Empty pixel data in {mammography_image.file_path}")
                return None

            # Convert to float32 for processing
            pixel_array = pixel_array.astype(np.float32)

            # Handle different photometric interpretations
            if hasattr(dataset, "PhotometricInterpretation"):
                if dataset.PhotometricInterpretation == "MONOCHROME1":
                    # Invert if needed (white background)
                    pixel_array = np.max(pixel_array) - pixel_array

            logger.debug(
                f"Read pixel data with shape {pixel_array.shape} from {mammography_image.file_path}"
            )
            return pixel_array

        except Exception as e:
            logger.error(
                f"Error reading pixel data from {mammography_image.file_path}: {e!s}"
            )
            return None

    def _apply_preprocessing_pipeline(
        self, pixel_array: np.ndarray, _mammography_image: MammographyImage
    ) -> Optional[np.ndarray]:
        """
        Apply complete preprocessing pipeline to pixel array.

        Educational Note: This method demonstrates the step-by-step
        preprocessing pipeline including border removal, normalization,
        and resizing.

        Args:
            pixel_array: Input pixel array
            mammography_image: MammographyImage instance for metadata

        Returns:
            np.ndarray: Preprocessed array if successful, None otherwise
        """
        try:
            processed_array = pixel_array.copy()

            # Step 1: Border removal (if enabled)
            if self.config.get("border_removal", True):
                processed_array = self._remove_borders(processed_array)
                if processed_array is None:
                    return None

            # Step 2: Normalization
            processed_array = self._normalize_image(processed_array)
            if processed_array is None:
                return None

            # Step 3: Resizing
            processed_array = self._resize_image(processed_array)
            if processed_array is None:
                return None

            return processed_array

        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e!s}")
            return None

    def _remove_borders(self, pixel_array: np.ndarray) -> Optional[np.ndarray]:
        """
        Remove image borders and markers to focus on breast tissue.

        Educational Note: Border removal improves clustering quality by
        eliminating non-breast tissue and acquisition artifacts.

        Args:
            pixel_array: Input pixel array

        Returns:
            np.ndarray: Array with borders removed, None if failed
        """
        try:
            # Convert to uint8 for morphological operations
            array_uint8 = self._normalize_to_uint8(pixel_array)

            # Create binary mask for breast tissue
            threshold = np.percentile(array_uint8, 5)  # Use 5th percentile as threshold
            binary_mask = array_uint8 > threshold

            # Remove small objects (noise)
            binary_mask = morphology.remove_small_objects(binary_mask, min_size=1000)

            # Fill holes
            binary_mask = ndimage.binary_fill_holes(binary_mask)

            # Find largest connected component (breast tissue)
            labeled_array, num_features = ndimage.label(binary_mask)
            if num_features == 0:
                logger.warning("No breast tissue detected, skipping border removal")
                return pixel_array

            # Get largest component
            component_sizes = [
                np.sum(labeled_array == i) for i in range(1, num_features + 1)
            ]
            largest_component = np.argmax(component_sizes) + 1
            breast_mask = labeled_array == largest_component

            # Check minimum breast area
            min_area = self.config.get("min_breast_area", 0.05)
            if np.sum(breast_mask) < min_area * pixel_array.size:
                logger.warning("Breast area too small, skipping border removal")
                return pixel_array

            # Apply mask to original array
            processed_array = pixel_array.copy()
            processed_array[~breast_mask] = 0

            logger.debug("Successfully removed borders")
            return processed_array

        except Exception as e:
            logger.error(f"Error in border removal: {e!s}")
            return None

    def _normalize_image(self, pixel_array: np.ndarray) -> Optional[np.ndarray]:
        """
        Normalize image using specified method.

        Educational Note: Different normalization methods have different
        effects on feature extraction and clustering quality.

        Args:
            pixel_array: Input pixel array

        Returns:
            np.ndarray: Normalized array if successful, None otherwise
        """
        try:
            normalization_method = self.config["normalization_method"]

            if normalization_method == "z_score_per_image":
                # Z-score normalization per image
                mean = np.mean(pixel_array)
                std = np.std(pixel_array)
                if std > 0:
                    normalized_array = (pixel_array - mean) / std
                else:
                    normalized_array = pixel_array - mean

            elif normalization_method == "fixed_window":
                # Fixed window normalization (common in medical imaging)
                window_center = np.percentile(pixel_array, 50)  # Median
                window_width = np.percentile(pixel_array, 95) - np.percentile(
                    pixel_array, 5
                )

                min_val = window_center - window_width / 2
                max_val = window_center + window_width / 2

                normalized_array = np.clip(
                    (pixel_array - min_val) / (max_val - min_val), 0, 1
                )

            elif normalization_method == "min_max_scaling":
                # Min-max scaling to [0, 1]
                min_val = np.min(pixel_array)
                max_val = np.max(pixel_array)
                if max_val > min_val:
                    normalized_array = (pixel_array - min_val) / (max_val - min_val)
                else:
                    normalized_array = np.zeros_like(pixel_array)

            elif normalization_method == "percentile_scaling":
                # Percentile-based scaling
                p2, p98 = np.percentile(pixel_array, [2, 98])
                normalized_array = np.clip((pixel_array - p2) / (p98 - p2), 0, 1)

            else:
                raise ValueError(
                    f"Unknown normalization method: {normalization_method}"
                )

            # Ensure no NaN or infinite values
            if np.any(np.isnan(normalized_array)) or np.any(np.isinf(normalized_array)):
                logger.warning(
                    "Normalization produced NaN or infinite values, using fallback"
                )
                normalized_array = np.clip(normalized_array, 0, 1)
                normalized_array = np.nan_to_num(
                    normalized_array, nan=0.0, posinf=1.0, neginf=0.0
                )

            logger.debug(f"Applied {normalization_method} normalization")
            return normalized_array

        except Exception as e:
            logger.error(f"Error in normalization: {e!s}")
            return None

    def _resize_image(self, pixel_array: np.ndarray) -> Optional[np.ndarray]:
        """
        Resize image to target dimensions.

        Educational Note: Resizing ensures consistent input dimensions
        for ResNet-50 while preserving aspect ratio when possible.

        Args:
            pixel_array: Input pixel array

        Returns:
            np.ndarray: Resized array if successful, None otherwise
        """
        try:
            target_size = self.config["target_size"]
            keep_aspect_ratio = self.config.get("keep_aspect_ratio", True)

            if keep_aspect_ratio:
                # Resize while preserving aspect ratio
                resized_array = self._resize_with_aspect_ratio(pixel_array, target_size)
            else:
                # Direct resize to target dimensions
                resized_array = cv2.resize(
                    pixel_array,
                    (target_size[1], target_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            logger.debug(
                f"Resized image from {pixel_array.shape} to {resized_array.shape}"
            )
            return resized_array

        except Exception as e:
            logger.error(f"Error in resizing: {e!s}")
            return None

    def _resize_with_aspect_ratio(
        self, pixel_array: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize image while preserving aspect ratio.

        Educational Note: Aspect ratio preservation maintains the
        original proportions of the mammography image.

        Args:
            pixel_array: Input pixel array
            target_size: Target dimensions (height, width)

        Returns:
            np.ndarray: Resized array with preserved aspect ratio
        """
        h, w = pixel_array.shape
        target_h, target_w = target_size

        # Calculate scaling factor
        scale_h = target_h / h
        scale_w = target_w / w
        scale = min(scale_h, scale_w)  # Use smaller scale to fit within target

        # Calculate new dimensions
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize image
        resized_array = cv2.resize(
            pixel_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        # Pad to target size
        padding_strategy = self.config.get("padding_strategy", "reflect")

        if padding_strategy == "reflect":
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
            resized_array = np.pad(
                resized_array,
                ((pad_h, target_h - new_h - pad_h), (pad_w, target_w - new_w - pad_w)),
                mode="reflect",
            )
        elif padding_strategy == "constant":
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
            resized_array = np.pad(
                resized_array,
                ((pad_h, target_h - new_h - pad_h), (pad_w, target_w - new_w - pad_w)),
                mode="constant",
                constant_values=0,
            )
        else:  # edge
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
            resized_array = np.pad(
                resized_array,
                ((pad_h, target_h - new_h - pad_h), (pad_w, target_w - new_w - pad_w)),
                mode="edge",
            )

        return resized_array

    def _array_to_tensor(self, pixel_array: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor.

        Educational Note: This conversion prepares the data for
        ResNet-50 feature extraction.

        Args:
            pixel_array: Input numpy array

        Returns:
            torch.Tensor: PyTorch tensor
        """
        # Convert to tensor and add channel dimension
        tensor = torch.from_numpy(pixel_array).unsqueeze(0)  # Add channel dimension
        return tensor.float()

    def _apply_input_adapter(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply input adapter to convert grayscale to RGB for ResNet-50.

        Educational Note: Input adapters handle the conversion from
        single-channel grayscale to 3-channel RGB format required by ResNet-50.

        Args:
            tensor: Input tensor (1, H, W)

        Returns:
            torch.Tensor: Output tensor (3, H, W)
        """
        input_adapter = self.config["input_adapter"]

        if input_adapter == "1to3_replication":
            # Replicate single channel to 3 channels
            return tensor.repeat(3, 1, 1)

        elif input_adapter == "conv1_adapted":
            # This would require modifying ResNet-50's first layer
            # For now, we'll use replication as a fallback
            logger.warning("conv1_adapted not implemented, using 1to3_replication")
            return tensor.repeat(3, 1, 1)

        else:
            raise ValueError(f"Unknown input adapter: {input_adapter}")

    def _normalize_to_uint8(self, pixel_array: np.ndarray) -> np.ndarray:
        """
        Normalize array to uint8 range for morphological operations.

        Educational Note: This normalization is used for binary operations
        like border detection and morphological processing.

        Args:
            pixel_array: Input float array

        Returns:
            np.ndarray: uint8 array
        """
        # Normalize to [0, 255] range
        normalized = (pixel_array - np.min(pixel_array)) / (
            np.max(pixel_array) - np.min(pixel_array)
        )
        return (normalized * 255).astype(np.uint8)


def create_image_preprocessor(config: Dict[str, Any]) -> ImagePreprocessor:
    """
    Factory function to create an ImagePreprocessor instance.

    Educational Note: This factory function provides a convenient way
    to create ImagePreprocessor instances with validated configurations.

    Args:
        config: Preprocessing configuration dictionary

    Returns:
        ImagePreprocessor: Configured ImagePreprocessor instance
    """
    return ImagePreprocessor(config)


def preprocess_single_image(
    mammography_image: MammographyImage, config: Dict[str, Any]
) -> Optional[PreprocessedTensor]:
    """
    Convenience function to preprocess a single mammography image.

    Educational Note: This function provides a simple interface for
    preprocessing individual images without creating an ImagePreprocessor instance.

    Args:
        mammography_image: MammographyImage instance to preprocess
        config: Preprocessing configuration dictionary

    Returns:
        PreprocessedTensor: Preprocessed tensor if successful, None otherwise
    """
    preprocessor = create_image_preprocessor(config)
    return preprocessor.preprocess_image(mammography_image)


def preprocess_batch_images(
    mammography_images: List[MammographyImage], config: Dict[str, Any]
) -> List[Optional[PreprocessedTensor]]:
    """
    Convenience function to preprocess a batch of mammography images.

    Educational Note: This function provides a simple interface for
    batch preprocessing without creating an ImagePreprocessor instance.

    Args:
        mammography_images: List of MammographyImage instances
        config: Preprocessing configuration dictionary

    Returns:
        List[Optional[PreprocessedTensor]]: List of preprocessed tensors
    """
    preprocessor = create_image_preprocessor(config)
    return preprocessor.preprocess_batch(mammography_images)
