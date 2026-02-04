"""
Comprehensive unit tests for preprocessing modules.

These tests validate preprocessing functionality including ImagePreprocessor
and PreprocessedTensor classes with thorough coverage of edge cases.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("pydicom")

from mammography.preprocess.image_preprocessor import (
    ImagePreprocessor,
    create_image_preprocessor,
    preprocess_single_image,
    preprocess_batch_images,
)
from mammography.preprocess.preprocessed_tensor import (
    PreprocessedTensor,
    create_preprocessed_tensor_from_config,
)


class TestImagePreprocessorConfig:
    """Tests for ImagePreprocessor configuration validation."""

    def test_valid_config(self):
        """Test initialization with valid configuration."""
        config = {
            "target_size": (512, 512),
            "normalization_method": "z_score_per_image",
            "input_adapter": "1to3_replication",
        }
        preprocessor = ImagePreprocessor(config)

        assert preprocessor.config["target_size"] == (512, 512)
        assert preprocessor.config["normalization_method"] == "z_score_per_image"
        assert preprocessor.config["input_adapter"] == "1to3_replication"

        # Check default values
        assert preprocessor.config["border_removal"] is True
        assert preprocessor.config["padding_strategy"] == "reflect"
        assert preprocessor.config["keep_aspect_ratio"] is True

    def test_missing_required_parameter(self):
        """Test validation fails with missing required parameters."""
        # Missing target_size
        config = {
            "normalization_method": "z_score_per_image",
            "input_adapter": "1to3_replication",
        }
        with pytest.raises(ValueError, match="Missing required configuration parameter: target_size"):
            ImagePreprocessor(config)

        # Missing normalization_method
        config = {
            "target_size": (512, 512),
            "input_adapter": "1to3_replication",
        }
        with pytest.raises(ValueError, match="Missing required configuration parameter: normalization_method"):
            ImagePreprocessor(config)

        # Missing input_adapter
        config = {
            "target_size": (512, 512),
            "normalization_method": "z_score_per_image",
        }
        with pytest.raises(ValueError, match="Missing required configuration parameter: input_adapter"):
            ImagePreprocessor(config)

    def test_invalid_target_size(self):
        """Test validation fails with invalid target size."""
        # Not a tuple/list
        config = {
            "target_size": 512,
            "normalization_method": "z_score_per_image",
            "input_adapter": "1to3_replication",
        }
        with pytest.raises(ValueError, match="target_size must be a tuple/list of 2 integers"):
            ImagePreprocessor(config)

        # Wrong length
        config = {
            "target_size": (512, 512, 3),
            "normalization_method": "z_score_per_image",
            "input_adapter": "1to3_replication",
        }
        with pytest.raises(ValueError, match="target_size must be a tuple/list of 2 integers"):
            ImagePreprocessor(config)

        # Negative values
        config = {
            "target_size": (-512, 512),
            "normalization_method": "z_score_per_image",
            "input_adapter": "1to3_replication",
        }
        with pytest.raises(ValueError, match="target_size values must be positive integers"):
            ImagePreprocessor(config)

    def test_invalid_normalization_method(self):
        """Test validation fails with invalid normalization method."""
        config = {
            "target_size": (512, 512),
            "normalization_method": "invalid_method",
            "input_adapter": "1to3_replication",
        }
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            ImagePreprocessor(config)

    def test_invalid_input_adapter(self):
        """Test validation fails with invalid input adapter."""
        config = {
            "target_size": (512, 512),
            "normalization_method": "z_score_per_image",
            "input_adapter": "invalid_adapter",
        }
        with pytest.raises(ValueError, match="Unsupported input adapter"):
            ImagePreprocessor(config)

    def test_supported_normalization_methods(self):
        """Test all supported normalization methods are valid."""
        for method in ImagePreprocessor.SUPPORTED_NORMALIZATION_METHODS:
            config = {
                "target_size": (512, 512),
                "normalization_method": method,
                "input_adapter": "1to3_replication",
            }
            preprocessor = ImagePreprocessor(config)
            assert preprocessor.config["normalization_method"] == method

    def test_supported_input_adapters(self):
        """Test all supported input adapters are valid."""
        for adapter in ImagePreprocessor.SUPPORTED_INPUT_ADAPTERS:
            config = {
                "target_size": (512, 512),
                "normalization_method": "z_score_per_image",
                "input_adapter": adapter,
            }
            preprocessor = ImagePreprocessor(config)
            assert preprocessor.config["input_adapter"] == adapter


class TestImagePreprocessorNormalization:
    """Tests for ImagePreprocessor normalization methods."""

    @pytest.fixture
    def preprocessor_config(self):
        """Create base preprocessor configuration."""
        return {
            "target_size": (512, 512),
            "normalization_method": "z_score_per_image",
            "input_adapter": "1to3_replication",
            "border_removal": False,
        }

    @pytest.fixture
    def sample_image(self):
        """Create a sample image array for testing."""
        np.random.seed(42)
        return np.random.uniform(0, 4095, (1024, 768)).astype(np.float32)

    def test_z_score_normalization(self, preprocessor_config, sample_image):
        """Test z-score normalization method."""
        preprocessor_config["normalization_method"] = "z_score_per_image"
        preprocessor = ImagePreprocessor(preprocessor_config)

        normalized = preprocessor._normalize_image(sample_image)

        assert normalized is not None
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))

        # Z-score should have approximately mean=0, std=1
        assert abs(np.mean(normalized)) < 1e-6
        assert abs(np.std(normalized) - 1.0) < 1e-6

    def test_fixed_window_normalization(self, preprocessor_config, sample_image):
        """Test fixed window normalization method."""
        preprocessor_config["normalization_method"] = "fixed_window"
        preprocessor = ImagePreprocessor(preprocessor_config)

        normalized = preprocessor._normalize_image(sample_image)

        assert normalized is not None
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 1

    def test_min_max_scaling(self, preprocessor_config, sample_image):
        """Test min-max scaling normalization method."""
        preprocessor_config["normalization_method"] = "min_max_scaling"
        preprocessor = ImagePreprocessor(preprocessor_config)

        normalized = preprocessor._normalize_image(sample_image)

        assert normalized is not None
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 1

        # Check that min and max values are properly scaled
        assert abs(np.min(normalized) - 0.0) < 1e-6
        assert abs(np.max(normalized) - 1.0) < 1e-6

    def test_percentile_scaling(self, preprocessor_config, sample_image):
        """Test percentile-based scaling normalization method."""
        preprocessor_config["normalization_method"] = "percentile_scaling"
        preprocessor = ImagePreprocessor(preprocessor_config)

        normalized = preprocessor._normalize_image(sample_image)

        assert normalized is not None
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 1

    def test_normalization_with_constant_image(self, preprocessor_config):
        """Test normalization handles constant (zero variance) images."""
        constant_image = np.ones((512, 512), dtype=np.float32) * 100.0

        preprocessor_config["normalization_method"] = "z_score_per_image"
        preprocessor = ImagePreprocessor(preprocessor_config)

        normalized = preprocessor._normalize_image(constant_image)

        assert normalized is not None
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))

    def test_normalization_with_nan_values(self, preprocessor_config):
        """Test normalization handles NaN values."""
        image_with_nan = np.random.uniform(0, 4095, (512, 512)).astype(np.float32)
        image_with_nan[100:110, 100:110] = np.nan

        preprocessor_config["normalization_method"] = "z_score_per_image"
        preprocessor = ImagePreprocessor(preprocessor_config)

        normalized = preprocessor._normalize_image(image_with_nan)

        # Should handle NaN values through fallback
        assert normalized is not None


class TestImagePreprocessorResizing:
    """Tests for ImagePreprocessor resizing functionality."""

    @pytest.fixture
    def preprocessor_config(self):
        """Create base preprocessor configuration."""
        return {
            "target_size": (512, 512),
            "normalization_method": "min_max_scaling",
            "input_adapter": "1to3_replication",
            "border_removal": False,
        }

    @pytest.fixture
    def sample_image(self):
        """Create a sample image array for testing."""
        return np.random.uniform(0, 1, (1024, 768)).astype(np.float32)

    def test_resize_with_aspect_ratio_preserved(self, preprocessor_config, sample_image):
        """Test resizing while preserving aspect ratio."""
        preprocessor_config["keep_aspect_ratio"] = True
        preprocessor = ImagePreprocessor(preprocessor_config)

        resized = preprocessor._resize_image(sample_image)

        assert resized is not None
        assert resized.shape == (512, 512)
        assert not np.any(np.isnan(resized))

    def test_resize_without_aspect_ratio(self, preprocessor_config, sample_image):
        """Test direct resizing without aspect ratio preservation."""
        preprocessor_config["keep_aspect_ratio"] = False
        preprocessor = ImagePreprocessor(preprocessor_config)

        resized = preprocessor._resize_image(sample_image)

        assert resized is not None
        assert resized.shape == (512, 512)
        assert not np.any(np.isnan(resized))

    def test_resize_with_different_padding_strategies(self, preprocessor_config, sample_image):
        """Test different padding strategies for aspect ratio preservation."""
        preprocessor_config["keep_aspect_ratio"] = True

        for strategy in ["reflect", "constant", "edge"]:
            preprocessor_config["padding_strategy"] = strategy
            preprocessor = ImagePreprocessor(preprocessor_config)

            resized = preprocessor._resize_image(sample_image)

            assert resized is not None
            assert resized.shape == (512, 512)
            assert not np.any(np.isnan(resized))

    def test_resize_different_target_sizes(self, preprocessor_config, sample_image):
        """Test resizing to different target dimensions."""
        for target_size in [(256, 256), (512, 512), (1024, 1024)]:
            preprocessor_config["target_size"] = target_size
            preprocessor = ImagePreprocessor(preprocessor_config)

            resized = preprocessor._resize_image(sample_image)

            assert resized is not None
            assert resized.shape == target_size


class TestImagePreprocessorInputAdapters:
    """Tests for ImagePreprocessor input adapter functionality."""

    @pytest.fixture
    def preprocessor_config(self):
        """Create base preprocessor configuration."""
        return {
            "target_size": (512, 512),
            "normalization_method": "min_max_scaling",
            "input_adapter": "1to3_replication",
        }

    def test_1to3_replication_adapter(self, preprocessor_config):
        """Test 1to3 replication input adapter."""
        preprocessor = ImagePreprocessor(preprocessor_config)

        # Create single-channel tensor
        single_channel = torch.randn(1, 512, 512)

        result = preprocessor._apply_input_adapter(single_channel)

        assert result.shape == (3, 512, 512)
        # All channels should be identical
        assert torch.allclose(result[0], result[1])
        assert torch.allclose(result[1], result[2])

    def test_conv1_adapted_fallback(self, preprocessor_config):
        """Test conv1_adapted falls back to replication."""
        preprocessor_config["input_adapter"] = "conv1_adapted"
        preprocessor = ImagePreprocessor(preprocessor_config)

        single_channel = torch.randn(1, 512, 512)

        result = preprocessor._apply_input_adapter(single_channel)

        # Should fallback to replication
        assert result.shape == (3, 512, 512)


class TestImagePreprocessorBorderRemoval:
    """Tests for ImagePreprocessor border removal functionality."""

    @pytest.fixture
    def preprocessor_config(self):
        """Create base preprocessor configuration."""
        return {
            "target_size": (512, 512),
            "normalization_method": "min_max_scaling",
            "input_adapter": "1to3_replication",
            "border_removal": True,
        }

    @pytest.fixture
    def sample_image_with_borders(self):
        """Create a sample image with border artifacts."""
        image = np.zeros((1024, 768), dtype=np.float32)

        # Add breast tissue region
        image[200:800, 150:600] = np.random.uniform(500, 3000, (600, 450))

        # Add border artifacts
        image[0:50, :] = np.random.uniform(0, 100, (50, 768))

        return image

    def test_border_removal_enabled(self, preprocessor_config, sample_image_with_borders):
        """Test border removal when enabled."""
        preprocessor = ImagePreprocessor(preprocessor_config)

        result = preprocessor._remove_borders(sample_image_with_borders)

        assert result is not None
        assert result.shape == sample_image_with_borders.shape

    def test_border_removal_disabled(self, preprocessor_config, sample_image_with_borders):
        """Test preprocessing with border removal disabled."""
        preprocessor_config["border_removal"] = False
        preprocessor = ImagePreprocessor(preprocessor_config)

        # Border removal should be skipped
        result = preprocessor._apply_preprocessing_pipeline(
            sample_image_with_borders, Mock()
        )

        assert result is not None

    def test_border_removal_with_no_tissue(self, preprocessor_config):
        """Test border removal with empty image (no tissue detected)."""
        empty_image = np.zeros((512, 512), dtype=np.float32)

        preprocessor = ImagePreprocessor(preprocessor_config)

        result = preprocessor._remove_borders(empty_image)

        # Should return original image when no tissue detected
        assert result is not None

    def test_border_removal_min_area_threshold(self, preprocessor_config):
        """Test border removal with minimum area threshold."""
        preprocessor_config["min_breast_area"] = 0.1

        # Create image with small breast region
        small_region_image = np.zeros((512, 512), dtype=np.float32)
        small_region_image[200:250, 200:250] = 1000

        preprocessor = ImagePreprocessor(preprocessor_config)
        result = preprocessor._remove_borders(small_region_image)

        # Should skip border removal due to small area
        assert result is not None


class TestImagePreprocessorHelpers:
    """Tests for ImagePreprocessor helper methods."""

    @pytest.fixture
    def preprocessor_config(self):
        """Create base preprocessor configuration."""
        return {
            "target_size": (512, 512),
            "normalization_method": "min_max_scaling",
            "input_adapter": "1to3_replication",
        }

    def test_normalize_to_uint8(self, preprocessor_config):
        """Test conversion to uint8 for morphological operations."""
        preprocessor = ImagePreprocessor(preprocessor_config)

        # Create float array
        float_array = np.random.uniform(0, 4095, (512, 512)).astype(np.float32)

        uint8_array = preprocessor._normalize_to_uint8(float_array)

        assert uint8_array.dtype == np.uint8
        assert np.min(uint8_array) >= 0
        assert np.max(uint8_array) <= 255

    def test_normalize_to_uint8_constant_image(self, preprocessor_config):
        """Test uint8 conversion with constant image."""
        preprocessor = ImagePreprocessor(preprocessor_config)

        constant_array = np.ones((512, 512), dtype=np.float32) * 100.0

        uint8_array = preprocessor._normalize_to_uint8(constant_array)

        assert uint8_array.dtype == np.uint8
        # Should be all zeros for constant image
        assert np.all(uint8_array == 0)

    def test_array_to_tensor(self, preprocessor_config):
        """Test conversion from numpy array to PyTorch tensor."""
        preprocessor = ImagePreprocessor(preprocessor_config)

        array = np.random.uniform(0, 1, (512, 512)).astype(np.float32)

        tensor = preprocessor._array_to_tensor(array)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 512, 512)
        assert tensor.dtype == torch.float32


class TestImagePreprocessorFactoryFunctions:
    """Tests for factory functions and convenience methods."""

    @pytest.fixture
    def valid_config(self):
        """Create valid configuration."""
        return {
            "target_size": (512, 512),
            "normalization_method": "z_score_per_image",
            "input_adapter": "1to3_replication",
        }

    def test_create_image_preprocessor(self, valid_config):
        """Test factory function for creating preprocessor."""
        preprocessor = create_image_preprocessor(valid_config)

        assert isinstance(preprocessor, ImagePreprocessor)
        assert preprocessor.config["target_size"] == (512, 512)


class TestPreprocessedTensorValidation:
    """Tests for PreprocessedTensor validation functionality."""

    @pytest.fixture
    def valid_tensor_data(self):
        """Create valid tensor data."""
        return torch.randn(3, 512, 512, dtype=torch.float32)

    @pytest.fixture
    def valid_config(self):
        """Create valid preprocessing configuration."""
        return {
            "target_size": (512, 512),
            "normalization_method": "z_score_per_image",
            "input_adapter": "1to3_replication",
        }

    def test_valid_preprocessed_tensor_creation(self, valid_tensor_data, valid_config):
        """Test creating PreprocessedTensor with valid parameters."""
        tensor = PreprocessedTensor(
            image_id="test_image_001",
            tensor_data=valid_tensor_data,
            preprocessing_config=valid_config,
            normalization_method="z_score_per_image",
            target_size=(512, 512),
            input_adapter="1to3_replication",
        )

        assert tensor.image_id == "test_image_001"
        assert tensor.tensor_data.shape == (3, 512, 512)
        assert tensor.normalization_method == "z_score_per_image"
        assert tensor.target_size == (512, 512)

    def test_invalid_image_id_empty(self, valid_tensor_data, valid_config):
        """Test validation fails with empty image ID."""
        with pytest.raises(ValueError, match="image_id cannot be empty"):
            PreprocessedTensor(
                image_id="",
                tensor_data=valid_tensor_data,
                preprocessing_config=valid_config,
                normalization_method="z_score_per_image",
                target_size=(512, 512),
                input_adapter="1to3_replication",
            )

    def test_invalid_image_id_type(self, valid_tensor_data, valid_config):
        """Test validation fails with non-string image ID."""
        with pytest.raises(TypeError, match="image_id must be a string"):
            PreprocessedTensor(
                image_id=123,
                tensor_data=valid_tensor_data,
                preprocessing_config=valid_config,
                normalization_method="z_score_per_image",
                target_size=(512, 512),
                input_adapter="1to3_replication",
            )

    def test_invalid_tensor_not_torch_tensor(self, valid_config):
        """Test validation fails with non-tensor data."""
        with pytest.raises(TypeError, match="tensor_data must be a torch.Tensor"):
            PreprocessedTensor(
                image_id="test_image",
                tensor_data=np.random.randn(3, 512, 512),
                preprocessing_config=valid_config,
                normalization_method="z_score_per_image",
                target_size=(512, 512),
                input_adapter="1to3_replication",
            )

    def test_invalid_tensor_wrong_dimensions(self, valid_config):
        """Test validation fails with wrong tensor dimensions."""
        # 2D tensor
        with pytest.raises(ValueError, match="tensor_data must be 3D"):
            PreprocessedTensor(
                image_id="test_image",
                tensor_data=torch.randn(512, 512),
                preprocessing_config=valid_config,
                normalization_method="z_score_per_image",
                target_size=(512, 512),
                input_adapter="1to3_replication",
            )

        # 4D tensor
        with pytest.raises(ValueError, match="tensor_data must be 3D"):
            PreprocessedTensor(
                image_id="test_image",
                tensor_data=torch.randn(1, 3, 512, 512),
                preprocessing_config=valid_config,
                normalization_method="z_score_per_image",
                target_size=(512, 512),
                input_adapter="1to3_replication",
            )

    def test_invalid_tensor_wrong_channels(self, valid_config):
        """Test validation fails with wrong number of channels."""
        with pytest.raises(ValueError, match="tensor_data must have 3 channels"):
            PreprocessedTensor(
                image_id="test_image",
                tensor_data=torch.randn(1, 512, 512),
                preprocessing_config=valid_config,
                normalization_method="z_score_per_image",
                target_size=(512, 512),
                input_adapter="1to3_replication",
            )

    def test_invalid_tensor_with_nan(self, valid_config):
        """Test validation fails with NaN values."""
        tensor_with_nan = torch.randn(3, 512, 512)
        tensor_with_nan[0, 100, 100] = float('nan')

        with pytest.raises(ValueError, match="tensor_data contains NaN values"):
            PreprocessedTensor(
                image_id="test_image",
                tensor_data=tensor_with_nan,
                preprocessing_config=valid_config,
                normalization_method="z_score_per_image",
                target_size=(512, 512),
                input_adapter="1to3_replication",
            )

    def test_invalid_tensor_with_inf(self, valid_config):
        """Test validation fails with infinite values."""
        tensor_with_inf = torch.randn(3, 512, 512)
        tensor_with_inf[0, 100, 100] = float('inf')

        with pytest.raises(ValueError, match="tensor_data contains infinite values"):
            PreprocessedTensor(
                image_id="test_image",
                tensor_data=tensor_with_inf,
                preprocessing_config=valid_config,
                normalization_method="z_score_per_image",
                target_size=(512, 512),
                input_adapter="1to3_replication",
            )

    def test_invalid_normalization_method(self, valid_tensor_data, valid_config):
        """Test validation fails with invalid normalization method."""
        with pytest.raises(ValueError, match="normalization_method must be one of"):
            PreprocessedTensor(
                image_id="test_image",
                tensor_data=valid_tensor_data,
                preprocessing_config=valid_config,
                normalization_method="invalid_method",
                target_size=(512, 512),
                input_adapter="1to3_replication",
            )

    def test_invalid_target_size_type(self, valid_tensor_data, valid_config):
        """Test validation fails with invalid target size type."""
        with pytest.raises(TypeError, match="target_size must be a tuple"):
            PreprocessedTensor(
                image_id="test_image",
                tensor_data=valid_tensor_data,
                preprocessing_config=valid_config,
                normalization_method="z_score_per_image",
                target_size=[512, 512],
                input_adapter="1to3_replication",
            )

    def test_invalid_target_size_length(self, valid_tensor_data, valid_config):
        """Test validation fails with wrong target size length."""
        with pytest.raises(ValueError, match="target_size must have exactly 2 elements"):
            PreprocessedTensor(
                image_id="test_image",
                tensor_data=valid_tensor_data,
                preprocessing_config=valid_config,
                normalization_method="z_score_per_image",
                target_size=(512,),
                input_adapter="1to3_replication",
            )

    def test_invalid_target_size_negative(self, valid_tensor_data, valid_config):
        """Test validation fails with negative target size."""
        with pytest.raises(ValueError, match="target_size.*must be positive"):
            PreprocessedTensor(
                image_id="test_image",
                tensor_data=valid_tensor_data,
                preprocessing_config=valid_config,
                normalization_method="z_score_per_image",
                target_size=(-512, 512),
                input_adapter="1to3_replication",
            )

    def test_invalid_input_adapter(self, valid_tensor_data, valid_config):
        """Test validation fails with invalid input adapter."""
        with pytest.raises(ValueError, match="input_adapter must be one of"):
            PreprocessedTensor(
                image_id="test_image",
                tensor_data=valid_tensor_data,
                preprocessing_config=valid_config,
                normalization_method="z_score_per_image",
                target_size=(512, 512),
                input_adapter="invalid_adapter",
            )

    def test_invalid_border_removed_type(self, valid_tensor_data, valid_config):
        """Test validation fails with non-boolean border_removed."""
        with pytest.raises(TypeError, match="border_removed must be a boolean"):
            PreprocessedTensor(
                image_id="test_image",
                tensor_data=valid_tensor_data,
                preprocessing_config=valid_config,
                normalization_method="z_score_per_image",
                target_size=(512, 512),
                input_adapter="1to3_replication",
                border_removed="true",
            )

    def test_missing_config_keys(self, valid_tensor_data):
        """Test validation fails with missing config keys."""
        incomplete_config = {
            "target_size": (512, 512),
        }

        with pytest.raises(ValueError, match="Missing required configuration key"):
            PreprocessedTensor(
                image_id="test_image",
                tensor_data=valid_tensor_data,
                preprocessing_config=incomplete_config,
                normalization_method="z_score_per_image",
                target_size=(512, 512),
                input_adapter="1to3_replication",
            )


class TestPreprocessedTensorStatistics:
    """Tests for PreprocessedTensor statistics and summaries."""

    @pytest.fixture
    def sample_tensor(self):
        """Create sample preprocessed tensor."""
        torch.manual_seed(42)
        tensor_data = torch.randn(3, 512, 512)
        config = {
            "target_size": (512, 512),
            "normalization_method": "z_score_per_image",
            "input_adapter": "1to3_replication",
        }

        return PreprocessedTensor(
            image_id="test_image_001",
            tensor_data=tensor_data,
            preprocessing_config=config,
            normalization_method="z_score_per_image",
            target_size=(512, 512),
            input_adapter="1to3_replication",
        )

    def test_get_tensor_stats(self, sample_tensor):
        """Test getting tensor statistics."""
        stats = sample_tensor.get_tensor_stats()

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "shape" in stats
        assert "dtype" in stats

        assert isinstance(stats["mean"], float)
        assert isinstance(stats["std"], float)
        assert stats["shape"] == [3, 512, 512]

    def test_get_preprocessing_summary(self, sample_tensor):
        """Test getting preprocessing summary."""
        summary = sample_tensor.get_preprocessing_summary()

        assert summary["image_id"] == "test_image_001"
        assert summary["target_size"] == (512, 512)
        assert summary["normalization_method"] == "z_score_per_image"
        assert summary["input_adapter"] == "1to3_replication"
        assert "tensor_stats" in summary
        assert "preprocessing_config" in summary

    def test_string_representations(self, sample_tensor):
        """Test string representation methods."""
        repr_str = repr(sample_tensor)
        str_str = str(sample_tensor)

        assert "test_image_001" in repr_str
        assert "test_image_001" in str_str
        assert "z_score_per_image" in repr_str
        assert "z_score_per_image" in str_str


class TestPreprocessedTensorSaveLoad:
    """Tests for PreprocessedTensor saving and loading functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def sample_tensor(self):
        """Create sample preprocessed tensor."""
        torch.manual_seed(42)
        tensor_data = torch.randn(3, 512, 512)
        config = {
            "target_size": (512, 512),
            "normalization_method": "z_score_per_image",
            "input_adapter": "1to3_replication",
        }

        return PreprocessedTensor(
            image_id="test_image_001",
            tensor_data=tensor_data,
            preprocessing_config=config,
            normalization_method="z_score_per_image",
            target_size=(512, 512),
            input_adapter="1to3_replication",
            border_removed=True,
            processing_time=1.5,
        )

    def test_save_tensor(self, sample_tensor, temp_dir):
        """Test saving tensor to file."""
        save_path = Path(temp_dir) / "test_tensor.pt"

        success = sample_tensor.save_tensor(save_path)

        assert success
        assert save_path.exists()

    def test_load_tensor(self, sample_tensor, temp_dir):
        """Test loading tensor from file."""
        save_path = Path(temp_dir) / "test_tensor.pt"
        sample_tensor.save_tensor(save_path)

        loaded_tensor = PreprocessedTensor.load_tensor(save_path)

        assert loaded_tensor.image_id == sample_tensor.image_id
        assert loaded_tensor.normalization_method == sample_tensor.normalization_method
        assert loaded_tensor.target_size == sample_tensor.target_size
        assert loaded_tensor.input_adapter == sample_tensor.input_adapter
        assert loaded_tensor.border_removed == sample_tensor.border_removed
        assert torch.allclose(loaded_tensor.tensor_data, sample_tensor.tensor_data)

    def test_save_load_round_trip(self, sample_tensor, temp_dir):
        """Test complete save/load round trip."""
        save_path = Path(temp_dir) / "round_trip_tensor.pt"

        # Save
        sample_tensor.save_tensor(save_path)

        # Load
        loaded_tensor = PreprocessedTensor.load_tensor(save_path)

        # Verify all attributes match
        assert loaded_tensor.image_id == sample_tensor.image_id
        assert loaded_tensor.normalization_method == sample_tensor.normalization_method
        assert loaded_tensor.target_size == sample_tensor.target_size
        assert loaded_tensor.input_adapter == sample_tensor.input_adapter
        assert loaded_tensor.border_removed == sample_tensor.border_removed
        assert loaded_tensor.processing_time == sample_tensor.processing_time

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            PreprocessedTensor.load_tensor("nonexistent_file.pt")

    def test_save_creates_directory(self, sample_tensor, temp_dir):
        """Test saving creates parent directories."""
        save_path = Path(temp_dir) / "subdir" / "test_tensor.pt"

        success = sample_tensor.save_tensor(save_path)

        assert success
        assert save_path.exists()
        assert save_path.parent.exists()


class TestPreprocessedTensorFactoryFunction:
    """Tests for PreprocessedTensor factory function."""

    def test_create_preprocessed_tensor_from_config(self):
        """Test factory function for creating PreprocessedTensor."""
        torch.manual_seed(42)
        tensor_data = torch.randn(3, 512, 512)
        config = {
            "target_size": (512, 512),
            "normalization_method": "z_score_per_image",
            "input_adapter": "1to3_replication",
            "border_removal": True,
        }

        tensor = create_preprocessed_tensor_from_config(
            image_id="test_image_001",
            tensor_data=tensor_data,
            config=config,
        )

        assert isinstance(tensor, PreprocessedTensor)
        assert tensor.image_id == "test_image_001"
        assert tensor.normalization_method == "z_score_per_image"
        assert tensor.target_size == (512, 512)
        assert tensor.input_adapter == "1to3_replication"

    def test_create_with_defaults(self):
        """Test factory function uses defaults when not specified."""
        torch.manual_seed(42)
        tensor_data = torch.randn(3, 512, 512)
        config = {}

        tensor = create_preprocessed_tensor_from_config(
            image_id="test_image_001",
            tensor_data=tensor_data,
            config=config,
        )

        # Should use default values
        assert tensor.normalization_method == "z_score_per_image"
        assert tensor.target_size == (512, 512)
        assert tensor.input_adapter == "1to3_replication"


class TestPreprocessedTensorEdgeCases:
    """Tests for PreprocessedTensor edge cases."""

    def test_tensor_dtype_conversion(self):
        """Test automatic dtype conversion to float32."""
        # Create int tensor
        tensor_data = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
        config = {
            "target_size": (512, 512),
            "normalization_method": "z_score_per_image",
            "input_adapter": "1to3_replication",
        }

        tensor = PreprocessedTensor(
            image_id="test_image",
            tensor_data=tensor_data,
            preprocessing_config=config,
            normalization_method="z_score_per_image",
            target_size=(512, 512),
            input_adapter="1to3_replication",
        )

        # Should be converted to float32
        assert tensor.tensor_data.dtype == torch.float32

    def test_whitespace_image_id(self):
        """Test validation of whitespace-only image ID."""
        tensor_data = torch.randn(3, 512, 512)
        config = {
            "target_size": (512, 512),
            "normalization_method": "z_score_per_image",
            "input_adapter": "1to3_replication",
        }

        with pytest.raises(ValueError, match="image_id cannot be empty"):
            PreprocessedTensor(
                image_id="   ",
                tensor_data=tensor_data,
                preprocessing_config=config,
                normalization_method="z_score_per_image",
                target_size=(512, 512),
                input_adapter="1to3_replication",
            )

    def test_image_id_stripped(self):
        """Test image ID is stripped of whitespace."""
        tensor_data = torch.randn(3, 512, 512)
        config = {
            "target_size": (512, 512),
            "normalization_method": "z_score_per_image",
            "input_adapter": "1to3_replication",
        }

        tensor = PreprocessedTensor(
            image_id="  test_image_001  ",
            tensor_data=tensor_data,
            preprocessing_config=config,
            normalization_method="z_score_per_image",
            target_size=(512, 512),
            input_adapter="1to3_replication",
        )

        assert tensor.image_id == "test_image_001"
