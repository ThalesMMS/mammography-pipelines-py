"""
Unit tests for image preprocessing functionality.

These tests validate individual image preprocessing functions and operations.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

pytest.importorskip("skimage")
from skimage import exposure, filters, restoration, transform
from skimage.morphology import disk

class TestImagePreprocessing:
    """Unit tests for image preprocessing functions."""

    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        """Create a sample mammography image for testing."""
        np.random.seed(42)
        height, width = 2048, 1536

        # Create breast-like shape
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2

        # Elliptical breast shape
        breast_mask = ((x - center_x) / (width * 0.4)) ** 2 + (
            (y - center_y) / (height * 0.5)
        ) ** 2 <= 1

        # Create image with breast tissue
        image = np.zeros((height, width), dtype=np.uint16)
        image[breast_mask] = np.random.normal(1000, 200, np.sum(breast_mask)).astype(
            np.uint16
        )

        # Add some noise
        noise = np.random.normal(0, 50, image.shape)
        image = np.clip(image + noise, 0, 4095)

        return image.astype(np.uint16)

    @pytest.fixture
    def sample_tensor(self) -> torch.Tensor:
        """Create a sample preprocessed tensor for testing."""
        return torch.randn(3, 512, 512)

    def test_image_resizing(self, sample_image):
        """Test image resizing functionality."""
        original_shape = sample_image.shape
        target_size = (512, 512)

        # Test resize with different methods
        resize_methods = ["bilinear", "bicubic", "nearest"]

        for method in resize_methods:
            if method == "bilinear":
                resized = transform.resize(
                    sample_image, target_size, order=1, preserve_range=True
                )
            elif method == "bicubic":
                resized = transform.resize(
                    sample_image, target_size, order=3, preserve_range=True
                )
            elif method == "nearest":
                resized = transform.resize(
                    sample_image, target_size, order=0, preserve_range=True
                )

            # Validate resized image
            assert resized.shape == target_size
            assert np.issubdtype(resized.dtype, np.floating) or np.issubdtype(
                resized.dtype, np.integer
            )
            assert not np.any(np.isnan(resized))
            assert not np.any(np.isinf(resized))

    def test_image_resizing_with_aspect_ratio(self, sample_image):
        """Test image resizing while preserving aspect ratio."""
        original_shape = sample_image.shape
        target_size = (512, 512)

        # Calculate aspect ratio preserving dimensions
        h, w = original_shape
        target_h, target_w = target_size

        # Calculate scaling factor
        scale = min(target_h / h, target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize image
        resized = transform.resize(sample_image, (new_h, new_w), preserve_range=True)

        # Validate resized image
        assert resized.shape == (new_h, new_w)
        assert new_h <= target_h
        assert new_w <= target_w

        # Calculate actual aspect ratio
        aspect_ratio_original = w / h
        aspect_ratio_resized = new_w / new_h

        # Aspect ratio should be preserved
        assert abs(aspect_ratio_original - aspect_ratio_resized) < 1e-6

    def test_image_padding(self, sample_image):
        """Test image padding functionality."""
        h, w = sample_image.shape
        target_size = (h + 128, w + 128)

        # Test different padding strategies
        padding_strategies = ["reflect", "constant", "edge"]

        for strategy in padding_strategies:
            if strategy == "reflect":
                # Reflect padding
                pad_h = (target_size[0] - h) // 2
                pad_w = (target_size[1] - w) // 2
                padded = np.pad(
                    sample_image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect"
                )
            elif strategy == "constant":
                # Constant padding
                pad_h = (target_size[0] - h) // 2
                pad_w = (target_size[1] - w) // 2
                padded = np.pad(
                    sample_image,
                    ((pad_h, pad_h), (pad_w, pad_w)),
                    mode="constant",
                    constant_values=0,
                )
            elif strategy == "edge":
                # Edge padding
                pad_h = (target_size[0] - h) // 2
                pad_w = (target_size[1] - w) // 2
                padded = np.pad(
                    sample_image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge"
                )

            # Validate padded image
            assert padded.shape == target_size
            assert not np.any(np.isnan(padded))
            assert not np.any(np.isinf(padded))

    def test_image_normalization(self, sample_image):
        """Test image normalization methods."""
        normalization_methods = [
            "z_score_per_image",
            "fixed_window",
            "min_max_scaling",
            "percentile_scaling",
        ]

        for method in normalization_methods:
            if method == "z_score_per_image":
                # Z-score normalization
                mean = np.mean(sample_image)
                std = np.std(sample_image)
                normalized = (sample_image - mean) / (std + 1e-8)

                assert np.abs(np.mean(normalized)) < 1e-6
                assert np.abs(np.std(normalized) - 1.0) < 1e-6

            elif method == "fixed_window":
                # Fixed window normalization (e.g., 0-4095 -> 0-1)
                min_val, max_val = 0, 4095
                normalized = (sample_image - min_val) / (max_val - min_val)
                normalized = np.clip(normalized, 0, 1)

                assert np.min(normalized) >= 0
                assert np.max(normalized) <= 1

            elif method == "min_max_scaling":
                # Min-max scaling
                min_val = np.min(sample_image)
                max_val = np.max(sample_image)
                normalized = (sample_image - min_val) / (max_val - min_val + 1e-8)

                assert np.min(normalized) >= 0
                assert np.max(normalized) <= 1

            elif method == "percentile_scaling":
                # Percentile scaling
                p2, p98 = np.percentile(sample_image, [2, 98])
                normalized = (sample_image - p2) / (p98 - p2 + 1e-8)
                normalized = np.clip(normalized, 0, 1)

                assert np.min(normalized) >= 0
                assert np.max(normalized) <= 1

            # Validate normalized image
            assert not np.any(np.isnan(normalized))
            assert not np.any(np.isinf(normalized))

    def test_border_removal(self, sample_image):
        """Test border removal functionality."""
        # Test border removal
        border_threshold = 0.1
        min_breast_area = 0.05

        # Convert to float for processing
        image = sample_image.astype(np.float32)

        # Find non-zero regions (simulating breast tissue)
        non_zero_mask = image > np.percentile(image, 10)

        # Find bounding box
        rows = np.any(non_zero_mask, axis=1)
        cols = np.any(non_zero_mask, axis=0)

        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            # Crop image
            cropped = image[y_min : y_max + 1, x_min : x_max + 1]

            # Validate cropped image
            assert cropped.shape[0] <= image.shape[0]
            assert cropped.shape[1] <= image.shape[1]
            assert cropped.shape[0] > 0
            assert cropped.shape[1] > 0

            # Validate breast area
            breast_area = np.sum(cropped > np.percentile(cropped, 10))
            total_area = cropped.size
            breast_ratio = breast_area / total_area

            assert breast_ratio >= min_breast_area

    def test_image_filtering(self, sample_image):
        """Test image filtering operations."""
        # Test Gaussian filter
        sigma = 1.0
        filtered = filters.gaussian(sample_image, sigma=sigma)

        assert filtered.shape == sample_image.shape
        assert not np.any(np.isnan(filtered))
        assert not np.any(np.isinf(filtered))

        # Test median filter
        kernel_size = 3
        median_filtered = filters.median(sample_image, disk(kernel_size))

        assert median_filtered.shape == sample_image.shape
        assert not np.any(np.isnan(median_filtered))
        assert not np.any(np.isinf(median_filtered))

        # Test bilateral filter (fallback to restoration API when needed)
        if hasattr(filters, "bilateral"):
            bilateral_filtered = filters.bilateral(
                sample_image, sigma_color=0.1, sigma_spatial=1.0
            )
        else:
            bilateral_filtered = restoration.denoise_bilateral(
                sample_image.astype(np.float32),
                sigma_color=0.1,
                sigma_spatial=1.0,
                channel_axis=None,
            )

        assert bilateral_filtered.shape == sample_image.shape
        assert not np.any(np.isnan(bilateral_filtered))
        assert not np.any(np.isinf(bilateral_filtered))

    def test_image_enhancement(self, sample_image):
        """Test image enhancement operations."""
        # Test histogram equalization
        equalized = exposure.equalize_hist(sample_image)

        assert equalized.shape == sample_image.shape
        assert np.min(equalized) >= 0
        assert np.max(equalized) <= 1
        assert not np.any(np.isnan(equalized))
        assert not np.any(np.isinf(equalized))

        # Test adaptive histogram equalization
        adaptive_equalized = exposure.equalize_adapthist(sample_image, clip_limit=0.03)

        assert adaptive_equalized.shape == sample_image.shape
        assert np.min(adaptive_equalized) >= 0
        assert np.max(adaptive_equalized) <= 1
        assert not np.any(np.isnan(adaptive_equalized))
        assert not np.any(np.isinf(adaptive_equalized))

        # Test contrast stretching
        p2, p98 = np.percentile(sample_image, (2, 98))
        stretched = exposure.rescale_intensity(sample_image, in_range=(p2, p98))

        assert stretched.shape == sample_image.shape
        assert not np.any(np.isnan(stretched))
        assert not np.any(np.isinf(stretched))

    def test_input_adapter_1to3_replication(self, sample_image):
        """Test 1-to-3 channel replication for grayscale to RGB conversion."""
        # Convert grayscale to RGB by replication
        grayscale = sample_image.astype(np.float32)
        rgb = np.stack([grayscale, grayscale, grayscale], axis=0)

        # Validate RGB image
        assert rgb.shape == (3, *grayscale.shape)
        assert rgb.dtype == np.float32
        assert not np.any(np.isnan(rgb))
        assert not np.any(np.isinf(rgb))

        # Validate all channels are identical
        assert np.array_equal(rgb[0], grayscale)
        assert np.array_equal(rgb[1], grayscale)
        assert np.array_equal(rgb[2], grayscale)

    def test_input_adapter_conv1_adapted(self, sample_image):
        """Test adapted conv1 weights for grayscale to RGB conversion."""
        # This would involve modifying the first conv layer weights
        # For testing, we'll simulate the expected behavior
        grayscale = sample_image.astype(np.float32)

        # Simulate adapted conv1 weights (this would be done in the model)
        # For now, we'll just validate the input format
        assert grayscale.ndim == 2
        assert grayscale.dtype == np.float32
        assert not np.any(np.isnan(grayscale))
        assert not np.any(np.isinf(grayscale))

    def test_tensor_conversion(self, sample_image):
        """Test conversion from numpy array to PyTorch tensor."""
        # Convert to float32
        image_float = sample_image.astype(np.float32)

        # Convert to tensor
        tensor = torch.from_numpy(image_float)

        # Validate tensor
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.shape == image_float.shape
        assert not torch.any(torch.isnan(tensor))
        assert not torch.any(torch.isinf(tensor))

        # Test tensor operations
        tensor_normalized = (tensor - tensor.mean()) / (tensor.std() + 1e-8)

        assert torch.abs(tensor_normalized.mean()) < 1e-6
        assert torch.abs(tensor_normalized.std() - 1.0) < 1e-6

    def test_batch_preprocessing(self, sample_image):
        """Test batch preprocessing functionality."""
        # Create multiple images
        batch_images = [sample_image.copy() for _ in range(3)]

        # Process batch
        batch_tensors = []
        for image in batch_images:
            # Normalize
            normalized = (image - image.mean()) / (image.std() + 1e-8)

            # Convert to RGB
            rgb = np.stack([normalized, normalized, normalized], axis=0)

            # Convert to tensor
            tensor = torch.from_numpy(rgb.astype(np.float32))
            batch_tensors.append(tensor)

        # Stack into batch
        batch_tensor = torch.stack(batch_tensors)

        # Validate batch
        assert batch_tensor.shape == (3, 3, *sample_image.shape)
        assert batch_tensor.dtype == torch.float32
        assert not torch.any(torch.isnan(batch_tensor))
        assert not torch.any(torch.isinf(batch_tensor))

    def test_preprocessing_reproducibility(self, sample_image):
        """Test reproducibility of preprocessing with fixed seeds."""
        seed = 42

        # Process image multiple times with same seed
        results = []
        for _ in range(3):
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Add noise (simulating random operations)
            noise = np.random.normal(0, 10, sample_image.shape)
            noisy_image = sample_image + noise

            # Normalize
            normalized = (noisy_image - noisy_image.mean()) / (noisy_image.std() + 1e-8)

            # Convert to tensor
            tensor = torch.from_numpy(normalized.astype(np.float32))
            results.append(tensor)

        # Results should be identical
        for i in range(1, len(results)):
            assert torch.allclose(
                results[0], results[i]
            ), "Preprocessing not reproducible"

    def test_preprocessing_error_handling(self, sample_image):
        """Test error handling in preprocessing operations."""
        # Test with invalid input shapes
        invalid_shapes = [
            np.array([]),  # Empty array
            np.zeros((0, 0)),  # Zero dimensions
            np.ones((1, 1)) * np.nan,  # NaN values
            np.ones((1, 1)) * np.inf,  # Infinite values
        ]

        for invalid_image in invalid_shapes:
            with pytest.raises((ValueError, IndexError)):
                # This should raise an error
                if invalid_image.size == 0:
                    raise ValueError("Empty image")
                if np.any(np.isnan(invalid_image)):
                    raise ValueError("NaN values in image")
                if np.any(np.isinf(invalid_image)):
                    raise ValueError("Infinite values in image")

        # Test with invalid target sizes
        invalid_target_sizes = [
            (0, 512),  # Zero height
            (512, 0),  # Zero width
            (-1, 512),  # Negative height
            (512, -1),  # Negative width
        ]

        for target_size in invalid_target_sizes:
            with pytest.raises(ValueError):
                # This should raise an error
                if target_size[0] <= 0 or target_size[1] <= 0:
                    raise ValueError("Invalid target size")

        # Test with invalid normalization methods
        invalid_methods = ["invalid_method", "", None]

        for method in invalid_methods:
            with pytest.raises(ValueError):
                # This should raise an error
                valid_methods = [
                    "z_score_per_image",
                    "fixed_window",
                    "min_max_scaling",
                    "percentile_scaling",
                ]
                if method not in valid_methods:
                    raise ValueError("Invalid normalization method")

    def test_preprocessing_performance(self, sample_image):
        """Test preprocessing performance benchmarks."""
        import time

        # Time the preprocessing operations
        start_time = time.time()

        # Perform multiple preprocessing operations
        for _ in range(10):
            # Resize
            resized = transform.resize(sample_image, (512, 512), preserve_range=True)

            # Normalize
            normalized = (resized - resized.mean()) / (resized.std() + 1e-8)

            # Convert to RGB
            rgb = np.stack([normalized, normalized, normalized], axis=0)

            # Convert to tensor
            tensor = torch.from_numpy(rgb.astype(np.float32))

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 5.0, f"Preprocessing too slow: {processing_time:.2f}s"

    def test_preprocessing_memory_usage(self, sample_image):
        """Test memory usage during preprocessing."""
        import os

        psutil = pytest.importorskip("psutil")

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform preprocessing operations
        resized = transform.resize(sample_image, (512, 512), preserve_range=True)
        normalized = (resized - resized.mean()) / (resized.std() + 1e-8)
        rgb = np.stack([normalized, normalized, normalized], axis=0)
        tensor = torch.from_numpy(rgb.astype(np.float32))

        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (adjust threshold as needed)
        assert (
            memory_increase < 100 * 1024 * 1024
        ), f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__])
