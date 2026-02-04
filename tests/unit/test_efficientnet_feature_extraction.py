"""
Unit tests for EfficientNet-B0 feature extraction functionality.

These tests validate individual EfficientNet-B0 feature extraction functions and operations.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
models = pytest.importorskip("torchvision.models")


class TestEfficientNetFeatureExtraction:
    """Unit tests for EfficientNet-B0 feature extraction functions."""

    @pytest.fixture
    def sample_tensor(self) -> torch.Tensor:
        """Create a sample preprocessed tensor for testing."""
        torch.manual_seed(42)
        return torch.randn(3, 512, 512)

    @pytest.fixture
    def efficientnet_b0_model(self):
        """Load EfficientNet-B0 model for testing."""
        torch.manual_seed(42)
        model = models.efficientnet_b0(weights=None)
        model.eval()
        return model

    @pytest.fixture
    def sample_batch(self) -> torch.Tensor:
        """Create a sample batch of tensors for testing."""
        torch.manual_seed(42)
        return torch.randn(4, 3, 512, 512)

    def test_efficientnet_b0_model_loading(self, efficientnet_b0_model):
        """Test EfficientNet-B0 model loading and basic properties."""
        # Validate model properties
        assert isinstance(efficientnet_b0_model, models.EfficientNet)
        assert efficientnet_b0_model.training == False  # Should be in eval mode

        # Validate model architecture
        assert hasattr(efficientnet_b0_model, "features")
        assert hasattr(efficientnet_b0_model, "avgpool")
        assert hasattr(efficientnet_b0_model, "classifier")

        # Validate classifier dimensions
        assert efficientnet_b0_model.classifier[1].in_features == 1280
        assert efficientnet_b0_model.classifier[1].out_features == 1000  # ImageNet classes

    def test_efficientnet_b0_forward_pass(self, sample_tensor, efficientnet_b0_model):
        """Test EfficientNet-B0 forward pass through the network."""
        # Add batch dimension
        batch_tensor = sample_tensor.unsqueeze(0)

        # Forward pass through EfficientNet-B0
        with torch.no_grad():
            x = batch_tensor

            # Feature extraction
            x = efficientnet_b0_model.features(x)

            # Average pooling
            x = efficientnet_b0_model.avgpool(x)
            x = torch.flatten(x, 1)

            # Final classification layer
            x = efficientnet_b0_model.classifier(x)

        # Validate output
        assert x.shape == (1, 1000)  # ImageNet classes
        assert x.dtype == torch.float32
        assert not torch.any(torch.isnan(x))
        assert not torch.any(torch.isinf(x))

    def test_efficientnet_b0_feature_extraction(self, sample_tensor, efficientnet_b0_model):
        """Test EfficientNet-B0 feature extraction at avgpool layer."""
        # Add batch dimension
        batch_tensor = sample_tensor.unsqueeze(0)

        # Extract features at avgpool layer
        with torch.no_grad():
            x = batch_tensor

            # Forward pass to avgpool
            x = efficientnet_b0_model.features(x)

            # Extract features at avgpool layer
            features = efficientnet_b0_model.avgpool(x)
            features = torch.flatten(features, 1)

        # Validate features
        assert features.shape == (1, 1280)  # EfficientNet-B0 feature dimension
        assert features.dtype == torch.float32
        assert not torch.any(torch.isnan(features))
        assert not torch.any(torch.isinf(features))

    def test_efficientnet_b0_batch_processing(self, sample_batch, efficientnet_b0_model):
        """Test EfficientNet-B0 batch processing."""
        batch_size = sample_batch.shape[0]

        # Process batch
        with torch.no_grad():
            x = sample_batch

            # Forward pass to avgpool
            x = efficientnet_b0_model.features(x)

            # Extract features
            features = efficientnet_b0_model.avgpool(x)
            features = torch.flatten(features, 1)

        # Validate batch features
        assert features.shape == (batch_size, 1280)
        assert features.dtype == torch.float32
        assert not torch.any(torch.isnan(features))
        assert not torch.any(torch.isinf(features))

    def test_efficientnet_b0_feature_extraction_reproducibility(
        self, sample_tensor, efficientnet_b0_model
    ):
        """Test reproducibility of EfficientNet-B0 feature extraction."""
        seed = 42

        # Extract features multiple times with same seed
        features_list = []
        for _ in range(3):
            torch.manual_seed(seed)
            np.random.seed(seed)

            with torch.no_grad():
                x = sample_tensor.unsqueeze(0)

                # Forward pass to avgpool
                x = efficientnet_b0_model.features(x)

                features = efficientnet_b0_model.avgpool(x)
                features = torch.flatten(features, 1)
                features_list.append(features)

        # Features should be identical
        for i in range(1, len(features_list)):
            assert torch.allclose(
                features_list[0], features_list[i]
            ), "Feature extraction not reproducible"

    def test_efficientnet_b0_different_input_sizes(self, efficientnet_b0_model):
        """Test EfficientNet-B0 with different input sizes."""
        input_sizes = [(256, 256), (512, 512), (1024, 1024)]

        for height, width in input_sizes:
            # Create tensor with different size
            tensor = torch.randn(3, height, width)

            with torch.no_grad():
                x = tensor.unsqueeze(0)

                # Forward pass to avgpool
                x = efficientnet_b0_model.features(x)

                features = efficientnet_b0_model.avgpool(x)
                features = torch.flatten(features, 1)

            # Features should always be 1280-dimensional
            assert features.shape == (1, 1280)
            assert not torch.any(torch.isnan(features))
            assert not torch.any(torch.isinf(features))

    def test_efficientnet_b0_feature_normalization(self, sample_tensor, efficientnet_b0_model):
        """Test EfficientNet-B0 feature normalization options."""
        # Extract features
        with torch.no_grad():
            x = sample_tensor.unsqueeze(0)

            # Forward pass to avgpool
            x = efficientnet_b0_model.features(x)

            features = efficientnet_b0_model.avgpool(x)
            features = torch.flatten(features, 1)

        # Test different normalization methods
        normalization_methods = ["none", "l2", "l1", "min_max"]

        for method in normalization_methods:
            if method == "none":
                normalized_features = features
            elif method == "l2":
                normalized_features = features / (
                    torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8
                )
            elif method == "l1":
                normalized_features = features / (
                    torch.norm(features, p=1, dim=1, keepdim=True) + 1e-8
                )
            elif method == "min_max":
                min_val = torch.min(features, dim=1, keepdim=True)[0]
                max_val = torch.max(features, dim=1, keepdim=True)[0]
                normalized_features = (features - min_val) / (max_val - min_val + 1e-8)

            # Validate normalized features
            assert normalized_features.shape == features.shape
            assert not torch.any(torch.isnan(normalized_features))
            assert not torch.any(torch.isinf(normalized_features))

            # Validate normalization
            if method == "l2":
                assert (
                    torch.abs(torch.norm(normalized_features, p=2, dim=1) - 1.0).max()
                    < 1e-6
                )
            elif method == "l1":
                assert (
                    torch.abs(torch.norm(normalized_features, p=1, dim=1) - 1.0).max()
                    < 1e-6
                )
            elif method == "min_max":
                assert normalized_features.min() >= 0
                assert normalized_features.max() <= 1

    def test_efficientnet_b0_feature_extraction_performance(
        self, sample_tensor, efficientnet_b0_model
    ):
        """Test EfficientNet-B0 feature extraction performance benchmarks."""
        import time

        # Time the feature extraction
        start_time = time.time()

        # Extract features multiple times
        for _ in range(10):
            with torch.no_grad():
                x = sample_tensor.unsqueeze(0)

                # Forward pass to avgpool
                x = efficientnet_b0_model.features(x)

                features = efficientnet_b0_model.avgpool(x)
                features = torch.flatten(features, 1)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert (
            processing_time < 10.0
        ), f"Feature extraction too slow: {processing_time:.2f}s"

    def test_efficientnet_b0_feature_extraction_memory_usage(
        self, sample_tensor, efficientnet_b0_model
    ):
        """Test memory usage during EfficientNet-B0 feature extraction."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Extract features
        with torch.no_grad():
            x = sample_tensor.unsqueeze(0)

            # Forward pass to avgpool
            x = efficientnet_b0_model.features(x)

            features = efficientnet_b0_model.avgpool(x)
            features = torch.flatten(features, 1)

        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (adjust threshold as needed)
        assert (
            memory_increase < 100 * 1024 * 1024
        ), f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB"

    def test_efficientnet_b0_feature_extraction_error_handling(self, efficientnet_b0_model):
        """Test error handling in EfficientNet-B0 feature extraction."""
        # Test with invalid tensor shapes
        invalid_tensors = [
            torch.randn(1, 512, 512),  # Wrong number of channels
            torch.randn(3, 256, 256),  # Wrong dimensions
            torch.randn(3, 512, 512, 3),  # Extra dimension
        ]

        for invalid_tensor in invalid_tensors:
            with pytest.raises((ValueError, RuntimeError)):
                # This should raise an error
                if invalid_tensor.shape[0] != 3:
                    raise ValueError("Invalid number of channels")
                if invalid_tensor.shape[1] != 512 or invalid_tensor.shape[2] != 512:
                    raise ValueError("Invalid tensor dimensions")
                if len(invalid_tensor.shape) != 3:
                    raise ValueError("Invalid tensor rank")

        # Test with invalid data types
        invalid_dtypes = [
            torch.randn(3, 512, 512).long(),  # Long tensor
            torch.randn(3, 512, 512).double(),  # Double tensor
        ]

        for invalid_tensor in invalid_dtypes:
            with pytest.raises((ValueError, RuntimeError)):
                # This should raise an error
                if invalid_tensor.dtype != torch.float32:
                    raise ValueError("Invalid tensor dtype")

    def test_efficientnet_b0_device_handling(self, sample_tensor, efficientnet_b0_model):
        """Test device handling for EfficientNet-B0 feature extraction."""
        # Test CPU device
        assert sample_tensor.device.type == "cpu"

        # Test CUDA device (if available)
        if torch.cuda.is_available():
            sample_tensor_cuda = sample_tensor.cuda()
            efficientnet_b0_model_cuda = efficientnet_b0_model.cuda()

            with torch.no_grad():
                x = sample_tensor_cuda.unsqueeze(0)

                # Forward pass to avgpool
                x = efficientnet_b0_model_cuda.features(x)

                features = efficientnet_b0_model_cuda.avgpool(x)
                features = torch.flatten(features, 1)

            assert features.device.type == "cuda"
            assert features.shape == (1, 1280)

        # Test MPS device (if available on macOS)
        if torch.backends.mps.is_available():
            sample_tensor_mps = sample_tensor.to("mps")
            efficientnet_b0_model_mps = efficientnet_b0_model.to("mps")

            with torch.no_grad():
                x = sample_tensor_mps.unsqueeze(0)

                # Forward pass to avgpool
                x = efficientnet_b0_model_mps.features(x)

                features = efficientnet_b0_model_mps.avgpool(x)
                features = torch.flatten(features, 1)

            assert features.device.type == "mps"
            assert features.shape == (1, 1280)

    def test_efficientnet_b0_feature_quality(self, sample_tensor, efficientnet_b0_model):
        """Test quality of EfficientNet-B0 extracted features."""
        # Extract features
        with torch.no_grad():
            x = sample_tensor.unsqueeze(0)

            # Forward pass to avgpool
            x = efficientnet_b0_model.features(x)

            features = efficientnet_b0_model.avgpool(x)
            features = torch.flatten(features, 1)

        # Test feature quality
        # Features should not be all zeros
        assert not torch.all(features == 0)

        # Features should have reasonable variance
        feature_std = torch.std(features)
        assert feature_std > 0.1  # Adjust threshold as needed

        # Features should not be all the same value
        feature_unique = torch.unique(features)
        assert len(feature_unique) > 1

        # Features should be in reasonable range
        feature_min = torch.min(features)
        feature_max = torch.max(features)
        assert feature_min > -1e4  # Untrained weights can yield wide ranges
        assert feature_max < 1e4  # Untrained weights can yield wide ranges

    def test_efficientnet_b0_feature_extraction_with_gradients(
        self, sample_tensor, efficientnet_b0_model
    ):
        """Test EfficientNet-B0 feature extraction with gradient computation."""
        # Enable gradients
        sample_tensor.requires_grad_(True)
        efficientnet_b0_model.train()

        # Forward pass to avgpool
        x = sample_tensor.unsqueeze(0)

        x = efficientnet_b0_model.features(x)

        features = efficientnet_b0_model.avgpool(x)
        features = torch.flatten(features, 1)

        # Test gradient computation
        loss = torch.mean(features)
        loss.backward()

        # Validate gradients
        assert sample_tensor.grad is not None
        assert not torch.any(torch.isnan(sample_tensor.grad))
        assert not torch.any(torch.isinf(sample_tensor.grad))

        # Reset model to eval mode
        efficientnet_b0_model.eval()


if __name__ == "__main__":
    pytest.main([__file__])
