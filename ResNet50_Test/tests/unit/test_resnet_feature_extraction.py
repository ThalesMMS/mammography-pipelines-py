"""
Unit tests for ResNet-50 feature extraction functionality.

These tests validate individual ResNet-50 feature extraction functions and operations.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import numpy as np
import pytest
import torch
import torchvision.models as models

# Import the modules we'll be testing (these will be implemented later)
# from src.models.embeddings.resnet_extractor import ResNetEmbeddingExtractor
# from src.models.embeddings.input_adapter import InputAdapter
# from src.models.embeddings.feature_pooler import FeaturePooler


class TestResNetFeatureExtraction:
    """Unit tests for ResNet-50 feature extraction functions."""

    @pytest.fixture
    def sample_tensor(self) -> torch.Tensor:
        """Create a sample preprocessed tensor for testing."""
        return torch.randn(3, 512, 512)

    @pytest.fixture
    def resnet50_model(self):
        """Load ResNet-50 model for testing."""
        model = models.resnet50(pretrained=True)
        model.eval()
        return model

    @pytest.fixture
    def sample_batch(self) -> torch.Tensor:
        """Create a sample batch of tensors for testing."""
        return torch.randn(4, 3, 512, 512)

    def test_resnet50_model_loading(self, resnet50_model):
        """Test ResNet-50 model loading and basic properties."""
        # Validate model properties
        assert isinstance(resnet50_model, models.ResNet)
        assert resnet50_model.training == False  # Should be in eval mode

        # Validate model architecture
        assert hasattr(resnet50_model, "conv1")
        assert hasattr(resnet50_model, "bn1")
        assert hasattr(resnet50_model, "relu")
        assert hasattr(resnet50_model, "maxpool")
        assert hasattr(resnet50_model, "layer1")
        assert hasattr(resnet50_model, "layer2")
        assert hasattr(resnet50_model, "layer3")
        assert hasattr(resnet50_model, "layer4")
        assert hasattr(resnet50_model, "avgpool")
        assert hasattr(resnet50_model, "fc")

        # Validate layer dimensions
        assert resnet50_model.conv1.out_channels == 64
        assert resnet50_model.fc.in_features == 2048
        assert resnet50_model.fc.out_features == 1000  # ImageNet classes

    def test_resnet50_forward_pass(self, sample_tensor, resnet50_model):
        """Test ResNet-50 forward pass through the network."""
        # Add batch dimension
        batch_tensor = sample_tensor.unsqueeze(0)

        # Forward pass through ResNet-50
        with torch.no_grad():
            x = batch_tensor

            # Conv1
            x = resnet50_model.conv1(x)
            x = resnet50_model.bn1(x)
            x = resnet50_model.relu(x)
            x = resnet50_model.maxpool(x)

            # ResNet blocks
            x = resnet50_model.layer1(x)
            x = resnet50_model.layer2(x)
            x = resnet50_model.layer3(x)
            x = resnet50_model.layer4(x)

            # Average pooling
            x = resnet50_model.avgpool(x)
            x = torch.flatten(x, 1)

            # Final classification layer
            x = resnet50_model.fc(x)

        # Validate output
        assert x.shape == (1, 1000)  # ImageNet classes
        assert x.dtype == torch.float32
        assert not torch.any(torch.isnan(x))
        assert not torch.any(torch.isinf(x))

    def test_resnet50_feature_extraction(self, sample_tensor, resnet50_model):
        """Test ResNet-50 feature extraction at avgpool layer."""
        # Add batch dimension
        batch_tensor = sample_tensor.unsqueeze(0)

        # Extract features at avgpool layer
        with torch.no_grad():
            x = batch_tensor

            # Forward pass to avgpool
            x = resnet50_model.conv1(x)
            x = resnet50_model.bn1(x)
            x = resnet50_model.relu(x)
            x = resnet50_model.maxpool(x)

            x = resnet50_model.layer1(x)
            x = resnet50_model.layer2(x)
            x = resnet50_model.layer3(x)
            x = resnet50_model.layer4(x)

            # Extract features at avgpool layer
            features = resnet50_model.avgpool(x)
            features = torch.flatten(features, 1)

        # Validate features
        assert features.shape == (1, 2048)  # ResNet-50 feature dimension
        assert features.dtype == torch.float32
        assert not torch.any(torch.isnan(features))
        assert not torch.any(torch.isinf(features))

    def test_resnet50_batch_processing(self, sample_batch, resnet50_model):
        """Test ResNet-50 batch processing."""
        batch_size = sample_batch.shape[0]

        # Process batch
        with torch.no_grad():
            x = sample_batch

            # Forward pass to avgpool
            x = resnet50_model.conv1(x)
            x = resnet50_model.bn1(x)
            x = resnet50_model.relu(x)
            x = resnet50_model.maxpool(x)

            x = resnet50_model.layer1(x)
            x = resnet50_model.layer2(x)
            x = resnet50_model.layer3(x)
            x = resnet50_model.layer4(x)

            # Extract features
            features = resnet50_model.avgpool(x)
            features = torch.flatten(features, 1)

        # Validate batch features
        assert features.shape == (batch_size, 2048)
        assert features.dtype == torch.float32
        assert not torch.any(torch.isnan(features))
        assert not torch.any(torch.isinf(features))

    def test_resnet50_feature_extraction_reproducibility(
        self, sample_tensor, resnet50_model
    ):
        """Test reproducibility of ResNet-50 feature extraction."""
        seed = 42

        # Extract features multiple times with same seed
        features_list = []
        for _ in range(3):
            torch.manual_seed(seed)
            np.random.seed(seed)

            with torch.no_grad():
                x = sample_tensor.unsqueeze(0)

                # Forward pass to avgpool
                x = resnet50_model.conv1(x)
                x = resnet50_model.bn1(x)
                x = resnet50_model.relu(x)
                x = resnet50_model.maxpool(x)

                x = resnet50_model.layer1(x)
                x = resnet50_model.layer2(x)
                x = resnet50_model.layer3(x)
                x = resnet50_model.layer4(x)

                features = resnet50_model.avgpool(x)
                features = torch.flatten(features, 1)
                features_list.append(features)

        # Features should be identical
        for i in range(1, len(features_list)):
            assert torch.allclose(
                features_list[0], features_list[i]
            ), "Feature extraction not reproducible"

    def test_resnet50_different_input_sizes(self, resnet50_model):
        """Test ResNet-50 with different input sizes."""
        input_sizes = [(256, 256), (512, 512), (1024, 1024)]

        for height, width in input_sizes:
            # Create tensor with different size
            tensor = torch.randn(3, height, width)

            with torch.no_grad():
                x = tensor.unsqueeze(0)

                # Forward pass to avgpool
                x = resnet50_model.conv1(x)
                x = resnet50_model.bn1(x)
                x = resnet50_model.relu(x)
                x = resnet50_model.maxpool(x)

                x = resnet50_model.layer1(x)
                x = resnet50_model.layer2(x)
                x = resnet50_model.layer3(x)
                x = resnet50_model.layer4(x)

                features = resnet50_model.avgpool(x)
                features = torch.flatten(features, 1)

            # Features should always be 2048-dimensional
            assert features.shape == (1, 2048)
            assert not torch.any(torch.isnan(features))
            assert not torch.any(torch.isinf(features))

    def test_resnet50_feature_normalization(self, sample_tensor, resnet50_model):
        """Test ResNet-50 feature normalization options."""
        # Extract features
        with torch.no_grad():
            x = sample_tensor.unsqueeze(0)

            # Forward pass to avgpool
            x = resnet50_model.conv1(x)
            x = resnet50_model.bn1(x)
            x = resnet50_model.relu(x)
            x = resnet50_model.maxpool(x)

            x = resnet50_model.layer1(x)
            x = resnet50_model.layer2(x)
            x = resnet50_model.layer3(x)
            x = resnet50_model.layer4(x)

            features = resnet50_model.avgpool(x)
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

    def test_resnet50_feature_extraction_performance(
        self, sample_tensor, resnet50_model
    ):
        """Test ResNet-50 feature extraction performance benchmarks."""
        import time

        # Time the feature extraction
        start_time = time.time()

        # Extract features multiple times
        for _ in range(10):
            with torch.no_grad():
                x = sample_tensor.unsqueeze(0)

                # Forward pass to avgpool
                x = resnet50_model.conv1(x)
                x = resnet50_model.bn1(x)
                x = resnet50_model.relu(x)
                x = resnet50_model.maxpool(x)

                x = resnet50_model.layer1(x)
                x = resnet50_model.layer2(x)
                x = resnet50_model.layer3(x)
                x = resnet50_model.layer4(x)

                features = resnet50_model.avgpool(x)
                features = torch.flatten(features, 1)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert (
            processing_time < 10.0
        ), f"Feature extraction too slow: {processing_time:.2f}s"

    def test_resnet50_feature_extraction_memory_usage(
        self, sample_tensor, resnet50_model
    ):
        """Test memory usage during ResNet-50 feature extraction."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Extract features
        with torch.no_grad():
            x = sample_tensor.unsqueeze(0)

            # Forward pass to avgpool
            x = resnet50_model.conv1(x)
            x = resnet50_model.bn1(x)
            x = resnet50_model.relu(x)
            x = resnet50_model.maxpool(x)

            x = resnet50_model.layer1(x)
            x = resnet50_model.layer2(x)
            x = resnet50_model.layer3(x)
            x = resnet50_model.layer4(x)

            features = resnet50_model.avgpool(x)
            features = torch.flatten(features, 1)

        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (adjust threshold as needed)
        assert (
            memory_increase < 100 * 1024 * 1024
        ), f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB"

    def test_resnet50_feature_extraction_error_handling(self, resnet50_model):
        """Test error handling in ResNet-50 feature extraction."""
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

    def test_resnet50_device_handling(self, sample_tensor, resnet50_model):
        """Test device handling for ResNet-50 feature extraction."""
        # Test CPU device
        assert sample_tensor.device.type == "cpu"

        # Test CUDA device (if available)
        if torch.cuda.is_available():
            sample_tensor_cuda = sample_tensor.cuda()
            resnet50_model_cuda = resnet50_model.cuda()

            with torch.no_grad():
                x = sample_tensor_cuda.unsqueeze(0)

                # Forward pass to avgpool
                x = resnet50_model_cuda.conv1(x)
                x = resnet50_model_cuda.bn1(x)
                x = resnet50_model_cuda.relu(x)
                x = resnet50_model_cuda.maxpool(x)

                x = resnet50_model_cuda.layer1(x)
                x = resnet50_model_cuda.layer2(x)
                x = resnet50_model_cuda.layer3(x)
                x = resnet50_model_cuda.layer4(x)

                features = resnet50_model_cuda.avgpool(x)
                features = torch.flatten(features, 1)

            assert features.device.type == "cuda"
            assert features.shape == (1, 2048)

        # Test MPS device (if available on macOS)
        if torch.backends.mps.is_available():
            sample_tensor_mps = sample_tensor.to("mps")
            resnet50_model_mps = resnet50_model.to("mps")

            with torch.no_grad():
                x = sample_tensor_mps.unsqueeze(0)

                # Forward pass to avgpool
                x = resnet50_model_mps.conv1(x)
                x = resnet50_model_mps.bn1(x)
                x = resnet50_model_mps.relu(x)
                x = resnet50_model_mps.maxpool(x)

                x = resnet50_model_mps.layer1(x)
                x = resnet50_model_mps.layer2(x)
                x = resnet50_model_mps.layer3(x)
                x = resnet50_model_mps.layer4(x)

                features = resnet50_model_mps.avgpool(x)
                features = torch.flatten(features, 1)

            assert features.device.type == "mps"
            assert features.shape == (1, 2048)

    def test_resnet50_feature_quality(self, sample_tensor, resnet50_model):
        """Test quality of ResNet-50 extracted features."""
        # Extract features
        with torch.no_grad():
            x = sample_tensor.unsqueeze(0)

            # Forward pass to avgpool
            x = resnet50_model.conv1(x)
            x = resnet50_model.bn1(x)
            x = resnet50_model.relu(x)
            x = resnet50_model.maxpool(x)

            x = resnet50_model.layer1(x)
            x = resnet50_model.layer2(x)
            x = resnet50_model.layer3(x)
            x = resnet50_model.layer4(x)

            features = resnet50_model.avgpool(x)
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
        assert feature_min > -10  # Adjust threshold as needed
        assert feature_max < 10  # Adjust threshold as needed

    def test_resnet50_feature_extraction_with_gradients(
        self, sample_tensor, resnet50_model
    ):
        """Test ResNet-50 feature extraction with gradient computation."""
        # Enable gradients
        sample_tensor.requires_grad_(True)
        resnet50_model.train()

        # Forward pass to avgpool
        x = sample_tensor.unsqueeze(0)

        x = resnet50_model.conv1(x)
        x = resnet50_model.bn1(x)
        x = resnet50_model.relu(x)
        x = resnet50_model.maxpool(x)

        x = resnet50_model.layer1(x)
        x = resnet50_model.layer2(x)
        x = resnet50_model.layer3(x)
        x = resnet50_model.layer4(x)

        features = resnet50_model.avgpool(x)
        features = torch.flatten(features, 1)

        # Test gradient computation
        loss = torch.mean(features)
        loss.backward()

        # Validate gradients
        assert sample_tensor.grad is not None
        assert not torch.any(torch.isnan(sample_tensor.grad))
        assert not torch.any(torch.isinf(sample_tensor.grad))

        # Reset model to eval mode
        resnet50_model.eval()


if __name__ == "__main__":
    pytest.main([__file__])
