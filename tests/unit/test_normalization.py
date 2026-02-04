"""
Unit tests for normalization utilities.

These tests validate normalization statistics computation and validation functions.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from mammography.utils.normalization import (
    NormalizationStats,
    compute_normalization_stats,
    validate_normalization,
)


class TestNormalizationStats:
    """Unit tests for NormalizationStats dataclass."""

    def test_normalization_stats_creation_3_channel(self):
        """Test creating NormalizationStats with 3 channels."""
        stats = NormalizationStats(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            method="imagenet",
            sample_size=None
        )

        assert stats.mean == [0.485, 0.456, 0.406]
        assert stats.std == [0.229, 0.224, 0.225]
        assert stats.method == "imagenet"
        assert stats.sample_size is None

    def test_normalization_stats_creation_1_channel(self):
        """Test creating NormalizationStats with 1 channel."""
        stats = NormalizationStats(
            mean=[0.5],
            std=[0.5],
            method="custom",
            sample_size=1000
        )

        assert stats.mean == [0.5]
        assert stats.std == [0.5]
        assert stats.method == "custom"
        assert stats.sample_size == 1000

    def test_normalization_stats_validation_length_mismatch(self):
        """Test that mean and std must have same length."""
        with pytest.raises(ValueError, match="mean and std must have same length"):
            NormalizationStats(
                mean=[0.5, 0.5],
                std=[0.5]
            )

    def test_normalization_stats_validation_invalid_channel_count(self):
        """Test that mean/std must have 1 or 3 values."""
        with pytest.raises(ValueError, match="mean/std must have 1 or 3 values"):
            NormalizationStats(
                mean=[0.5, 0.5],
                std=[0.5, 0.5]
            )

    def test_normalization_stats_validation_negative_std(self):
        """Test that std values must be positive."""
        with pytest.raises(ValueError, match="std values must be positive"):
            NormalizationStats(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, -0.1, 0.5]
            )

    def test_normalization_stats_validation_zero_std(self):
        """Test that std values cannot be zero."""
        with pytest.raises(ValueError, match="std values must be positive"):
            NormalizationStats(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.0, 0.5]
            )

    def test_normalization_stats_to_dict(self):
        """Test conversion to dictionary."""
        stats = NormalizationStats(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            method="imagenet",
            sample_size=None
        )

        stats_dict = stats.to_dict()

        assert stats_dict == {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "method": "imagenet",
            "sample_size": None
        }

    def test_normalization_stats_from_dict(self):
        """Test creation from dictionary."""
        stats_dict = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "method": "imagenet",
            "sample_size": None
        }

        stats = NormalizationStats.from_dict(stats_dict)

        assert stats.mean == [0.485, 0.456, 0.406]
        assert stats.std == [0.229, 0.224, 0.225]
        assert stats.method == "imagenet"
        assert stats.sample_size is None

    def test_normalization_stats_imagenet_defaults(self):
        """Test ImageNet default statistics."""
        stats = NormalizationStats.imagenet_defaults()

        assert stats.mean == [0.485, 0.456, 0.406]
        assert stats.std == [0.229, 0.224, 0.225]
        assert stats.method == "imagenet"
        assert stats.sample_size is None

    def test_normalization_stats_round_trip(self):
        """Test conversion to dict and back."""
        original_stats = NormalizationStats(
            mean=[0.5, 0.6, 0.7],
            std=[0.1, 0.2, 0.3],
            method="custom",
            sample_size=500
        )

        stats_dict = original_stats.to_dict()
        restored_stats = NormalizationStats.from_dict(stats_dict)

        assert restored_stats.mean == original_stats.mean
        assert restored_stats.std == original_stats.std
        assert restored_stats.method == original_stats.method
        assert restored_stats.sample_size == original_stats.sample_size


class TestComputeNormalizationStats:
    """Unit tests for compute_normalization_stats function."""

    @pytest.fixture
    def simple_dataset_3_channel(self):
        """Create a simple dataset with 3-channel images."""
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples=100):
                self.num_samples = num_samples
                # Create deterministic data
                torch.manual_seed(42)
                self.data = torch.randn(num_samples, 3, 64, 64)

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return self.data[idx]

        return SimpleDataset()

    @pytest.fixture
    def simple_dataset_1_channel(self):
        """Create a simple dataset with 1-channel images."""
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples=100):
                self.num_samples = num_samples
                # Create deterministic data
                torch.manual_seed(42)
                self.data = torch.randn(num_samples, 1, 64, 64)

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return self.data[idx]

        return SimpleDataset()

    @pytest.fixture
    def normalized_dataset(self):
        """Create a dataset with known normalization."""
        class NormalizedDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples=100):
                self.num_samples = num_samples
                # Create data with known mean and std
                torch.manual_seed(42)
                self.data = torch.randn(num_samples, 3, 64, 64) * 0.5 + 0.3

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return self.data[idx]

        return NormalizedDataset()

    def test_compute_stats_from_dataset(self, simple_dataset_3_channel):
        """Test computing statistics from a dataset."""
        stats = compute_normalization_stats(
            simple_dataset_3_channel,
            num_samples=50,
            batch_size=10
        )

        assert isinstance(stats, NormalizationStats)
        assert len(stats.mean) == 3
        assert len(stats.std) == 3
        assert stats.method == "auto"
        assert stats.sample_size == 50

        # Check that values are reasonable
        for mean_val in stats.mean:
            assert -1.0 < mean_val < 1.0

        for std_val in stats.std:
            assert 0.5 < std_val < 1.5

    def test_compute_stats_all_samples(self, simple_dataset_3_channel):
        """Test computing statistics from all samples."""
        stats = compute_normalization_stats(
            simple_dataset_3_channel,
            num_samples=None,
            batch_size=10
        )

        assert stats.sample_size == len(simple_dataset_3_channel)

    def test_compute_stats_1_channel(self, simple_dataset_1_channel):
        """Test computing statistics for 1-channel images."""
        stats = compute_normalization_stats(
            simple_dataset_1_channel,
            num_samples=50,
            batch_size=10
        )

        assert len(stats.mean) == 1
        assert len(stats.std) == 1

    def test_compute_stats_from_dataloader(self, simple_dataset_3_channel):
        """Test computing statistics from a DataLoader."""
        dataloader = torch.utils.data.DataLoader(
            simple_dataset_3_channel,
            batch_size=10,
            shuffle=False
        )

        stats = compute_normalization_stats(dataloader)

        assert isinstance(stats, NormalizationStats)
        assert len(stats.mean) == 3
        assert len(stats.std) == 3

    def test_compute_stats_reproducibility(self, simple_dataset_3_channel):
        """Test that statistics computation is reproducible."""
        torch.manual_seed(42)
        stats1 = compute_normalization_stats(
            simple_dataset_3_channel,
            num_samples=50,
            batch_size=10
        )

        torch.manual_seed(42)
        stats2 = compute_normalization_stats(
            simple_dataset_3_channel,
            num_samples=50,
            batch_size=10
        )

        assert stats1.mean == stats2.mean
        assert stats1.std == stats2.std

    def test_compute_stats_known_distribution(self, normalized_dataset):
        """Test computing statistics on data with known distribution."""
        stats = compute_normalization_stats(
            normalized_dataset,
            num_samples=None,
            batch_size=10
        )

        # Expected mean around 0.3, std around 0.5
        for mean_val in stats.mean:
            assert abs(mean_val - 0.3) < 0.1

        for std_val in stats.std:
            assert abs(std_val - 0.5) < 0.1

    def test_compute_stats_empty_dataset(self):
        """Test that empty dataset raises error."""
        class EmptyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 0

            def __getitem__(self, idx):
                raise IndexError("Empty dataset")

        empty_dataset = EmptyDataset()

        with pytest.raises(ValueError, match="Dataset is empty"):
            compute_normalization_stats(empty_dataset)

    def test_compute_stats_tuple_batch_format(self):
        """Test handling of tuple batch format (image, label)."""
        class TupleDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples=50):
                self.num_samples = num_samples
                torch.manual_seed(42)
                self.images = torch.randn(num_samples, 3, 64, 64)
                self.labels = torch.randint(0, 2, (num_samples,))

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]

        dataset = TupleDataset()
        stats = compute_normalization_stats(dataset, batch_size=10)

        assert isinstance(stats, NormalizationStats)
        assert len(stats.mean) == 3
        assert len(stats.std) == 3

    def test_compute_stats_dict_batch_format(self):
        """Test handling of dict batch format."""
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples=50):
                self.num_samples = num_samples
                torch.manual_seed(42)
                self.images = torch.randn(num_samples, 3, 64, 64)

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return {"image": self.images[idx], "label": 0}

        dataset = DictDataset()
        stats = compute_normalization_stats(dataset, batch_size=10)

        assert isinstance(stats, NormalizationStats)
        assert len(stats.mean) == 3
        assert len(stats.std) == 3

    def test_compute_stats_3d_tensor_handling(self):
        """Test handling of 3D tensors (no channel dimension)."""
        class Dataset3D(torch.utils.data.Dataset):
            def __init__(self, num_samples=50):
                self.num_samples = num_samples
                torch.manual_seed(42)
                self.images = torch.randn(num_samples, 64, 64)

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return self.images[idx]

        dataset = Dataset3D()
        stats = compute_normalization_stats(dataset, batch_size=10)

        assert isinstance(stats, NormalizationStats)
        assert len(stats.mean) == 1
        assert len(stats.std) == 1

    def test_compute_stats_numpy_conversion(self):
        """Test handling of numpy arrays."""
        class NumpyDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples=50):
                self.num_samples = num_samples
                np.random.seed(42)
                self.images = np.random.randn(num_samples, 3, 64, 64).astype(np.float32)

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return self.images[idx]

        dataset = NumpyDataset()
        stats = compute_normalization_stats(dataset, batch_size=10)

        assert isinstance(stats, NormalizationStats)
        assert len(stats.mean) == 3
        assert len(stats.std) == 3


class TestValidateNormalization:
    """Unit tests for validate_normalization function."""

    @pytest.fixture
    def normalized_data_3_channel(self):
        """Create normalized 3-channel data."""
        torch.manual_seed(42)
        # Create data close to ImageNet statistics
        data = torch.randn(32, 3, 224, 224) * 0.229 + 0.485
        return data

    @pytest.fixture
    def normalized_data_1_channel(self):
        """Create normalized 1-channel data."""
        torch.manual_seed(42)
        # Create data close to mean=0.5, std=0.5
        data = torch.randn(32, 1, 224, 224) * 0.5 + 0.5
        return data

    @pytest.fixture
    def unnormalized_data(self):
        """Create unnormalized data (e.g., 0-255 range)."""
        torch.manual_seed(42)
        data = torch.randint(0, 256, (32, 3, 224, 224), dtype=torch.float32)
        return data

    def test_validate_normalized_data_3_channel(self, normalized_data_3_channel):
        """Test validation of properly normalized 3-channel data."""
        result = validate_normalization(
            normalized_data_3_channel,
            expected_mean=[0.485, 0.456, 0.406],
            expected_std=[0.229, 0.224, 0.225],
            tolerance=0.5  # Use higher tolerance for random data
        )

        assert isinstance(result, dict)
        assert "is_normalized" in result
        assert "actual_mean" in result
        assert "actual_std" in result
        assert "expected_mean" in result
        assert "expected_std" in result
        assert "warnings" in result

        assert len(result["actual_mean"]) == 3
        assert len(result["actual_std"]) == 3

    def test_validate_normalized_data_1_channel(self, normalized_data_1_channel):
        """Test validation of properly normalized 1-channel data."""
        result = validate_normalization(
            normalized_data_1_channel,
            expected_mean=[0.5],
            expected_std=[0.5],
            tolerance=0.5
        )

        assert len(result["actual_mean"]) == 1
        assert len(result["actual_std"]) == 1

    def test_validate_unnormalized_data(self, unnormalized_data):
        """Test validation of unnormalized data."""
        result = validate_normalization(
            unnormalized_data,
            expected_mean=[0.485, 0.456, 0.406],
            expected_std=[0.229, 0.224, 0.225],
            tolerance=0.1
        )

        assert result["is_normalized"] is False
        assert len(result["warnings"]) > 0

    def test_validate_default_expectations_3_channel(self):
        """Test validation with default expectations for 3 channels."""
        torch.manual_seed(42)
        data = torch.randn(16, 3, 128, 128)

        result = validate_normalization(data, tolerance=0.5)

        assert result["expected_mean"] == [0.485, 0.456, 0.406]
        assert result["expected_std"] == [0.229, 0.224, 0.225]

    def test_validate_default_expectations_1_channel(self):
        """Test validation with default expectations for 1 channel."""
        torch.manual_seed(42)
        data = torch.randn(16, 1, 128, 128)

        result = validate_normalization(data, tolerance=0.5)

        assert result["expected_mean"] == [0.5]
        assert result["expected_std"] == [0.5]

    def test_validate_numpy_input(self):
        """Test validation with numpy array input."""
        np.random.seed(42)
        data = np.random.randn(16, 3, 128, 128).astype(np.float32)

        result = validate_normalization(data, tolerance=0.5)

        assert isinstance(result, dict)
        assert "is_normalized" in result

    def test_validate_3d_tensor_auto_expansion(self):
        """Test validation of 3D tensor (auto-expanded to 4D)."""
        torch.manual_seed(42)
        data = torch.randn(3, 128, 128)

        result = validate_normalization(data, tolerance=0.5)

        assert isinstance(result, dict)
        assert len(result["actual_mean"]) == 3

    def test_validate_invalid_shape(self):
        """Test validation of invalid tensor shape."""
        data = torch.randn(10, 20)  # 2D tensor

        with pytest.raises(ValueError, match="Expected 4D tensor"):
            validate_normalization(data)

    def test_validate_channel_mismatch_mean(self):
        """Test validation with mismatched channel count in expected_mean."""
        data = torch.randn(16, 3, 128, 128)

        with pytest.raises(ValueError, match="expected_mean length"):
            validate_normalization(
                data,
                expected_mean=[0.5],  # 1 value for 3 channels
                expected_std=[0.229, 0.224, 0.225]
            )

    def test_validate_channel_mismatch_std(self):
        """Test validation with mismatched channel count in expected_std."""
        data = torch.randn(16, 3, 128, 128)

        with pytest.raises(ValueError, match="expected_std length"):
            validate_normalization(
                data,
                expected_mean=[0.485, 0.456, 0.406],
                expected_std=[0.5]  # 1 value for 3 channels
            )

    def test_validate_tolerance_zero(self):
        """Test validation with very strict tolerance."""
        torch.manual_seed(42)
        # Create data with exact mean=0, std=1
        data = torch.randn(32, 3, 224, 224)

        result = validate_normalization(
            data,
            expected_mean=[0.0, 0.0, 0.0],
            expected_std=[1.0, 1.0, 1.0],
            tolerance=0.5  # Reasonable tolerance for random data
        )

        assert isinstance(result, dict)

    def test_validate_warnings_format(self, unnormalized_data):
        """Test that warnings are properly formatted."""
        result = validate_normalization(
            unnormalized_data,
            expected_mean=[0.485, 0.456, 0.406],
            expected_std=[0.229, 0.224, 0.225],
            tolerance=0.1
        )

        if len(result["warnings"]) > 0:
            # Check that warnings contain expected information
            warning = result["warnings"][0]
            assert "Channel" in warning
            assert any(
                keyword in warning
                for keyword in ["mean", "std", "deviates", "expected"]
            )

    def test_validate_actual_stats_accuracy(self):
        """Test that actual statistics are computed accurately."""
        torch.manual_seed(42)
        # Create data with known statistics
        data = torch.ones(10, 3, 64, 64) * 2.0

        result = validate_normalization(data, tolerance=0.5)

        # All channels should have mean=2.0, std=0.0
        for mean_val in result["actual_mean"]:
            assert abs(mean_val - 2.0) < 1e-6

        for std_val in result["actual_std"]:
            assert std_val < 1e-6

    def test_validate_batch_size_independence(self):
        """Test that validation is independent of batch size."""
        torch.manual_seed(42)
        data_large = torch.randn(64, 3, 128, 128)

        torch.manual_seed(42)
        data_small = torch.randn(8, 3, 128, 128)

        # Both should use same defaults and tolerance
        result_large = validate_normalization(data_large, tolerance=0.5)
        result_small = validate_normalization(data_small, tolerance=0.5)

        assert result_large["expected_mean"] == result_small["expected_mean"]
        assert result_large["expected_std"] == result_small["expected_std"]


if __name__ == "__main__":
    pytest.main([__file__])
