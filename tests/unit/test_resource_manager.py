#
# test_resource_manager.py
# mammography-pipelines
#
# Tests for ResourceManager in tuning/resource_manager.py
#
# Medical Disclaimer: This is an educational research project.
# Not intended for clinical use or medical diagnosis.
#
from unittest.mock import MagicMock, patch

import pytest
import torch

from mammography.tuning.resource_manager import ResourceManager


class TestResourceManagerInit:
    """Test ResourceManager initialization and device detection."""

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_init_cpu_device(self, mock_mps, mock_cuda):
        """Test initialization defaults to CPU when no GPU available."""
        rm = ResourceManager()
        assert rm.device.type == "cpu"
        assert rm.has_gpu is False
        assert rm.gpu_memory_gb == 0.0
        assert rm.cpu_count >= 1

    @patch("torch.cuda.is_available", return_value=True)
    def test_init_cuda_device(self, mock_cuda):
        """Test initialization detects CUDA device."""
        # Mock CUDA properties
        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024 ** 3)  # 16 GB
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager()
            assert rm.device.type == "cuda"
            assert rm.has_gpu is True
            assert rm.gpu_memory_gb == 16.0

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_init_mps_device(self, mock_mps, mock_cuda):
        """Test initialization detects MPS device (Apple Silicon)."""
        rm = ResourceManager()
        assert rm.device.type == "mps"
        assert rm.has_gpu is True
        assert rm.gpu_memory_gb == 8.0  # MPS default estimate

    def test_init_explicit_cpu_override(self):
        """Test explicit CPU device override."""
        rm = ResourceManager(device="cpu")
        assert rm.device.type == "cpu"
        assert rm.has_gpu is False
        assert rm.gpu_memory_gb == 0.0

    @patch("torch.cuda.is_available", return_value=True)
    def test_init_explicit_cuda_override(self, mock_cuda):
        """Test explicit CUDA device override."""
        # Mock CUDA properties
        mock_props = MagicMock()
        mock_props.total_memory = 24 * (1024 ** 3)  # 24 GB
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            assert rm.device.type == "cuda"
            assert rm.has_gpu is True

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_init_explicit_mps_override(self, mock_mps, mock_cuda):
        """Test explicit MPS device override."""
        rm = ResourceManager(device="mps")
        assert rm.device.type == "mps"
        assert rm.has_gpu is True
        assert rm.gpu_memory_gb == 8.0

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_init_invalid_device_override(self, mock_mps, mock_cuda):
        """Test that invalid device override falls back to auto-detection."""
        rm = ResourceManager(device="invalid_device")
        # Should fall back to CPU (since mocked no GPU available)
        assert rm.device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_init_device_case_insensitive(self, mock_mps, mock_cuda):
        """Test that device override is case-insensitive."""
        rm = ResourceManager(device="CPU")
        assert rm.device.type == "cpu"


class TestGPUMemoryDetection:
    """Test GPU memory detection for different device types."""

    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_memory_detection(self, mock_cuda):
        """Test CUDA GPU memory detection."""
        mock_props = MagicMock()
        mock_props.total_memory = 12 * (1024 ** 3)  # 12 GB
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            assert rm.gpu_memory_gb == 12.0

    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_memory_detection_failure(self, mock_cuda):
        """Test CUDA memory detection gracefully handles exceptions."""
        with patch("torch.cuda.get_device_properties", side_effect=RuntimeError("No GPU")):
            rm = ResourceManager(device="cuda")
            assert rm.gpu_memory_gb == 0.0  # Fallback on error

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_mps_memory_default(self, mock_mps, mock_cuda):
        """Test MPS device returns default 8 GB estimate."""
        rm = ResourceManager(device="mps")
        assert rm.gpu_memory_gb == 8.0

    def test_cpu_memory_zero(self):
        """Test CPU device returns 0 GB GPU memory."""
        rm = ResourceManager(device="cpu")
        assert rm.gpu_memory_gb == 0.0


class TestCPUCountDetection:
    """Test CPU core count detection."""

    def test_cpu_count_positive(self):
        """Test that CPU count is at least 1."""
        rm = ResourceManager(device="cpu")
        assert rm.cpu_count >= 1

    @patch("os.cpu_count", return_value=None)
    def test_cpu_count_none_fallback(self, mock_cpu_count):
        """Test that None from os.cpu_count() falls back to 1."""
        rm = ResourceManager(device="cpu")
        assert rm.cpu_count == 1

    @patch("os.cpu_count", return_value=8)
    def test_cpu_count_normal(self, mock_cpu_count):
        """Test normal CPU count detection."""
        rm = ResourceManager(device="cpu")
        assert rm.cpu_count == 8


class TestMaxBatchSize:
    """Test maximum batch size recommendations based on GPU memory."""

    def test_max_batch_size_cpu(self):
        """Test max batch size for CPU-only mode."""
        rm = ResourceManager(device="cpu")
        assert rm.get_max_batch_size() == 8

    @patch("torch.cuda.is_available", return_value=True)
    def test_max_batch_size_24gb_gpu(self, mock_cuda):
        """Test max batch size for 24+ GB GPU."""
        mock_props = MagicMock()
        mock_props.total_memory = 24 * (1024 ** 3)
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            assert rm.get_max_batch_size() == 64

    @patch("torch.cuda.is_available", return_value=True)
    def test_max_batch_size_16gb_gpu(self, mock_cuda):
        """Test max batch size for 16-24 GB GPU."""
        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024 ** 3)
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            assert rm.get_max_batch_size() == 48

    @patch("torch.cuda.is_available", return_value=True)
    def test_max_batch_size_12gb_gpu(self, mock_cuda):
        """Test max batch size for 12-16 GB GPU."""
        mock_props = MagicMock()
        mock_props.total_memory = 12 * (1024 ** 3)
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            assert rm.get_max_batch_size() == 32

    @patch("torch.cuda.is_available", return_value=True)
    def test_max_batch_size_8gb_gpu(self, mock_cuda):
        """Test max batch size for 8-12 GB GPU."""
        mock_props = MagicMock()
        mock_props.total_memory = 8 * (1024 ** 3)
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            assert rm.get_max_batch_size() == 24

    @patch("torch.cuda.is_available", return_value=True)
    def test_max_batch_size_6gb_gpu(self, mock_cuda):
        """Test max batch size for 6-8 GB GPU."""
        mock_props = MagicMock()
        mock_props.total_memory = 6 * (1024 ** 3)
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            assert rm.get_max_batch_size() == 16

    @patch("torch.cuda.is_available", return_value=True)
    def test_max_batch_size_small_gpu(self, mock_cuda):
        """Test max batch size for <6 GB GPU."""
        mock_props = MagicMock()
        mock_props.total_memory = 4 * (1024 ** 3)
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            assert rm.get_max_batch_size() == 8

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_max_batch_size_mps(self, mock_mps, mock_cuda):
        """Test max batch size for MPS device (8 GB default)."""
        rm = ResourceManager(device="mps")
        assert rm.get_max_batch_size() == 24


class TestMaxNumWorkers:
    """Test maximum DataLoader workers recommendation."""

    @patch("os.cpu_count", return_value=4)
    def test_max_num_workers_4_cores(self, mock_cpu_count):
        """Test max workers with 4 CPU cores."""
        rm = ResourceManager(device="cpu")
        assert rm.get_max_num_workers() == 4

    @patch("os.cpu_count", return_value=16)
    def test_max_num_workers_capped_at_8(self, mock_cpu_count):
        """Test that max workers is capped at 8."""
        rm = ResourceManager(device="cpu")
        assert rm.get_max_num_workers() == 8

    @patch("os.cpu_count", return_value=1)
    def test_max_num_workers_single_core(self, mock_cpu_count):
        """Test max workers with single CPU core."""
        rm = ResourceManager(device="cpu")
        assert rm.get_max_num_workers() == 1

    @patch("os.cpu_count", return_value=8)
    def test_max_num_workers_exactly_8_cores(self, mock_cpu_count):
        """Test max workers with exactly 8 CPU cores."""
        rm = ResourceManager(device="cpu")
        assert rm.get_max_num_workers() == 8


class TestResourceSummary:
    """Test resource summary dictionary generation."""

    def test_resource_summary_keys(self):
        """Test that resource summary contains all expected keys."""
        rm = ResourceManager(device="cpu")
        summary = rm.get_resource_summary()

        expected_keys = {
            "device",
            "gpu_memory_gb",
            "cpu_count",
            "has_gpu",
            "max_batch_size",
            "max_num_workers",
        }
        assert set(summary.keys()) == expected_keys

    def test_resource_summary_types(self):
        """Test that resource summary values have correct types."""
        rm = ResourceManager(device="cpu")
        summary = rm.get_resource_summary()

        assert isinstance(summary["device"], str)
        assert isinstance(summary["gpu_memory_gb"], float)
        assert isinstance(summary["cpu_count"], int)
        assert isinstance(summary["has_gpu"], bool)
        assert isinstance(summary["max_batch_size"], int)
        assert isinstance(summary["max_num_workers"], int)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("os.cpu_count", return_value=8)
    def test_resource_summary_values_gpu(self, mock_cpu_count, mock_cuda):
        """Test resource summary values for GPU system."""
        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024 ** 3)
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            summary = rm.get_resource_summary()

            assert summary["device"] == "cuda"
            assert summary["gpu_memory_gb"] == 16.0
            assert summary["cpu_count"] == 8
            assert summary["has_gpu"] is True
            assert summary["max_batch_size"] == 48
            assert summary["max_num_workers"] == 8

    @patch("os.cpu_count", return_value=4)
    def test_resource_summary_values_cpu(self, mock_cpu_count):
        """Test resource summary values for CPU-only system."""
        rm = ResourceManager(device="cpu")
        summary = rm.get_resource_summary()

        assert summary["device"] == "cpu"
        assert summary["gpu_memory_gb"] == 0.0
        assert summary["cpu_count"] == 4
        assert summary["has_gpu"] is False
        assert summary["max_batch_size"] == 8
        assert summary["max_num_workers"] == 4


class TestArchitectureFiltering:
    """Test architecture filtering based on resource constraints."""

    def test_filter_architectures_cpu_only(self):
        """Test that CPU-only mode filters out large models."""
        rm = ResourceManager(device="cpu")
        architectures = ["efficientnet_b0", "resnet50", "vit_b_16", "vit_l_16"]

        filtered = rm.filter_architectures(architectures)
        # CPU mode should only allow lightweight models
        assert set(filtered) == {"efficientnet_b0", "resnet50"}

    def test_filter_architectures_empty_list(self):
        """Test architecture filtering with empty list."""
        rm = ResourceManager(device="cpu")
        filtered = rm.filter_architectures([])
        assert filtered == []

    @patch("torch.cuda.is_available", return_value=True)
    def test_filter_architectures_small_gpu(self, mock_cuda):
        """Test filtering with small GPU (4 GB) - only smallest models fit."""
        mock_props = MagicMock()
        mock_props.total_memory = 4 * (1024 ** 3)  # 4 GB
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            architectures = ["efficientnet_b0", "resnet50", "vit_b_16", "vit_l_16"]

            filtered = rm.filter_architectures(architectures)
            # 4 GB should only fit efficientnet_b0 (requires 4*1.25=5 GB threshold)
            assert filtered == ["efficientnet_b0"]

    @patch("torch.cuda.is_available", return_value=True)
    def test_filter_architectures_medium_gpu(self, mock_cuda):
        """Test filtering with medium GPU (8 GB) - resnet and smaller ViTs fit."""
        mock_props = MagicMock()
        mock_props.total_memory = 8 * (1024 ** 3)  # 8 GB
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            architectures = ["efficientnet_b0", "resnet50", "vit_b_32", "vit_b_16", "vit_l_16"]

            filtered = rm.filter_architectures(architectures)
            # 8 GB should fit: efficientnet_b0, resnet50 (needs 7.5GB), possibly vit_b_32
            # vit_b_32 needs 8*1.25=10GB threshold, so won't fit
            # resnet50 needs 6*1.25=7.5GB, should fit
            assert "efficientnet_b0" in filtered
            assert "resnet50" in filtered
            assert "vit_l_16" not in filtered  # Too large

    @patch("torch.cuda.is_available", return_value=True)
    def test_filter_architectures_large_gpu(self, mock_cuda):
        """Test filtering with large GPU (24 GB) - all models fit."""
        mock_props = MagicMock()
        mock_props.total_memory = 24 * (1024 ** 3)  # 24 GB
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            architectures = ["efficientnet_b0", "resnet50", "vit_b_16", "vit_b_32", "vit_l_16"]

            filtered = rm.filter_architectures(architectures)
            # 24 GB should fit all models
            assert set(filtered) == set(architectures)

    @patch("torch.cuda.is_available", return_value=True)
    def test_filter_architectures_unknown_arch(self, mock_cuda):
        """Test filtering with unknown architecture defaults to 8 GB requirement."""
        mock_props = MagicMock()
        mock_props.total_memory = 12 * (1024 ** 3)  # 12 GB
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            architectures = ["efficientnet_b0", "unknown_model", "vit_l_16"]

            filtered = rm.filter_architectures(architectures)
            # unknown_model should use default 8 GB requirement (10 GB threshold)
            # 12 GB should fit efficientnet_b0 and unknown_model
            assert "efficientnet_b0" in filtered
            assert "unknown_model" in filtered
            assert "vit_l_16" not in filtered  # Requires 20 GB threshold

    @patch("torch.cuda.is_available", return_value=True)
    def test_filter_architectures_fallback_to_smallest(self, mock_cuda):
        """Test that if all architectures are filtered, smallest is kept as fallback."""
        mock_props = MagicMock()
        mock_props.total_memory = 2 * (1024 ** 3)  # 2 GB - very small
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            # All models require more than 2 GB
            architectures = ["efficientnet_b0", "resnet50", "vit_b_16"]

            filtered = rm.filter_architectures(architectures)
            # Should fallback to efficientnet_b0 as smallest
            assert filtered == ["efficientnet_b0"]

    @patch("torch.cuda.is_available", return_value=True)
    def test_filter_architectures_fallback_without_efficientnet(self, mock_cuda):
        """Test fallback when efficientnet_b0 is not in the list."""
        mock_props = MagicMock()
        mock_props.total_memory = 2 * (1024 ** 3)  # 2 GB - very small
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            # All models require more than 2 GB, and no efficientnet_b0
            architectures = ["resnet50", "vit_b_16"]

            filtered = rm.filter_architectures(architectures)
            # Should fallback to first architecture
            assert filtered == ["resnet50"]

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_filter_architectures_mps_device(self, mock_mps, mock_cuda):
        """Test filtering on MPS device (Apple Silicon) with 8 GB default."""
        rm = ResourceManager(device="mps")
        architectures = ["efficientnet_b0", "resnet50", "vit_b_16", "vit_l_16"]

        filtered = rm.filter_architectures(architectures)
        # MPS has 8 GB, should fit smaller models
        assert "efficientnet_b0" in filtered
        assert "resnet50" in filtered
        # vit_l_16 requires 20 GB threshold, won't fit
        assert "vit_l_16" not in filtered


class TestConfigValidation:
    """Test configuration validation based on resource limits."""

    @patch("os.cpu_count", return_value=8)
    def test_validate_config_valid(self, mock_cpu_count):
        """Test validation passes for valid configuration."""
        rm = ResourceManager(device="cpu")
        # CPU mode: max_batch=8, max_workers=8
        assert rm.validate_config(batch_size=8, num_workers=4) is True

    @patch("os.cpu_count", return_value=8)
    def test_validate_config_batch_size_exceeds(self, mock_cpu_count):
        """Test validation fails when batch size exceeds maximum."""
        rm = ResourceManager(device="cpu")
        # CPU mode: max_batch=8
        assert rm.validate_config(batch_size=16, num_workers=4) is False

    @patch("os.cpu_count", return_value=4)
    def test_validate_config_num_workers_exceeds(self, mock_cpu_count):
        """Test validation fails when num_workers exceeds maximum."""
        rm = ResourceManager(device="cpu")
        # max_workers=4
        assert rm.validate_config(batch_size=8, num_workers=8) is False

    @patch("torch.cuda.is_available", return_value=True)
    @patch("os.cpu_count", return_value=8)
    def test_validate_config_both_exceed(self, mock_cpu_count, mock_cuda):
        """Test validation fails when both batch size and num_workers exceed limits."""
        mock_props = MagicMock()
        mock_props.total_memory = 4 * (1024 ** 3)  # 4 GB - small GPU
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            # max_batch=8, max_workers=8
            assert rm.validate_config(batch_size=32, num_workers=16) is False

    @patch("torch.cuda.is_available", return_value=True)
    @patch("os.cpu_count", return_value=16)
    def test_validate_config_valid_gpu(self, mock_cpu_count, mock_cuda):
        """Test validation passes for valid GPU configuration."""
        mock_props = MagicMock()
        mock_props.total_memory = 24 * (1024 ** 3)  # 24 GB GPU
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            rm = ResourceManager(device="cuda")
            # max_batch=64, max_workers=8 (capped)
            assert rm.validate_config(batch_size=32, num_workers=8) is True

    @patch("os.cpu_count", return_value=8)
    def test_validate_config_boundary_valid(self, mock_cpu_count):
        """Test validation at exact boundary (should pass)."""
        rm = ResourceManager(device="cpu")
        # max_batch=8, max_workers=8
        assert rm.validate_config(batch_size=8, num_workers=8) is True

    @patch("os.cpu_count", return_value=8)
    def test_validate_config_boundary_invalid(self, mock_cpu_count):
        """Test validation just over boundary (should fail)."""
        rm = ResourceManager(device="cpu")
        # max_batch=8
        assert rm.validate_config(batch_size=9, num_workers=8) is False
