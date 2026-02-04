"""
Unit tests for device detection utilities.

Tests cover:
- Device detection (CPU/CUDA/MPS)
- Device configuration retrieval
- Device optimization
- Memory tracking
- Device-agnostic tensor operations
"""

import pytest
import torch
import platform
from mammography.utils.device_detection import (
    DeviceDetector,
    get_optimal_device,
    get_device_config,
    print_device_status,
)


class TestDeviceDetector:
    """Test the DeviceDetector class."""

    def test_device_detector_initialization(self):
        """Test that DeviceDetector initializes correctly."""
        detector = DeviceDetector()
        assert detector.system_info is not None
        assert detector.available_devices is not None
        assert detector.best_device is not None
        assert "platform" in detector.system_info
        assert "cpu" in detector.available_devices
        assert detector.available_devices["cpu"] is True

    def test_get_device_returns_torch_device(self):
        """Test that get_device returns a torch.device object."""
        detector = DeviceDetector()
        device = detector.get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]

    def test_get_device_config_returns_dict(self):
        """Test that get_device_config returns a configuration dictionary."""
        detector = DeviceDetector()
        config = detector.get_device_config()
        assert isinstance(config, dict)
        assert "batch_size" in config
        assert "mixed_precision" in config
        assert "num_workers" in config
        assert "pin_memory" in config

    def test_cpu_always_available(self):
        """Test that CPU device is always detected as available."""
        detector = DeviceDetector()
        assert detector.available_devices["cpu"] is True

    def test_device_selection_priority(self):
        """Test device selection follows priority: CUDA > MPS > CPU."""
        detector = DeviceDetector()
        best_device = detector.best_device

        # If CUDA is available, it should be selected
        if detector.available_devices.get("cuda", False):
            assert best_device == "cuda"
        # If MPS is available (and CUDA isn't), it should be selected
        elif detector.available_devices.get("mps", False):
            assert best_device == "mps"
        # Otherwise, CPU should be selected
        else:
            assert best_device == "cpu"

    def test_cuda_detection(self):
        """Test CUDA detection matches torch.cuda.is_available()."""
        detector = DeviceDetector()
        cuda_available = torch.cuda.is_available()
        assert detector.available_devices["cuda"] == cuda_available

    def test_mps_detection(self):
        """Test MPS detection (Apple Silicon)."""
        detector = DeviceDetector()
        if hasattr(torch.backends, "mps"):
            mps_available = torch.backends.mps.is_available()
            assert detector.available_devices["mps"] == mps_available
        else:
            assert detector.available_devices["mps"] is False

    def test_optimize_for_device_cpu(self):
        """Test CPU-specific optimizations."""
        detector = DeviceDetector()
        if detector.best_device == "cpu":
            # Should not raise any errors
            detector.optimize_for_device()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_optimize_for_device_cuda(self):
        """Test CUDA-specific optimizations."""
        # Force CUDA device
        detector = DeviceDetector()
        if detector.best_device == "cuda":
            detector.optimize_for_device()
            # Check CUDA optimizations were applied
            assert torch.backends.cudnn.benchmark is True

    def test_get_memory_info_structure(self):
        """Test that get_memory_info returns expected structure."""
        detector = DeviceDetector()
        memory_info = detector.get_memory_info()
        assert isinstance(memory_info, dict)
        assert "device" in memory_info
        assert "total_memory" in memory_info
        assert "allocated_memory" in memory_info
        assert "cached_memory" in memory_info

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_memory_info_cuda(self):
        """Test CUDA memory information retrieval."""
        detector = DeviceDetector()
        if detector.best_device == "cuda":
            memory_info = detector.get_memory_info()
            assert memory_info["device"] == "cuda"
            assert memory_info["total_memory"] > 0

    def test_print_device_status_no_errors(self):
        """Test that print_device_status executes without errors."""
        detector = DeviceDetector()
        # Should not raise any errors
        detector.print_device_status()

    def test_system_info_completeness(self):
        """Test that system info contains all required fields."""
        detector = DeviceDetector()
        required_fields = ["platform", "architecture", "processor", "python_version", "torch_version"]
        for field in required_fields:
            assert field in detector.system_info
            assert detector.system_info[field] is not None


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    def test_get_optimal_device(self):
        """Test get_optimal_device function."""
        device = get_optimal_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]

    def test_get_device_config(self):
        """Test get_device_config function."""
        config = get_device_config()
        assert isinstance(config, dict)
        assert "batch_size" in config
        assert config["batch_size"] > 0
        assert isinstance(config["mixed_precision"], bool)
        assert isinstance(config["num_workers"], int)
        assert isinstance(config["pin_memory"], bool)

    def test_print_device_status_function(self):
        """Test print_device_status function."""
        # Should not raise any errors
        print_device_status()


class TestDeviceAgnosticOperations:
    """Test device-agnostic tensor operations."""

    def test_tensor_creation_on_detected_device(self):
        """Test creating tensors directly on detected device."""
        device = get_optimal_device()
        tensor = torch.randn(10, 10, device=device)
        assert tensor.device.type == device.type

    def test_tensor_movement_to_device(self):
        """Test moving tensors between devices."""
        device = get_optimal_device()
        # Create on CPU
        tensor_cpu = torch.randn(10, 10)
        assert tensor_cpu.device.type == "cpu"

        # Move to detected device
        tensor_device = tensor_cpu.to(device)
        assert tensor_device.device.type == device.type

    def test_model_to_device(self):
        """Test moving a model to detected device."""
        device = get_optimal_device()
        model = torch.nn.Linear(10, 5)
        model = model.to(device)

        # Check model parameters are on correct device
        for param in model.parameters():
            assert param.device.type == device.type

    def test_operations_on_same_device(self):
        """Test that operations work when tensors are on same device."""
        device = get_optimal_device()
        a = torch.randn(10, device=device)
        b = torch.randn(10, device=device)

        # Should not raise device mismatch error
        c = a + b
        assert c.device.type == device.type

    def test_label_tensor_conversion(self):
        """Test converting labels to tensors on correct device."""
        device = get_optimal_device()

        # Simulate labels from dataloader (list or numpy array)
        labels = [0, 1, 2, 3]

        # Convert to tensor on device
        y_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        assert y_tensor.device.type == device.type
        assert y_tensor.dtype == torch.long

    def test_mixed_precision_compatibility(self):
        """Test that mixed precision is compatible with detected device."""
        device = get_optimal_device()
        config = get_device_config()

        # If mixed precision is recommended, test it works
        if config["mixed_precision"]:
            with torch.autocast(device_type=device.type, enabled=True):
                model = torch.nn.Linear(10, 5).to(device)
                x = torch.randn(2, 10, device=device)
                output = model(x)
                assert output.device.type == device.type


class TestDeviceConfiguration:
    """Test device-specific configuration recommendations."""

    def test_cpu_config_values(self):
        """Test CPU configuration has appropriate values."""
        detector = DeviceDetector()
        if detector.best_device == "cpu":
            config = detector.get_device_config()
            assert config["batch_size"] <= 8  # CPU should have smaller batch sizes
            assert config["mixed_precision"] is False  # AMP less beneficial on CPU
            assert config["pin_memory"] is False  # No GPU to pin to
            assert config["gpu_memory_limit"] == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_config_values(self):
        """Test CUDA configuration has appropriate values."""
        detector = DeviceDetector()
        if detector.best_device == "cuda":
            config = detector.get_device_config()
            assert config["batch_size"] >= 8  # CUDA can handle larger batches
            assert config["mixed_precision"] is True  # AMP beneficial on CUDA
            assert config["pin_memory"] is True  # Pin memory for faster transfers
            assert config["gpu_memory_limit"] > 0

    @pytest.mark.skipif(
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_mps_config_values(self):
        """Test MPS (Apple Silicon) configuration has appropriate values."""
        detector = DeviceDetector()
        if detector.best_device == "mps":
            config = detector.get_device_config()
            assert config["batch_size"] >= 4
            assert config["mixed_precision"] is False  # AMP not supported on MPS
            assert config["pin_memory"] is False


class TestErrorHandling:
    """Test error handling in device detection."""

    def test_no_crash_on_missing_cuda(self):
        """Test that missing CUDA doesn't crash the detector."""
        detector = DeviceDetector()
        # Should not raise any errors even if CUDA is not available
        assert detector.best_device in ["cpu", "cuda", "mps"]

    def test_graceful_fallback_to_cpu(self):
        """Test graceful fallback to CPU when accelerators unavailable."""
        detector = DeviceDetector()
        # CPU should always be available as fallback
        assert detector.available_devices["cpu"] is True
        # Best device should be valid
        assert detector.best_device in detector.available_devices.keys()
