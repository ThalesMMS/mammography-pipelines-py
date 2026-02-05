"""
Unit tests for smart defaults utilities.

Tests cover:
- SmartDefaults class initialization
- Batch size recommendations
- Worker count recommendations
- Cache mode selection
- Epoch recommendations
- Learning rate scaling
- Training and inference defaults
- Dataset-aware configuration
- Comprehensive defaults aggregation
"""

import pytest
from unittest.mock import Mock, patch
from mammography.utils.smart_defaults import SmartDefaults, get_smart_defaults
from mammography.utils.device_detection import DeviceDetector


class TestSmartDefaults:
    """Test the SmartDefaults class."""

    def test_smart_defaults_initialization(self):
        """Test that SmartDefaults initializes correctly."""
        defaults = SmartDefaults()
        assert defaults.device_detector is not None
        assert defaults.device_type is not None
        assert defaults.device_config is not None
        assert defaults.system_memory_gb > 0
        assert defaults.device_type in ["cpu", "cuda", "mps"]

    def test_initialization_with_custom_detector(self):
        """Test initialization with a custom DeviceDetector instance."""
        detector = DeviceDetector()
        defaults = SmartDefaults(device_detector=detector)
        assert defaults.device_detector is detector
        assert defaults.device_type == detector.best_device

    def test_initialization_with_dataset_format(self):
        """Test initialization with DatasetFormat for dataset-aware defaults."""
        # Mock DatasetFormat
        mock_format = Mock()
        mock_format.dataset_type = "archive"
        mock_format.image_format = "dicom"
        mock_format.image_count = 1000
        mock_format.warnings = []
        mock_format.suggestions = []

        defaults = SmartDefaults(dataset_format=mock_format)
        assert defaults.dataset_format is mock_format
        assert defaults.dataset_format.dataset_type == "archive"

    def test_system_memory_detection(self):
        """Test that system memory is detected or estimated."""
        defaults = SmartDefaults()
        memory_gb = defaults.system_memory_gb
        assert isinstance(memory_gb, float)
        assert memory_gb > 0
        # Should be a reasonable value (at least 4GB, typically 8GB+)
        assert memory_gb >= 4.0


class TestBatchSizeRecommendations:
    """Test batch size recommendation logic."""

    def test_get_batch_size_train_default(self):
        """Test default batch size for training."""
        defaults = SmartDefaults()
        batch_size = defaults.get_batch_size(task="train")
        assert isinstance(batch_size, int)
        assert batch_size >= 1
        # Should be a power of 2
        assert batch_size & (batch_size - 1) == 0

    def test_get_batch_size_inference_larger(self):
        """Test that inference batch size is larger than training."""
        defaults = SmartDefaults()
        train_batch = defaults.get_batch_size(task="train")
        inference_batch = defaults.get_batch_size(task="inference")
        # Inference should typically be larger (no gradients)
        assert inference_batch >= train_batch

    def test_get_batch_size_embed(self):
        """Test batch size for embedding extraction."""
        defaults = SmartDefaults()
        embed_batch = defaults.get_batch_size(task="embed")
        assert isinstance(embed_batch, int)
        assert embed_batch >= 1

    def test_get_batch_size_large_image_adjustment(self):
        """Test that larger images result in smaller batch sizes."""
        defaults = SmartDefaults()
        small_image_batch = defaults.get_batch_size(image_size=224)
        large_image_batch = defaults.get_batch_size(image_size=512)
        # Larger images should have smaller or equal batch size
        assert large_image_batch <= small_image_batch

    def test_get_batch_size_small_dataset_adjustment(self):
        """Test batch size adjustment for small datasets."""
        defaults = SmartDefaults()
        small_dataset_batch = defaults.get_batch_size(dataset_size=100)
        # Should have reasonable batch size for small dataset
        assert small_dataset_batch >= 4
        assert small_dataset_batch <= 100

    def test_get_batch_size_power_of_2(self):
        """Test that returned batch sizes are powers of 2."""
        defaults = SmartDefaults()
        for task in ["train", "inference", "embed"]:
            batch_size = defaults.get_batch_size(task=task)
            # Check if power of 2
            assert batch_size & (batch_size - 1) == 0

    def test_get_batch_size_device_limits(self):
        """Test device-specific batch size limits."""
        defaults = SmartDefaults()
        batch_size = defaults.get_batch_size()

        if defaults.device_type == "cpu":
            # CPU should have reasonable limits
            assert batch_size <= 8
        elif defaults.device_type == "mps":
            # MPS should have reasonable limits
            assert batch_size <= 16


class TestNumWorkersRecommendations:
    """Test number of workers recommendation logic."""

    def test_get_num_workers_train(self):
        """Test worker count for training."""
        defaults = SmartDefaults()
        num_workers = defaults.get_num_workers(task="train")
        assert isinstance(num_workers, int)
        assert num_workers >= 0

    def test_get_num_workers_inference(self):
        """Test worker count for inference."""
        defaults = SmartDefaults()
        train_workers = defaults.get_num_workers(task="train")
        inference_workers = defaults.get_num_workers(task="inference")
        # Inference typically uses fewer workers (less I/O bound)
        assert inference_workers >= 0
        assert isinstance(inference_workers, int)

    def test_get_num_workers_cpu_adjustment(self):
        """Test that CPU device uses appropriate worker count."""
        defaults = SmartDefaults()
        num_workers = defaults.get_num_workers()
        # Should be non-negative
        assert num_workers >= 0
        # Should be reasonable for the system
        if defaults.device_type == "cpu":
            import os
            cpu_count = os.cpu_count() or 4
            # CPU can use more workers, but not more than CPUs available
            assert num_workers <= cpu_count


class TestCacheModeRecommendations:
    """Test cache mode recommendation logic."""

    def test_get_cache_mode_no_dataset_size(self):
        """Test cache mode when dataset size is unknown."""
        defaults = SmartDefaults()
        cache_mode = defaults.get_cache_mode()
        assert cache_mode == "auto"

    def test_get_cache_mode_small_dataset(self):
        """Test cache mode for small datasets."""
        defaults = SmartDefaults()
        cache_mode = defaults.get_cache_mode(dataset_size=500)
        # Small datasets should use memory cache
        assert cache_mode == "memory"

    def test_get_cache_mode_medium_dataset(self):
        """Test cache mode for medium datasets."""
        defaults = SmartDefaults()
        cache_mode = defaults.get_cache_mode(dataset_size=5000)
        # Medium datasets should use disk cache
        assert cache_mode == "disk"

    def test_get_cache_mode_large_dataset(self):
        """Test cache mode for large datasets."""
        defaults = SmartDefaults()
        cache_mode = defaults.get_cache_mode(dataset_size=20000)
        # Large datasets should use tensor-disk
        assert cache_mode == "tensor-disk"

    def test_get_cache_mode_with_dataset_format(self):
        """Test cache mode uses dataset format info when available."""
        mock_format = Mock()
        mock_format.dataset_type = "archive"
        mock_format.image_format = "dicom"
        mock_format.image_count = 500
        mock_format.warnings = []
        mock_format.suggestions = []

        defaults = SmartDefaults(dataset_format=mock_format)
        cache_mode = defaults.get_cache_mode()
        # Should use dataset_format.image_count
        assert cache_mode in ["memory", "disk", "tensor-disk", "auto"]


class TestEpochRecommendations:
    """Test epoch recommendation logic."""

    def test_get_epochs_no_dataset_size(self):
        """Test epoch recommendation when dataset size is unknown."""
        defaults = SmartDefaults()
        epochs = defaults.get_epochs()
        assert epochs == 30  # Conservative default

    def test_get_epochs_very_small_dataset(self):
        """Test epoch recommendation for very small datasets."""
        defaults = SmartDefaults()
        epochs = defaults.get_epochs(dataset_size=300)
        # Very small datasets need more epochs
        assert epochs == 50

    def test_get_epochs_small_dataset(self):
        """Test epoch recommendation for small datasets."""
        defaults = SmartDefaults()
        epochs = defaults.get_epochs(dataset_size=1000)
        # Small datasets need moderate epochs
        assert epochs == 30

    def test_get_epochs_medium_dataset(self):
        """Test epoch recommendation for medium datasets."""
        defaults = SmartDefaults()
        epochs = defaults.get_epochs(dataset_size=3000)
        # Medium datasets need fewer epochs
        assert epochs == 20

    def test_get_epochs_large_dataset(self):
        """Test epoch recommendation for large datasets."""
        defaults = SmartDefaults()
        epochs = defaults.get_epochs(dataset_size=10000)
        # Large datasets need even fewer epochs
        assert epochs == 15


class TestConfigurationDefaults:
    """Test configuration default getters."""

    def test_get_mixed_precision(self):
        """Test mixed precision recommendation."""
        defaults = SmartDefaults()
        mixed_precision = defaults.get_mixed_precision()
        assert isinstance(mixed_precision, bool)
        # Should match device config
        assert mixed_precision == defaults.device_config.get("mixed_precision", False)

    def test_get_pin_memory(self):
        """Test pin memory recommendation."""
        defaults = SmartDefaults()
        pin_memory = defaults.get_pin_memory()
        assert isinstance(pin_memory, bool)
        # Should match device config
        assert pin_memory == defaults.device_config.get("pin_memory", False)

    def test_get_learning_rate_default(self):
        """Test default learning rate calculation."""
        defaults = SmartDefaults()
        lr = defaults.get_learning_rate()
        assert isinstance(lr, float)
        assert lr > 0
        # Should be in reasonable range
        assert 1e-5 <= lr <= 1e-2

    def test_get_learning_rate_batch_size_scaling(self):
        """Test learning rate scales with batch size."""
        defaults = SmartDefaults()
        lr_small = defaults.get_learning_rate(batch_size=8)
        lr_large = defaults.get_learning_rate(batch_size=32)
        # Larger batch size should have larger learning rate (linear scaling)
        assert lr_large > lr_small

    def test_get_learning_rate_clamping(self):
        """Test learning rate is clamped to reasonable range."""
        defaults = SmartDefaults()
        # Even with extreme batch sizes, should stay in range
        lr_tiny = defaults.get_learning_rate(batch_size=1)
        lr_huge = defaults.get_learning_rate(batch_size=256)
        assert 1e-5 <= lr_tiny <= 1e-2
        assert 1e-5 <= lr_huge <= 1e-2


class TestTrainingDefaults:
    """Test training defaults aggregation."""

    def test_get_training_defaults_structure(self):
        """Test that training defaults returns expected structure."""
        defaults = SmartDefaults()
        config = defaults.get_training_defaults()
        assert isinstance(config, dict)
        # Check all expected keys are present
        required_keys = [
            "device",
            "batch_size",
            "num_workers",
            "epochs",
            "lr",
            "mixed_precision",
            "pin_memory",
            "cache_mode",
        ]
        for key in required_keys:
            assert key in config

    def test_get_training_defaults_values(self):
        """Test that training defaults have valid values."""
        defaults = SmartDefaults()
        config = defaults.get_training_defaults(dataset_size=1000)
        assert config["device"] in ["cpu", "cuda", "mps"]
        assert isinstance(config["batch_size"], int)
        assert config["batch_size"] >= 1
        assert isinstance(config["num_workers"], int)
        assert config["num_workers"] >= 0
        assert isinstance(config["epochs"], int)
        assert config["epochs"] > 0
        assert isinstance(config["lr"], float)
        assert config["lr"] > 0
        assert isinstance(config["mixed_precision"], bool)
        assert isinstance(config["pin_memory"], bool)
        assert config["cache_mode"] in ["memory", "disk", "tensor-disk", "auto"]

    def test_get_training_defaults_with_image_size(self):
        """Test training defaults with custom image size."""
        defaults = SmartDefaults()
        config = defaults.get_training_defaults(image_size=512)
        assert isinstance(config["batch_size"], int)
        # Batch size should be adjusted for larger images
        assert config["batch_size"] >= 1


class TestInferenceDefaults:
    """Test inference defaults aggregation."""

    def test_get_inference_defaults_structure(self):
        """Test that inference defaults returns expected structure."""
        defaults = SmartDefaults()
        config = defaults.get_inference_defaults()
        assert isinstance(config, dict)
        # Check expected keys
        required_keys = [
            "device",
            "batch_size",
            "num_workers",
            "mixed_precision",
            "pin_memory",
        ]
        for key in required_keys:
            assert key in config

    def test_get_inference_defaults_values(self):
        """Test that inference defaults have valid values."""
        defaults = SmartDefaults()
        config = defaults.get_inference_defaults()
        assert config["device"] in ["cpu", "cuda", "mps"]
        assert isinstance(config["batch_size"], int)
        assert config["batch_size"] >= 1
        assert isinstance(config["num_workers"], int)
        assert isinstance(config["mixed_precision"], bool)
        assert isinstance(config["pin_memory"], bool)


class TestDatasetAwareDefaults:
    """Test dataset-aware default calculations."""

    def test_get_dataset_info_without_format(self):
        """Test get_dataset_info returns empty dict without dataset format."""
        defaults = SmartDefaults()
        info = defaults.get_dataset_info()
        assert isinstance(info, dict)
        assert len(info) == 0

    def test_get_dataset_info_with_format(self):
        """Test get_dataset_info with dataset format."""
        mock_format = Mock()
        mock_format.dataset_type = "archive"
        mock_format.image_format = "dicom"
        mock_format.image_count = 1000
        mock_format.format_counts = {"dicom": 1000}
        mock_format.csv_path = "/path/to/classificacao.csv"
        mock_format.dicom_root = "/path/to/archive"
        mock_format.has_features_txt = False
        mock_format.has_csv = True
        mock_format.warnings = ["Warning 1"]
        mock_format.suggestions = ["Suggestion 1"]

        defaults = SmartDefaults(dataset_format=mock_format)
        info = defaults.get_dataset_info()

        assert isinstance(info, dict)
        assert info["dataset_type"] == "archive"
        assert info["image_format"] == "dicom"
        assert info["image_count"] == 1000
        assert info["warnings"] == ["Warning 1"]
        assert info["suggestions"] == ["Suggestion 1"]

    def test_get_warnings_without_format(self):
        """Test get_warnings returns empty list without dataset format."""
        defaults = SmartDefaults()
        warnings = defaults.get_warnings()
        assert isinstance(warnings, list)
        assert len(warnings) == 0

    def test_get_warnings_with_format(self):
        """Test get_warnings with dataset format."""
        mock_format = Mock()
        mock_format.dataset_type = "archive"
        mock_format.image_format = "dicom"
        mock_format.image_count = 1000
        mock_format.warnings = ["Warning 1", "Warning 2"]
        mock_format.suggestions = []

        defaults = SmartDefaults(dataset_format=mock_format)
        warnings = defaults.get_warnings()
        assert warnings == ["Warning 1", "Warning 2"]

    def test_get_suggestions_without_format(self):
        """Test get_suggestions returns empty list without dataset format."""
        defaults = SmartDefaults()
        suggestions = defaults.get_suggestions()
        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_get_suggestions_with_format(self):
        """Test get_suggestions with dataset format."""
        mock_format = Mock()
        mock_format.dataset_type = "mamografias"
        mock_format.image_format = "png"
        mock_format.image_count = 500
        mock_format.warnings = []
        mock_format.suggestions = ["Suggestion 1", "Suggestion 2"]

        defaults = SmartDefaults(dataset_format=mock_format)
        suggestions = defaults.get_suggestions()
        assert suggestions == ["Suggestion 1", "Suggestion 2"]


class TestComprehensiveDefaults:
    """Test comprehensive defaults aggregation."""

    def test_get_comprehensive_defaults_structure(self):
        """Test comprehensive defaults returns expected structure."""
        defaults = SmartDefaults()
        config = defaults.get_comprehensive_defaults(dataset_size=1000)
        assert isinstance(config, dict)
        assert "hardware" in config
        assert "training" in config

    def test_get_comprehensive_defaults_hardware(self):
        """Test hardware section of comprehensive defaults."""
        defaults = SmartDefaults()
        config = defaults.get_comprehensive_defaults()
        hardware = config["hardware"]
        assert isinstance(hardware, dict)
        assert "device" in hardware
        assert "system_memory_gb" in hardware
        assert "mixed_precision_available" in hardware
        assert hardware["device"] in ["cpu", "cuda", "mps"]
        assert isinstance(hardware["system_memory_gb"], float)
        assert isinstance(hardware["mixed_precision_available"], bool)

    def test_get_comprehensive_defaults_training(self):
        """Test training section of comprehensive defaults."""
        defaults = SmartDefaults()
        config = defaults.get_comprehensive_defaults()
        training = config["training"]
        assert isinstance(training, dict)
        assert "device" in training
        assert "batch_size" in training
        assert "lr" in training

    def test_get_comprehensive_defaults_with_dataset(self):
        """Test comprehensive defaults includes dataset info when available."""
        mock_format = Mock()
        mock_format.dataset_type = "archive"
        mock_format.image_format = "dicom"
        mock_format.image_count = 1000
        mock_format.format_counts = {"dicom": 1000}
        mock_format.csv_path = "/path/to/classificacao.csv"
        mock_format.dicom_root = "/path/to/archive"
        mock_format.has_features_txt = False
        mock_format.has_csv = True
        mock_format.warnings = []
        mock_format.suggestions = []

        defaults = SmartDefaults(dataset_format=mock_format)
        config = defaults.get_comprehensive_defaults()
        assert "dataset" in config
        assert config["dataset"]["dataset_type"] == "archive"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_nearest_power_of_2(self):
        """Test nearest power of 2 calculation."""
        assert SmartDefaults._nearest_power_of_2(1) == 1
        assert SmartDefaults._nearest_power_of_2(2) == 2
        assert SmartDefaults._nearest_power_of_2(3) == 4
        assert SmartDefaults._nearest_power_of_2(5) == 4
        assert SmartDefaults._nearest_power_of_2(7) == 8
        assert SmartDefaults._nearest_power_of_2(15) == 16
        assert SmartDefaults._nearest_power_of_2(17) == 16
        assert SmartDefaults._nearest_power_of_2(31) == 32
        assert SmartDefaults._nearest_power_of_2(33) == 32

    def test_nearest_power_of_2_edge_cases(self):
        """Test nearest power of 2 with edge cases."""
        # Zero and negative should return 1
        assert SmartDefaults._nearest_power_of_2(0) == 1
        assert SmartDefaults._nearest_power_of_2(-5) == 1
        # Powers of 2 should return themselves
        assert SmartDefaults._nearest_power_of_2(16) == 16
        assert SmartDefaults._nearest_power_of_2(32) == 32
        assert SmartDefaults._nearest_power_of_2(64) == 64

    def test_print_recommendations_no_errors(self):
        """Test that print_recommendations executes without errors."""
        defaults = SmartDefaults()
        # Should not raise any errors
        defaults.print_recommendations()

    def test_print_recommendations_with_dataset_format(self):
        """Test print_recommendations with dataset format."""
        mock_format = Mock()
        mock_format.dataset_type = "archive"
        mock_format.image_format = "dicom"
        mock_format.image_count = 1000
        mock_format.warnings = ["Warning 1"]
        mock_format.suggestions = ["Suggestion 1"]

        defaults = SmartDefaults(dataset_format=mock_format)
        # Should not raise any errors
        defaults.print_recommendations(dataset_size=1000)


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    def test_get_smart_defaults_function(self):
        """Test get_smart_defaults convenience function."""
        config = get_smart_defaults(dataset_size=1000)
        assert isinstance(config, dict)
        assert "device" in config
        assert "batch_size" in config
        assert "num_workers" in config
        assert "epochs" in config
        assert "lr" in config

    def test_get_smart_defaults_with_image_size(self):
        """Test get_smart_defaults with custom image size."""
        config = get_smart_defaults(dataset_size=500, image_size=512)
        assert isinstance(config, dict)
        assert isinstance(config["batch_size"], int)
        assert config["batch_size"] >= 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_image_size(self):
        """Test with very large image size."""
        defaults = SmartDefaults()
        batch_size = defaults.get_batch_size(image_size=1024)
        # Should still return valid batch size
        assert isinstance(batch_size, int)
        assert batch_size >= 1

    def test_very_small_dataset(self):
        """Test with very small dataset."""
        defaults = SmartDefaults()
        config = defaults.get_training_defaults(dataset_size=10)
        assert config["batch_size"] >= 1
        assert config["batch_size"] <= 10
        assert config["epochs"] > 0

    def test_very_large_dataset(self):
        """Test with very large dataset."""
        defaults = SmartDefaults()
        config = defaults.get_training_defaults(dataset_size=100000)
        assert config["cache_mode"] == "tensor-disk"
        assert config["epochs"] == 15  # Large datasets need fewer epochs

    def test_multiple_tasks_consistency(self):
        """Test that multiple calls return consistent results."""
        defaults = SmartDefaults()
        batch1 = defaults.get_batch_size(task="train")
        batch2 = defaults.get_batch_size(task="train")
        assert batch1 == batch2

        workers1 = defaults.get_num_workers(task="train")
        workers2 = defaults.get_num_workers(task="train")
        assert workers1 == workers2
