"""
Integration tests for wizard smart defaults with hardware detection.

Tests cover:
- GPU hardware detection in wizard
- Smart defaults calculation based on detected hardware
- Batch size recommendations (GPU vs CPU)
- Worker count recommendations
- Cache mode selection
- AMP (Automatic Mixed Precision) settings
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography.utils.smart_defaults import SmartDefaults
from mammography.utils.device_detection import DeviceDetector


class TestWizardGPUDetection:
    """Test wizard GPU hardware detection and smart defaults."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_detected_suggests_larger_batch_size(self):
        """Test that GPU detection suggests larger batch sizes."""
        # Initialize SmartDefaults with real hardware
        defaults = SmartDefaults()

        # If GPU is available, should suggest larger batch size
        if defaults.device_type == "cuda":
            batch_size = defaults.get_batch_size(task="train", image_size=224)
            # GPU should suggest batch size >= 8 (typically 16)
            assert batch_size >= 8
            # Should be power of 2
            assert batch_size & (batch_size - 1) == 0

    def test_cpu_suggests_smaller_batch_size(self):
        """Test that CPU-only systems suggest smaller batch sizes."""
        # Mock DeviceDetector to force CPU mode
        with patch("mammography.utils.smart_defaults.DeviceDetector") as mock_detector_class:
            mock_detector = Mock(spec=DeviceDetector)
            mock_detector.best_device = "cpu"
            mock_detector.get_device_config.return_value = {
                "batch_size": 4,
                "num_workers": 4,
                "pin_memory": False,
                "mixed_precision": False,
            }
            mock_detector_class.return_value = mock_detector

            defaults = SmartDefaults()
            batch_size = defaults.get_batch_size(task="train", image_size=224)

            # CPU should suggest smaller batch size (typically 4-8)
            assert batch_size <= 8
            # Should be power of 2
            assert batch_size & (batch_size - 1) == 0

    def test_gpu_batch_size_larger_than_cpu(self):
        """Test that GPU batch size is larger than CPU batch size."""
        # Test CPU batch size
        with patch("mammography.utils.smart_defaults.DeviceDetector") as mock_detector_class:
            mock_detector = Mock(spec=DeviceDetector)
            mock_detector.best_device = "cpu"
            mock_detector.get_device_config.return_value = {
                "batch_size": 4,
                "num_workers": 4,
                "pin_memory": False,
                "mixed_precision": False,
            }
            mock_detector_class.return_value = mock_detector

            cpu_defaults = SmartDefaults()
            cpu_batch_size = cpu_defaults.get_batch_size(task="train", image_size=224)

        # Test GPU batch size
        with patch("mammography.utils.smart_defaults.DeviceDetector") as mock_detector_class:
            mock_detector = Mock(spec=DeviceDetector)
            mock_detector.best_device = "cuda"
            mock_detector.get_device_config.return_value = {
                "batch_size": 16,
                "num_workers": 4,
                "pin_memory": True,
                "mixed_precision": True,
            }
            mock_detector_class.return_value = mock_detector

            # Mock CUDA properties
            with patch("torch.cuda.get_device_properties") as mock_props:
                mock_device_props = MagicMock()
                mock_device_props.total_memory = 8 * 1024**3  # 8GB VRAM
                mock_props.return_value = mock_device_props

                gpu_defaults = SmartDefaults()
                gpu_batch_size = gpu_defaults.get_batch_size(task="train", image_size=224)

        # GPU should have larger batch size than CPU
        assert gpu_batch_size > cpu_batch_size


class TestWizardNumWorkers:
    """Test wizard worker count recommendations."""

    def test_cpu_suggests_more_workers(self):
        """Test that CPU mode suggests more workers for data loading."""
        with patch("mammography.utils.smart_defaults.DeviceDetector") as mock_detector_class:
            mock_detector = Mock(spec=DeviceDetector)
            mock_detector.best_device = "cpu"
            mock_detector.get_device_config.return_value = {
                "batch_size": 4,
                "num_workers": 4,
                "pin_memory": False,
                "mixed_precision": False,
            }
            mock_detector_class.return_value = mock_detector

            with patch("os.cpu_count", return_value=8):
                defaults = SmartDefaults()
                num_workers = defaults.get_num_workers(task="train")

                # CPU should use more workers (cpu_count - 1, max 8)
                assert num_workers >= 4
                assert num_workers <= 8

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_num_workers_reasonable(self):
        """Test that GPU mode suggests reasonable worker count."""
        defaults = SmartDefaults()

        if defaults.device_type == "cuda":
            num_workers = defaults.get_num_workers(task="train")

            # GPU should suggest reasonable worker count (typically 4)
            assert num_workers >= 0
            assert num_workers <= 16

    def test_inference_uses_fewer_workers(self):
        """Test that inference task uses fewer workers."""
        defaults = SmartDefaults()

        train_workers = defaults.get_num_workers(task="train")
        inference_workers = defaults.get_num_workers(task="inference")

        # Inference should use fewer workers (less I/O bound)
        assert inference_workers <= train_workers


class TestWizardCacheMode:
    """Test wizard cache mode recommendations."""

    def test_small_dataset_suggests_memory_cache(self):
        """Test that small datasets suggest memory caching."""
        defaults = SmartDefaults()

        # Small dataset (500 images)
        cache_mode = defaults.get_cache_mode(dataset_size=500)

        # Should suggest memory cache for small datasets
        assert cache_mode in ["memory", "auto"]

    def test_large_dataset_suggests_disk_cache(self):
        """Test that large datasets suggest disk caching."""
        defaults = SmartDefaults()

        # Large dataset (10000 images)
        cache_mode = defaults.get_cache_mode(dataset_size=10000)

        # Should suggest disk or tensor-disk for large datasets
        assert cache_mode in ["disk", "tensor-disk", "auto"]

    def test_cache_mode_with_dataset_format(self):
        """Test cache mode recommendations with dataset format info."""
        # Mock DatasetFormat for DICOM dataset
        mock_format = Mock()
        mock_format.dataset_type = "archive"
        mock_format.image_format = "dicom"
        mock_format.image_count = 2000
        mock_format.warnings = []
        mock_format.suggestions = []

        defaults = SmartDefaults(dataset_format=mock_format)
        cache_mode = defaults.get_cache_mode()

        # Should use dataset_format.image_count for recommendation
        assert cache_mode in ["memory", "disk", "tensor-disk", "auto"]


class TestWizardAMPRecommendations:
    """Test Automatic Mixed Precision recommendations."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_enables_amp_in_config(self):
        """Test that GPU systems enable AMP in device config."""
        detector = DeviceDetector()

        if detector.best_device == "cuda":
            config = detector.get_device_config()

            # GPU should enable mixed precision
            assert "mixed_precision" in config
            assert config["mixed_precision"] is True

    def test_cpu_disables_amp_in_config(self):
        """Test that CPU systems disable AMP in device config."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                detector = DeviceDetector()
                config = detector.get_device_config()

                # CPU should disable mixed precision
                assert "mixed_precision" in config
                assert config["mixed_precision"] is False


class TestWizardEpochRecommendations:
    """Test epoch recommendations based on dataset size."""

    def test_small_dataset_more_epochs(self):
        """Test that small datasets get more epochs recommended."""
        defaults = SmartDefaults()

        # Small dataset (500 images)
        epochs = defaults.get_epochs(dataset_size=500)

        # Small datasets need more epochs
        assert epochs >= 30

    def test_large_dataset_fewer_epochs(self):
        """Test that large datasets get fewer epochs recommended."""
        defaults = SmartDefaults()

        # Large dataset (10000 images)
        epochs = defaults.get_epochs(dataset_size=10000)

        # Large datasets need fewer epochs
        assert epochs <= 30


class TestWizardLearningRate:
    """Test learning rate scaling recommendations."""

    def test_get_learning_rate_returns_float(self):
        """Test that learning rate is returned as float."""
        defaults = SmartDefaults()
        lr = defaults.get_learning_rate()

        assert isinstance(lr, float)
        assert lr > 0

    def test_learning_rate_scales_with_batch_size(self):
        """Test that learning rate scales with batch size."""
        defaults = SmartDefaults()

        # Get LR for different batch sizes
        small_batch_lr = defaults.get_learning_rate(batch_size=4)
        large_batch_lr = defaults.get_learning_rate(batch_size=32)

        # Larger batch size should have larger LR (linear scaling rule)
        assert large_batch_lr >= small_batch_lr


class TestWizardComprehensiveDefaults:
    """Test comprehensive defaults aggregation."""

    def test_get_training_defaults_returns_complete_dict(self):
        """Test that training defaults returns all required keys."""
        defaults = SmartDefaults()
        training_defaults = defaults.get_training_defaults()

        required_keys = ["batch_size", "num_workers", "cache_mode", "epochs", "lr"]
        for key in required_keys:
            assert key in training_defaults

    def test_get_inference_defaults_returns_complete_dict(self):
        """Test that inference defaults returns all required keys."""
        defaults = SmartDefaults()
        inference_defaults = defaults.get_inference_defaults()

        required_keys = ["batch_size", "num_workers", "cache_mode"]
        for key in required_keys:
            assert key in inference_defaults

    def test_inference_batch_larger_than_training(self):
        """Test that inference batch size is larger than training."""
        defaults = SmartDefaults()

        training_defaults = defaults.get_training_defaults()
        inference_defaults = defaults.get_inference_defaults()

        # Inference can use larger batches (no gradients)
        assert inference_defaults["batch_size"] >= training_defaults["batch_size"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_comprehensive_defaults(self):
        """Test comprehensive defaults on GPU system."""
        defaults = SmartDefaults()

        if defaults.device_type == "cuda":
            all_defaults = defaults.get_comprehensive_defaults()

            # Should have training and inference sections
            assert "training" in all_defaults
            assert "inference" in all_defaults
            assert "hardware" in all_defaults

            # Hardware should show CUDA
            assert all_defaults["hardware"]["device_type"] == "cuda"


class TestWizardDatasetAwareDefaults:
    """Test dataset-aware default recommendations."""

    def test_dicom_dataset_recommendations(self):
        """Test recommendations for DICOM datasets."""
        # Mock DatasetFormat for DICOM dataset
        mock_format = Mock()
        mock_format.dataset_type = "archive"
        mock_format.image_format = "dicom"
        mock_format.image_count = 1000
        mock_format.warnings = []
        mock_format.suggestions = ["Consider preprocessing DICOM files"]

        defaults = SmartDefaults(dataset_format=mock_format)

        # Get dataset info
        info = defaults.get_dataset_info()

        assert info["dataset_type"] == "archive"
        assert info["image_format"] == "dicom"
        assert info["image_count"] == 1000

    def test_png_dataset_recommendations(self):
        """Test recommendations for PNG/JPG datasets."""
        # Mock DatasetFormat for PNG dataset
        mock_format = Mock()
        mock_format.dataset_type = "mamografias"
        mock_format.image_format = "png"
        mock_format.image_count = 2000
        mock_format.warnings = []
        mock_format.suggestions = []

        defaults = SmartDefaults(dataset_format=mock_format)

        # Get cache mode - PNG datasets can use more aggressive caching
        cache_mode = defaults.get_cache_mode()

        assert cache_mode in ["memory", "disk", "tensor-disk", "auto"]

    def test_dataset_warnings_propagate(self):
        """Test that dataset warnings are accessible."""
        # Mock DatasetFormat with warnings
        mock_format = Mock()
        mock_format.dataset_type = "archive"
        mock_format.image_format = "dicom"
        mock_format.image_count = 50
        mock_format.warnings = ["Very small dataset - results may not generalize"]
        mock_format.suggestions = ["Consider data augmentation"]

        defaults = SmartDefaults(dataset_format=mock_format)

        # Get warnings
        warnings = defaults.get_warnings()

        assert len(warnings) > 0
        assert any("small dataset" in w.lower() for w in warnings)

    def test_dataset_suggestions_accessible(self):
        """Test that dataset suggestions are accessible."""
        # Mock DatasetFormat with suggestions
        mock_format = Mock()
        mock_format.dataset_type = "mamografias"
        mock_format.image_format = "png"
        mock_format.image_count = 1000
        mock_format.warnings = []
        mock_format.suggestions = ["Images are already preprocessed", "Ready for training"]

        defaults = SmartDefaults(dataset_format=mock_format)

        # Get suggestions
        suggestions = defaults.get_suggestions()

        assert len(suggestions) > 0

    def test_patches_completo_dataset_recommendations(self):
        """Test recommendations for patches_completo dataset preset."""
        # Mock DatasetFormat for patches_completo dataset
        mock_format = Mock()
        mock_format.dataset_type = "patches_completo"
        mock_format.image_format = "png"
        mock_format.image_count = 5000
        mock_format.warnings = []
        mock_format.suggestions = ["Patch-level dataset detected", "Ensure patch labels are correct"]

        defaults = SmartDefaults(dataset_format=mock_format)

        # Get dataset info
        info = defaults.get_dataset_info()

        assert info["dataset_type"] == "patches_completo"
        assert info["image_format"] == "png"
        assert info["image_count"] == 5000

        # Patches can use more aggressive caching than full images
        cache_mode = defaults.get_cache_mode()
        assert cache_mode in ["memory", "disk", "tensor-disk", "auto"]

    def test_archive_vs_mamografias_preset_differences(self):
        """Test that archive (DICOM) and mamografias (PNG) presets have different recommendations."""
        # Mock archive preset (DICOM, larger files)
        archive_format = Mock()
        archive_format.dataset_type = "archive"
        archive_format.image_format = "dicom"
        archive_format.image_count = 1000
        archive_format.warnings = []
        archive_format.suggestions = []

        archive_defaults = SmartDefaults(dataset_format=archive_format)

        # Mock mamografias preset (PNG, smaller files)
        mammo_format = Mock()
        mammo_format.dataset_type = "mamografias"
        mammo_format.image_format = "png"
        mammo_format.image_count = 1000
        mammo_format.warnings = []
        mammo_format.suggestions = []

        mammo_defaults = SmartDefaults(dataset_format=mammo_format)

        # Both should provide valid recommendations
        archive_cache = archive_defaults.get_cache_mode()
        mammo_cache = mammo_defaults.get_cache_mode()

        assert archive_cache in ["memory", "disk", "tensor-disk", "auto"]
        assert mammo_cache in ["memory", "disk", "tensor-disk", "auto"]

        # Archive dataset info should reflect DICOM
        archive_info = archive_defaults.get_dataset_info()
        assert archive_info["image_format"] == "dicom"

        # Mamografias dataset info should reflect PNG
        mammo_info = mammo_defaults.get_dataset_info()
        assert mammo_info["image_format"] == "png"

    def test_all_three_presets_valid_configs(self):
        """Test that all three dataset presets (archive, mamografias, patches_completo) produce valid configs."""
        presets = [
            {"type": "archive", "format": "dicom", "count": 1200},
            {"type": "mamografias", "format": "png", "count": 2000},
            {"type": "patches_completo", "format": "png", "count": 5000},
        ]

        for preset in presets:
            # Mock DatasetFormat
            mock_format = Mock()
            mock_format.dataset_type = preset["type"]
            mock_format.image_format = preset["format"]
            mock_format.image_count = preset["count"]
            mock_format.warnings = []
            mock_format.suggestions = []

            defaults = SmartDefaults(dataset_format=mock_format)

            # Get comprehensive defaults
            config = defaults.get_comprehensive_defaults()

            # Verify all required sections present
            assert "training" in config
            assert "inference" in config
            assert "hardware" in config

            # Verify training config has required keys
            assert "batch_size" in config["training"]
            assert "num_workers" in config["training"]
            assert "cache_mode" in config["training"]
            assert "epochs" in config["training"]
            assert "lr" in config["training"]

            # Verify batch size is reasonable power of 2
            batch_size = config["training"]["batch_size"]
            assert batch_size > 0
            assert batch_size & (batch_size - 1) == 0  # Power of 2 check

            # Verify epochs reasonable based on dataset size
            epochs = config["training"]["epochs"]
            assert 10 <= epochs <= 100

    def test_dataset_format_validation_workflow(self):
        """Test the complete dataset format validation workflow."""
        # Simulate archive preset with validation issues
        archive_format = Mock()
        archive_format.dataset_type = "archive"
        archive_format.image_format = "dicom"
        archive_format.image_count = 100  # Small dataset
        archive_format.warnings = [
            "Small dataset detected (100 images)",
            "Consider data augmentation",
        ]
        archive_format.suggestions = [
            "Enable aggressive augmentation for small datasets",
            "Consider view-specific training",
        ]

        defaults = SmartDefaults(dataset_format=archive_format)

        # Verify warnings propagate
        warnings = defaults.get_warnings()
        assert len(warnings) == 2
        assert any("100 images" in w for w in warnings)

        # Verify suggestions propagate
        suggestions = defaults.get_suggestions()
        assert len(suggestions) == 2
        assert any("augmentation" in s.lower() for s in suggestions)

        # Small dataset should get more epochs
        epochs = defaults.get_epochs()
        assert epochs >= 30

        # Verify dataset info accessible
        info = defaults.get_dataset_info()
        assert info["dataset_type"] == "archive"
        assert info["image_count"] == 100
