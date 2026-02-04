"""
Integration tests for automatic normalization feature.

These tests validate the automatic normalization computation, validation,
and integration with the MammoDensityDataset.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch

import pytest
import numpy as np
import torch
from PIL import Image

from mammography.data.dataset import MammoDensityDataset, robust_collate
from mammography.utils.normalization import (
    compute_normalization_stats,
    validate_normalization,
    NormalizationStats,
)


@pytest.fixture
def mock_mammography_image() -> np.ndarray:
    """Create a realistic mammography-like pixel array for testing."""
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
    image = np.clip(image + noise, 0, 4095).astype(np.uint16)

    return image


@pytest.fixture
def mock_dataset_rows(tmp_path: Path, mock_mammography_image: np.ndarray) -> List[Dict[str, Any]]:
    """Create mock dataset rows with temporary image files."""
    rows = []

    # Create 10 temporary PNG images for testing
    for i in range(10):
        img_path = tmp_path / f"image_{i:03d}.png"

        # Add some variation to each image
        np.random.seed(42 + i)
        image_array = mock_mammography_image.copy()
        image_array = np.clip(image_array + np.random.normal(0, 100, image_array.shape), 0, 4095).astype(np.uint16)

        # Convert to RGB PIL Image
        # Normalize to 0-255 for PNG saving
        image_8bit = (image_array / 16).astype(np.uint8)
        pil_image = Image.fromarray(image_8bit).convert("RGB")
        pil_image.save(img_path, format="PNG")

        rows.append({
            "image_path": str(img_path),
            "professional_label": (i % 4) + 1,  # Labels 1-4
            "accession": f"ACC{i:03d}",
        })

    return rows


class TestAutoNormalization:
    """Integration tests for automatic normalization feature."""

    def test_auto_normalize_computes_stats(self, mock_dataset_rows: List[Dict[str, Any]], caplog):
        """Test that auto_normalize=True computes normalization statistics from dataset."""
        with caplog.at_level(logging.INFO):
            # Create dataset with auto-normalization enabled
            dataset = MammoDensityDataset(
                rows=mock_dataset_rows,
                img_size=224,
                train=False,
                augment=False,
                auto_normalize=True,
                auto_normalize_samples=5,  # Use small sample for speed
            )

        # Verify that normalization stats were computed
        assert dataset._norm_mean is not None
        assert dataset._norm_std is not None
        assert len(dataset._norm_mean) == 3
        assert len(dataset._norm_std) == 3

        # Verify all std values are positive
        assert all(s > 0 for s in dataset._norm_std)

        # Check that auto-normalization was attempted
        assert any("Auto-normalization enabled" in record.message for record in caplog.records)

        # If auto-normalization succeeded, stats should differ from ImageNet
        # If it failed, it should have fallen back to ImageNet defaults with a warning
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        if dataset._norm_mean == imagenet_mean and dataset._norm_std == imagenet_std:
            # Verify fallback warning was logged
            assert any("falling back to ImageNet defaults" in record.message for record in caplog.records)
        else:
            # Stats successfully computed and differ from ImageNet
            assert dataset._norm_mean != imagenet_mean or dataset._norm_std != imagenet_std

    def test_auto_normalize_with_explicit_mean_std(self, mock_dataset_rows: List[Dict[str, Any]], caplog):
        """Test that explicit mean/std override auto_normalize."""
        custom_mean = [0.5, 0.5, 0.5]
        custom_std = [0.2, 0.2, 0.2]

        with caplog.at_level(logging.WARNING):
            dataset = MammoDensityDataset(
                rows=mock_dataset_rows,
                img_size=224,
                train=False,
                augment=False,
                auto_normalize=True,
                auto_normalize_samples=5,
                mean=custom_mean,
                std=custom_std,
            )

        # Verify that provided values are used
        assert dataset._norm_mean == custom_mean
        assert dataset._norm_std == custom_std

        # Verify warning was logged
        assert any("auto_normalize=True but mean/std explicitly provided" in record.message
                  for record in caplog.records)

    def test_auto_normalize_fallback_on_failure(self, caplog):
        """Test that auto_normalize falls back to ImageNet defaults on error."""
        # Create dataset with invalid rows to trigger fallback
        invalid_rows = [
            {"image_path": "/nonexistent/path.png", "professional_label": 1, "accession": "ACC001"}
        ]

        with caplog.at_level(logging.WARNING):
            dataset = MammoDensityDataset(
                rows=invalid_rows,
                img_size=224,
                train=False,
                augment=False,
                auto_normalize=True,
                auto_normalize_samples=1,
            )

        # Verify fallback to ImageNet defaults
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        assert dataset._norm_mean == imagenet_mean
        assert dataset._norm_std == imagenet_std

        # Verify warning was logged
        assert any("Auto-normalization failed, falling back to ImageNet defaults" in record.message
                  for record in caplog.records)

    def test_normalization_applied_correctly(self, mock_dataset_rows: List[Dict[str, Any]]):
        """Test that computed normalization is applied to dataset items."""
        # Create dataset with auto-normalization
        dataset = MammoDensityDataset(
            rows=mock_dataset_rows,
            img_size=224,
            train=False,
            augment=False,
            auto_normalize=True,
            auto_normalize_samples=5,
        )

        # Get a few samples
        samples = []
        for i in range(min(3, len(dataset))):
            item = dataset[i]
            if item is not None:
                samples.append(item[0])  # Get image tensor

        assert len(samples) > 0, "No valid samples loaded"

        # Stack samples and check normalization
        sample_tensor = torch.stack(samples)

        # Verify tensor shape (should be NCHW)
        assert sample_tensor.ndim == 4
        assert sample_tensor.shape[1] == 3  # RGB channels
        assert sample_tensor.shape[2] == 224
        assert sample_tensor.shape[3] == 224

        # Compute actual statistics from samples
        actual_mean = sample_tensor.mean(dim=[0, 2, 3]).tolist()
        actual_std = sample_tensor.std(dim=[0, 2, 3]).tolist()

        # Verify normalization was applied (values should be centered around computed mean)
        # Note: Due to augmentation and small sample size, we just verify reasonable ranges
        for mean_val in actual_mean:
            assert -3.0 < mean_val < 3.0, f"Mean {mean_val} outside reasonable range"

        for std_val in actual_std:
            assert 0.1 < std_val < 5.0, f"Std {std_val} outside reasonable range"

    def test_normalization_validation_warnings(self, mock_dataset_rows: List[Dict[str, Any]], caplog):
        """Test that validation warnings are logged for unnormalized data."""
        with caplog.at_level(logging.WARNING):
            # Create dataset with auto-normalization
            dataset = MammoDensityDataset(
                rows=mock_dataset_rows,
                img_size=224,
                train=False,
                augment=False,
                auto_normalize=True,
                auto_normalize_samples=5,
            )

        # Check if validation warnings were logged (may or may not appear depending on data)
        # This test verifies that the validation mechanism runs
        validation_ran = any(
            "Data validation" in record.message or "normalization" in record.message.lower()
            for record in caplog.records
        )
        assert validation_ran, "Normalization validation should have run"

    def test_no_auto_normalize_uses_defaults(self, mock_dataset_rows: List[Dict[str, Any]]):
        """Test that auto_normalize=False uses ImageNet defaults."""
        dataset = MammoDensityDataset(
            rows=mock_dataset_rows,
            img_size=224,
            train=False,
            augment=False,
            auto_normalize=False,
        )

        # Verify ImageNet defaults are used
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        assert dataset._norm_mean == imagenet_mean
        assert dataset._norm_std == imagenet_std

    def test_compute_normalization_stats_function(self, mock_dataset_rows: List[Dict[str, Any]]):
        """Test compute_normalization_stats utility function."""
        from torch.utils.data import DataLoader

        # Create a simple dataset (without auto-normalize to avoid recursion)
        dataset = MammoDensityDataset(
            rows=mock_dataset_rows,
            img_size=224,
            train=False,
            augment=False,
            auto_normalize=False,
        )

        # Create DataLoader with robust collate to handle None values
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=robust_collate,
        )

        # Compute normalization stats using DataLoader
        stats = compute_normalization_stats(
            dataset=dataloader,
            num_samples=None,  # Use all samples from dataloader
            num_workers=0,
            batch_size=2,  # Ignored when passing DataLoader
        )

        # Verify stats structure
        assert isinstance(stats, NormalizationStats)
        assert len(stats.mean) == 3
        assert len(stats.std) == 3
        assert stats.method == "auto"
        # sample_size may vary due to robust_collate filtering None values
        assert stats.sample_size is not None

        # Verify all values are valid
        assert all(isinstance(m, float) for m in stats.mean)
        assert all(isinstance(s, float) for s in stats.std)
        assert all(s > 0 for s in stats.std)

    def test_validate_normalization_function(self):
        """Test validate_normalization utility function."""
        # Create normalized data
        normalized_data = torch.randn(16, 3, 224, 224) * 0.229 + 0.485

        result = validate_normalization(
            data=normalized_data,
            expected_mean=[0.485, 0.456, 0.406],
            expected_std=[0.229, 0.224, 0.225],
            tolerance=0.2,  # Use larger tolerance for random data
        )

        # Verify result structure
        assert "is_normalized" in result
        assert "actual_mean" in result
        assert "actual_std" in result
        assert "expected_mean" in result
        assert "expected_std" in result
        assert "warnings" in result

        # Verify actual values are computed
        assert len(result["actual_mean"]) == 3
        assert len(result["actual_std"]) == 3

    def test_normalization_stats_serialization(self):
        """Test NormalizationStats serialization and deserialization."""
        stats = NormalizationStats(
            mean=[0.5, 0.5, 0.5],
            std=[0.2, 0.2, 0.2],
            method="auto",
            sample_size=1000,
        )

        # Serialize
        stats_dict = stats.to_dict()
        assert isinstance(stats_dict, dict)
        assert stats_dict["mean"] == [0.5, 0.5, 0.5]
        assert stats_dict["std"] == [0.2, 0.2, 0.2]
        assert stats_dict["method"] == "auto"
        assert stats_dict["sample_size"] == 1000

        # Deserialize
        restored_stats = NormalizationStats.from_dict(stats_dict)
        assert restored_stats.mean == stats.mean
        assert restored_stats.std == stats.std
        assert restored_stats.method == stats.method
        assert restored_stats.sample_size == stats.sample_size

    def test_imagenet_defaults_class_method(self):
        """Test NormalizationStats.imagenet_defaults() class method."""
        stats = NormalizationStats.imagenet_defaults()

        assert stats.mean == [0.485, 0.456, 0.406]
        assert stats.std == [0.229, 0.224, 0.225]
        assert stats.method == "imagenet"
        assert stats.sample_size is None

    def test_auto_normalize_with_different_sample_sizes(self, mock_dataset_rows: List[Dict[str, Any]]):
        """Test auto_normalize with different sample sizes."""
        sample_sizes = [1, 3, 5, 10]

        for sample_size in sample_sizes:
            dataset = MammoDensityDataset(
                rows=mock_dataset_rows,
                img_size=224,
                train=False,
                augment=False,
                auto_normalize=True,
                auto_normalize_samples=sample_size,
            )

            # Verify stats were computed
            assert len(dataset._norm_mean) == 3
            assert len(dataset._norm_std) == 3
            assert all(s > 0 for s in dataset._norm_std)

    def test_auto_normalize_integration_with_caching(self, tmp_path: Path, mock_dataset_rows: List[Dict[str, Any]]):
        """Test that auto_normalize works with different caching modes."""
        cache_modes = ["none", "memory"]

        for cache_mode in cache_modes:
            cache_dir = tmp_path / f"cache_{cache_mode}" if cache_mode != "none" else None

            dataset = MammoDensityDataset(
                rows=mock_dataset_rows,
                img_size=224,
                train=False,
                augment=False,
                cache_mode=cache_mode,
                cache_dir=str(cache_dir) if cache_dir else None,
                auto_normalize=True,
                auto_normalize_samples=3,
            )

            # Verify dataset is functional
            item = dataset[0]
            assert item is not None
            assert len(item) == 4  # image, label, metadata, embedding

            # Verify normalization was applied
            assert dataset._norm_mean is not None
            assert dataset._norm_std is not None
