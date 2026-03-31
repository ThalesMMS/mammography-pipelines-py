"""
Unit tests for MammoDensityDataset cache mode validation.

These tests validate the cache_mode parameter and related validation logic
in the MammoDensityDataset class.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mammography.data.csv_loader import resolve_dataset_cache_mode
from mammography.data.dataset import MammoDensityDataset


class TestCacheModeValidation:
    """Test cache_mode parameter validation."""

    @pytest.fixture
    def minimal_rows(self):
        """Create minimal row data for dataset initialization."""
        return [
            {
                "image_path": "dummy_image_001.png",
                "density_label": 1,
                "accession": "ACC001",
            }
        ]

    def test_valid_cache_mode_none(self, minimal_rows):
        """Test that cache_mode='none' is valid."""
        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="none",
        )

        assert dataset.cache_mode == "none"
        assert dataset.cache_dir is None

    def test_valid_cache_mode_memory(self, minimal_rows):
        """Test that cache_mode='memory' is valid."""
        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="memory",
        )

        assert dataset.cache_mode == "memory"
        assert dataset._image_cache is not None
        assert isinstance(dataset._image_cache, dict)

    def test_valid_cache_mode_disk(self, minimal_rows, tmp_path):
        """Test that cache_mode='disk' is valid with cache_dir."""
        cache_dir = tmp_path / "cache"

        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="disk",
            cache_dir=str(cache_dir),
        )

        assert dataset.cache_mode == "disk"
        assert dataset.cache_dir == cache_dir

    def test_valid_cache_mode_tensor_disk(self, minimal_rows, tmp_path):
        """Test that cache_mode='tensor-disk' is valid with cache_dir."""
        cache_dir = tmp_path / "cache"

        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="tensor-disk",
            cache_dir=str(cache_dir),
        )

        assert dataset.cache_mode == "tensor-disk"
        assert dataset.cache_dir == cache_dir

    def test_valid_cache_mode_tensor_memmap(self, minimal_rows, tmp_path):
        """Test that cache_mode='tensor-memmap' is valid with cache_dir."""
        cache_dir = tmp_path / "cache"

        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="tensor-memmap",
            cache_dir=str(cache_dir),
        )

        assert dataset.cache_mode == "tensor-memmap"
        assert dataset.cache_dir == cache_dir

    def test_cache_mode_case_insensitive(self, minimal_rows):
        """Test that cache_mode is normalized to lowercase."""
        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="MEMORY",
        )

        assert dataset.cache_mode == "memory"

    def test_cache_mode_mixed_case(self, minimal_rows):
        """Test that cache_mode handles mixed case."""
        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="NoNe",
        )

        assert dataset.cache_mode == "none"

    def test_invalid_cache_mode_raises_error(self, minimal_rows):
        """Test that invalid cache_mode raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MammoDensityDataset(
                rows=minimal_rows,
                img_size=224,
                train=False,
                cache_mode="invalid_mode",
            )

        assert "cache_mode inválido" in str(exc_info.value)
        assert "invalid_mode" in str(exc_info.value)

    def test_invalid_cache_mode_empty_string(self, minimal_rows):
        """Test that empty string cache_mode defaults to 'none'."""
        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="",
        )

        assert dataset.cache_mode == "none"

    def test_invalid_cache_mode_typo(self, minimal_rows):
        """Test that common typos in cache_mode raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MammoDensityDataset(
                rows=minimal_rows,
                img_size=224,
                train=False,
                cache_mode="mem",  # typo for 'memory'
            )

        assert "cache_mode inválido" in str(exc_info.value)

    def test_invalid_cache_mode_with_hyphen_typo(self, minimal_rows):
        """Test that hyphen variations raise ValueError for unsupported modes."""
        with pytest.raises(ValueError) as exc_info:
            MammoDensityDataset(
                rows=minimal_rows,
                img_size=224,
                train=False,
                cache_mode="tensor_disk",  # underscore instead of hyphen
            )

        assert "cache_mode inválido" in str(exc_info.value)


class TestCacheDirValidation:
    """Test cache_dir parameter validation for disk-based cache modes."""

    @pytest.fixture
    def minimal_rows(self):
        """Create minimal row data for dataset initialization."""
        return [
            {
                "image_path": "dummy_image_001.png",
                "density_label": 1,
                "accession": "ACC001",
            }
        ]

    def test_disk_cache_mode_requires_cache_dir(self, minimal_rows):
        """Test that cache_mode='disk' requires cache_dir."""
        with pytest.raises(ValueError) as exc_info:
            MammoDensityDataset(
                rows=minimal_rows,
                img_size=224,
                train=False,
                cache_mode="disk",
                cache_dir=None,
            )

        assert "cache_dir é obrigatório" in str(exc_info.value)
        assert "persistência em disco" in str(exc_info.value)

    def test_tensor_disk_cache_mode_requires_cache_dir(self, minimal_rows):
        """Test that cache_mode='tensor-disk' requires cache_dir."""
        with pytest.raises(ValueError) as exc_info:
            MammoDensityDataset(
                rows=minimal_rows,
                img_size=224,
                train=False,
                cache_mode="tensor-disk",
                cache_dir=None,
            )

        assert "cache_dir é obrigatório" in str(exc_info.value)

    def test_tensor_memmap_cache_mode_requires_cache_dir(self, minimal_rows):
        """Test that cache_mode='tensor-memmap' requires cache_dir."""
        with pytest.raises(ValueError) as exc_info:
            MammoDensityDataset(
                rows=minimal_rows,
                img_size=224,
                train=False,
                cache_mode="tensor-memmap",
                cache_dir=None,
            )

        assert "cache_dir é obrigatório" in str(exc_info.value)

    def test_memory_cache_mode_allows_none_cache_dir(self, minimal_rows):
        """Test that cache_mode='memory' works with cache_dir=None."""
        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="memory",
            cache_dir=None,
        )

        assert dataset.cache_mode == "memory"
        assert dataset.cache_dir is None

    def test_none_cache_mode_allows_none_cache_dir(self, minimal_rows):
        """Test that cache_mode='none' works with cache_dir=None."""
        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="none",
            cache_dir=None,
        )

        assert dataset.cache_mode == "none"
        assert dataset.cache_dir is None

    def test_disk_cache_dir_accepts_path_object(self, minimal_rows, tmp_path):
        """Test that cache_dir accepts Path object for disk cache."""
        cache_dir = tmp_path / "cache"

        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="disk",
            cache_dir=cache_dir,
        )

        assert dataset.cache_dir == cache_dir

    def test_disk_cache_dir_accepts_string(self, minimal_rows, tmp_path):
        """Test that cache_dir accepts string path for disk cache."""
        cache_dir = tmp_path / "cache"

        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="disk",
            cache_dir=str(cache_dir),
        )

        assert dataset.cache_dir == cache_dir

    def test_cache_dir_converted_to_path(self, minimal_rows, tmp_path):
        """Test that string cache_dir is converted to Path object."""
        cache_dir_str = str(tmp_path / "cache")

        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="disk",
            cache_dir=cache_dir_str,
        )

        assert isinstance(dataset.cache_dir, Path)
        assert str(dataset.cache_dir) == cache_dir_str


class TestCacheModeDefaults:
    """Test default values and edge cases for cache_mode."""

    @pytest.fixture
    def minimal_rows(self):
        """Create minimal row data for dataset initialization."""
        return [
            {
                "image_path": "dummy_image_001.png",
                "density_label": 1,
                "accession": "ACC001",
            }
        ]

    def test_default_cache_mode_is_none(self, minimal_rows):
        """Test that default cache_mode is 'none'."""
        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
        )

        assert dataset.cache_mode == "none"

    def test_explicit_none_cache_mode(self, minimal_rows):
        """Test that None as cache_mode defaults to 'none'."""
        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode=None,
        )

        assert dataset.cache_mode == "none"

    def test_cache_mode_whitespace_handling(self, minimal_rows):
        """Test that cache_mode with whitespace is handled correctly."""
        # Note: The current implementation does .lower() but doesn't strip
        # This test documents the actual behavior
        with pytest.raises(ValueError) as exc_info:
            MammoDensityDataset(
                rows=minimal_rows,
                img_size=224,
                train=False,
                cache_mode=" none ",  # whitespace around 'none'
            )

        assert "cache_mode inválido" in str(exc_info.value)


class TestCacheIndexInitialization:
    """Test that cache indexes are initialized correctly based on cache_mode."""

    @pytest.fixture
    def minimal_rows(self):
        """Create minimal row data for dataset initialization."""
        return [
            {
                "image_path": "dummy_image_001.png",
                "density_label": 1,
                "accession": "ACC001",
            }
        ]

    def test_none_cache_mode_no_image_cache(self, minimal_rows):
        """Test that cache_mode='none' doesn't create image cache."""
        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="none",
        )

        assert dataset._image_cache is None

    def test_memory_cache_mode_creates_image_cache(self, minimal_rows):
        """Test that cache_mode='memory' creates empty image cache dict."""
        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="memory",
        )

        assert dataset._image_cache is not None
        assert isinstance(dataset._image_cache, dict)
        assert len(dataset._image_cache) == 0

    def test_disk_cache_mode_creates_disk_index(self, minimal_rows, tmp_path):
        """Test that cache_mode='disk' creates disk cache index."""
        cache_dir = tmp_path / "cache"

        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="disk",
            cache_dir=str(cache_dir),
        )

        assert hasattr(dataset, "_disk_cache_index")
        assert isinstance(dataset._disk_cache_index, dict)

    def test_tensor_disk_cache_mode_creates_tensor_index(self, minimal_rows, tmp_path):
        """Test that cache_mode='tensor-disk' creates tensor disk index."""
        cache_dir = tmp_path / "cache"

        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="tensor-disk",
            cache_dir=str(cache_dir),
        )

        assert hasattr(dataset, "_tensor_disk_index")
        assert isinstance(dataset._tensor_disk_index, dict)

    def test_tensor_memmap_cache_mode_creates_memmap_index(
        self, minimal_rows, tmp_path
    ):
        """Test that cache_mode='tensor-memmap' creates memmap index."""
        cache_dir = tmp_path / "cache"

        dataset = MammoDensityDataset(
            rows=minimal_rows,
            img_size=224,
            train=False,
            cache_mode="tensor-memmap",
            cache_dir=str(cache_dir),
        )

        assert hasattr(dataset, "_tensor_memmap_index")
        assert isinstance(dataset._tensor_memmap_index, dict)


class TestResolveDatasetCacheMode:
    """Test resolve_dataset_cache_mode() function."""

    def test_non_auto_mode_passthrough_none(self):
        """Test that non-auto mode 'none' is returned as-is."""
        rows = [{"image_path": "test.png"}]
        result = resolve_dataset_cache_mode("none", rows)
        assert result == "none"

    def test_non_auto_mode_passthrough_memory(self):
        """Test that non-auto mode 'memory' is returned as-is."""
        rows = [{"image_path": "test.png"}]
        result = resolve_dataset_cache_mode("memory", rows)
        assert result == "memory"

    def test_non_auto_mode_passthrough_disk(self):
        """Test that non-auto mode 'disk' is returned as-is."""
        rows = [{"image_path": "test.png"}]
        result = resolve_dataset_cache_mode("disk", rows)
        assert result == "disk"

    def test_non_auto_mode_passthrough_tensor_disk(self):
        """Test that non-auto mode 'tensor-disk' is returned as-is."""
        rows = [{"image_path": "test.png"}]
        result = resolve_dataset_cache_mode("tensor-disk", rows)
        assert result == "tensor-disk"

    def test_non_auto_mode_passthrough_tensor_memmap(self):
        """Test that non-auto mode 'tensor-memmap' is returned as-is."""
        rows = [{"image_path": "test.png"}]
        result = resolve_dataset_cache_mode("tensor-memmap", rows)
        assert result == "tensor-memmap"

    def test_non_auto_mode_case_insensitive(self):
        """Test that non-auto mode is normalized to lowercase."""
        rows = [{"image_path": "test.png"}]
        result = resolve_dataset_cache_mode("MEMORY", rows)
        assert result == "memory"

    def test_auto_mode_empty_rows(self):
        """Test that auto mode returns 'none' for empty rows."""
        rows = []
        result = resolve_dataset_cache_mode("auto", rows)
        assert result == "none"

    def test_auto_mode_no_image_paths(self):
        """Test that auto mode returns 'none' when no image_path in rows."""
        rows = [{"density_label": 1, "accession": "ACC001"}]
        result = resolve_dataset_cache_mode("auto", rows)
        assert result == "none"

    def test_auto_mode_small_dataset_non_dicom(self):
        """Test auto mode returns 'memory' for small non-DICOM dataset (<= 1000)."""
        rows = [{"image_path": f"image_{i:04d}.png"} for i in range(100)]
        result = resolve_dataset_cache_mode("auto", rows)
        assert result == "memory"

    def test_auto_mode_small_dataset_dicom(self):
        """Test auto mode returns 'disk' for small DICOM dataset (<= 6000)."""
        rows = [{"image_path": f"image_{i:04d}.dcm"} for i in range(100)]
        result = resolve_dataset_cache_mode("auto", rows)
        assert result == "disk"

    def test_auto_mode_boundary_1000_non_dicom(self):
        """Test auto mode at exactly 1000 rows with non-DICOM files."""
        rows = [{"image_path": f"image_{i:04d}.png"} for i in range(1000)]
        result = resolve_dataset_cache_mode("auto", rows)
        assert result == "memory"

    def test_auto_mode_boundary_1001_non_dicom(self):
        """Test auto mode at 1001 rows with non-DICOM returns 'none'."""
        rows = [{"image_path": f"image_{i:04d}.png"} for i in range(1001)]
        result = resolve_dataset_cache_mode("auto", rows)
        assert result == "none"

    def test_auto_mode_medium_dataset_dicom(self):
        """Test auto mode returns 'disk' for medium DICOM dataset (1001-6000)."""
        rows = [{"image_path": f"image_{i:04d}.dcm"} for i in range(3000)]
        result = resolve_dataset_cache_mode("auto", rows)
        assert result == "disk"

    def test_auto_mode_boundary_6000_dicom(self):
        """Test auto mode at exactly 6000 rows with DICOM files."""
        rows = [{"image_path": f"image_{i:05d}.dcm"} for i in range(6000)]
        result = resolve_dataset_cache_mode("auto", rows)
        assert result == "disk"

    def test_auto_mode_boundary_6001_dicom(self):
        """Test auto mode at 6001 rows with DICOM returns 'tensor-disk'."""
        rows = [{"image_path": f"image_{i:05d}.dcm"} for i in range(6001)]
        result = resolve_dataset_cache_mode("auto", rows)
        assert result == "tensor-disk"

    def test_auto_mode_large_dataset_dicom(self):
        """Test auto mode returns 'tensor-disk' for large DICOM dataset (> 6000)."""
        rows = [{"image_path": f"image_{i:05d}.dcm"} for i in range(10000)]
        result = resolve_dataset_cache_mode("auto", rows)
        assert result == "tensor-disk"

    def test_auto_mode_dicom_extension_case_insensitive(self):
        """Test auto mode detects DICOM files with various case extensions."""
        rows = [
            {"image_path": "image_001.DCM"},
            {"image_path": "image_002.dcm"},
            {"image_path": "image_003.DICOM"},
        ]
        result = resolve_dataset_cache_mode("auto", rows)
        assert result == "disk"

    def test_auto_mode_mixed_dicom_and_png(self):
        """Test auto mode with mixed DICOM and PNG files (has_dicom=True)."""
        rows = [
            {"image_path": "image_001.dcm"},
            {"image_path": "image_002.png"},
        ]
        result = resolve_dataset_cache_mode("auto", rows)
        # DICOM detected -> always use disk caching to avoid excessive RAM
        assert result == "disk"

    def test_auto_mode_mixed_files_medium_size(self):
        """Test auto mode with mixed files at medium size uses disk cache."""
        rows = [{"image_path": f"image_{i:04d}.dcm"} for i in range(1500)]
        rows.extend([{"image_path": f"image_{i:04d}.png"} for i in range(100)])
        result = resolve_dataset_cache_mode("auto", rows)
        # Total 1600, has DICOM, so should be 'disk'
        assert result == "disk"

    def test_env_override_memory(self):
        """Test that MAMMO_CACHE_MODE environment variable overrides requested mode."""
        rows = [{"image_path": f"image_{i:04d}.dcm"} for i in range(10000)]
        with patch.dict(os.environ, {"MAMMO_CACHE_MODE": "memory"}):
            result = resolve_dataset_cache_mode("auto", rows)
            assert result == "memory"

    def test_env_override_disk(self):
        """Test that MAMMO_CACHE_MODE can override to 'disk'."""
        rows = [{"image_path": "test.png"}]
        with patch.dict(os.environ, {"MAMMO_CACHE_MODE": "disk"}):
            result = resolve_dataset_cache_mode("none", rows)
            assert result == "disk"

    def test_env_override_none(self):
        """Test that MAMMO_CACHE_MODE can override to 'none'."""
        rows = [{"image_path": f"image_{i:04d}.dcm"} for i in range(100)]
        with patch.dict(os.environ, {"MAMMO_CACHE_MODE": "none"}):
            result = resolve_dataset_cache_mode("auto", rows)
            assert result == "none"

    def test_env_override_tensor_disk(self):
        """Test that MAMMO_CACHE_MODE can override to 'tensor-disk'."""
        rows = [{"image_path": "test.png"}]
        with patch.dict(os.environ, {"MAMMO_CACHE_MODE": "tensor-disk"}):
            result = resolve_dataset_cache_mode("memory", rows)
            assert result == "tensor-disk"

    def test_env_override_tensor_memmap(self):
        """Test that MAMMO_CACHE_MODE can override to 'tensor-memmap'."""
        rows = [{"image_path": "test.png"}]
        with patch.dict(os.environ, {"MAMMO_CACHE_MODE": "tensor-memmap"}):
            result = resolve_dataset_cache_mode("memory", rows)
            assert result == "tensor-memmap"

    def test_env_override_auto(self):
        """Test that MAMMO_CACHE_MODE can be set to 'auto'."""
        rows = [{"image_path": f"image_{i:04d}.dcm"} for i in range(100)]
        with patch.dict(os.environ, {"MAMMO_CACHE_MODE": "auto"}):
            result = resolve_dataset_cache_mode("memory", rows)
            # Override to auto, which evaluates to 'disk' for DICOM datasets
            assert result == "disk"

    def test_env_override_case_insensitive(self):
        """Test that MAMMO_CACHE_MODE override is case-insensitive."""
        rows = [{"image_path": "test.png"}]
        with patch.dict(os.environ, {"MAMMO_CACHE_MODE": "MEMORY"}):
            result = resolve_dataset_cache_mode("none", rows)
            assert result == "memory"

    def test_env_override_with_whitespace(self):
        """Test that MAMMO_CACHE_MODE override handles whitespace."""
        rows = [{"image_path": "test.png"}]
        with patch.dict(os.environ, {"MAMMO_CACHE_MODE": "  disk  "}):
            result = resolve_dataset_cache_mode("none", rows)
            assert result == "disk"

    def test_env_override_invalid_value_ignored(self):
        """Test that invalid MAMMO_CACHE_MODE value is ignored."""
        rows = [{"image_path": f"image_{i:04d}.dcm"} for i in range(100)]
        with patch.dict(os.environ, {"MAMMO_CACHE_MODE": "invalid_mode"}):
            result = resolve_dataset_cache_mode("auto", rows)
            # Should fall back to auto mode logic (disk for DICOM)
            assert result == "disk"

    def test_default_mode_none_normalized(self):
        """Test that None as requested mode defaults to 'none'."""
        rows = [{"image_path": "test.png"}]
        result = resolve_dataset_cache_mode(None, rows)
        assert result == "none"

    def test_empty_string_mode_normalized(self):
        """Test that empty string as requested mode defaults to 'none'."""
        rows = [{"image_path": "test.png"}]
        result = resolve_dataset_cache_mode("", rows)
        assert result == "none"

    def test_auto_mode_with_dataframe(self):
        """Test auto mode works with pandas DataFrame input."""
        import pandas as pd

        df = pd.DataFrame([{"image_path": f"image_{i:04d}.png"} for i in range(100)])
        result = resolve_dataset_cache_mode("auto", df)
        assert result == "memory"

    def test_auto_mode_with_dataframe_dicom(self):
        """Test auto mode with DataFrame and DICOM files."""
        import pandas as pd

        df = pd.DataFrame([{"image_path": f"image_{i:04d}.dcm"} for i in range(3000)])
        result = resolve_dataset_cache_mode("auto", df)
        assert result == "disk"

    def test_auto_mode_rows_with_missing_image_path(self):
        """Test auto mode handles rows with missing image_path."""
        rows = [
            {"image_path": "image_001.dcm"},
            {"density_label": 1},  # No image_path
            {"image_path": "image_002.dcm"},
        ]
        result = resolve_dataset_cache_mode("auto", rows)
        # DICOM detected -> always use disk caching
        assert result == "disk"

    def test_auto_mode_rows_with_none_image_path(self):
        """Test auto mode handles rows with None as image_path."""
        rows = [
            {"image_path": "image_001.dcm"},
            {"image_path": None},
            {"image_path": "image_002.dcm"},
        ]
        result = resolve_dataset_cache_mode("auto", rows)
        # DICOM detected -> always use disk caching
        assert result == "disk"
