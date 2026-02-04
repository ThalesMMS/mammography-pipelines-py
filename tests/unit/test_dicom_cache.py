"""
Unit tests for DICOM LRU cache functionality.

These tests validate the DicomLRUCache class and its caching behavior.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import copy
import json
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pydicom = pytest.importorskip("pydicom")

from mammography.io.dicom_cache import DicomLRUCache


def _save_dicom(dataset, path: str) -> None:
    """Helper to save DICOM dataset to file."""
    try:
        dataset.save_as(path, enforce_file_format=True)
    except TypeError:
        dataset.save_as(path, write_like_original=False)


class TestDicomLRUCacheInitialization:
    """Test DicomLRUCache initialization."""

    def test_valid_initialization(self):
        """Test that cache initializes with valid max_size."""
        cache = DicomLRUCache(max_size=10)

        assert cache.max_size == 10
        assert cache.size == 0
        assert cache.cache_dir is None
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.evictions == 0

    def test_initialization_with_cache_dir(self, tmp_path):
        """Test that cache initializes with cache_dir and creates directory."""
        cache_dir = tmp_path / "cache"
        cache = DicomLRUCache(max_size=10, cache_dir=cache_dir)

        assert cache.max_size == 10
        assert cache.cache_dir == cache_dir
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_initialization_with_existing_cache_dir(self, tmp_path):
        """Test initialization with existing cache_dir doesn't fail."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache = DicomLRUCache(max_size=10, cache_dir=cache_dir)

        assert cache.cache_dir == cache_dir
        assert cache_dir.exists()

    def test_initialization_with_string_cache_dir(self, tmp_path):
        """Test that cache_dir can be provided as string."""
        cache_dir = tmp_path / "cache"
        cache = DicomLRUCache(max_size=10, cache_dir=str(cache_dir))

        assert cache.cache_dir == cache_dir
        assert cache_dir.exists()

    def test_invalid_max_size_negative(self):
        """Test that negative max_size raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            DicomLRUCache(max_size=-1)

        assert "must be a positive integer" in str(exc_info.value)

    def test_invalid_max_size_zero(self):
        """Test that zero max_size raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            DicomLRUCache(max_size=0)

        assert "must be a positive integer" in str(exc_info.value)

    def test_invalid_max_size_non_integer(self):
        """Test that non-integer max_size raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            DicomLRUCache(max_size=10.5)

        assert "must be a positive integer" in str(exc_info.value)


class TestDicomLRUCacheBehavior:
    """Test LRU cache behavior."""

    @pytest.fixture
    def valid_dicom_dataset(self):
        """Create a valid DICOM dataset for testing."""
        dataset = pydicom.Dataset()

        # Required fields
        dataset.PatientID = "TEST_PATIENT_001"
        dataset.StudyInstanceUID = "1.2.840.12345.123456789"
        dataset.SeriesInstanceUID = "1.2.840.12345.987654321"
        dataset.SOPInstanceUID = "1.2.840.12345.456789123"
        dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"

        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dataset.file_meta = file_meta

        # Image metadata
        dataset.Manufacturer = "SIEMENS"
        dataset.PixelSpacing = [0.1, 0.1]
        dataset.BitsStored = 16
        dataset.BitsAllocated = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.Rows = 64
        dataset.Columns = 64

        # Create pixel data with fixed seed for reproducibility
        rng = np.random.RandomState(42)
        dataset.PixelData = rng.randint(
            0, 4095, (64, 64), dtype=np.uint16
        ).tobytes()

        return dataset

    @pytest.fixture
    def dicom_file(self, valid_dicom_dataset, tmp_path):
        """Create a temporary DICOM file for testing."""
        filepath = tmp_path / "test_image.dcm"
        _save_dicom(valid_dicom_dataset, str(filepath))
        return filepath

    @pytest.fixture
    def multiple_dicom_files(self, valid_dicom_dataset, tmp_path):
        """Create multiple DICOM files for testing."""
        files = []
        rng = np.random.RandomState(43)
        for i in range(5):
            # Create a copy for each file to avoid mutation
            dataset_copy = copy.deepcopy(valid_dicom_dataset)
            dataset_copy.PatientID = f"TEST_PATIENT_{i:03d}"
            dataset_copy.SOPInstanceUID = f"1.2.840.12345.456789{i:03d}"
            # Update pixel data to make each file unique
            dataset_copy.PixelData = rng.randint(
                0, 4095, (64, 64), dtype=np.uint16
            ).tobytes()
            filepath = tmp_path / f"test_image_{i}.dcm"
            _save_dicom(dataset_copy, str(filepath))
            files.append(filepath)
        return files

    def test_cache_miss_on_first_access(self, dicom_file):
        """Test that first access to a file is a cache miss."""
        cache = DicomLRUCache(max_size=10)

        assert cache.misses == 0
        assert cache.hits == 0

        # First access should be a miss
        ds = cache.get(dicom_file)

        assert cache.misses == 1
        assert cache.hits == 0
        assert cache.size == 1
        assert ds.PatientID == "TEST_PATIENT_001"

    def test_cache_hit_on_second_access(self, dicom_file):
        """Test that second access to same file is a cache hit."""
        cache = DicomLRUCache(max_size=10)

        # First access (miss)
        ds1 = cache.get(dicom_file)
        assert cache.misses == 1
        assert cache.hits == 0

        # Second access (hit)
        ds2 = cache.get(dicom_file)
        assert cache.misses == 1
        assert cache.hits == 1
        assert cache.size == 1

        # Should be same object
        assert ds1 is ds2

    def test_lru_eviction_when_full(self, multiple_dicom_files):
        """Test that LRU eviction occurs when cache is full."""
        cache = DicomLRUCache(max_size=3)

        # Fill cache with 3 files
        for i in range(3):
            cache.get(multiple_dicom_files[i])

        assert cache.size == 3
        assert cache.evictions == 0

        # Add 4th file - should evict first file
        cache.get(multiple_dicom_files[3])

        assert cache.size == 3
        assert cache.evictions == 1

        # First file should no longer be in cache
        assert multiple_dicom_files[0] not in cache
        # Other files should still be in cache
        assert multiple_dicom_files[1] in cache
        assert multiple_dicom_files[2] in cache
        assert multiple_dicom_files[3] in cache

    def test_lru_order_maintained(self, multiple_dicom_files):
        """Test that LRU order is maintained correctly."""
        cache = DicomLRUCache(max_size=3)

        # Fill cache with files 0, 1, 2
        for i in range(3):
            cache.get(multiple_dicom_files[i])

        # Access file 0 again (makes it most recently used)
        cache.get(multiple_dicom_files[0])

        # Add file 3 - should evict file 1 (least recently used)
        cache.get(multiple_dicom_files[3])

        assert cache.size == 3
        assert cache.evictions == 1

        # File 1 should be evicted
        assert multiple_dicom_files[1] not in cache
        # Files 0, 2, 3 should remain
        assert multiple_dicom_files[0] in cache
        assert multiple_dicom_files[2] in cache
        assert multiple_dicom_files[3] in cache

    def test_multiple_evictions(self, multiple_dicom_files):
        """Test that multiple evictions work correctly."""
        cache = DicomLRUCache(max_size=2)

        # Add files one by one
        for i in range(5):
            cache.get(multiple_dicom_files[i])

        assert cache.size == 2
        assert cache.evictions == 3

        # Only last 2 files should be in cache
        assert multiple_dicom_files[0] not in cache
        assert multiple_dicom_files[1] not in cache
        assert multiple_dicom_files[2] not in cache
        assert multiple_dicom_files[3] in cache
        assert multiple_dicom_files[4] in cache

    def test_stop_before_pixels_parameter(self, dicom_file):
        """Test that stop_before_pixels parameter is passed through."""
        cache = DicomLRUCache(max_size=10)

        # Load without pixel data
        ds = cache.get(dicom_file, stop_before_pixels=True)

        assert ds is not None
        assert cache.size == 1

        # Note: With stop_before_pixels=True, PixelData attribute exists
        # but pixel_array may not be readily available without reloading


class TestDicomLRUCacheStatistics:
    """Test cache statistics tracking."""

    @pytest.fixture
    def dicom_file(self, tmp_path):
        """Create a temporary DICOM file for testing."""
        dataset = pydicom.Dataset()
        dataset.PatientID = "TEST"
        dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        dataset.SOPInstanceUID = "1.2.3.4"

        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dataset.file_meta = file_meta

        dataset.Rows = 32
        dataset.Columns = 32
        dataset.BitsStored = 16
        dataset.BitsAllocated = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        rng = np.random.RandomState(44)
        dataset.PixelData = rng.randint(0, 4095, (32, 32), dtype=np.uint16).tobytes()

        filepath = tmp_path / "test.dcm"
        _save_dicom(dataset, str(filepath))
        return filepath

    def test_hit_rate_calculation(self, dicom_file):
        """Test that hit rate is calculated correctly."""
        cache = DicomLRUCache(max_size=10)

        # Initially 0% hit rate
        assert cache.hit_rate == 0.0

        # First access (miss) - 0% hit rate
        cache.get(dicom_file)
        assert cache.hit_rate == 0.0

        # Second access (hit) - 50% hit rate
        cache.get(dicom_file)
        assert cache.hit_rate == 0.5

        # Third access (hit) - 66.7% hit rate
        cache.get(dicom_file)
        assert abs(cache.hit_rate - 2/3) < 0.001

    def test_stats_dictionary(self, dicom_file):
        """Test that stats property returns correct dictionary."""
        cache = DicomLRUCache(max_size=10)

        # Access file twice
        cache.get(dicom_file)
        cache.get(dicom_file)

        stats = cache.stats

        assert stats["size"] == 1
        assert stats["max_size"] == 10
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["evictions"] == 0
        assert stats["hit_rate"] == 0.5

    def test_reset_stats(self, dicom_file):
        """Test that reset_stats clears statistics."""
        cache = DicomLRUCache(max_size=10)

        # Generate some stats
        cache.get(dicom_file)
        cache.get(dicom_file)

        assert cache.hits > 0
        assert cache.misses > 0

        # Reset stats
        cache.reset_stats()

        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.evictions == 0
        assert cache.hit_rate == 0.0

        # Cache should still contain the file
        assert cache.size == 1


class TestDicomLRUCacheOperations:
    """Test cache operations."""

    @pytest.fixture
    def dicom_file(self, tmp_path):
        """Create a temporary DICOM file for testing."""
        dataset = pydicom.Dataset()
        dataset.PatientID = "TEST"
        dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        dataset.SOPInstanceUID = "1.2.3.4"

        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dataset.file_meta = file_meta

        dataset.Rows = 32
        dataset.Columns = 32
        dataset.BitsStored = 16
        dataset.BitsAllocated = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        rng = np.random.RandomState(44)
        dataset.PixelData = rng.randint(0, 4095, (32, 32), dtype=np.uint16).tobytes()

        filepath = tmp_path / "test.dcm"
        _save_dicom(dataset, str(filepath))
        return filepath

    def test_clear_operation(self, dicom_file):
        """Test that clear removes all entries from cache."""
        cache = DicomLRUCache(max_size=10)

        # Add file to cache
        cache.get(dicom_file)
        assert cache.size == 1

        # Clear cache
        cache.clear()

        assert cache.size == 0
        assert dicom_file not in cache

        # Stats should be preserved
        assert cache.misses == 1

    def test_evict_operation_existing_file(self, dicom_file):
        """Test that evict removes specific file from cache."""
        cache = DicomLRUCache(max_size=10)

        # Add file to cache
        cache.get(dicom_file)
        assert cache.size == 1
        assert dicom_file in cache

        # Evict file
        result = cache.evict(dicom_file)

        assert result is True
        assert cache.size == 0
        assert dicom_file not in cache

    def test_evict_operation_non_existing_file(self, tmp_path):
        """Test that evict returns False for non-cached file."""
        cache = DicomLRUCache(max_size=10)

        non_existent = tmp_path / "non_existent.dcm"
        result = cache.evict(non_existent)

        assert result is False
        assert cache.size == 0

    def test_contains_operator(self, dicom_file):
        """Test __contains__ operator."""
        cache = DicomLRUCache(max_size=10)

        # File not in cache initially
        assert dicom_file not in cache

        # Add file to cache
        cache.get(dicom_file)

        # File should now be in cache
        assert dicom_file in cache

    def test_len_operator(self, dicom_file):
        """Test __len__ operator."""
        cache = DicomLRUCache(max_size=10)

        assert len(cache) == 0

        cache.get(dicom_file)

        assert len(cache) == 1

    def test_repr(self):
        """Test __repr__ method."""
        cache = DicomLRUCache(max_size=10)

        repr_str = repr(cache)

        assert "DicomLRUCache" in repr_str
        assert "max_size=10" in repr_str
        assert "size=0" in repr_str
        assert "hit_rate" in repr_str


class TestDicomLRUCachePersistence:
    """Test disk persistence functionality."""

    @pytest.fixture
    def dicom_file(self, tmp_path):
        """Create a temporary DICOM file for testing."""
        dataset = pydicom.Dataset()
        dataset.PatientID = "TEST"
        dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        dataset.SOPInstanceUID = "1.2.3.4"

        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dataset.file_meta = file_meta

        dataset.Rows = 32
        dataset.Columns = 32
        dataset.BitsStored = 16
        dataset.BitsAllocated = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        rng = np.random.RandomState(44)
        dataset.PixelData = rng.randint(0, 4095, (32, 32), dtype=np.uint16).tobytes()

        filepath = tmp_path / "test.dcm"
        _save_dicom(dataset, str(filepath))
        return filepath

    def test_save_with_cache_dir(self, tmp_path, dicom_file):
        """Test that save creates metadata file."""
        cache_dir = tmp_path / "cache"
        cache = DicomLRUCache(max_size=10, cache_dir=cache_dir)

        # Add file to cache
        cache.get(dicom_file)

        # Save cache
        cache.save()

        # Check metadata file exists
        metadata_file = cache_dir / "cache_metadata.json"
        assert metadata_file.exists()

        # Check metadata content
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        assert "cache_keys" in metadata
        assert "max_size" in metadata
        assert "stats" in metadata
        assert len(metadata["cache_keys"]) == 1
        assert metadata["max_size"] == 10

    def test_save_without_cache_dir(self):
        """Test that save raises ValueError without cache_dir."""
        cache = DicomLRUCache(max_size=10)

        with pytest.raises(ValueError) as exc_info:
            cache.save()

        assert "cache_dir was not specified" in str(exc_info.value)

    def test_load_restores_metadata(self, tmp_path, dicom_file):
        """Test that load restores cache metadata."""
        cache_dir = tmp_path / "cache"
        cache1 = DicomLRUCache(max_size=10, cache_dir=cache_dir)

        # Add file and generate stats
        cache1.get(dicom_file)
        cache1.get(dicom_file)

        # Save cache
        cache1.save()

        # Create new cache and load
        cache2 = DicomLRUCache(max_size=5, cache_dir=cache_dir)
        cache2.load()

        # Check restored values
        assert cache2.max_size == 10  # Restored from metadata
        assert cache2.hits == 1
        assert cache2.misses == 1
        assert cache2.evictions == 0

    def test_load_without_cache_dir(self):
        """Test that load raises ValueError without cache_dir."""
        cache = DicomLRUCache(max_size=10)

        with pytest.raises(ValueError) as exc_info:
            cache.load()

        assert "cache_dir was not specified" in str(exc_info.value)

    def test_load_missing_metadata_file(self, tmp_path):
        """Test that load raises FileNotFoundError for missing metadata."""
        cache_dir = tmp_path / "cache"
        cache = DicomLRUCache(max_size=10, cache_dir=cache_dir)

        with pytest.raises(FileNotFoundError) as exc_info:
            cache.load()

        assert "Cache metadata file not found" in str(exc_info.value)

    def test_save_preserves_lru_order(self, tmp_path):
        """Test that save preserves LRU order of cache keys."""
        cache_dir = tmp_path / "cache"
        cache = DicomLRUCache(max_size=10, cache_dir=cache_dir)

        # Create multiple DICOM files
        files = []
        rng = np.random.RandomState(45)
        for i in range(3):
            dataset = pydicom.Dataset()
            dataset.PatientID = f"TEST_{i}"
            dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
            dataset.SOPInstanceUID = f"1.2.3.{i}"

            file_meta = pydicom.dataset.FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
            file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            dataset.file_meta = file_meta

            dataset.Rows = 32
            dataset.Columns = 32
            dataset.BitsStored = 16
            dataset.BitsAllocated = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = 0
            dataset.SamplesPerPixel = 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.PixelData = rng.randint(0, 4095, (32, 32), dtype=np.uint16).tobytes()

            filepath = tmp_path / f"test_{i}.dcm"
            _save_dicom(dataset, str(filepath))
            files.append(filepath)

        # Add files to cache in order
        for f in files:
            cache.get(f)

        # Save cache
        cache.save()

        # Read metadata
        metadata_file = cache_dir / "cache_metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Check that keys are in correct order
        assert len(metadata["cache_keys"]) == 3
        # Keys should be absolute paths
        for i, key in enumerate(metadata["cache_keys"]):
            assert str(files[i].resolve()) == key


class TestDicomLRUCacheErrorHandling:
    """Test error handling."""

    def test_file_not_found_error(self, tmp_path):
        """Test that FileNotFoundError is raised for non-existent file."""
        cache = DicomLRUCache(max_size=10)

        non_existent = tmp_path / "does_not_exist.dcm"

        with pytest.raises(FileNotFoundError) as exc_info:
            cache.get(non_existent)

        assert "DICOM file not found" in str(exc_info.value)
        assert str(non_existent) in str(exc_info.value)

    def test_invalid_dicom_error(self, tmp_path):
        """Test that InvalidDicomError is raised for invalid DICOM file."""
        cache = DicomLRUCache(max_size=10)

        # Create invalid DICOM file
        invalid_file = tmp_path / "invalid.dcm"
        invalid_file.write_text("This is not a DICOM file")

        with pytest.raises(pydicom.errors.InvalidDicomError) as exc_info:
            cache.get(invalid_file)

        assert "Failed to read DICOM file" in str(exc_info.value)

    def test_path_normalization(self, tmp_path):
        """Test that different path formats refer to same cache entry."""
        # Create DICOM file
        dataset = pydicom.Dataset()
        dataset.PatientID = "TEST"
        dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        dataset.SOPInstanceUID = "1.2.3.4"

        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dataset.file_meta = file_meta

        dataset.Rows = 32
        dataset.Columns = 32
        dataset.BitsStored = 16
        dataset.BitsAllocated = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        rng = np.random.RandomState(46)
        dataset.PixelData = rng.randint(0, 4095, (32, 32), dtype=np.uint16).tobytes()

        filepath = tmp_path / "test.dcm"
        _save_dicom(dataset, str(filepath))

        cache = DicomLRUCache(max_size=10)

        # Access with Path object
        cache.get(filepath)
        assert cache.misses == 1
        assert cache.hits == 0

        # Access with string path (should be cache hit)
        cache.get(str(filepath))
        assert cache.misses == 1
        assert cache.hits == 1

        # Should only have one entry
        assert cache.size == 1
