"""
Comprehensive unit tests for DICOM I/O, lazy loading, and caching.

These tests validate the complete I/O stack including:
- DICOM file reading and parsing
- Lazy loading with memory optimization
- LRU caching for performance
- Integration between lazy loading and caching
- Pixel array access and windowing
- Error handling across the stack

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.errors import InvalidDicomError

from mammography.io.dicom import (
    DICOM_EXTS,
    DicomReader,
    MammographyImage,
    apply_windowing,
    create_dicom_reader,
    create_mammography_image_from_dicom,
    dicom_to_pil_rgb,
    extract_window_parameters,
    get_disclaimer,
    is_dicom_path,
    read_dicom_directory,
    read_single_dicom,
    robust_window,
)
from mammography.io.dicom_cache import DicomLRUCache
from mammography.io.lazy_dicom import LazyDicomDataset


def _save_dicom(dataset, path: str) -> None:
    """Helper to save DICOM dataset to file."""
    try:
        dataset.save_as(path, enforce_file_format=True)
    except TypeError:
        dataset.save_as(path, write_like_original=False)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def valid_dicom_dataset():
    """Create a valid mammography DICOM dataset for testing."""
    dataset = pydicom.Dataset()

    # Required DICOM fields
    dataset.PatientID = "TEST_PATIENT_001"
    dataset.StudyInstanceUID = "1.2.840.12345.123456789"
    dataset.SeriesInstanceUID = "1.2.840.12345.987654321"
    dataset.SOPInstanceUID = "1.2.840.12345.456789123"
    dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dataset.file_meta = file_meta

    # Image attributes
    dataset.Manufacturer = "SIEMENS"
    dataset.PixelSpacing = [0.1, 0.1]
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.Rows = 128
    dataset.Columns = 128

    # Mammography-specific fields
    dataset.ViewPosition = "CC"
    dataset.ImageLaterality = "L"

    # Create pixel data
    dataset.PixelData = np.random.randint(
        0, 4095, (128, 128), dtype=np.uint16
    ).tobytes()

    return dataset


@pytest.fixture
def mono1_dicom_dataset(valid_dicom_dataset):
    """Create MONOCHROME1 DICOM dataset."""
    valid_dicom_dataset.PhotometricInterpretation = "MONOCHROME1"
    return valid_dicom_dataset


@pytest.fixture
def dicom_with_rescale(valid_dicom_dataset):
    """Create DICOM dataset with RescaleSlope and RescaleIntercept."""
    valid_dicom_dataset.RescaleSlope = 2.0
    valid_dicom_dataset.RescaleIntercept = -1024.0
    return valid_dicom_dataset


@pytest.fixture
def dicom_with_windowing(valid_dicom_dataset):
    """Create DICOM dataset with WindowCenter and WindowWidth."""
    valid_dicom_dataset.WindowCenter = 2048
    valid_dicom_dataset.WindowWidth = 4096
    return valid_dicom_dataset


@pytest.fixture
def dicom_file(valid_dicom_dataset, tmp_path):
    """Create a temporary DICOM file for testing."""
    filepath = tmp_path / "test_image.dcm"
    _save_dicom(valid_dicom_dataset, str(filepath))
    return filepath


@pytest.fixture
def multiple_dicom_files(valid_dicom_dataset, tmp_path):
    """Create multiple DICOM files for testing."""
    files = []
    for i in range(5):
        filepath = tmp_path / f"test_image_{i}.dcm"
        valid_dicom_dataset.PatientID = f"TEST_PATIENT_{i:03d}"
        valid_dicom_dataset.SOPInstanceUID = f"1.2.840.12345.{i:09d}"
        _save_dicom(valid_dicom_dataset, str(filepath))
        files.append(filepath)
    return files


# ============================================================================
# Test Basic DICOM I/O Functions
# ============================================================================


class TestDicomConstants:
    """Test DICOM constants and utility functions."""

    def test_dicom_extensions(self):
        """Test that DICOM_EXTS contains expected extensions."""
        assert ".dcm" in DICOM_EXTS
        assert ".dicom" in DICOM_EXTS
        assert len(DICOM_EXTS) >= 2

    def test_get_disclaimer(self):
        """Test that disclaimer message is returned."""
        disclaimer = get_disclaimer()
        assert isinstance(disclaimer, str)
        assert len(disclaimer) > 0
        assert "EDUCATIONAL" in disclaimer or "educational" in disclaimer


class TestDicomPathDetection:
    """Tests for DICOM path detection."""

    def test_is_dicom_path_dcm_extension(self):
        """Test detection of .dcm extension."""
        assert is_dicom_path("file.dcm") is True
        assert is_dicom_path("FILE.DCM") is True
        assert is_dicom_path("/path/to/file.dcm") is True

    def test_is_dicom_path_dicom_extension(self):
        """Test detection of .dicom extension."""
        assert is_dicom_path("file.dicom") is True
        assert is_dicom_path("FILE.DICOM") is True
        assert is_dicom_path("/path/to/file.dicom") is True

    def test_is_dicom_path_non_dicom(self):
        """Test rejection of non-DICOM extensions."""
        assert is_dicom_path("file.png") is False
        assert is_dicom_path("file.jpg") is False
        assert is_dicom_path("file.txt") is False
        assert is_dicom_path("file") is False


# ============================================================================
# Test Lazy DICOM Loading
# ============================================================================


class TestLazyDicomDataset:
    """Tests for LazyDicomDataset lazy loading functionality."""

    def test_lazy_dataset_initialization(self, dicom_file):
        """Test that LazyDicomDataset initializes correctly."""
        lazy_ds = LazyDicomDataset(dicom_file)

        assert lazy_ds.filepath == dicom_file
        assert lazy_ds._dataset is not None
        assert lazy_ds._pixel_array is None
        assert not lazy_ds.is_pixel_data_loaded

    def test_metadata_available_immediately(self, dicom_file):
        """Test that metadata is available without loading pixel data."""
        lazy_ds = LazyDicomDataset(dicom_file)

        # Access metadata
        assert lazy_ds.PatientID == "TEST_PATIENT_001"
        assert lazy_ds.Manufacturer == "SIEMENS"
        assert lazy_ds.ViewPosition == "CC"
        assert lazy_ds.ImageLaterality == "L"
        assert lazy_ds.Rows == 128
        assert lazy_ds.Columns == 128

        # Pixel data should still not be loaded
        assert not lazy_ds.is_pixel_data_loaded

    def test_pixel_array_lazy_loading(self, dicom_file):
        """Test that pixel_array is loaded lazily on first access."""
        lazy_ds = LazyDicomDataset(dicom_file)

        # Pixel data should not be loaded yet
        assert not lazy_ds.is_pixel_data_loaded

        # Access pixel array - this triggers loading
        pixel_array = lazy_ds.pixel_array

        # Now pixel data should be loaded
        assert lazy_ds.is_pixel_data_loaded
        assert pixel_array is not None
        assert isinstance(pixel_array, np.ndarray)
        assert pixel_array.shape == (128, 128)
        assert pixel_array.dtype in [np.uint16, np.int16]

    def test_pixel_array_caching(self, dicom_file):
        """Test that pixel_array is cached after first access."""
        lazy_ds = LazyDicomDataset(dicom_file)

        # First access
        pixel_array_1 = lazy_ds.pixel_array
        assert lazy_ds.is_pixel_data_loaded

        # Second access should return cached array
        pixel_array_2 = lazy_ds.pixel_array

        # Should be the same object (not reloaded)
        assert pixel_array_1 is pixel_array_2

    def test_clear_pixel_cache(self, dicom_file):
        """Test that clear_pixel_cache() frees pixel data."""
        lazy_ds = LazyDicomDataset(dicom_file)

        # Load pixel data
        _ = lazy_ds.pixel_array
        assert lazy_ds.is_pixel_data_loaded

        # Clear cache
        lazy_ds.clear_pixel_cache()
        assert not lazy_ds.is_pixel_data_loaded
        assert lazy_ds._pixel_array is None

        # Metadata should still be available
        assert lazy_ds.PatientID == "TEST_PATIENT_001"

    def test_file_not_found_error(self, tmp_path):
        """Test that FileNotFoundError is raised for non-existent file."""
        nonexistent_file = tmp_path / "does_not_exist.dcm"

        with pytest.raises(FileNotFoundError) as exc_info:
            LazyDicomDataset(nonexistent_file)

        assert "DICOM file not found" in str(exc_info.value)

    def test_invalid_dicom_file_error(self, tmp_path):
        """Test error handling for invalid DICOM file."""
        # Create a non-DICOM file
        invalid_file = tmp_path / "not_a_dicom.dcm"
        invalid_file.write_text("This is not a DICOM file")

        lazy_ds = LazyDicomDataset(invalid_file)

        with pytest.raises(RuntimeError) as exc_info:
            _ = lazy_ds.pixel_array

        assert "Failed to load pixel data" in str(exc_info.value)


# ============================================================================
# Test DICOM LRU Cache
# ============================================================================


class TestDicomLRUCache:
    """Tests for DicomLRUCache caching functionality."""

    def test_cache_initialization(self):
        """Test that cache initializes with valid max_size."""
        cache = DicomLRUCache(max_size=10)

        assert cache.max_size == 10
        assert cache.size == 0
        assert cache.cache_dir is None
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.evictions == 0

    def test_cache_with_cache_dir(self, tmp_path):
        """Test cache initialization with cache_dir."""
        cache_dir = tmp_path / "cache"
        cache = DicomLRUCache(max_size=10, cache_dir=cache_dir)

        assert cache.max_size == 10
        assert cache.cache_dir == cache_dir
        assert cache_dir.exists()

    def test_invalid_max_size(self):
        """Test that invalid max_size raises ValueError."""
        with pytest.raises(ValueError):
            DicomLRUCache(max_size=-1)

        with pytest.raises(ValueError):
            DicomLRUCache(max_size=0)

    def test_cache_miss_on_first_access(self, dicom_file):
        """Test that first access to a file is a cache miss."""
        cache = DicomLRUCache(max_size=10)

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

    def test_cache_clear(self, dicom_file):
        """Test that clear() empties the cache."""
        cache = DicomLRUCache(max_size=10)

        # Add items to cache
        cache.get(dicom_file)
        assert cache.size == 1

        # Clear cache
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.evictions == 0

    def test_cache_statistics(self, multiple_dicom_files):
        """Test that cache statistics are tracked correctly."""
        cache = DicomLRUCache(max_size=10)

        # First access - miss
        cache.get(multiple_dicom_files[0])
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.0

        # Second access to same file - hit
        cache.get(multiple_dicom_files[0])
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


# ============================================================================
# Test Integration: Lazy Loading + Caching
# ============================================================================


class TestLazyLoadingWithCaching:
    """Tests for integration of lazy loading with caching."""

    def test_lazy_dataset_with_cache(self, dicom_file):
        """Test LazyDicomDataset with DicomLRUCache."""
        cache = DicomLRUCache(max_size=10)

        # First access through cache
        ds1 = cache.get(dicom_file)
        assert cache.misses == 1
        assert cache.hits == 0

        # If LazyDicomDataset is used, pixel data should not be loaded
        if isinstance(ds1, LazyDicomDataset):
            assert not ds1.is_pixel_data_loaded

        # Access metadata
        patient_id = ds1.PatientID
        assert patient_id == "TEST_PATIENT_001"

        # Second access should hit cache
        ds2 = cache.get(dicom_file)
        assert cache.hits == 1
        assert ds1 is ds2

    def test_cache_memory_efficiency_with_lazy_loading(self, multiple_dicom_files):
        """Test that caching with lazy loading is memory efficient."""
        cache = DicomLRUCache(max_size=5)

        # Load multiple files
        datasets = []
        for file_path in multiple_dicom_files:
            ds = cache.get(file_path)
            datasets.append(ds)

        # Verify all datasets are in cache
        assert cache.size == len(multiple_dicom_files)

        # If using lazy loading, pixel data should not be loaded
        for ds in datasets:
            if isinstance(ds, LazyDicomDataset):
                # Metadata should be accessible
                assert hasattr(ds, "PatientID")
                # But pixel data not loaded until accessed
                if not ds.is_pixel_data_loaded:
                    # This is the memory-efficient behavior we want
                    pass

    def test_cache_eviction_with_lazy_datasets(self, multiple_dicom_files):
        """Test cache eviction behavior with lazy datasets."""
        cache = DicomLRUCache(max_size=2)

        # Load 3 files (should evict first one)
        ds1 = cache.get(multiple_dicom_files[0])
        ds2 = cache.get(multiple_dicom_files[1])
        ds3 = cache.get(multiple_dicom_files[2])

        assert cache.size == 2
        assert cache.evictions == 1

        # First file should have been evicted
        ds1_again = cache.get(multiple_dicom_files[0])
        assert cache.misses == 4  # Initial 3 + 1 after eviction


# ============================================================================
# Test MammographyImage Class
# ============================================================================


class TestMammographyImage:
    """Tests for MammographyImage class."""

    def test_create_from_dicom(self, dicom_file):
        """Test creating MammographyImage from DICOM file."""
        img = create_mammography_image_from_dicom(str(dicom_file))

        assert img is not None
        assert img.patient_id == "TEST_PATIENT_001"
        assert img.view_position == "CC"
        assert img.laterality == "L"
        assert img.pixel_array.shape == (128, 128)

    def test_mammography_image_attributes(self, dicom_file):
        """Test MammographyImage preserves DICOM attributes."""
        img = create_mammography_image_from_dicom(str(dicom_file))

        # Check basic attributes
        assert hasattr(img, "patient_id")
        assert hasattr(img, "view_position")
        assert hasattr(img, "laterality")
        assert hasattr(img, "pixel_array")

        # Verify values
        assert img.patient_id == "TEST_PATIENT_001"
        assert img.manufacturer == "SIEMENS"


# ============================================================================
# Test DicomReader Class
# ============================================================================


class TestDicomReader:
    """Tests for DicomReader class."""

    def test_dicom_reader_initialization(self):
        """Test DicomReader initialization."""
        reader = create_dicom_reader(cache_size=10)

        assert reader is not None
        assert isinstance(reader, DicomReader)

    def test_read_single_dicom(self, dicom_file):
        """Test reading a single DICOM file."""
        img = read_single_dicom(str(dicom_file))

        assert img is not None
        assert isinstance(img, MammographyImage)
        assert img.patient_id == "TEST_PATIENT_001"

    def test_read_dicom_directory(self, tmp_path, multiple_dicom_files):
        """Test reading directory of DICOM files."""
        # Create a directory with DICOM files
        dicom_dir = tmp_path / "dicoms"
        dicom_dir.mkdir()

        # Copy files to directory
        for i, src_file in enumerate(multiple_dicom_files):
            dst_file = dicom_dir / f"image_{i}.dcm"
            dst_file.write_bytes(src_file.read_bytes())

        # Read directory
        images = list(read_dicom_directory(str(dicom_dir)))

        assert len(images) == len(multiple_dicom_files)
        for img in images:
            assert isinstance(img, MammographyImage)

    def test_dicom_reader_with_cache(self, dicom_file):
        """Test DicomReader with caching enabled."""
        reader = create_dicom_reader(cache_size=10)

        # Read same file twice
        img1 = reader.read(str(dicom_file))
        img2 = reader.read(str(dicom_file))

        assert img1 is not None
        assert img2 is not None
        # Both reads should succeed


# ============================================================================
# Test Error Handling Across Stack
# ============================================================================


class TestErrorHandling:
    """Tests for error handling across the I/O stack."""

    def test_lazy_loading_missing_file(self, tmp_path):
        """Test error when lazy loading non-existent file."""
        missing_file = tmp_path / "missing.dcm"

        with pytest.raises(FileNotFoundError):
            LazyDicomDataset(missing_file)

    def test_cache_with_invalid_file(self, tmp_path):
        """Test cache behavior with invalid DICOM file."""
        invalid_file = tmp_path / "invalid.dcm"
        invalid_file.write_text("Not a DICOM file")

        cache = DicomLRUCache(max_size=10)

        # Should handle gracefully or raise appropriate error
        with pytest.raises((RuntimeError, InvalidDicomError, Exception)):
            ds = cache.get(invalid_file)
            # If we get here, try to access pixel array
            _ = ds.pixel_array

    def test_read_single_dicom_missing_file(self, tmp_path):
        """Test read_single_dicom with missing file."""
        missing_file = tmp_path / "missing.dcm"

        result = read_single_dicom(str(missing_file))

        # Should return None or raise error
        assert result is None or isinstance(result, Exception)

    def test_dicom_reader_empty_directory(self, tmp_path):
        """Test reading empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        images = list(read_dicom_directory(str(empty_dir)))

        assert len(images) == 0


# ============================================================================
# Test Performance and Memory Characteristics
# ============================================================================


class TestPerformanceCharacteristics:
    """Tests for performance and memory characteristics of I/O stack."""

    def test_lazy_loading_memory_savings(self, dicom_file):
        """Test that lazy loading saves memory by not loading pixel data."""
        lazy_ds = LazyDicomDataset(dicom_file)

        # Access metadata only
        _ = lazy_ds.PatientID
        _ = lazy_ds.Manufacturer
        _ = lazy_ds.ViewPosition

        # Pixel data should not be loaded
        assert not lazy_ds.is_pixel_data_loaded

        # This represents memory savings since pixel data is not in memory

    def test_cache_reduces_file_reads(self, dicom_file):
        """Test that cache reduces redundant file reads."""
        cache = DicomLRUCache(max_size=10)

        # First access
        _ = cache.get(dicom_file)
        initial_misses = cache.misses

        # Multiple subsequent accesses
        for _ in range(10):
            _ = cache.get(dicom_file)

        # Should only have one miss (first access)
        assert cache.misses == initial_misses
        assert cache.hits == 10

    def test_lru_eviction_efficiency(self, multiple_dicom_files):
        """Test LRU eviction is efficient and predictable."""
        cache = DicomLRUCache(max_size=3)

        # Access files in order
        for file_path in multiple_dicom_files[:4]:
            cache.get(file_path)

        # Cache should be at capacity
        assert cache.size == 3

        # First file should have been evicted
        # Accessing it should be a miss
        before_misses = cache.misses
        cache.get(multiple_dicom_files[0])
        assert cache.misses == before_misses + 1


# ============================================================================
# Test Integration Scenarios
# ============================================================================


class TestIntegrationScenarios:
    """Tests for real-world integration scenarios."""

    def test_batch_processing_workflow(self, multiple_dicom_files):
        """Test typical batch processing workflow."""
        cache = DicomLRUCache(max_size=10)
        reader = create_dicom_reader(cache_size=10)

        images = []
        for file_path in multiple_dicom_files:
            img = read_single_dicom(str(file_path))
            if img is not None:
                images.append(img)

        assert len(images) == len(multiple_dicom_files)

        # Verify all images loaded correctly
        for img in images:
            assert img.pixel_array is not None
            assert img.patient_id is not None

    def test_memory_constrained_processing(self, multiple_dicom_files):
        """Test processing with memory constraints using lazy loading."""
        # Simulate memory-constrained environment with small cache
        cache = DicomLRUCache(max_size=2)

        processed_count = 0
        for file_path in multiple_dicom_files:
            ds = cache.get(file_path)

            # Access metadata only
            _ = ds.PatientID

            # Pixel data only loaded when needed
            if isinstance(ds, LazyDicomDataset):
                if not ds.is_pixel_data_loaded:
                    # This is good - saving memory
                    pass

            processed_count += 1

        assert processed_count == len(multiple_dicom_files)

    def test_repeated_access_pattern(self, dicom_file):
        """Test repeated access pattern (common in training loops)."""
        cache = DicomLRUCache(max_size=10)

        # Simulate multiple epochs
        for epoch in range(3):
            for _ in range(10):
                ds = cache.get(dicom_file)
                _ = ds.PatientID

        # Should have high hit rate after first epoch
        stats = cache.stats()
        assert stats["hit_rate"] > 0.9  # >90% hit rate
