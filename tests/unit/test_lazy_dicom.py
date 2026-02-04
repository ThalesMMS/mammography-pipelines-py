"""
Unit tests for lazy DICOM loading functionality.

These tests validate the LazyDicomDataset class and its lazy loading behavior.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import copy
import os
import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pydicom = pytest.importorskip("pydicom")

from mammography.io.lazy_dicom import LazyDicomDataset


def _save_dicom(dataset, path: str) -> None:
    """Helper to save DICOM dataset to file."""
    try:
        dataset.save_as(path, enforce_file_format=True)
    except TypeError:
        dataset.save_as(path, write_like_original=False)


class TestLazyDicomDataset:
    """Unit tests for LazyDicomDataset class."""

    @pytest.fixture
    def valid_dicom_dataset(self):
        """Create a valid DICOM dataset for testing."""
        dataset = pydicom.Dataset()

        # Required fields for mammography
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

        # Mammography-specific fields
        dataset.Manufacturer = "SIEMENS"
        dataset.PixelSpacing = [0.1, 0.1]
        dataset.BitsStored = 16
        dataset.BitsAllocated = 16
        dataset.HighBit = dataset.BitsStored - 1
        dataset.PixelRepresentation = 0
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.Rows = 128
        dataset.Columns = 128

        # Projection information
        dataset.ViewPosition = "CC"
        dataset.ImageLaterality = "L"

        # Create pixel data (smaller for faster tests) with fixed seed for reproducibility
        rng = np.random.RandomState(42)
        dataset.PixelData = rng.randint(
            0, 4095, (128, 128), dtype=np.uint16
        ).tobytes()

        return dataset

    @pytest.fixture
    def dicom_file(self, valid_dicom_dataset, tmp_path):
        """Create a temporary DICOM file for testing."""
        filepath = tmp_path / "test_image.dcm"
        _save_dicom(valid_dicom_dataset, str(filepath))
        return filepath

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

    def test_pixel_array_reload_after_clear(self, dicom_file):
        """Test that pixel_array can be reloaded after clearing cache."""
        lazy_ds = LazyDicomDataset(dicom_file)

        # Load pixel data
        pixel_array_1 = lazy_ds.pixel_array
        assert lazy_ds.is_pixel_data_loaded

        # Clear cache
        lazy_ds.clear_pixel_cache()
        assert not lazy_ds.is_pixel_data_loaded

        # Reload pixel data
        pixel_array_2 = lazy_ds.pixel_array
        assert lazy_ds.is_pixel_data_loaded

        # Arrays should have same content but be different objects
        assert pixel_array_1 is not pixel_array_2
        assert np.array_equal(pixel_array_1, pixel_array_2)

    def test_file_not_found_error(self, tmp_path):
        """Test that FileNotFoundError is raised for non-existent file."""
        nonexistent_file = tmp_path / "does_not_exist.dcm"

        with pytest.raises(FileNotFoundError) as exc_info:
            LazyDicomDataset(nonexistent_file)

        assert "DICOM file not found" in str(exc_info.value)
        assert str(nonexistent_file) in str(exc_info.value)

    def test_invalid_dicom_file_error(self, tmp_path):
        """Test that error is raised when accessing pixel_array of invalid DICOM file."""
        # Create a non-DICOM file
        invalid_file = tmp_path / "not_a_dicom.dcm"
        invalid_file.write_text("This is not a DICOM file")

        # With force=True, pydicom may not raise error on init
        # but accessing pixel_array should fail
        lazy_ds = LazyDicomDataset(invalid_file)

        with pytest.raises(RuntimeError) as exc_info:
            _ = lazy_ds.pixel_array

        assert "Failed to load pixel data" in str(exc_info.value)

    def test_getattr_delegation(self, dicom_file):
        """Test that __getattr__ delegates to underlying dataset."""
        lazy_ds = LazyDicomDataset(dicom_file)

        # Test various DICOM attributes
        assert lazy_ds.PatientID == "TEST_PATIENT_001"
        assert lazy_ds.StudyInstanceUID == "1.2.840.12345.123456789"
        assert lazy_ds.Manufacturer == "SIEMENS"
        assert lazy_ds.BitsStored == 16

    def test_getattr_nonexistent_attribute(self, dicom_file):
        """Test that __getattr__ raises AttributeError for non-existent attributes."""
        lazy_ds = LazyDicomDataset(dicom_file)

        with pytest.raises(AttributeError) as exc_info:
            _ = lazy_ds.NonExistentAttribute

        assert "LazyDicomDataset" in str(exc_info.value)
        assert "NonExistentAttribute" in str(exc_info.value)

    def test_contains_operator(self, dicom_file):
        """Test that __contains__ checks for DICOM tag existence."""
        lazy_ds = LazyDicomDataset(dicom_file)

        # Existing tags
        assert "PatientID" in lazy_ds
        assert "StudyInstanceUID" in lazy_ds
        assert "Manufacturer" in lazy_ds

        # Non-existent tags
        assert "NonExistentTag" not in lazy_ds

    def test_repr_without_pixel_data(self, dicom_file):
        """Test __repr__ when pixel data is not loaded."""
        lazy_ds = LazyDicomDataset(dicom_file)

        repr_str = repr(lazy_ds)

        assert "LazyDicomDataset" in repr_str
        assert str(dicom_file) in repr_str
        assert "not loaded" in repr_str

    def test_repr_with_pixel_data(self, dicom_file):
        """Test __repr__ when pixel data is loaded."""
        lazy_ds = LazyDicomDataset(dicom_file)
        _ = lazy_ds.pixel_array  # Load pixel data

        repr_str = repr(lazy_ds)

        assert "LazyDicomDataset" in repr_str
        assert str(dicom_file) in repr_str
        assert "loaded" in repr_str
        assert "not loaded" not in repr_str

    def test_stop_before_pixels_parameter(self, dicom_file):
        """Test that stop_before_pixels parameter works correctly."""
        # With stop_before_pixels=True (default)
        lazy_ds_true = LazyDicomDataset(dicom_file, stop_before_pixels=True)
        assert not lazy_ds_true.is_pixel_data_loaded

        # With stop_before_pixels=False (defeats lazy loading)
        lazy_ds_false = LazyDicomDataset(dicom_file, stop_before_pixels=False)
        assert not lazy_ds_false.is_pixel_data_loaded

        # Access pixel array in both cases
        pixels_true = lazy_ds_true.pixel_array
        pixels_false = lazy_ds_false.pixel_array

        # Both should successfully load pixel data
        assert pixels_true is not None
        assert pixels_false is not None
        assert np.array_equal(pixels_true, pixels_false)

    def test_multiple_lazy_datasets_independent(self, dicom_file):
        """Test that multiple LazyDicomDataset instances are independent."""
        lazy_ds_1 = LazyDicomDataset(dicom_file)
        lazy_ds_2 = LazyDicomDataset(dicom_file)

        # Load pixel data in first instance only
        _ = lazy_ds_1.pixel_array

        # Check independence
        assert lazy_ds_1.is_pixel_data_loaded
        assert not lazy_ds_2.is_pixel_data_loaded

        # Clear cache in first instance
        lazy_ds_1.clear_pixel_cache()
        assert not lazy_ds_1.is_pixel_data_loaded

        # Load pixel data in second instance
        _ = lazy_ds_2.pixel_array
        assert lazy_ds_2.is_pixel_data_loaded
        assert not lazy_ds_1.is_pixel_data_loaded

    def test_pixel_spacing_access(self, dicom_file):
        """Test access to PixelSpacing without loading pixel data."""
        lazy_ds = LazyDicomDataset(dicom_file)

        pixel_spacing = list(lazy_ds.PixelSpacing)

        assert len(pixel_spacing) == 2
        assert pixel_spacing == [0.1, 0.1]
        assert not lazy_ds.is_pixel_data_loaded

    def test_filepath_types(self, dicom_file):
        """Test that both str and Path types work for filepath."""
        # Test with Path
        lazy_ds_path = LazyDicomDataset(dicom_file)
        assert lazy_ds_path.filepath == dicom_file

        # Test with str
        lazy_ds_str = LazyDicomDataset(str(dicom_file))
        assert lazy_ds_str.filepath == Path(dicom_file)

        # Both should work identically
        assert lazy_ds_path.PatientID == lazy_ds_str.PatientID


class TestLazyLoadingMemoryBehavior:
    """Tests focused on memory behavior of lazy loading."""

    @pytest.fixture
    def large_dicom_file(self, tmp_path):
        """Create a larger DICOM file to test memory behavior."""
        dataset = pydicom.Dataset()

        # Required metadata
        dataset.PatientID = "TEST_PATIENT_002"
        dataset.StudyInstanceUID = "1.2.840.12345.111111111"
        dataset.SeriesInstanceUID = "1.2.840.12345.222222222"
        dataset.SOPInstanceUID = "1.2.840.12345.333333333"
        dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"

        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dataset.file_meta = file_meta

        # Image metadata
        dataset.Manufacturer = "GE"
        dataset.PixelSpacing = [0.05, 0.05]
        dataset.BitsStored = 16
        dataset.BitsAllocated = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.Rows = 512
        dataset.Columns = 512
        dataset.ViewPosition = "MLO"
        dataset.ImageLaterality = "R"

        # Create larger pixel data with fixed seed for reproducibility
        rng = np.random.RandomState(47)
        dataset.PixelData = rng.randint(
            0, 65535, (512, 512), dtype=np.uint16
        ).tobytes()

        filepath = tmp_path / "large_image.dcm"
        _save_dicom(dataset, str(filepath))
        return filepath

    def test_metadata_loading_without_pixel_data(self, large_dicom_file):
        """Test that metadata can be accessed without loading large pixel data."""
        lazy_ds = LazyDicomDataset(large_dicom_file)

        # Access multiple metadata fields
        patient_id = lazy_ds.PatientID
        manufacturer = lazy_ds.Manufacturer
        rows = lazy_ds.Rows
        cols = lazy_ds.Columns
        pixel_spacing = lazy_ds.PixelSpacing

        # Verify metadata
        assert patient_id == "TEST_PATIENT_002"
        assert manufacturer == "GE"
        assert rows == 512
        assert cols == 512
        assert list(pixel_spacing) == [0.05, 0.05]

        # Pixel data should still not be loaded
        assert not lazy_ds.is_pixel_data_loaded

    def test_pixel_data_size(self, large_dicom_file):
        """Test that pixel data has expected size when loaded."""
        lazy_ds = LazyDicomDataset(large_dicom_file)

        # Load pixel data
        pixel_array = lazy_ds.pixel_array

        # Verify size
        assert pixel_array.shape == (512, 512)
        assert pixel_array.nbytes == 512 * 512 * 2  # 2 bytes per pixel

    def test_clear_cache_frees_memory(self, large_dicom_file):
        """Test that clearing cache allows pixel data to be garbage collected."""
        lazy_ds = LazyDicomDataset(large_dicom_file)

        # Load pixel data
        _ = lazy_ds.pixel_array
        assert lazy_ds.is_pixel_data_loaded

        # Clear cache
        lazy_ds.clear_pixel_cache()
        assert not lazy_ds.is_pixel_data_loaded

        # Internal cache should be None
        assert lazy_ds._pixel_array is None
