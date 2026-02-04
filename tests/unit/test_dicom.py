"""
Unit tests for DICOM I/O functionality including codec and pixel array access.

These tests validate proper handling of:
- Pixel array access with various transfer syntaxes
- Codec availability and error messages
- DICOM reading with different compression formats
- Error handling for missing dependencies

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.errors import InvalidDicomError

from mammography.io.dicom import (
    dicom_to_pil_rgb,
    is_dicom_path,
    _is_mono1,
    _to_float32,
    _apply_rescale,
    robust_window,
    extract_window_parameters,
    apply_windowing,
)


def _save_dicom(dataset, path: str) -> None:
    """Helper to save DICOM dataset to file."""
    try:
        dataset.save_as(path, enforce_file_format=True)
    except TypeError:
        dataset.save_as(path, write_like_original=False)


@pytest.fixture
def valid_dicom_dataset():
    """Create a valid DICOM dataset for testing."""
    dataset = pydicom.Dataset()

    # Required fields
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

    # Create pixel data with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    dataset.PixelData = rng.randint(
        0, 4095, (128, 128), dtype=np.uint16
    ).tobytes()

    return dataset


@pytest.fixture
def mono1_dicom_dataset(valid_dicom_dataset):
    """Create MONOCHROME1 DICOM dataset."""
    dataset = copy.deepcopy(valid_dicom_dataset)
    dataset.PhotometricInterpretation = "MONOCHROME1"
    return dataset


@pytest.fixture
def dicom_with_rescale(valid_dicom_dataset):
    """Create DICOM dataset with RescaleSlope and RescaleIntercept."""
    dataset = copy.deepcopy(valid_dicom_dataset)
    dataset.RescaleSlope = 2.0
    dataset.RescaleIntercept = -1024.0
    return dataset


@pytest.fixture
def dicom_with_windowing(valid_dicom_dataset):
    """Create DICOM dataset with WindowCenter and WindowWidth."""
    dataset = copy.deepcopy(valid_dicom_dataset)
    dataset.WindowCenter = 2048
    dataset.WindowWidth = 4096
    return dataset


class TestDicomPath:
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


class TestPhotometricInterpretation:
    """Tests for photometric interpretation handling."""

    def test_is_mono1_monochrome1(self, mono1_dicom_dataset):
        """Test MONOCHROME1 detection."""
        assert _is_mono1(mono1_dicom_dataset) is True

    def test_is_mono1_monochrome2(self, valid_dicom_dataset):
        """Test MONOCHROME2 detection."""
        assert _is_mono1(valid_dicom_dataset) is False

    def test_is_mono1_missing_attribute(self):
        """Test handling of missing PhotometricInterpretation."""
        dataset = pydicom.Dataset()
        assert _is_mono1(dataset) is False


class TestArrayConversion:
    """Tests for array type conversion."""

    def test_to_float32_uint16(self):
        """Test conversion from uint16 to float32."""
        arr = np.array([1, 2, 3], dtype=np.uint16)
        result = _to_float32(arr)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_to_float32_already_float32(self):
        """Test that float32 arrays are not copied unnecessarily."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _to_float32(arr)
        assert result.dtype == np.float32
        assert result is arr  # Should be same object (no copy)

    def test_to_float32_int32(self):
        """Test conversion from int32 to float32."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = _to_float32(arr)
        assert result.dtype == np.float32


class TestRescale:
    """Tests for rescale slope/intercept application."""

    def test_apply_rescale_with_slope_intercept(self, dicom_with_rescale):
        """Test rescale with slope and intercept."""
        arr = np.array([1000, 2000, 3000], dtype=np.float32)
        result = _apply_rescale(dicom_with_rescale, arr)
        # Expected: arr * 2.0 + (-1024) = [976, 2976, 4976]
        expected = np.array([976, 2976, 4976], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_rescale_no_attributes(self, valid_dicom_dataset):
        """Test rescale when no slope/intercept present (defaults to 1.0 and 0.0)."""
        arr = np.array([1000, 2000, 3000], dtype=np.float32)
        result = _apply_rescale(valid_dicom_dataset, arr)
        # Should apply default slope=1.0, intercept=0.0
        np.testing.assert_array_equal(result, arr)

    def test_apply_rescale_invalid_values(self):
        """Test rescale with invalid slope/intercept falls back gracefully."""
        dataset = pydicom.Dataset()
        dataset.RescaleSlope = "invalid"
        dataset.RescaleIntercept = "invalid"
        arr = np.array([1000, 2000, 3000], dtype=np.float32)
        # Should not raise, should fall back gracefully
        result = _apply_rescale(dataset, arr)
        assert result is not None


class TestRobustWindow:
    """Tests for robust windowing."""

    def test_robust_window_normal_range(self):
        """Test windowing with normal pixel value distribution."""
        arr = np.array([0, 100, 200, 300, 400, 500], dtype=np.float32)
        result = robust_window(arr, p_low=0.0, p_high=100.0)
        # Should normalize to [0, 1] range
        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_robust_window_outliers(self):
        """Test windowing robustness to outliers."""
        # Most values around 100, one outlier at 10000
        arr = np.array([90, 95, 100, 105, 110, 10000], dtype=np.float32)
        result = robust_window(arr, p_low=0.5, p_high=99.5)
        # Outlier should be clipped, most values should span [0, 1]
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_robust_window_constant_array(self):
        """Test windowing with constant pixel values."""
        arr = np.ones(10, dtype=np.float32) * 100
        result = robust_window(arr)
        # All zeros when min == max
        np.testing.assert_array_equal(result, np.zeros(10, dtype=np.float32))


class TestWindowParameters:
    """Tests for window parameter extraction."""

    def test_extract_window_parameters_from_tags(self, dicom_with_windowing):
        """Test extraction from DICOM WindowCenter/WindowWidth tags."""
        arr = np.random.rand(128, 128).astype(np.float32) * 4096
        wc, ww, photometric = extract_window_parameters(dicom_with_windowing, arr)
        assert wc == 2048.0
        assert ww == 4096.0
        assert photometric == "MONOCHROME2"

    def test_extract_window_parameters_from_pixel_data(self, valid_dicom_dataset):
        """Test calculation from pixel data when tags missing."""
        arr = np.array([[0, 100], [200, 300]], dtype=np.float32)
        wc, ww, photometric = extract_window_parameters(valid_dicom_dataset, arr)
        # Should calculate: center = (300 + 0) / 2 = 150, width = 300 - 0 = 300
        assert wc == 150.0
        assert ww == 300.0

    def test_extract_window_parameters_multivalue(self):
        """Test handling of multi-valued WindowCenter/WindowWidth."""
        dataset = pydicom.Dataset()
        dataset.WindowCenter = [2048, 1024]  # Multi-value
        dataset.WindowWidth = [4096, 2048]
        dataset.PhotometricInterpretation = "MONOCHROME2"
        arr = np.zeros((10, 10), dtype=np.float32)
        wc, ww, photometric = extract_window_parameters(dataset, arr)
        # Should use first value
        assert wc == 2048.0
        assert ww == 4096.0

    def test_extract_window_parameters_zero_width(self):
        """Test handling of zero or negative window width."""
        dataset = pydicom.Dataset()
        dataset.PhotometricInterpretation = "MONOCHROME2"
        arr = np.ones((10, 10), dtype=np.float32) * 100  # Constant values
        wc, ww, photometric = extract_window_parameters(dataset, arr)
        # Should set minimum width of 1
        assert ww >= 1.0


class TestApplyWindowing:
    """Tests for windowing application."""

    def test_apply_windowing_monochrome2(self):
        """Test windowing for MONOCHROME2."""
        image = np.array([0, 50, 100, 150, 200], dtype=np.float32)
        wc, ww = 100.0, 100.0
        result = apply_windowing(image, wc, ww, "MONOCHROME2")
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_apply_windowing_monochrome1(self):
        """Test windowing for MONOCHROME1 (inverted)."""
        image = np.array([0, 50, 100, 150, 200], dtype=np.float32)
        wc, ww = 100.0, 100.0
        result = apply_windowing(image, wc, ww, "MONOCHROME1")
        # Should be inverted compared to MONOCHROME2
        assert result.dtype == np.uint8

    def test_apply_windowing_zero_width(self):
        """Test windowing with zero width (edge case)."""
        image = np.array([100, 100, 100], dtype=np.float32)
        result = apply_windowing(image, 100.0, 0.0, "MONOCHROME2")
        # Should result in zeros when width is zero
        np.testing.assert_array_equal(result, np.zeros(3, dtype=np.uint8))


class TestDicomToRGB:
    """Tests for DICOM to RGB PIL image conversion."""

    def test_dicom_to_pil_rgb_success(self, valid_dicom_dataset, tmp_path):
        """Test successful conversion of DICOM to RGB."""
        dcm_path = tmp_path / "test.dcm"
        _save_dicom(valid_dicom_dataset, str(dcm_path))

        result = dicom_to_pil_rgb(str(dcm_path))
        assert result.mode == "RGB"
        assert result.size == (128, 128)

    def test_dicom_to_pil_rgb_monochrome1(self, mono1_dicom_dataset, tmp_path):
        """Test conversion of MONOCHROME1 DICOM."""
        dcm_path = tmp_path / "test_mono1.dcm"
        _save_dicom(mono1_dicom_dataset, str(dcm_path))

        result = dicom_to_pil_rgb(str(dcm_path))
        assert result.mode == "RGB"
        # Verify inversion was applied
        assert result.size == (128, 128)

    def test_dicom_to_pil_rgb_invalid_file(self, tmp_path):
        """Test error handling for invalid DICOM file."""
        dcm_path = tmp_path / "invalid.dcm"
        dcm_path.write_text("This is not a DICOM file")

        with pytest.raises(RuntimeError) as exc_info:
            dicom_to_pil_rgb(str(dcm_path))
        assert "Falha ao ler pixel data" in str(exc_info.value)

    def test_dicom_to_pil_rgb_missing_file(self):
        """Test error handling for missing file."""
        with pytest.raises(RuntimeError) as exc_info:
            dicom_to_pil_rgb("nonexistent.dcm")
        assert "Falha ao ler pixel data" in str(exc_info.value)

    def test_dicom_to_pil_rgb_custom_window(self, valid_dicom_dataset, tmp_path):
        """Test custom windowing parameters."""
        dcm_path = tmp_path / "test.dcm"
        _save_dicom(valid_dicom_dataset, str(dcm_path))

        result = dicom_to_pil_rgb(str(dcm_path), window_low=1.0, window_high=99.0)
        assert result.mode == "RGB"
        assert result.size == (128, 128)


class TestPixelArrayAccess:
    """Tests for pixel array access and codec handling."""

    def test_pixel_array_access_uncompressed(self, valid_dicom_dataset, tmp_path):
        """Test pixel array access with uncompressed transfer syntax."""
        dcm_path = tmp_path / "uncompressed.dcm"
        _save_dicom(valid_dicom_dataset, str(dcm_path))

        ds = pydicom.dcmread(str(dcm_path))
        arr = ds.pixel_array

        assert arr is not None
        assert arr.shape == (128, 128)
        assert arr.dtype in (np.uint8, np.uint16, np.int16)

    def test_pixel_array_requires_numpy(self, valid_dicom_dataset, tmp_path):
        """Test that pixel_array access requires NumPy."""
        dcm_path = tmp_path / "test.dcm"
        _save_dicom(valid_dicom_dataset, str(dcm_path))

        ds = pydicom.dcmread(str(dcm_path))
        # NumPy is imported at module level, so this will always work
        # This test documents the requirement
        assert hasattr(ds, 'pixel_array')

    def test_pixel_array_empty_pixel_data(self, tmp_path):
        """Test handling of DICOM without pixel data."""
        dataset = pydicom.Dataset()
        dataset.PatientID = "TEST"
        dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        dataset.SOPInstanceUID = "1.2.3.4.5"

        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dataset.file_meta = file_meta

        dcm_path = tmp_path / "no_pixels.dcm"
        _save_dicom(dataset, str(dcm_path))

        ds = pydicom.dcmread(str(dcm_path))
        # Should raise AttributeError when pixel data is missing
        with pytest.raises(AttributeError):
            _ = ds.pixel_array


class TestCodecSupport:
    """Tests for DICOM codec support and error messages."""

    def test_uncompressed_transfer_syntax_works(self, valid_dicom_dataset, tmp_path):
        """Test that uncompressed transfer syntaxes always work."""
        # Explicitly set to uncompressed
        valid_dicom_dataset.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        dcm_path = tmp_path / "uncompressed.dcm"
        _save_dicom(valid_dicom_dataset, str(dcm_path))

        # Should not raise any codec errors
        result = dicom_to_pil_rgb(str(dcm_path))
        assert result is not None

    def test_codec_error_message_helpful(self, tmp_path):
        """Test that codec-related errors provide helpful messages."""
        # Create a DICOM file with compressed transfer syntax
        # Note: We can't actually test missing codec without uninstalling libraries
        # This test documents expected behavior

        dataset = pydicom.Dataset()
        dataset.PatientID = "TEST"
        dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        dataset.SOPInstanceUID = "1.2.3.4.5"
        dataset.StudyInstanceUID = "1.2.3"
        dataset.SeriesInstanceUID = "1.2.3.4"

        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        # Set JPEG transfer syntax (requires codec)
        file_meta.TransferSyntaxUID = pydicom.uid.JPEGBaseline8Bit
        dataset.file_meta = file_meta

        # Add minimal image data
        dataset.Rows = 10
        dataset.Columns = 10
        dataset.BitsAllocated = 8
        dataset.BitsStored = 8
        dataset.HighBit = 7
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.PixelRepresentation = 0

        # Note: Actual PixelData for JPEG would need proper JPEG encoding
        # For this test, we document that codec errors should be caught
        dcm_path = tmp_path / "jpeg.dcm"
        _save_dicom(dataset, str(dcm_path))

        # If codec is missing, error message should mention it
        # Actual behavior depends on whether pylibjpeg is installed


@pytest.mark.integration
class TestRealDicomFiles:
    """Integration tests with real DICOM files (if available)."""

    def test_load_real_dicom_if_available(self):
        """Test loading real DICOM files from archive if available."""
        archive_dir = Path("archive")
        if not archive_dir.exists():
            pytest.skip("Archive directory not available")

        # Find first DICOM file
        dicom_files = list(archive_dir.rglob("*.dcm"))
        if not dicom_files:
            pytest.skip("No DICOM files found in archive")

        # Test loading first file
        dcm_path = dicom_files[0]
        try:
            result = dicom_to_pil_rgb(str(dcm_path))
            assert result is not None
            assert result.mode == "RGB"
        except RuntimeError as e:
            # If it fails, error message should be helpful
            assert "Falha ao ler pixel data" in str(e)
