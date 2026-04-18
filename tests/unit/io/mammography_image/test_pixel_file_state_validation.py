# ruff: noqa
"""
Unit tests for MammographyImage constructor validation.

These tests validate the MammographyImage class initialization and state machine.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import logging
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pydicom = pytest.importorskip("pydicom")

from mammography.io.dicom import MammographyImage, create_mammography_image_from_dicom
import mammography.io.dicom.metadata as metadata_module

def _save_dicom(dataset, path: str) -> None:
    """Helper to save DICOM dataset to file."""
    try:
        dataset.save_as(path, enforce_file_format=True)
    except TypeError:
        dataset.save_as(path, write_like_original=False)

@pytest.fixture
def valid_dicom_dataset():
    """Create a valid mammography DICOM dataset for testing."""
    dataset = pydicom.Dataset()

    # Required DICOM fields
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
def dicom_file(valid_dicom_dataset, tmp_path):
    """Create a temporary DICOM file for testing."""
    filepath = tmp_path / "test_image.dcm"
    _save_dicom(valid_dicom_dataset, str(filepath))
    return filepath

@pytest.fixture
def valid_params(dicom_file):
    """Create valid parameters for MammographyImage initialization."""
    return {
        "patient_id": "TEST_PATIENT_001",
        "study_id": "1.2.840.12345.123456789",
        "series_id": "1.2.840.12345.987654321",
        "instance_id": "1.2.840.12345.456789123",
        "projection_type": "CC",
        "laterality": "L",
        "manufacturer": "SIEMENS",
        "pixel_spacing": (0.1, 0.1),
        "bits_stored": 16,
        "file_path": str(dicom_file),
    }

class TestPixelSpacingValidation:
    """Test pixel_spacing parameter validation."""

    def test_invalid_pixel_spacing_type_list(self, valid_params):
        """Test that list pixel_spacing raises TypeError."""
        valid_params["pixel_spacing"] = [0.1, 0.1]
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "pixel_spacing must be a tuple" in str(exc_info.value)

    def test_invalid_pixel_spacing_type_string(self, valid_params):
        """Test that string pixel_spacing raises TypeError."""
        valid_params["pixel_spacing"] = "0.1, 0.1"
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "pixel_spacing must be a tuple" in str(exc_info.value)

    def test_invalid_pixel_spacing_type_none(self, valid_params):
        """Test that None pixel_spacing raises TypeError."""
        valid_params["pixel_spacing"] = None
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "pixel_spacing must be a tuple" in str(exc_info.value)

    def test_invalid_pixel_spacing_length_one(self, valid_params):
        """Test that single-element pixel_spacing raises ValueError."""
        valid_params["pixel_spacing"] = (0.1,)
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "pixel_spacing must have exactly 2 elements" in str(exc_info.value)

    def test_invalid_pixel_spacing_length_three(self, valid_params):
        """Test that three-element pixel_spacing raises ValueError."""
        valid_params["pixel_spacing"] = (0.1, 0.1, 0.1)
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "pixel_spacing must have exactly 2 elements" in str(exc_info.value)

    def test_invalid_pixel_spacing_element_type_string(self, valid_params):
        """Test that non-numeric pixel_spacing element raises TypeError."""
        valid_params["pixel_spacing"] = ("0.1", 0.1)
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "pixel_spacing[0] must be a number" in str(exc_info.value)

    def test_invalid_pixel_spacing_element_type_none(self, valid_params):
        """Test that None pixel_spacing element raises TypeError."""
        valid_params["pixel_spacing"] = (0.1, None)
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "pixel_spacing[1] must be a number" in str(exc_info.value)

    def test_invalid_pixel_spacing_negative_value(self, valid_params):
        """Test that negative pixel_spacing raises ValueError."""
        valid_params["pixel_spacing"] = (-0.1, 0.1)
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "pixel_spacing[0] must be positive" in str(exc_info.value)

    def test_invalid_pixel_spacing_zero_value(self, valid_params):
        """Test that zero pixel_spacing raises ValueError."""
        valid_params["pixel_spacing"] = (0.1, 0.0)
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "pixel_spacing[1] must be positive" in str(exc_info.value)

    def test_pixel_spacing_converts_int_to_float(self, valid_params):
        """Test that integer pixel_spacing values are converted to float."""
        valid_params["pixel_spacing"] = (1, 2)
        img = MammographyImage(**valid_params)

        assert img.pixel_spacing == (1.0, 2.0)
        assert isinstance(img.pixel_spacing[0], float)
        assert isinstance(img.pixel_spacing[1], float)

class TestBitsStoredValidation:
    """Test bits_stored parameter validation."""

    def test_invalid_bits_stored_type_string(self, valid_params):
        """Test that non-integer bits_stored raises TypeError."""
        valid_params["bits_stored"] = "16"
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "bits_stored must be an integer" in str(exc_info.value)

    def test_invalid_bits_stored_type_float(self, valid_params):
        """Test that float bits_stored raises TypeError."""
        valid_params["bits_stored"] = 16.5
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "bits_stored must be an integer" in str(exc_info.value)

    def test_invalid_bits_stored_type_none(self, valid_params):
        """Test that None bits_stored raises TypeError."""
        valid_params["bits_stored"] = None
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "bits_stored must be an integer" in str(exc_info.value)

    def test_invalid_bits_stored_negative(self, valid_params):
        """Test that negative bits_stored raises ValueError."""
        valid_params["bits_stored"] = -1
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "bits_stored must be positive" in str(exc_info.value)

    def test_invalid_bits_stored_zero(self, valid_params):
        """Test that zero bits_stored raises ValueError."""
        valid_params["bits_stored"] = 0
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "bits_stored must be positive" in str(exc_info.value)

    def test_invalid_bits_stored_exceeds_max(self, valid_params):
        """Test that bits_stored > 32 raises ValueError."""
        valid_params["bits_stored"] = 33
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "bits_stored must be <= 32" in str(exc_info.value)

    def test_valid_bits_stored_boundary_values(self, valid_params):
        """Test valid bits_stored boundary values."""
        # Test minimum valid value
        valid_params["bits_stored"] = 1
        img = MammographyImage(**valid_params)
        assert img.bits_stored == 1

        # Test maximum valid value
        valid_params["bits_stored"] = 32
        img = MammographyImage(**valid_params)
        assert img.bits_stored == 32

class TestFilePathValidation:
    """Test file_path parameter validation."""

    def test_invalid_file_path_type_integer(self, valid_params):
        """Test that non-string file_path raises TypeError."""
        valid_params["file_path"] = 12345
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "file_path must be a string" in str(exc_info.value)

    def test_invalid_file_path_type_none(self, valid_params):
        """Test that None file_path raises TypeError."""
        valid_params["file_path"] = None
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "file_path must be a string" in str(exc_info.value)

    def test_invalid_file_path_empty_string(self, valid_params):
        """Test that empty file_path raises ValueError."""
        valid_params["file_path"] = ""
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "file_path cannot be empty or whitespace" in str(exc_info.value)

    def test_invalid_file_path_whitespace(self, valid_params):
        """Test that whitespace-only file_path raises ValueError."""
        valid_params["file_path"] = "   "
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "file_path cannot be empty or whitespace" in str(exc_info.value)

    def test_invalid_file_path_nonexistent(self, valid_params):
        """Test that non-existent file_path raises ValueError."""
        valid_params["file_path"] = "/nonexistent/path/to/file.dcm"
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "DICOM file does not exist" in str(exc_info.value)
        assert valid_params["file_path"] not in str(exc_info.value)

    def test_invalid_file_path_directory(self, valid_params, tmp_path):
        """Test that directory file_path raises ValueError."""
        valid_params["file_path"] = str(tmp_path)
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "Path is not a file" in str(exc_info.value)
        assert valid_params["file_path"] not in str(exc_info.value)

    def test_file_path_converts_to_absolute(self, valid_params, tmp_path):
        """Test that file_path is converted to absolute path."""
        # Create a file
        test_file = tmp_path / "test.dcm"
        test_file.touch()

        valid_params["file_path"] = str(test_file)
        img = MammographyImage(**valid_params)

        assert img.file_path == str(test_file.absolute())
        assert Path(img.file_path).is_absolute()

class TestStateValidation:
    """Test state parameter validation."""

    def test_invalid_state_type_integer(self, valid_params):
        """Test that non-string state raises TypeError."""
        valid_params["state"] = 123
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "state must be a string" in str(exc_info.value)

    def test_invalid_state_type_none(self, valid_params):
        """Test that None state raises TypeError."""
        valid_params["state"] = None
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "state must be a string" in str(exc_info.value)

    def test_invalid_state_value(self, valid_params):
        """Test that invalid state value raises ValueError."""
        valid_params["state"] = "INVALID"
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "state must be one of" in str(exc_info.value)
        assert "raw" in str(exc_info.value)

    def test_all_valid_states(self, valid_params):
        """Test that all valid states can be used."""
        valid_states = ["raw", "validated", "preprocessed", "embedded", "clustered"]

        for state in valid_states:
            valid_params["state"] = state
            img = MammographyImage(**valid_params)
            assert img.state == state
