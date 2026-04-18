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

class TestPatientIdValidation:
    """Test patient_id parameter validation."""

    def test_invalid_patient_id_type_integer(self, valid_params):
        """Test that non-string patient_id raises TypeError."""
        valid_params["patient_id"] = 12345
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "patient_id must be a string" in str(exc_info.value)

    def test_invalid_patient_id_type_none(self, valid_params):
        """Test that None patient_id raises TypeError."""
        valid_params["patient_id"] = None
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "patient_id must be a string" in str(exc_info.value)

    def test_invalid_patient_id_empty_string(self, valid_params):
        """Test that empty patient_id raises ValueError."""
        valid_params["patient_id"] = ""
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "patient_id cannot be empty or whitespace" in str(exc_info.value)

    def test_invalid_patient_id_whitespace(self, valid_params):
        """Test that whitespace-only patient_id raises ValueError."""
        valid_params["patient_id"] = "   "
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "patient_id cannot be empty or whitespace" in str(exc_info.value)

    def test_patient_id_strips_whitespace(self, valid_params):
        """Test that patient_id strips leading/trailing whitespace."""
        valid_params["patient_id"] = "  TEST_PATIENT_001  "
        img = MammographyImage(**valid_params)

        assert img.patient_id == "TEST_PATIENT_001"

class TestUidValidation:
    """Test UID parameter validation (study_id, series_id, instance_id)."""

    @pytest.mark.parametrize("uid_field", ["study_id", "series_id", "instance_id"])
    def test_invalid_uid_type_integer(self, valid_params, uid_field):
        """Test that non-string UID raises TypeError."""
        valid_params[uid_field] = 123456789
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert f"{uid_field} must be a string" in str(exc_info.value)

    @pytest.mark.parametrize("uid_field", ["study_id", "series_id", "instance_id"])
    def test_invalid_uid_type_none(self, valid_params, uid_field):
        """Test that None UID raises TypeError."""
        valid_params[uid_field] = None
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert f"{uid_field} must be a string" in str(exc_info.value)

    @pytest.mark.parametrize("uid_field", ["study_id", "series_id", "instance_id"])
    def test_invalid_uid_empty_string(self, valid_params, uid_field):
        """Test that empty UID raises ValueError."""
        valid_params[uid_field] = ""
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert f"{uid_field} cannot be empty or whitespace" in str(exc_info.value)

    @pytest.mark.parametrize("uid_field", ["study_id", "series_id", "instance_id"])
    def test_invalid_uid_whitespace(self, valid_params, uid_field):
        """Test that whitespace-only UID raises ValueError."""
        valid_params[uid_field] = "   "
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert f"{uid_field} cannot be empty or whitespace" in str(exc_info.value)

    @pytest.mark.parametrize("uid_field", ["study_id", "series_id", "instance_id"])
    def test_invalid_uid_format_letters(self, valid_params, uid_field):
        """Test that UID with letters raises ValueError."""
        valid_params[uid_field] = "1.2.840.ABC.123"
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert f"{uid_field} must be a valid DICOM UID format" in str(exc_info.value)

    @pytest.mark.parametrize("uid_field", ["study_id", "series_id", "instance_id"])
    def test_invalid_uid_format_special_chars(self, valid_params, uid_field):
        """Test that UID with special characters raises ValueError."""
        valid_params[uid_field] = "1.2.840-12345-123"
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert f"{uid_field} must be a valid DICOM UID format" in str(exc_info.value)

    @pytest.mark.parametrize("uid_field", ["study_id", "series_id", "instance_id"])
    @pytest.mark.parametrize("malformed_uid", ["1..2.840", "1.02.840", "1.2 .840"])
    def test_invalid_uid_format_rejected_by_pydicom_validator(
        self, valid_params, uid_field, malformed_uid
    ):
        """Malformed numeric UIDs should be rejected by pydicom.uid.UID."""
        valid_params[uid_field] = malformed_uid
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert f"{uid_field} must be a valid DICOM UID format" in str(exc_info.value)

    @pytest.mark.parametrize("uid_field", ["study_id", "series_id", "instance_id"])
    def test_uid_strips_whitespace(self, valid_params, uid_field):
        """Test that UID strips leading/trailing whitespace."""
        original_value = valid_params[uid_field]
        valid_params[uid_field] = f"  {original_value}  "
        img = MammographyImage(**valid_params)

        assert getattr(img, uid_field) == original_value

class TestProjectionTypeValidation:
    """Test projection_type parameter validation."""

    def test_invalid_projection_type_integer(self, valid_params):
        """Test that non-string projection_type raises TypeError."""
        valid_params["projection_type"] = 123
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "projection_type must be a string" in str(exc_info.value)

    def test_invalid_projection_type_none(self, valid_params):
        """Test that None projection_type raises TypeError."""
        valid_params["projection_type"] = None
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "projection_type must be a string" in str(exc_info.value)

    def test_invalid_projection_type_value(self, valid_params):
        """Test that invalid projection_type value raises ValueError."""
        valid_params["projection_type"] = "INVALID"
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "projection_type must be one of" in str(exc_info.value)
        assert "INVALID" in str(exc_info.value)

    def test_invalid_projection_type_lowercase(self, valid_params):
        """Test that lowercase projection_type raises ValueError."""
        valid_params["projection_type"] = "cc"
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "projection_type must be one of" in str(exc_info.value)

    def test_invalid_projection_type_empty_string(self, valid_params):
        """Test that empty projection_type raises ValueError."""
        valid_params["projection_type"] = ""
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "projection_type must be one of" in str(exc_info.value)

class TestLateralityValidation:
    """Test laterality parameter validation."""

    def test_invalid_laterality_type_integer(self, valid_params):
        """Test that non-string laterality raises TypeError."""
        valid_params["laterality"] = 1
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "laterality must be a string" in str(exc_info.value)

    def test_invalid_laterality_type_none(self, valid_params):
        """Test that None laterality raises TypeError."""
        valid_params["laterality"] = None
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "laterality must be a string" in str(exc_info.value)

    def test_invalid_laterality_value(self, valid_params):
        """Test that invalid laterality value raises ValueError."""
        valid_params["laterality"] = "INVALID"
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "laterality must be one of" in str(exc_info.value)
        assert "['L', 'R']" in str(exc_info.value)

    def test_invalid_laterality_lowercase(self, valid_params):
        """Test that lowercase laterality raises ValueError."""
        valid_params["laterality"] = "l"
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "laterality must be one of" in str(exc_info.value)

    def test_invalid_laterality_empty_string(self, valid_params):
        """Test that empty laterality raises ValueError."""
        valid_params["laterality"] = ""
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "laterality must be one of" in str(exc_info.value)

class TestManufacturerValidation:
    """Test manufacturer parameter validation."""

    def test_invalid_manufacturer_type_integer(self, valid_params):
        """Test that non-string manufacturer raises TypeError."""
        valid_params["manufacturer"] = 12345
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "manufacturer must be a string" in str(exc_info.value)

    def test_invalid_manufacturer_type_none(self, valid_params):
        """Test that None manufacturer raises TypeError."""
        valid_params["manufacturer"] = None
        with pytest.raises(TypeError) as exc_info:
            MammographyImage(**valid_params)

        assert "manufacturer must be a string" in str(exc_info.value)

    def test_invalid_manufacturer_empty_string(self, valid_params):
        """Test that empty manufacturer raises ValueError."""
        valid_params["manufacturer"] = ""
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "manufacturer cannot be empty or whitespace" in str(exc_info.value)

    def test_invalid_manufacturer_whitespace(self, valid_params):
        """Test that whitespace-only manufacturer raises ValueError."""
        valid_params["manufacturer"] = "   "
        with pytest.raises(ValueError) as exc_info:
            MammographyImage(**valid_params)

        assert "manufacturer cannot be empty or whitespace" in str(exc_info.value)

    def test_manufacturer_strips_whitespace(self, valid_params):
        """Test that manufacturer strips leading/trailing whitespace."""
        valid_params["manufacturer"] = "  SIEMENS  "
        img = MammographyImage(**valid_params)

        assert img.manufacturer == "SIEMENS"
