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

class TestMammographyImageStateMachine:
    """Test state machine transition logic."""

    def test_valid_transition_raw_to_validated(self, valid_params):
        """Test valid transition from raw to validated."""
        img = MammographyImage(**valid_params, state="raw")
        result = img.transition_to("validated")

        assert result is True
        assert img.state == "validated"
        assert len(img.validation_errors) == 0

    def test_valid_transition_validated_to_preprocessed(self, valid_params):
        """Test valid transition from validated to preprocessed."""
        img = MammographyImage(**valid_params, state="validated")
        result = img.transition_to("preprocessed")

        assert result is True
        assert img.state == "preprocessed"
        assert len(img.validation_errors) == 0

    def test_valid_transition_preprocessed_to_embedded(self, valid_params):
        """Test valid transition from preprocessed to embedded."""
        img = MammographyImage(**valid_params, state="preprocessed")
        result = img.transition_to("embedded")

        assert result is True
        assert img.state == "embedded"
        assert len(img.validation_errors) == 0

    def test_valid_transition_embedded_to_clustered(self, valid_params):
        """Test valid transition from embedded to clustered."""
        img = MammographyImage(**valid_params, state="embedded")
        result = img.transition_to("clustered")

        assert result is True
        assert img.state == "clustered"
        assert len(img.validation_errors) == 0

    def test_complete_state_chain(self, valid_params):
        """Test complete state transition chain from raw to clustered."""
        img = MammographyImage(**valid_params, state="raw")

        # raw -> validated
        assert img.transition_to("validated") is True
        assert img.state == "validated"

        # validated -> preprocessed
        assert img.transition_to("preprocessed") is True
        assert img.state == "preprocessed"

        # preprocessed -> embedded
        assert img.transition_to("embedded") is True
        assert img.state == "embedded"

        # embedded -> clustered
        assert img.transition_to("clustered") is True
        assert img.state == "clustered"

        # No errors accumulated
        assert len(img.validation_errors) == 0

    def test_invalid_transition_raw_to_preprocessed(self, valid_params):
        """Test invalid transition from raw to preprocessed."""
        img = MammographyImage(**valid_params, state="raw")
        result = img.transition_to("preprocessed")

        assert result is False
        assert img.state == "raw"
        assert len(img.validation_errors) > 0
        assert "Invalid transition" in img.validation_errors[0]
        assert "raw" in img.validation_errors[0]
        assert "preprocessed" in img.validation_errors[0]

    def test_invalid_transition_raw_to_embedded(self, valid_params):
        """Test invalid transition from raw to embedded."""
        img = MammographyImage(**valid_params, state="raw")
        result = img.transition_to("embedded")

        assert result is False
        assert img.state == "raw"
        assert len(img.validation_errors) > 0
        assert "Invalid transition" in img.validation_errors[0]

    def test_invalid_transition_raw_to_clustered(self, valid_params):
        """Test invalid transition from raw to clustered."""
        img = MammographyImage(**valid_params, state="raw")
        result = img.transition_to("clustered")

        assert result is False
        assert img.state == "raw"
        assert len(img.validation_errors) > 0

    def test_invalid_transition_validated_to_embedded(self, valid_params):
        """Test invalid transition from validated to embedded."""
        img = MammographyImage(**valid_params, state="validated")
        result = img.transition_to("embedded")

        assert result is False
        assert img.state == "validated"
        assert len(img.validation_errors) > 0

    def test_invalid_transition_validated_to_clustered(self, valid_params):
        """Test invalid transition from validated to clustered."""
        img = MammographyImage(**valid_params, state="validated")
        result = img.transition_to("clustered")

        assert result is False
        assert img.state == "validated"
        assert len(img.validation_errors) > 0

    def test_invalid_transition_preprocessed_to_validated(self, valid_params):
        """Test invalid backward transition from preprocessed to validated."""
        img = MammographyImage(**valid_params, state="preprocessed")
        result = img.transition_to("validated")

        assert result is False
        assert img.state == "preprocessed"
        assert len(img.validation_errors) > 0

    def test_invalid_transition_preprocessed_to_clustered(self, valid_params):
        """Test invalid transition from preprocessed to clustered."""
        img = MammographyImage(**valid_params, state="preprocessed")
        result = img.transition_to("clustered")

        assert result is False
        assert img.state == "preprocessed"
        assert len(img.validation_errors) > 0

    def test_invalid_transition_embedded_to_preprocessed(self, valid_params):
        """Test invalid backward transition from embedded to preprocessed."""
        img = MammographyImage(**valid_params, state="embedded")
        result = img.transition_to("preprocessed")

        assert result is False
        assert img.state == "embedded"
        assert len(img.validation_errors) > 0

    def test_invalid_transition_clustered_has_no_next(self, valid_params):
        """Test that clustered state has no valid transitions."""
        img = MammographyImage(**valid_params, state="clustered")

        # Try to transition to any state - all should fail
        for target_state in ["raw", "validated", "preprocessed", "embedded"]:
            result = img.transition_to(target_state)
            assert result is False
            assert img.state == "clustered"

    def test_invalid_transition_clustered_to_raw(self, valid_params):
        """Test invalid backward transition from clustered to raw."""
        img = MammographyImage(**valid_params, state="clustered")
        result = img.transition_to("raw")

        assert result is False
        assert img.state == "clustered"
        assert len(img.validation_errors) > 0

    def test_invalid_transition_to_invalid_state(self, valid_params):
        """Test transition to invalid state raises ValueError."""
        img = MammographyImage(**valid_params, state="raw")

        with pytest.raises(ValueError) as exc_info:
            img.transition_to("INVALID")

        assert "Invalid state" in str(exc_info.value)
        assert "INVALID" in str(exc_info.value)

    def test_invalid_transition_to_empty_string(self, valid_params):
        """Test transition to empty string raises ValueError."""
        img = MammographyImage(**valid_params, state="raw")

        with pytest.raises(ValueError) as exc_info:
            img.transition_to("")

        assert "Invalid state" in str(exc_info.value)

    def test_transition_updates_timestamp(self, valid_params):
        """Test that valid transition updates updated_at timestamp."""
        img = MammographyImage(**valid_params, state="raw")
        old_updated_at = img.updated_at

        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.01)

        img.transition_to("validated")

        assert img.updated_at > old_updated_at

    def test_invalid_transition_does_not_update_timestamp(self, valid_params):
        """Test that invalid transition does not update timestamp."""
        img = MammographyImage(**valid_params, state="raw")
        old_updated_at = img.updated_at

        # Small delay to ensure we could detect a change
        import time
        time.sleep(0.01)

        # Try invalid transition
        img.transition_to("preprocessed")

        # Timestamp should not change for invalid transition
        assert img.updated_at == old_updated_at

    def test_validation_errors_accumulate(self, valid_params):
        """Test that validation errors accumulate across failed transitions."""
        img = MammographyImage(**valid_params, state="raw")

        # First invalid transition
        img.transition_to("preprocessed")
        assert len(img.validation_errors) == 1

        # Second invalid transition
        img.transition_to("embedded")
        assert len(img.validation_errors) == 2

        # Third invalid transition
        img.transition_to("clustered")
        assert len(img.validation_errors) == 3

        # All errors should contain "Invalid transition"
        for error in img.validation_errors:
            assert "Invalid transition" in error

    def test_same_state_transition_fails(self, valid_params):
        """Test that transitioning to the same state fails."""
        img = MammographyImage(**valid_params, state="validated")
        result = img.transition_to("validated")

        assert result is False
        assert img.state == "validated"
        assert len(img.validation_errors) > 0

class TestStringRepresentations:
    """Test __repr__ and __str__ methods."""

    def test_repr_contains_key_fields(self, valid_params):
        """Test that __repr__ contains only safe key fields."""
        img = MammographyImage(**valid_params)
        repr_str = repr(img)
        uid_fingerprint = metadata_module.fingerprint_uid(valid_params["instance_id"])

        assert "MammographyImage" in repr_str
        assert valid_params["instance_id"] not in repr_str
        assert uid_fingerprint in repr_str
        assert "CC" in repr_str
        assert "L" in repr_str
        assert "raw" in repr_str
        assert "TEST_PATIENT_001" not in repr_str
        assert valid_params["file_path"] not in repr_str

    def test_repr_with_different_state(self, valid_params):
        """Test that __repr__ reflects different states."""
        img = MammographyImage(**valid_params, state="validated")
        repr_str = repr(img)

        assert "validated" in repr_str
        assert "MammographyImage" in repr_str

    def test_repr_with_mlo_projection(self, valid_params):
        """Test __repr__ with MLO projection type."""
        valid_params["projection_type"] = "MLO"
        img = MammographyImage(**valid_params)
        repr_str = repr(img)

        assert "MLO" in repr_str
        assert "MammographyImage" in repr_str

    def test_repr_with_right_laterality(self, valid_params):
        """Test __repr__ with right laterality."""
        valid_params["laterality"] = "R"
        img = MammographyImage(**valid_params)
        repr_str = repr(img)

        assert "R" in repr_str
        assert "MammographyImage" in repr_str

    def test_repr_is_valid_python_syntax(self, valid_params):
        """Test that __repr__ output is valid Python-like syntax."""
        img = MammographyImage(**valid_params)
        repr_str = repr(img)

        # Should start with class name and open parenthesis
        assert repr_str.startswith("MammographyImage(")
        assert repr_str.endswith(")")
        # Should contain key=value pairs
        assert "instance_id=" in repr_str
        assert "projection_type=" in repr_str
        assert "laterality=" in repr_str
        assert "state=" in repr_str
        assert "patient_id=" not in repr_str

    def test_str_contains_readable_info(self, valid_params):
        """Test that __str__ contains human-readable information."""
        img = MammographyImage(**valid_params)
        str_output = str(img)
        uid_fingerprint = metadata_module.fingerprint_uid(valid_params["instance_id"])

        assert "Mammography Image" in str_output
        assert valid_params["instance_id"] not in str_output
        assert uid_fingerprint in str_output
        assert "CC" in str_output
        assert "L" in str_output
        assert "raw" in str_output
        assert "TEST_PATIENT_001" not in str_output
        assert valid_params["file_path"] not in str_output

    def test_str_multiline_format(self, valid_params):
        """Test that __str__ returns multiline formatted output."""
        img = MammographyImage(**valid_params)
        str_output = str(img)

        # Should contain newlines
        lines = str_output.split("\n")
        assert len(lines) >= 4

        # Check each line contains expected content
        assert any("Mammography Image" in line for line in lines)
        assert any("Projection" in line for line in lines)
        assert any("Laterality" in line for line in lines)
        assert any("State" in line for line in lines)
        assert not any("Patient" in line for line in lines)
        assert not any("File" in line for line in lines)

    def test_str_with_different_state(self, valid_params):
        """Test that __str__ reflects different states."""
        img = MammographyImage(**valid_params, state="preprocessed")
        str_output = str(img)

        assert "preprocessed" in str_output

    def test_str_different_from_repr(self, valid_params):
        """Test that __str__ and __repr__ produce different outputs."""
        img = MammographyImage(**valid_params)

        str_output = str(img)
        repr_output = repr(img)

        # They should be different
        assert str_output != repr_output

        # __str__ is more human-readable
        assert "Mammography Image:" in str_output
        assert "Mammography Image:" not in repr_output

        # __repr__ is more code-like
        assert "MammographyImage(" in repr_output
        assert "MammographyImage(" not in str_output

class TestMetadataDict:
    """Test get_metadata_dict method."""

    def test_metadata_dict_contains_all_fields(self, valid_params):
        """Test that metadata dict contains all expected fields."""
        img = MammographyImage(**valid_params)
        metadata = img.get_metadata_dict()

        assert metadata["patient_id"] == "TEST_PATIENT_001"
        assert metadata["study_id"] == "1.2.840.12345.123456789"
        assert metadata["series_id"] == "1.2.840.12345.987654321"
        assert metadata["instance_id"] == "1.2.840.12345.456789123"
        assert metadata["projection_type"] == "CC"
        assert metadata["laterality"] == "L"
        assert metadata["manufacturer"] == "SIEMENS"
        assert metadata["pixel_spacing"] == (0.1, 0.1)
        assert metadata["bits_stored"] == 16
        assert metadata["file_path"] == valid_params["file_path"]
        assert metadata["state"] == "raw"
        assert metadata["validation_errors"] == []
        assert "created_at" in metadata
        assert "updated_at" in metadata
        assert "acquisition_date" in metadata

    def test_metadata_dict_acquisition_date_isoformat(self, valid_params):
        """Test that acquisition_date is converted to isoformat string."""
        acquisition_date = datetime(2023, 1, 15, 10, 30, 0)
        img = MammographyImage(**valid_params, acquisition_date=acquisition_date)
        metadata = img.get_metadata_dict()

        assert metadata["acquisition_date"] == "2023-01-15T10:30:00"

    def test_metadata_dict_timestamps_isoformat(self, valid_params):
        """Test that timestamps are converted to isoformat strings."""
        img = MammographyImage(**valid_params)
        metadata = img.get_metadata_dict()

        # Verify they are strings in ISO format
        assert isinstance(metadata["created_at"], str)
        assert isinstance(metadata["updated_at"], str)
        # Should be parseable as datetime
        datetime.fromisoformat(metadata["created_at"])
        datetime.fromisoformat(metadata["updated_at"])

    def test_metadata_dict_with_validation_errors(self, valid_params):
        """Test that metadata dict includes validation errors."""
        img = MammographyImage(**valid_params)
        # Add some validation errors
        img.validation_errors.append("Test error 1")
        img.validation_errors.append("Test error 2")

        metadata = img.get_metadata_dict()

        assert len(metadata["validation_errors"]) == 2
        assert "Test error 1" in metadata["validation_errors"]
        assert "Test error 2" in metadata["validation_errors"]

    def test_metadata_dict_validation_errors_are_snapshot(self, valid_params):
        """Metadata dict should not expose the live validation_errors list."""
        img = MammographyImage(**valid_params)
        img.validation_errors.append("Original error")

        metadata = img.get_metadata_dict()
        img.validation_errors.append("Later error")
        metadata["validation_errors"].append("External mutation")

        assert metadata["validation_errors"] == ["Original error", "External mutation"]
        assert img.validation_errors == ["Original error", "Later error"]

    def test_metadata_dict_with_different_state(self, valid_params):
        """Test that metadata dict reflects current state."""
        img = MammographyImage(**valid_params, state="validated")
        metadata = img.get_metadata_dict()

        assert metadata["state"] == "validated"

    def test_metadata_dict_is_serializable(self, valid_params):
        """Test that metadata dict can be serialized to JSON."""
        import json
        img = MammographyImage(**valid_params)
        metadata = img.get_metadata_dict()

        # Should not raise exception
        json_str = json.dumps(metadata)
        assert isinstance(json_str, str)
        # Should be deserializable
        deserialized = json.loads(json_str)
        assert deserialized["patient_id"] == "TEST_PATIENT_001"
