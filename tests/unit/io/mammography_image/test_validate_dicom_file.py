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

class TestValidateDicomFile:
    """Test validate_dicom_file method."""

    def test_successful_validation_with_matching_metadata(self, valid_params, valid_dicom_dataset):
        """Test that validation succeeds when DICOM file matches stored metadata."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        assert img.state == "raw"
        assert img.validation_errors == []

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is True
        assert img.state == "validated"
        assert img.validation_errors == []

    def test_validation_reads_file_when_dataset_not_provided(self, valid_params):
        """Test that validation reads DICOM file when dataset is not provided."""
        img = MammographyImage(**valid_params)

        # Don't provide dataset - should read from file_path
        result = img.validate_dicom_file()

        assert result is True
        assert img.state == "validated"

    def test_validation_fallback_uses_tolerant_force_read(
        self, valid_params, valid_dicom_dataset, monkeypatch
    ):
        """File validation fallback should use force=True under the tolerant context."""
        calls = []
        img = MammographyImage(**valid_params)

        def fake_dcmread(path, **kwargs):
            calls.append((path, kwargs))
            return valid_dicom_dataset

        monkeypatch.setattr(metadata_module.pydicom, "dcmread", fake_dcmread)

        assert img.validate_dicom_file(validate_pixel_data=False) is True
        assert calls == [
            (
                img.file_path,
                {"force": True, "stop_before_pixels": True},
            )
        ]

    def test_validation_failure_for_missing_patient_id(self, valid_params, valid_dicom_dataset):
        """Test validation fails when PatientID tag is missing."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        # Remove PatientID from dataset
        delattr(valid_dicom_dataset, "PatientID")

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        assert img.state == "raw"  # State should not change
        assert len(img.validation_errors) > 0
        assert any("Missing required DICOM tag: PatientID" in err for err in img.validation_errors)

    def test_validation_failure_for_mismatched_patient_id(self, valid_params, valid_dicom_dataset):
        """Test validation fails when PatientID doesn't match."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        # Change PatientID in dataset
        valid_dicom_dataset.PatientID = "DIFFERENT_PATIENT"

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        assert img.state == "raw"
        assert len(img.validation_errors) > 0
        assert any("PatientID mismatch" in err for err in img.validation_errors)

    def test_validation_redacts_sensitive_identifier_mismatches(
        self, valid_params, valid_dicom_dataset, caplog
    ):
        """Test validation errors redact identifier values before storage/logging."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        valid_dicom_dataset.PatientID = "WRONG_PATIENT"
        valid_dicom_dataset.StudyInstanceUID = "9.9.999.99999.999999999"

        with caplog.at_level(logging.WARNING, logger="mammography.io.dicom.metadata"):
            result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        errors_text = "\n".join(img.validation_errors)
        assert result is False
        assert "PatientID mismatch" in errors_text
        assert "StudyInstanceUID mismatch" in errors_text
        assert "<REDACTED>" in errors_text
        assert "WRONG_PATIENT" not in errors_text
        assert "TEST_PATIENT_001" not in errors_text
        assert "9.9.999.99999.999999999" not in errors_text
        assert "WRONG_PATIENT" not in caplog.text
        assert "9.9.999.99999.999999999" not in caplog.text

    def test_validation_failure_for_missing_study_uid(self, valid_params, valid_dicom_dataset):
        """Test validation fails when StudyInstanceUID is missing."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        delattr(valid_dicom_dataset, "StudyInstanceUID")

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        assert any("Missing required DICOM tag: StudyInstanceUID" in err for err in img.validation_errors)

    def test_validation_failure_for_mismatched_study_uid(self, valid_params, valid_dicom_dataset):
        """Test validation fails when StudyInstanceUID doesn't match."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        valid_dicom_dataset.StudyInstanceUID = "9.9.999.99999.999999999"

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        assert any("StudyInstanceUID mismatch" in err for err in img.validation_errors)

    def test_validation_failure_for_mismatched_series_uid(self, valid_params, valid_dicom_dataset):
        """Test validation fails when SeriesInstanceUID doesn't match."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        valid_dicom_dataset.SeriesInstanceUID = "9.9.999.99999.888888888"

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        assert any("SeriesInstanceUID mismatch" in err for err in img.validation_errors)

    def test_validation_failure_for_mismatched_sop_instance_uid(self, valid_params, valid_dicom_dataset):
        """Test validation fails when SOPInstanceUID doesn't match."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        valid_dicom_dataset.SOPInstanceUID = "9.9.999.99999.777777777"

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        assert any("SOPInstanceUID mismatch" in err for err in img.validation_errors)

    def test_validation_failure_for_mismatched_manufacturer(self, valid_params, valid_dicom_dataset):
        """Test validation fails when Manufacturer doesn't match."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        valid_dicom_dataset.Manufacturer = "GE"

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        assert any("Manufacturer mismatch" in err for err in img.validation_errors)

    def test_validation_failure_for_mismatched_pixel_spacing(self, valid_params, valid_dicom_dataset):
        """Test validation fails when PixelSpacing doesn't match."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        valid_dicom_dataset.PixelSpacing = [0.2, 0.2]

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        assert any("PixelSpacing mismatch" in err for err in img.validation_errors)

    def test_validation_failure_for_mismatched_bits_stored(self, valid_params, valid_dicom_dataset):
        """Test validation fails when BitsStored doesn't match."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        valid_dicom_dataset.BitsStored = 12

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        assert any("BitsStored mismatch" in err for err in img.validation_errors)

    def test_validation_failure_for_mismatched_view_position(self, valid_params, valid_dicom_dataset):
        """Test validation fails when ViewPosition doesn't match."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        valid_dicom_dataset.ViewPosition = "MLO"

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        assert any("ViewPosition mismatch" in err for err in img.validation_errors)

    def test_validation_failure_for_mismatched_image_laterality(self, valid_params, valid_dicom_dataset):
        """Test validation fails when ImageLaterality doesn't match."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        valid_dicom_dataset.ImageLaterality = "R"

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        assert any("ImageLaterality mismatch" in err for err in img.validation_errors)

    def test_validation_failure_for_missing_pixel_data(self, valid_params, valid_dicom_dataset):
        """Test validation fails when PixelData is missing."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        delattr(valid_dicom_dataset, "PixelData")

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        assert any("Missing PixelData" in err for err in img.validation_errors)

    def test_validation_failure_for_empty_pixel_data(self, valid_params, valid_dicom_dataset):
        """Test validation fails when PixelData is empty."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        # Set empty pixel data
        valid_dicom_dataset.PixelData = np.array([], dtype=np.uint16).tobytes()

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        assert img.validation_errors.count("PixelData has zero size") == 1
        assert "PixelData is empty" not in img.validation_errors

    def test_validation_sanitizes_pixel_data_exception(
        self, valid_params, valid_dicom_dataset, caplog
    ):
        """PixelData exceptions should not expose raw exception text in errors."""
        sensitive_detail = r"C:\phi\patient001.dcm"

        class BrokenPixelDataDataset:
            def __init__(self, source):
                for tag_name in (
                    "PatientID",
                    "StudyInstanceUID",
                    "SeriesInstanceUID",
                    "SOPInstanceUID",
                    "Manufacturer",
                    "PixelSpacing",
                    "BitsStored",
                    "ViewPosition",
                    "ImageLaterality",
                ):
                    setattr(self, tag_name, getattr(source, tag_name))
                self.PixelData = b"not-empty"

            @property
            def pixel_array(self):
                raise RuntimeError(f"failed reading {sensitive_detail}")

        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)
        dataset = BrokenPixelDataDataset(valid_dicom_dataset)

        with caplog.at_level(logging.ERROR, logger=metadata_module.logger.name):
            result = img.validate_dicom_file(dataset=dataset)

        assert result is False
        assert img.validation_errors == ["RuntimeError: error reading PixelData"]
        assert sensitive_detail not in "\n".join(img.validation_errors)
        assert not any(record.exc_info for record in caplog.records)

    def test_validation_sanitizes_invalid_dicom_exception(
        self, valid_params, monkeypatch, caplog
    ):
        """Invalid DICOM exceptions should not expose raw exception text in errors."""
        sensitive_detail = r"C:\phi\invalid_patient.dcm"
        img = MammographyImage(**valid_params)

        def fail_dcmread(path, **kwargs):
            raise metadata_module.InvalidDicomError(f"invalid file {sensitive_detail}")

        monkeypatch.setattr(metadata_module.pydicom, "dcmread", fail_dcmread)

        with caplog.at_level(logging.ERROR, logger=metadata_module.logger.name):
            result = img.validate_dicom_file()

        assert result is False
        assert img.validation_errors == ["InvalidDicomError: invalid DICOM file"]
        assert sensitive_detail not in "\n".join(img.validation_errors)
        assert not any(record.exc_info for record in caplog.records)

    def test_validation_sanitizes_generic_exception(
        self, valid_params, monkeypatch, caplog
    ):
        """Generic validation exceptions should not expose raw exception text."""
        sensitive_detail = r"C:\phi\validation_patient.dcm"
        img = MammographyImage(**valid_params)

        def fail_dcmread(path, **kwargs):
            raise RuntimeError(f"failed on {sensitive_detail}")

        monkeypatch.setattr(metadata_module.pydicom, "dcmread", fail_dcmread)

        with caplog.at_level(logging.ERROR, logger=metadata_module.logger.name):
            result = img.validate_dicom_file()

        assert result is False
        assert img.validation_errors == ["RuntimeError: validation error"]
        assert sensitive_detail not in "\n".join(img.validation_errors)
        assert not any(record.exc_info for record in caplog.records)

    def test_validation_handles_invalid_dicom_file(self, valid_params, tmp_path):
        """Test validation handles invalid DICOM file gracefully."""
        # Create a non-DICOM file
        invalid_file = tmp_path / "invalid.dcm"
        invalid_file.write_text("Not a DICOM file")

        # Create params with invalid file
        invalid_params = valid_params.copy()
        invalid_params["file_path"] = str(invalid_file)

        # This should fail during MammographyImage initialization
        # because file_path validation checks file exists
        # Let's use a valid DICOM file for initialization but corrupt it after
        img = MammographyImage(**valid_params)

        # Overwrite the file with invalid data
        Path(img.file_path).write_text("Corrupted DICOM")

        result = img.validate_dicom_file()

        assert result is False
        assert len(img.validation_errors) > 0

    def test_validation_errors_accumulate(self, valid_params, valid_dicom_dataset):
        """Test that multiple validation errors accumulate."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        # Create multiple mismatches
        valid_dicom_dataset.PatientID = "WRONG_PATIENT"
        valid_dicom_dataset.Manufacturer = "WRONG_MANUFACTURER"
        valid_dicom_dataset.ViewPosition = "MLO"

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        assert len(img.validation_errors) >= 3
        assert any("PatientID mismatch" in err for err in img.validation_errors)
        assert any("Manufacturer mismatch" in err for err in img.validation_errors)
        assert any("ViewPosition mismatch" in err for err in img.validation_errors)

    def test_validation_clears_previous_errors(self, valid_params, valid_dicom_dataset):
        """Test that successful validation after failed validation clears errors."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        # First validation with errors
        valid_dicom_dataset.PatientID = "WRONG_PATIENT"
        result1 = img.validate_dicom_file(dataset=valid_dicom_dataset)
        assert result1 is False
        error_count = len(img.validation_errors)
        assert error_count > 0

        # Fix the data and validate again
        valid_dicom_dataset.PatientID = "TEST_PATIENT_001"

        result2 = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result2 is True
        assert img.state == "validated"
        assert img.validation_errors == []

    def test_successful_validation_is_idempotent(self, valid_params, valid_dicom_dataset):
        """Test repeated successful validation does not add transition errors."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        result1 = img.validate_dicom_file(dataset=valid_dicom_dataset)
        result2 = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result1 is True
        assert result2 is True
        assert img.state == "validated"
        assert img.validation_errors == []

    def test_validation_returns_false_when_state_transition_fails(
        self, valid_params, valid_dicom_dataset
    ):
        """Test validation checks transition_to before reporting success."""
        img = MammographyImage(
            **valid_params,
            state="preprocessed",
            dataset=valid_dicom_dataset,
        )

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        assert img.state == "preprocessed"
        assert any("Invalid transition" in err for err in img.validation_errors)

    def test_validation_updates_timestamp_on_success(self, valid_params, valid_dicom_dataset):
        """Test that successful validation updates timestamp."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)
        old_updated_at = img.updated_at

        import time
        time.sleep(0.01)

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is True
        assert img.updated_at > old_updated_at

    def test_validation_does_not_update_timestamp_on_failure(self, valid_params, valid_dicom_dataset):
        """Test that failed validation does not update timestamp."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)
        old_updated_at = img.updated_at

        import time
        time.sleep(0.01)

        # Make validation fail
        valid_dicom_dataset.PatientID = "WRONG"

        result = img.validate_dicom_file(dataset=valid_dicom_dataset)

        assert result is False
        # Timestamp should not be updated because state transition failed
        assert img.updated_at == old_updated_at
