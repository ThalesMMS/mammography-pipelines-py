# ruff: noqa
"""
Unit tests for MammographyImage constructor validation.

These tests validate the MammographyImage class initialization and state machine.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import logging
import io
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

class TestMammographyImageValidInitialization:
    """Test valid MammographyImage initialization."""

    def test_valid_initialization_with_all_required_params(self, valid_params):
        """Test that MammographyImage initializes with valid parameters."""
        img = MammographyImage(**valid_params)

        assert img.patient_id == "TEST_PATIENT_001"
        assert img.study_id == "1.2.840.12345.123456789"
        assert img.series_id == "1.2.840.12345.987654321"
        assert img.instance_id == "1.2.840.12345.456789123"
        assert img.projection_type == "CC"
        assert img.laterality == "L"
        assert img.manufacturer == "SIEMENS"
        assert img.pixel_spacing == (0.1, 0.1)
        assert img.bits_stored == 16
        assert img.file_path == valid_params["file_path"]
        assert img.state == "raw"
        assert img.validation_errors == []
        assert isinstance(img.created_at, datetime)
        assert isinstance(img.updated_at, datetime)

    def test_initialization_with_optional_acquisition_date(self, valid_params):
        """Test initialization with acquisition_date parameter."""
        acquisition_date = datetime(2023, 1, 15)
        img = MammographyImage(**valid_params, acquisition_date=acquisition_date)

        assert img.acquisition_date == acquisition_date

    def test_initialization_without_acquisition_date(self, valid_params):
        """Test that a missing acquisition_date remains missing."""
        img = MammographyImage(**valid_params)

        assert img.acquisition_date is None

    def test_initialization_with_custom_state(self, valid_params):
        """Test initialization with custom valid state."""
        img = MammographyImage(**valid_params, state="validated")

        assert img.state == "validated"

    def test_initialization_with_dataset(self, valid_params, valid_dicom_dataset):
        """Test initialization with optional dataset parameter."""
        img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)

        assert img.dataset == valid_dicom_dataset

    def test_pixel_array_rereads_lazy_dataset_without_pixel_data(self, valid_params):
        """pixel_array should re-read when the cached dataset omits PixelData."""
        lazy_dataset = pydicom.dcmread(valid_params["file_path"], stop_before_pixels=True)
        assert not hasattr(lazy_dataset, "PixelData")
        img = MammographyImage(**valid_params, dataset=lazy_dataset)

        pixel_array = img.pixel_array

        assert pixel_array.shape == (128, 128)
        assert hasattr(img.dataset, "PixelData")

    def test_pixel_array_reread_uses_tolerant_force_read(
        self, valid_params, monkeypatch
    ):
        """pixel_array fallback should use the tolerant forced read path."""
        calls = []
        lazy_dataset = pydicom.Dataset()
        img = MammographyImage(**valid_params, dataset=lazy_dataset)

        class FakeDataset:
            PixelData = b"1"
            pixel_array = np.array([[1]], dtype=np.uint16)

        def fake_dcmread(path, **kwargs):
            calls.append((path, kwargs))
            return FakeDataset()

        monkeypatch.setattr(metadata_module.pydicom, "dcmread", fake_dcmread)

        pixel_array = img.pixel_array

        assert pixel_array.shape == (1, 1)
        assert calls == [
            (
                img.file_path,
                {"force": True, "stop_before_pixels": False},
            )
        ]

    def test_pixel_array_sanitizes_reread_failure(
        self, valid_params, monkeypatch, caplog
    ):
        """pixel_array reread failures should not expose raw exception text."""
        sensitive_detail = r"C:\phi\pixel_read_patient.dcm"
        img = MammographyImage(**valid_params, dataset=pydicom.Dataset())

        def fail_dcmread(path, **kwargs):
            raise RuntimeError(f"failed reading {sensitive_detail}")

        monkeypatch.setattr(metadata_module.pydicom, "dcmread", fail_dcmread)

        with caplog.at_level(logging.ERROR, logger=metadata_module.logger.name):
            with pytest.raises(metadata_module.DicomValidationError) as exc_info:
                _ = img.pixel_array

        assert "failed to read DICOM pixel data" in str(exc_info.value)
        assert sensitive_detail not in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert not any(record.exc_info for record in caplog.records)
        assert sensitive_detail not in "\n".join(
            record.getMessage() for record in caplog.records
        )

    def test_pixel_array_sanitizes_decode_failure(self, valid_params, caplog):
        """pixel_array decode failures should not expose raw exception text."""
        sensitive_detail = r"C:\phi\pixel_decode_patient.dcm"

        class BrokenDataset:
            PixelData = b"not-empty"

            @property
            def pixel_array(self):
                raise RuntimeError(f"decode failed for {sensitive_detail}")

        img = MammographyImage(**valid_params, dataset=BrokenDataset())

        with caplog.at_level(logging.ERROR, logger=metadata_module.logger.name):
            with pytest.raises(metadata_module.DicomValidationError) as exc_info:
                _ = img.pixel_array

        assert "failed to read DICOM pixel data" in str(exc_info.value)
        assert sensitive_detail not in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert not any(record.exc_info for record in caplog.records)
        assert sensitive_detail not in "\n".join(
            record.getMessage() for record in caplog.records
        )

    def test_repr_and_str_do_not_expose_phi(self, valid_params):
        """String representations should omit patient identifiers and full file paths."""
        img = MammographyImage(**valid_params)
        uid_fingerprint = metadata_module.fingerprint_uid(valid_params["instance_id"])

        repr_text = repr(img)
        str_text = str(img)

        assert img.instance_id not in repr_text
        assert img.instance_id not in str_text
        assert uid_fingerprint in repr_text
        assert uid_fingerprint in str_text
        assert valid_params["patient_id"] not in repr_text
        assert valid_params["patient_id"] not in str_text
        assert valid_params["file_path"] not in repr_text
        assert valid_params["file_path"] not in str_text

    def test_logs_fingerprint_instance_id(
        self, valid_params, valid_dicom_dataset, caplog, capsys
    ):
        """Logs should use a deterministic fingerprint instead of raw SOP UID."""
        uid_fingerprint = metadata_module.fingerprint_uid(valid_params["instance_id"])

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        metadata_module.logger.addHandler(handler)
        try:
            with caplog.at_level(logging.INFO, logger=metadata_module.logger.name):
                img = MammographyImage(**valid_params, dataset=valid_dicom_dataset)
                img.validate_dicom_file(dataset=valid_dicom_dataset)
                valid_dicom_dataset.PatientID = "DIFFERENT_PATIENT"
                img.validate_dicom_file(dataset=valid_dicom_dataset)
        finally:
            metadata_module.logger.removeHandler(handler)

        captured = capsys.readouterr()
        log_text = "\n".join(
            [
                *(record.getMessage() for record in caplog.records),
                captured.err,
                log_stream.getvalue(),
            ]
        )
        assert valid_params["instance_id"] not in log_text
        assert uid_fingerprint in log_text

    def test_factory_reports_missing_bits_stored(self, valid_dicom_dataset, dicom_file):
        """Factory should report a missing BitsStored tag explicitly."""
        del valid_dicom_dataset.BitsStored

        with pytest.raises(ValueError, match="Missing required DICOM tag 'BitsStored'"):
            create_mammography_image_from_dicom(
                str(dicom_file),
                dataset=valid_dicom_dataset,
            )

    def test_factory_raises_when_validation_fails(self, valid_dicom_dataset, dicom_file):
        """Factory should not return a MammographyImage when validation fails."""
        del valid_dicom_dataset.PixelData

        with pytest.raises(
            metadata_module.DicomValidationError,
            match="DICOM validation failed",
        ):
            create_mammography_image_from_dicom(
                str(dicom_file),
                dataset=valid_dicom_dataset,
                validate=True,
            )

    def test_factory_invalid_dicom_error_omits_path(self, dicom_file, monkeypatch):
        """InvalidDicomError wrapper should not include filesystem paths."""
        file_path = str(dicom_file)

        def fail_dcmread(path, **kwargs):
            raise metadata_module.InvalidDicomError(f"invalid content at {path}")

        monkeypatch.setattr(metadata_module.pydicom, "dcmread", fail_dcmread)

        with pytest.raises(metadata_module.InvalidDicomError) as exc_info:
            create_mammography_image_from_dicom(file_path)

        assert str(exc_info.value) == "Invalid DICOM file"
        assert file_path not in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    def test_factory_generic_error_omits_path(self, dicom_file, monkeypatch):
        """Generic factory wrapper should not include filesystem paths."""
        file_path = str(dicom_file)

        def fail_dcmread(path, **kwargs):
            raise RuntimeError(f"boom at {path}")

        monkeypatch.setattr(metadata_module.pydicom, "dcmread", fail_dcmread)

        with pytest.raises(ValueError) as exc_info:
            create_mammography_image_from_dicom(file_path)

        assert str(exc_info.value) == "Error creating MammographyImage"
        assert file_path not in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    def test_factory_validate_false_skips_validate_dicom_file(
        self, valid_dicom_dataset, dicom_file, monkeypatch
    ):
        """Factory should not validate when validate=False."""

        def fail_validate(self, *args, **kwargs):
            raise AssertionError("validate_dicom_file should not be called")

        monkeypatch.setattr(MammographyImage, "validate_dicom_file", fail_validate)

        img = create_mammography_image_from_dicom(
            str(dicom_file),
            dataset=valid_dicom_dataset,
            validate=False,
        )

        assert isinstance(img, MammographyImage)

    def test_factory_fallback_uses_tolerant_force_read(
        self, valid_dicom_dataset, dicom_file, monkeypatch
    ):
        """Factory fallback should read with force=True under the tolerant context."""
        calls = []

        def fake_dcmread(path, **kwargs):
            calls.append((path, kwargs))
            return valid_dicom_dataset

        monkeypatch.setattr(metadata_module.pydicom, "dcmread", fake_dcmread)

        img = create_mammography_image_from_dicom(
            str(dicom_file),
            validate=False,
            validate_pixel_data=False,
        )

        assert isinstance(img, MammographyImage)
        assert calls == [
            (
                str(dicom_file),
                {"force": True, "stop_before_pixels": True},
            )
        ]

    @pytest.mark.parametrize("uid_field", ["study_id", "series_id", "instance_id"])
    @pytest.mark.parametrize("malformed_uid", ["1..2.840", "1.02.840", "1.2 .840"])
    def test_factory_rejects_malformed_uid_when_validating(
        self, valid_dicom_dataset, dicom_file, uid_field, malformed_uid
    ):
        """Factory validation should reject malformed UIDs via MammographyImage."""
        dicom_tag_by_field = {
            "study_id": "StudyInstanceUID",
            "series_id": "SeriesInstanceUID",
            "instance_id": "SOPInstanceUID",
        }
        setattr(valid_dicom_dataset, dicom_tag_by_field[uid_field], malformed_uid)

        with pytest.raises(ValueError) as exc_info:
            create_mammography_image_from_dicom(
                str(dicom_file),
                dataset=valid_dicom_dataset,
                validate=True,
            )

        assert f"{uid_field} must be a valid DICOM UID format" in str(exc_info.value)

    def test_mlo_projection_type(self, valid_params):
        """Test initialization with MLO projection type."""
        valid_params["projection_type"] = "MLO"
        img = MammographyImage(**valid_params)

        assert img.projection_type == "MLO"

    def test_right_laterality(self, valid_params):
        """Test initialization with right laterality."""
        valid_params["laterality"] = "R"
        img = MammographyImage(**valid_params)

        assert img.laterality == "R"

    def test_different_pixel_spacing(self, valid_params):
        """Test initialization with different pixel spacing values."""
        valid_params["pixel_spacing"] = (0.2, 0.15)
        img = MammographyImage(**valid_params)

        assert img.pixel_spacing == (0.2, 0.15)

    def test_different_bits_stored(self, valid_params):
        """Test initialization with different bits_stored values."""
        valid_params["bits_stored"] = 12
        img = MammographyImage(**valid_params)

        assert img.bits_stored == 12
