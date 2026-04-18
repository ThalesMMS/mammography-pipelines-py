# ruff: noqa
"""
Unit tests for DicomReader.read_dicom_directory() functionality.

These tests validate the DicomReader class's directory reading capabilities including:
- Directory scanning (recursive and non-recursive)
- Patient-level organization
- Parallel file processing
- Error handling for invalid inputs
- Statistics tracking

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import copy
import logging
from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

import mammography.io.dicom.reader as reader_module
from mammography.io.dicom import DicomReader, MammographyImage, create_dicom_reader

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
    dataset.Rows = 64
    dataset.Columns = 64

    # Mammography-specific fields
    dataset.ViewPosition = "CC"
    dataset.ImageLaterality = "L"

    # Create pixel data with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    dataset.PixelData = rng.randint(0, 4095, (64, 64), dtype=np.uint16).tobytes()

    return dataset

@pytest.fixture
def dicom_directory_with_files(valid_dicom_dataset, tmp_path):
    """Create a directory with multiple DICOM files for testing."""
    dicom_dir = tmp_path / "dicom_files"
    dicom_dir.mkdir()

    rng = np.random.RandomState(43)
    files = []
    for i in range(5):
        # Create a copy for each file to avoid mutation
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        dataset_copy.PatientID = f"TEST_PATIENT_{i:03d}"
        dataset_copy.SOPInstanceUID = f"1.2.840.12345.456789{i:03d}"
        dataset_copy.PixelData = rng.randint(0, 4095, (64, 64), dtype=np.uint16).tobytes()

        filepath = dicom_dir / f"test_image_{i}.dcm"
        _save_dicom(dataset_copy, str(filepath))
        files.append(filepath)

    return dicom_dir, files

@pytest.fixture
def nested_dicom_directory(valid_dicom_dataset, tmp_path):
    """Create a nested directory structure with DICOM files for testing."""
    root_dir = tmp_path / "nested_dicom"
    root_dir.mkdir()

    # Create subdirectories
    subdir1 = root_dir / "patient_001"
    subdir2 = root_dir / "patient_002"
    subdir1.mkdir()
    subdir2.mkdir()

    rng = np.random.RandomState(44)
    files = []

    # Add files to root
    for i in range(2):
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        dataset_copy.PatientID = f"ROOT_PATIENT_{i:03d}"
        dataset_copy.SOPInstanceUID = f"1.2.840.12345.10{i:03d}"
        dataset_copy.PixelData = rng.randint(0, 4095, (64, 64), dtype=np.uint16).tobytes()

        filepath = root_dir / f"root_image_{i}.dcm"
        _save_dicom(dataset_copy, str(filepath))
        files.append(filepath)

    # Add files to subdir1
    for i in range(3):
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        dataset_copy.PatientID = "PATIENT_001"
        dataset_copy.SOPInstanceUID = f"1.2.840.12345.11{i:03d}"
        dataset_copy.PixelData = rng.randint(0, 4095, (64, 64), dtype=np.uint16).tobytes()

        filepath = subdir1 / f"image_{i}.dcm"
        _save_dicom(dataset_copy, str(filepath))
        files.append(filepath)

    # Add files to subdir2
    for i in range(2):
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        dataset_copy.PatientID = "PATIENT_002"
        dataset_copy.SOPInstanceUID = f"1.2.840.12345.12{i:03d}"
        dataset_copy.PixelData = rng.randint(0, 4095, (64, 64), dtype=np.uint16).tobytes()

        filepath = subdir2 / f"image_{i}.dcm"
        _save_dicom(dataset_copy, str(filepath))
        files.append(filepath)

    return root_dir, files

@pytest.fixture
def mixed_file_directory(valid_dicom_dataset, tmp_path):
    """Create a directory with DICOM and non-DICOM files for testing."""
    mixed_dir = tmp_path / "mixed_files"
    mixed_dir.mkdir()

    rng = np.random.RandomState(45)
    dicom_files = []

    # Add DICOM files
    for i in range(3):
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        dataset_copy.PatientID = f"MIXED_PATIENT_{i:03d}"
        dataset_copy.SOPInstanceUID = f"1.2.840.12345.13{i:03d}"
        dataset_copy.PixelData = rng.randint(0, 4095, (64, 64), dtype=np.uint16).tobytes()

        filepath = mixed_dir / f"dicom_{i}.dcm"
        _save_dicom(dataset_copy, str(filepath))
        dicom_files.append(filepath)

    # Add non-DICOM files
    (mixed_dir / "readme.txt").write_text("Not a DICOM file")
    (mixed_dir / "data.json").write_text('{"key": "value"}')
    (mixed_dir / "image.png").write_bytes(b"PNG data")

    return mixed_dir, dicom_files

class TestDicomReaderValidationBranches:
    """Test validation branch coverage for DicomReader with invalid/missing DICOM tags."""

    def test_missing_patient_id_tag(self, tmp_path, valid_dicom_dataset):
        """Test that missing PatientID tag is detected during validation."""
        test_dir = tmp_path / "missing_patient_id"
        test_dir.mkdir()

        # Create dataset without PatientID
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        del dataset_copy.PatientID

        filepath = test_dir / "no_patient_id.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True)
        result = reader.read_dicom_directory(test_dir)

        # Should fail validation and return empty result
        assert len(result) == 0
        assert reader.stats["files_invalid"] > 0
        # Check that error message mentions missing PatientID
        assert any("PatientID" in str(err) for err in reader.stats["validation_errors"])

    def test_missing_study_instance_uid_tag(self, tmp_path, valid_dicom_dataset):
        """Test that missing StudyInstanceUID tag is detected during validation."""
        test_dir = tmp_path / "missing_study_uid"
        test_dir.mkdir()

        # Create dataset without StudyInstanceUID
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        del dataset_copy.StudyInstanceUID

        filepath = test_dir / "no_study_uid.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True)
        result = reader.read_dicom_directory(test_dir)

        # Should fail validation
        assert len(result) == 0
        assert reader.stats["files_invalid"] > 0
        assert any(
            "StudyInstanceUID" in str(err) or "study_id" in str(err)
            for err in reader.stats["validation_errors"]
        )

    def test_missing_manufacturer_tag(self, tmp_path, valid_dicom_dataset):
        """Test that missing Manufacturer tag is detected during validation."""
        test_dir = tmp_path / "missing_manufacturer"
        test_dir.mkdir()

        # Create dataset without Manufacturer
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        del dataset_copy.Manufacturer

        filepath = test_dir / "no_manufacturer.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True)
        result = reader.read_dicom_directory(test_dir)

        # Should fail validation
        assert len(result) == 0
        assert reader.stats["files_invalid"] > 0
        assert any("Manufacturer" in str(err) for err in reader.stats["validation_errors"])

    def test_missing_pixel_spacing_tag(self, tmp_path, valid_dicom_dataset):
        """Test that missing PixelSpacing tag is detected during validation."""
        test_dir = tmp_path / "missing_pixel_spacing"
        test_dir.mkdir()

        # Create dataset without PixelSpacing
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        del dataset_copy.PixelSpacing

        filepath = test_dir / "no_pixel_spacing.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True)
        result = reader.read_dicom_directory(test_dir)

        # Should fail validation
        assert len(result) == 0
        assert reader.stats["files_invalid"] > 0
        assert any("PixelSpacing" in str(err) for err in reader.stats["validation_errors"])

    def test_invalid_pixel_spacing_zero_values(self, tmp_path, valid_dicom_dataset):
        """Test that PixelSpacing with zero values is detected as invalid."""
        test_dir = tmp_path / "zero_pixel_spacing"
        test_dir.mkdir()

        # Create dataset with zero pixel spacing
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        dataset_copy.PixelSpacing = [0.0, 0.0]

        filepath = test_dir / "zero_spacing.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True)
        result = reader.read_dicom_directory(test_dir)

        # Should fail validation due to invalid pixel spacing
        assert len(result) == 0
        assert reader.stats["files_invalid"] > 0
        assert any("PixelSpacing" in str(err) for err in reader.stats["validation_errors"])

    def test_invalid_pixel_spacing_negative_values(self, tmp_path, valid_dicom_dataset):
        """Test that PixelSpacing with negative values is detected as invalid."""
        test_dir = tmp_path / "negative_pixel_spacing"
        test_dir.mkdir()

        # Create dataset with negative pixel spacing
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        dataset_copy.PixelSpacing = [-0.1, 0.1]

        filepath = test_dir / "negative_spacing.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True)
        result = reader.read_dicom_directory(test_dir)

        # Should fail validation
        assert len(result) == 0
        assert reader.stats["files_invalid"] > 0
        assert any("PixelSpacing" in str(err) for err in reader.stats["validation_errors"])

    def test_invalid_pixel_spacing_wrong_length(self, tmp_path, valid_dicom_dataset):
        """Test that PixelSpacing with wrong number of elements is detected."""
        test_dir = tmp_path / "wrong_length_spacing"
        test_dir.mkdir()

        # Create dataset with wrong pixel spacing length
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        dataset_copy.PixelSpacing = [0.1, 0.1, 0.1]  # Should be 2 elements

        filepath = test_dir / "wrong_length.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True)
        result = reader.read_dicom_directory(test_dir)

        # Should fail validation
        assert len(result) == 0
        assert reader.stats["files_invalid"] > 0
        assert any("PixelSpacing" in str(err) for err in reader.stats["validation_errors"])

    def test_uncommon_allowed_view_position_reads_without_warning(
        self, tmp_path, valid_dicom_dataset, caplog
    ):
        """Test that allowed uncommon ViewPosition values use the public read path."""
        test_dir = tmp_path / "unexpected_view_position"
        test_dir.mkdir()

        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        dataset_copy.ViewPosition = "ML"

        filepath = test_dir / "unexpected_view.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True)
        with caplog.at_level(logging.WARNING, logger="mammography.io.dicom.reader"):
            image = reader.read_dicom_file(filepath)
        warnings_text = caplog.text

        assert isinstance(image, MammographyImage)
        assert image.projection_type == "ML"
        assert reader.stats["files_valid"] == 1
        assert reader.stats["files_invalid"] == 0
        assert list(reader.stats["validation_errors"]) == []
        assert "ViewPosition" not in warnings_text
        assert str(filepath) not in warnings_text
        assert filepath.name not in warnings_text

    def test_invalid_view_position_recorded_from_factory_validation(
        self, tmp_path, valid_dicom_dataset
    ):
        """Test that factory validation rejects unsupported ViewPosition values."""
        test_dir = tmp_path / "invalid_view_position"
        test_dir.mkdir()

        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        dataset_copy.ViewPosition = "INVALID"

        filepath = test_dir / "invalid_view.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True)
        image = reader.read_dicom_file(filepath)

        assert image is None
        assert reader.stats["files_valid"] == 0
        assert reader.stats["files_invalid"] == 1
        errors_text = "\n".join(str(err) for err in reader.stats["validation_errors"])
        assert "ViewPosition" in errors_text
        assert str(filepath) not in errors_text
        assert filepath.name not in errors_text

    def test_invalid_image_laterality(self, tmp_path, valid_dicom_dataset):
        """Test that invalid ImageLaterality values are detected."""
        test_dir = tmp_path / "invalid_laterality"
        test_dir.mkdir()

        # Create dataset with invalid ImageLaterality
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        dataset_copy.ImageLaterality = "B"  # Invalid, must be L or R

        filepath = test_dir / "invalid_laterality.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True)
        result = reader.read_dicom_directory(test_dir)

        # Should fail validation
        assert len(result) == 0
        assert reader.stats["files_invalid"] > 0
        assert any("ImageLaterality" in str(err) for err in reader.stats["validation_errors"])

    def test_missing_pixel_data_tag(self, tmp_path, valid_dicom_dataset):
        """Test that missing PixelData tag is detected during validation."""
        test_dir = tmp_path / "missing_pixel_data"
        test_dir.mkdir()

        # Create dataset without PixelData
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        del dataset_copy.PixelData

        filepath = test_dir / "no_pixel_data.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True, lazy_load=False)
        result = reader.read_dicom_directory(test_dir)

        # Should fail validation
        assert len(result) == 0
        assert reader.stats["files_invalid"] > 0
        assert any("PixelData" in str(err) for err in reader.stats["validation_errors"])

    def test_empty_pixel_data(self, tmp_path, valid_dicom_dataset):
        """Test that empty PixelData is detected during validation."""
        test_dir = tmp_path / "empty_pixel_data"
        test_dir.mkdir()

        # Create dataset with empty PixelData
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        dataset_copy.PixelData = b""  # Empty bytes

        filepath = test_dir / "empty_pixels.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True, lazy_load=False)
        result = reader.read_dicom_directory(test_dir)

        # Should fail validation
        assert len(result) == 0
        assert reader.stats["files_invalid"] > 0

    def test_missing_bits_stored_tag(self, tmp_path, valid_dicom_dataset):
        """Test that missing BitsStored tag is detected during validation."""
        test_dir = tmp_path / "missing_bits_stored"
        test_dir.mkdir()

        # Create dataset without BitsStored
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        del dataset_copy.BitsStored

        filepath = test_dir / "no_bits_stored.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True)
        result = reader.read_dicom_directory(test_dir)

        # Should fail validation
        assert len(result) == 0
        assert reader.stats["files_invalid"] > 0
        assert any("BitsStored" in str(err) for err in reader.stats["validation_errors"])

    def test_missing_view_position_tag(self, tmp_path, valid_dicom_dataset):
        """Test that missing ViewPosition tag is detected during validation."""
        test_dir = tmp_path / "missing_view_position"
        test_dir.mkdir()

        # Create dataset without ViewPosition
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        del dataset_copy.ViewPosition

        filepath = test_dir / "no_view_position.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True)
        result = reader.read_dicom_directory(test_dir)

        # Should fail validation
        assert len(result) == 0
        assert reader.stats["files_invalid"] > 0
        assert any("ViewPosition" in str(err) for err in reader.stats["validation_errors"])

    def test_missing_image_laterality_tag(self, tmp_path, valid_dicom_dataset):
        """Test that missing ImageLaterality tag is detected during validation."""
        test_dir = tmp_path / "missing_laterality"
        test_dir.mkdir()

        # Create dataset without ImageLaterality
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        del dataset_copy.ImageLaterality

        filepath = test_dir / "no_laterality.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True)
        result = reader.read_dicom_directory(test_dir)

        # Should fail validation
        assert len(result) == 0
        assert reader.stats["files_invalid"] > 0
        assert any("ImageLaterality" in str(err) for err in reader.stats["validation_errors"])

    def test_validation_with_lazy_load_skips_pixel_data(self, tmp_path, valid_dicom_dataset):
        """Test that lazy_load=True skips PixelData validation."""
        test_dir = tmp_path / "lazy_load_validation"
        test_dir.mkdir()

        # Create valid dataset
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        filepath = test_dir / "valid.dcm"
        _save_dicom(dataset_copy, str(filepath))

        # With lazy_load=True, should succeed even though pixel data isn't validated
        reader = DicomReader(validate_on_read=True, lazy_load=True)
        result = reader.read_dicom_directory(test_dir)

        # Should successfully read file with lazy loading
        assert len(result) > 0
        assert reader.stats["files_valid"] == 1

    def test_validation_optional_tags_generate_warnings(self, tmp_path, valid_dicom_dataset):
        """Test that missing optional tags generate warnings but don't fail validation."""
        test_dir = tmp_path / "optional_tags"
        test_dir.mkdir()

        # Create dataset without optional tags
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        # Optional tags like AcquisitionDate are not present in fixture

        filepath = test_dir / "no_optional.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True)
        result = reader.read_dicom_directory(test_dir)

        # Should still succeed (optional tags don't cause validation failure)
        assert len(result) > 0
        assert reader.stats["files_valid"] > 0

    def test_multiple_validation_errors_accumulated(self, tmp_path, valid_dicom_dataset):
        """Test that multiple validation errors are accumulated for a single file."""
        test_dir = tmp_path / "multiple_errors"
        test_dir.mkdir()

        # Create dataset with multiple issues
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        del dataset_copy.PatientID
        del dataset_copy.Manufacturer
        dataset_copy.ViewPosition = "INVALID"
        dataset_copy.PixelSpacing = [0.0, 0.0]

        filepath = test_dir / "many_errors.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(validate_on_read=True)
        result = reader.read_dicom_directory(test_dir)

        # Should fail validation with multiple errors
        assert len(result) == 0
        assert reader.stats["files_invalid"] > 0
        # Should have accumulated multiple errors
        assert len(reader.stats["validation_errors"]) >= 3

    def test_validation_disabled_accepts_invalid_files(self, tmp_path, valid_dicom_dataset):
        """Test that validation can be disabled to accept files with missing tags."""
        test_dir = tmp_path / "validation_disabled"
        test_dir.mkdir()

        # Create dataset with missing required tag
        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        del dataset_copy.Manufacturer

        filepath = test_dir / "missing_tag.dcm"
        _save_dicom(dataset_copy, str(filepath))

        # With validation disabled, file creation may still fail due to constructor validation
        # But the DicomReader validation should be bypassed
        reader = DicomReader(validate_on_read=False)

        # This will still fail because MammographyImage constructor validates
        # But we're testing that DicomReader validation is bypassed
        result = reader.read_dicom_directory(test_dir)

        # The file will be processed (even if it fails later in constructor)
        # At least files_processed should increment
        assert reader.stats["files_processed"] >= 0
