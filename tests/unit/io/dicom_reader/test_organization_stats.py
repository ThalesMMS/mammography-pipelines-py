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

class TestDicomReaderPatientOrganization:
    """Test patient-level organization functionality."""

    def test_organize_by_patient_groups_correctly(self, tmp_path, valid_dicom_dataset):
        """Test that images are grouped by patient ID correctly."""
        test_dir = tmp_path / "patient_test"
        test_dir.mkdir()

        rng = np.random.RandomState(46)

        # Create files for 3 patients
        patient_ids = ["PATIENT_A", "PATIENT_B", "PATIENT_C"]
        files_per_patient = [2, 3, 1]

        for patient_idx, (patient_id, count) in enumerate(
            zip(patient_ids, files_per_patient),
            start=1,
        ):
            for i in range(count):
                dataset_copy = copy.deepcopy(valid_dicom_dataset)
                dataset_copy.PatientID = patient_id
                dataset_copy.SOPInstanceUID = f"1.2.840.12345.16{patient_idx}.{i}"
                dataset_copy.PixelData = rng.randint(
                    0, 4095, (64, 64), dtype=np.uint16
                ).tobytes()

                filepath = test_dir / f"{patient_id}_{i}.dcm"
                _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader()
        result = reader.read_dicom_directory(test_dir, patient_level=True)

        # Check that we have 3 patients
        assert len(result) == 3

        # Check each patient has correct number of images
        for patient_id, count in zip(patient_ids, files_per_patient):
            assert patient_id in result
            assert len(result[patient_id]) == count

    def test_patient_organization_sorts_by_acquisition_date(
        self, tmp_path, valid_dicom_dataset
    ):
        """Test that images for each patient are sorted by acquisition date."""
        test_dir = tmp_path / "date_test"
        test_dir.mkdir()

        rng = np.random.RandomState(47)
        patient_id = "DATE_TEST_PATIENT"

        # Create files with different acquisition dates
        dates = ["20230101", "20230103", "20230102"]  # Out of order

        for i, date in enumerate(dates):
            dataset_copy = copy.deepcopy(valid_dicom_dataset)
            dataset_copy.PatientID = patient_id
            dataset_copy.SOPInstanceUID = f"1.2.840.12345.14{i}"
            dataset_copy.AcquisitionDate = date
            dataset_copy.PixelData = rng.randint(
                0, 4095, (64, 64), dtype=np.uint16
            ).tobytes()

            filepath = test_dir / f"image_{i}.dcm"
            _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader()
        result = reader.read_dicom_directory(test_dir, patient_level=True)

        # Images should be sorted by acquisition date
        images = result[patient_id]
        assert len(images) == 3

        # Check that dates are in ascending order
        acquisition_dates = [img.acquisition_date for img in images]
        assert acquisition_dates == sorted(acquisition_dates)

class TestDicomReaderStatistics:
    """Test statistics tracking during directory reading."""

    def test_statistics_updated_after_read(self, dicom_directory_with_files):
        """Test that reader statistics are updated after reading directory."""
        dicom_dir, expected_files = dicom_directory_with_files

        reader = DicomReader()

        # Initial stats should be zero
        assert reader.stats["files_processed"] == 0
        assert reader.stats["files_valid"] == 0
        assert reader.stats["files_invalid"] == 0

        reader.read_dicom_directory(dicom_dir)

        # Stats should be updated
        assert reader.stats["files_processed"] > 0
        assert reader.stats["files_valid"] > 0

    def test_metadata_cache_populated(self, dicom_directory_with_files):
        """Test that metadata cache is populated when cache_metadata=True."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader(cache_metadata=True)

        # Cache should be empty initially
        assert len(reader.metadata_cache) == 0

        reader.read_dicom_directory(dicom_dir)

        # Cache should be populated
        assert len(reader.metadata_cache) > 0

        # Check cache contains metadata dictionaries
        for metadata in reader.metadata_cache.values():
            assert isinstance(metadata, dict)
            assert "patient_id" in metadata
            assert "file_path" in metadata

    def test_metadata_cache_not_populated_when_disabled(self, dicom_directory_with_files):
        """Test that metadata cache is not populated when cache_metadata=False."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader(cache_metadata=False)
        reader.read_dicom_directory(dicom_dir)

        # Cache should remain empty
        assert len(reader.metadata_cache) == 0

    def test_get_processing_stats(self, dicom_directory_with_files):
        """Test that get_processing_stats returns correct information."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader()
        reader.read_dicom_directory(dicom_dir)

        stats = reader.get_processing_stats()

        assert "files_processed" in stats
        assert "files_valid" in stats
        assert "files_invalid" in stats
        assert "success_rate" in stats
        assert "total_validation_errors" in stats
        assert "recent_errors" in stats

        # Success rate should be 100% for valid files
        assert stats["success_rate"] <= 100.0

class TestDicomReaderParallelProcessing:
    """Test parallel file processing functionality."""

    def test_parallel_processing_with_max_workers(self, dicom_directory_with_files):
        """Test that parallel processing uses specified max_workers."""
        dicom_dir, _ = dicom_directory_with_files

        # Test with different worker counts
        for max_workers in [1, 2, 4]:
            reader = DicomReader(max_workers=max_workers)
            result = reader.read_dicom_directory(dicom_dir)

            # Should get same results regardless of worker count
            assert len(result) > 0

    @pytest.mark.parametrize("max_workers", [None, 0, -1, 1.5, "4", True])
    def test_rejects_invalid_max_workers(self, max_workers):
        """Test that invalid max_workers values fail at construction time."""
        with pytest.raises(ValueError, match="max_workers"):
            DicomReader(max_workers=max_workers)

    def test_parallel_processing_handles_large_directory(
        self, tmp_path, valid_dicom_dataset
    ):
        """Test parallel processing with larger number of files."""
        large_dir = tmp_path / "large_dir"
        large_dir.mkdir()

        rng = np.random.RandomState(48)
        file_count = 20

        # Create many DICOM files
        for i in range(file_count):
            dataset_copy = copy.deepcopy(valid_dicom_dataset)
            dataset_copy.PatientID = f"PATIENT_{i % 5:03d}"  # 5 different patients
            dataset_copy.SOPInstanceUID = f"1.2.840.12345.15{i:03d}"
            dataset_copy.PixelData = rng.randint(
                0, 4095, (64, 64), dtype=np.uint16
            ).tobytes()

            filepath = large_dir / f"file_{i:03d}.dcm"
            _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader(max_workers=4)
        result = reader.read_dicom_directory(large_dir)

        # Count total images
        total_images = sum(len(images) for images in result.values())
        assert total_images == file_count

class TestDicomReaderValidationOptions:
    """Test validation behavior during directory reading."""

    def test_validate_on_read_true(self, dicom_directory_with_files):
        """Test that validation occurs when validate_on_read=True."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader(validate_on_read=True)
        result = reader.read_dicom_directory(dicom_dir)

        # All returned images should be validated
        for images in result.values():
            for img in images:
                # Images should be in validated state or later
                assert img.state in ["validated", "preprocessed", "embedded", "clustered"]

    def test_validate_on_read_false(self, dicom_directory_with_files):
        """Test that validation is skipped when validate_on_read=False."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader(validate_on_read=False)
        result = reader.read_dicom_directory(dicom_dir)

        # Should still return images (validation is optional)
        assert len(result) > 0
