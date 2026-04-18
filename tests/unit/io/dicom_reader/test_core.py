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
import io
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

def test_dicom_reader_tag_constants_are_immutable() -> None:
    """Class-level reader constants should not be mutable shared lists."""
    assert isinstance(DicomReader.SUPPORTED_EXTENSIONS, tuple)
    assert DicomReader.SUPPORTED_EXTENSIONS == ()
    assert isinstance(DicomReader.REQUIRED_TAGS, tuple)
    assert isinstance(DicomReader.OPTIONAL_TAGS, tuple)

def test_dicom_reader_redacts_missing_file_path_in_errors(tmp_path, caplog) -> None:
    """Missing-file errors should not expose PHI-bearing path components."""
    reader = DicomReader()
    missing_file = tmp_path / "PATIENT_JANE_DOE" / "Jane_Doe_12345.dcm"

    with caplog.at_level(logging.ERROR, logger="mammography.io.dicom.reader"):
        with pytest.raises(FileNotFoundError) as exc_info:
            reader.read_dicom_file(missing_file)

    stats = reader.get_processing_stats()
    combined_text = "\n".join(
        [
            str(exc_info.value),
            caplog.text,
            *stats["recent_errors"],
        ]
    )

    assert str(missing_file) not in combined_text
    assert missing_file.name not in combined_text
    assert "PATIENT_JANE_DOE" not in combined_text
    assert "<dicom-path:" in combined_text

def test_dicom_reader_redacts_validation_error_paths(
    tmp_path, valid_dicom_dataset, caplog, capsys
) -> None:
    """Validation errors stored in stats should redact source file paths."""
    patient_dir = tmp_path / "PATIENT_JANE_DOE"
    patient_dir.mkdir()
    dicom_file = patient_dir / "Jane_Doe_12345.dcm"
    dataset_copy = copy.deepcopy(valid_dicom_dataset)
    del dataset_copy.PatientID
    _save_dicom(dataset_copy, str(dicom_file))

    reader = DicomReader(validate_on_read=True)
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    reader_module.logger.addHandler(handler)
    try:
        with caplog.at_level(logging.INFO, logger="mammography.io.dicom.reader"):
            result = reader.read_dicom_directory(patient_dir)
    finally:
        reader_module.logger.removeHandler(handler)
    captured = capsys.readouterr()
    stats = reader.get_processing_stats()
    errors_text = "\n".join(
        [caplog.text, captured.err, log_stream.getvalue(), *stats["recent_errors"]]
    )

    assert result == {}
    assert "PatientID" in errors_text
    assert str(dicom_file) not in errors_text
    assert dicom_file.name not in errors_text
    assert "PATIENT_JANE_DOE" not in errors_text
    assert "<dicom-path:" in errors_text

class TestDicomReaderReadDirectoryBasics:
    """Test basic directory reading functionality."""

    def test_read_empty_directory(self, tmp_path):
        """Test reading an empty directory returns empty dict."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        reader = DicomReader()
        result = reader.read_dicom_directory(empty_dir)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_read_directory_with_files(self, dicom_directory_with_files):
        """Test reading a directory with DICOM files."""
        dicom_dir, expected_files = dicom_directory_with_files

        reader = DicomReader()
        result = reader.read_dicom_directory(dicom_dir)

        assert isinstance(result, dict)
        assert len(result) > 0

        # With patient_level=True (default), keys are patient IDs
        for patient_id, images in result.items():
            assert isinstance(patient_id, str)
            assert isinstance(images, list)
            assert all(isinstance(img, MammographyImage) for img in images)

    def test_read_directory_patient_level_true(self, dicom_directory_with_files):
        """Test that patient_level=True organizes by patient ID."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader()
        result = reader.read_dicom_directory(dicom_dir, patient_level=True)

        # Each key should be a patient ID
        for patient_id, images in result.items():
            # All images for this patient should have the same patient_id
            for img in images:
                assert img.patient_id == patient_id

    def test_read_directory_patient_level_false(self, dicom_directory_with_files):
        """Test that patient_level=False organizes by file path."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader()
        result = reader.read_dicom_directory(dicom_dir, patient_level=False)

        # Each key should be a file path
        for file_path, images in result.items():
            assert isinstance(file_path, str)
            assert len(images) == 1  # One image per file
            assert images[0].file_path == str(Path(file_path).absolute())

    def test_read_directory_returns_mammography_images(self, dicom_directory_with_files):
        """Test that all returned items are MammographyImage instances."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader()
        result = reader.read_dicom_directory(dicom_dir)

        for images in result.values():
            for img in images:
                assert isinstance(img, MammographyImage)
                assert hasattr(img, "patient_id")
                assert hasattr(img, "file_path")
                assert hasattr(img, "projection_type")
                assert hasattr(img, "laterality")

    def test_create_reader_wires_cache_size(self):
        """Factory should pass cache_size through to DicomReader."""
        reader = create_dicom_reader(cache_size=2)

        assert reader.cache_size == 2

    def test_metadata_cache_honors_cache_size(self, dicom_directory_with_files):
        """Metadata cache should remain bounded by cache_size."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader(cache_size=1)
        reader.read_dicom_directory(dicom_dir)

        assert len(reader.metadata_cache) <= 1

    def test_metadata_cache_size_zero_disables_retention(self, dicom_directory_with_files):
        """cache_size=0 should keep metadata retention disabled without failing reads."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader(cache_size=0)
        reader.read_dicom_directory(dicom_dir)

        assert len(reader.metadata_cache) == 0

    @pytest.mark.parametrize("cache_size", [-1, 1.5, "2", True])
    def test_rejects_invalid_cache_size(self, cache_size):
        """Invalid cache_size values should fail at construction time."""
        with pytest.raises(ValueError, match="cache_size"):
            DicomReader(cache_size=cache_size)

class TestDicomReaderRecursiveBehavior:
    """Test recursive and non-recursive directory scanning."""

    def test_read_directory_recursive_true(self, nested_dicom_directory):
        """Test that recursive=True finds files in subdirectories."""
        root_dir, all_files = nested_dicom_directory

        reader = DicomReader()
        result = reader.read_dicom_directory(root_dir, recursive=True)

        # Count total images across all patients
        total_images = sum(len(images) for images in result.values())

        # Should find all files (root + subdirectories)
        assert total_images == len(all_files)

    def test_read_directory_recursive_false(self, nested_dicom_directory):
        """Test that recursive=False only finds files in root directory."""
        root_dir, all_files = nested_dicom_directory

        reader = DicomReader()
        result = reader.read_dicom_directory(root_dir, recursive=False)

        # Count total images across all patients
        total_images = sum(len(images) for images in result.values())

        # Should only find files in root directory (2 files)
        assert total_images == 2
        assert total_images < len(all_files)

    def test_recursive_finds_deeply_nested_files(self, tmp_path, valid_dicom_dataset):
        """Test that recursive search finds files in deeply nested directories."""
        # Create a deeply nested structure
        deep_dir = tmp_path / "level1" / "level2" / "level3"
        deep_dir.mkdir(parents=True)

        dataset_copy = copy.deepcopy(valid_dicom_dataset)
        filepath = deep_dir / "deep_file.dcm"
        _save_dicom(dataset_copy, str(filepath))

        reader = DicomReader()
        result = reader.read_dicom_directory(tmp_path / "level1", recursive=True)

        # Should find the deeply nested file
        total_images = sum(len(images) for images in result.values())
        assert total_images == 1

class TestDicomReaderDirectoryErrorHandling:
    """Test error handling for invalid inputs."""

    def test_read_nonexistent_directory(self, tmp_path):
        """Test that reading non-existent directory raises FileNotFoundError."""
        nonexistent_dir = tmp_path / "PATIENT_JANE_DOE" / "does_not_exist"

        reader = DicomReader()

        with pytest.raises(FileNotFoundError) as exc_info:
            reader.read_dicom_directory(nonexistent_dir)

        assert "Directory not found" in str(exc_info.value)
        assert str(nonexistent_dir) not in str(exc_info.value)
        assert "PATIENT_JANE_DOE" not in str(exc_info.value)
        assert "<dicom-path:" in str(exc_info.value)

    def test_read_file_instead_of_directory(self, tmp_path):
        """Test that reading a file instead of directory raises ValueError."""
        # Create a file
        test_file = tmp_path / "Jane_Doe_12345.txt"
        test_file.write_text("Not a directory")

        reader = DicomReader()

        with pytest.raises(ValueError) as exc_info:
            reader.read_dicom_directory(test_file)

        assert "not a directory" in str(exc_info.value).lower()
        assert str(test_file) not in str(exc_info.value)
        assert test_file.name not in str(exc_info.value)
        assert "<dicom-path:" in str(exc_info.value)

    def test_read_directory_with_invalid_dicoms(self, tmp_path):
        """Test that invalid DICOM files are skipped gracefully."""
        invalid_dir = tmp_path / "invalid_dicoms"
        invalid_dir.mkdir()

        # Create files with .dcm extension but invalid content
        for i in range(3):
            invalid_file = invalid_dir / f"invalid_{i}.dcm"
            invalid_file.write_text(f"This is not a valid DICOM file {i}")

        reader = DicomReader()
        result = reader.read_dicom_directory(invalid_dir)

        # Should return empty dict (no valid DICOM files)
        assert isinstance(result, dict)
        # May be empty or have filtered out invalid files
        total_images = sum(len(images) for images in result.values())
        assert total_images == 0

    def test_read_directory_with_mixed_files(self, mixed_file_directory):
        """Test that non-DICOM files are ignored."""
        mixed_dir, dicom_files = mixed_file_directory

        reader = DicomReader()
        result = reader.read_dicom_directory(mixed_dir)

        # Count total valid images
        total_images = sum(len(images) for images in result.values())

        # Should only find DICOM files, not txt/json/png files
        assert total_images == len(dicom_files)
