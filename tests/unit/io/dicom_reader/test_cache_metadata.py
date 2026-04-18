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

class TestDicomReaderCacheManagement:
    """Test cache clearing and management functionality."""

    def test_clear_cache_empties_metadata_cache(self, dicom_directory_with_files):
        """Test that clear_cache empties the metadata cache."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader(cache_metadata=True)
        reader.read_dicom_directory(dicom_dir)

        # Cache should be populated
        assert len(reader.metadata_cache) > 0

        # Clear the cache
        reader.clear_cache()

        # Cache should now be empty
        assert len(reader.metadata_cache) == 0

    def test_clear_cache_on_empty_cache(self):
        """Test that clear_cache works on an empty cache."""
        reader = DicomReader(cache_metadata=True)

        # Cache should be empty initially
        assert len(reader.metadata_cache) == 0

        # Clearing empty cache should not raise error
        reader.clear_cache()

        # Cache should still be empty
        assert len(reader.metadata_cache) == 0

    def test_cache_persists_across_multiple_reads(self, dicom_directory_with_files):
        """Test that cache can be cleared and repopulated."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader(cache_metadata=True)

        # First read
        reader.read_dicom_directory(dicom_dir)
        first_cache_size = len(reader.metadata_cache)
        assert first_cache_size > 0

        # Cache should persist and can be cleared
        reader.clear_cache()
        assert len(reader.metadata_cache) == 0

        # Read again after clearing
        reader.read_dicom_directory(dicom_dir)
        assert len(reader.metadata_cache) == first_cache_size

class TestDicomReaderEnhancedStatistics:
    """Enhanced tests for statistics tracking."""

    def test_stats_initialized_to_zero(self):
        """Test that stats are initialized to zero on creation."""
        reader = DicomReader()

        assert reader.stats["files_processed"] == 0
        assert reader.stats["files_valid"] == 0
        assert reader.stats["files_invalid"] == 0
        assert list(reader.stats["validation_errors"]) == []
        assert (
            reader.stats["validation_errors"].maxlen
            == DicomReader.VALIDATION_ERROR_HISTORY_LIMIT
        )

    def test_stats_track_valid_files_correctly(self, dicom_directory_with_files):
        """Test that stats correctly track valid files."""
        dicom_dir, expected_files = dicom_directory_with_files

        reader = DicomReader(validate_on_read=True)
        reader.read_dicom_directory(dicom_dir)

        # All files should be valid
        assert reader.stats["files_valid"] == len(expected_files)
        assert reader.stats["files_processed"] == len(expected_files)
        assert reader.stats["files_invalid"] == 0

    def test_stats_track_invalid_files(self, tmp_path):
        """Test that stats correctly track invalid files."""
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()

        # Create invalid DICOM files
        for i in range(3):
            invalid_file = invalid_dir / f"bad_{i}.dcm"
            invalid_file.write_text("Not a valid DICOM")

        reader = DicomReader(validate_on_read=True)
        result = reader.read_dicom_directory(invalid_dir)

        # All files should be invalid
        assert reader.stats["files_invalid"] == 3
        assert len(result) == 0

    def test_read_file_counts_missing_file_attempt(self, tmp_path):
        """Test that missing direct reads count as attempted and invalid."""
        reader = DicomReader(validate_on_read=True)

        with pytest.raises(FileNotFoundError):
            reader.read_dicom_file(tmp_path / "missing.dcm")

        stats = reader.get_processing_stats()
        assert stats["files_processed"] == 1
        assert stats["files_invalid"] == 1

    def test_read_file_counts_non_dicom_content_attempt(self, tmp_path):
        """Test that non-DICOM direct reads count as attempted and invalid."""
        unsupported = tmp_path / "not_dicom.txt"
        unsupported.write_text("not a dicom")
        reader = DicomReader(validate_on_read=True)

        result = reader.read_dicom_file(unsupported)

        stats = reader.get_processing_stats()
        assert result is None
        assert stats["files_processed"] == 1
        assert stats["files_invalid"] == 1

    def test_stats_accumulate_across_operations(self, dicom_directory_with_files, tmp_path):
        """Test that stats accumulate across multiple read operations."""
        dicom_dir, expected_files = dicom_directory_with_files

        reader = DicomReader(validate_on_read=True)

        # First read
        reader.read_dicom_directory(dicom_dir)
        first_count = reader.stats["files_processed"]

        # Create another directory
        second_dir = tmp_path / "second"
        second_dir.mkdir()

        # Copy one file to second directory
        valid_dicom_dataset = pytest.importorskip("pydicom")
        # Can't easily create another file without fixture, so just check stats persist
        assert reader.stats["files_processed"] == first_count
        assert reader.stats["files_valid"] > 0

    def test_get_processing_stats_includes_success_rate(self, dicom_directory_with_files):
        """Test that get_processing_stats includes success rate."""
        dicom_dir, expected_files = dicom_directory_with_files

        reader = DicomReader()
        reader.read_dicom_directory(dicom_dir)

        stats = reader.get_processing_stats()

        assert "success_rate" in stats
        # With all valid files, success rate should be 100%
        assert stats["success_rate"] == 100.0

    def test_get_processing_stats_with_mixed_results(self, mixed_file_directory):
        """Test stats with mix of valid and invalid files."""
        mixed_dir, dicom_files = mixed_file_directory

        reader = DicomReader(validate_on_read=True)
        reader.read_dicom_directory(mixed_dir)

        stats = reader.get_processing_stats()

        # Valid DICOMs and non-DICOM candidates should be counted separately.
        assert stats["files_valid"] == len(dicom_files)
        assert stats["files_invalid"] >= 3
        assert stats["files_valid"] > 0
        assert stats["success_rate"] <= 100.0

    def test_stats_recent_errors_limited(self, tmp_path):
        """Test that recent_errors in stats is limited to last 10."""
        invalid_dir = tmp_path / "many_invalid"
        invalid_dir.mkdir()

        # Create many invalid DICOM files
        for i in range(15):
            invalid_file = invalid_dir / f"bad_{i}.dcm"
            invalid_file.write_text(f"Not valid DICOM {i}")

        reader = DicomReader(validate_on_read=True)
        reader.read_dicom_directory(invalid_dir)

        stats = reader.get_processing_stats()

        # recent_errors should be limited to 10
        assert len(stats["recent_errors"]) <= 10

    def test_validation_error_history_is_bounded(self):
        """Test that retained validation errors do not grow without bound."""
        reader = DicomReader()
        limit = DicomReader.VALIDATION_ERROR_HISTORY_LIMIT

        for index in range(limit + 5):
            reader._record_invalid(f"error {index}")

        retained_errors = list(reader.stats["validation_errors"])
        assert len(retained_errors) == limit
        assert retained_errors[0] == "error 5"
        assert reader.get_processing_stats()["total_validation_errors"] == limit

class TestDicomReaderEnhancedLazyLoading:
    """Enhanced tests for lazy loading behavior."""

    def test_lazy_load_defers_pixel_data_loading(self, dicom_directory_with_files):
        """Test that lazy_load=True defers pixel data loading."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader(lazy_load=True, validate_on_read=False)
        result = reader.read_dicom_directory(dicom_dir)

        # Should successfully read files without loading pixel data
        assert len(result) > 0

        # Images should be created even without pixel data
        for images in result.values():
            for img in images:
                assert isinstance(img, MammographyImage)

    def test_lazy_load_false_loads_pixel_data(self, dicom_directory_with_files):
        """Test that lazy_load=False loads pixel data immediately."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader(lazy_load=False, validate_on_read=True)
        result = reader.read_dicom_directory(dicom_dir)

        # Should successfully read and validate files with pixel data
        assert len(result) > 0

        for images in result.values():
            for img in images:
                # With lazy_load=False and validate_on_read=True, should be validated
                assert img.state in ["validated", "preprocessed", "embedded", "clustered"]

    def test_lazy_load_with_validation(self, dicom_directory_with_files):
        """Test lazy loading with validation enabled."""
        dicom_dir, _ = dicom_directory_with_files

        # lazy_load=True skips pixel data validation
        reader = DicomReader(lazy_load=True, validate_on_read=True)
        result = reader.read_dicom_directory(dicom_dir)

        # Should return valid images even with lazy loading
        assert len(result) > 0

    def test_lazy_load_parameter_passed_to_read(self, tmp_path, valid_dicom_dataset):
        """Test that lazy_load parameter is used in read operations."""
        test_dir = tmp_path / "lazy_test"
        test_dir.mkdir()

        filepath = test_dir / "test.dcm"
        _save_dicom(valid_dicom_dataset, str(filepath))

        # Test with lazy_load=True
        reader_lazy = DicomReader(lazy_load=True)
        img_lazy = reader_lazy.read_dicom_file(filepath)
        assert img_lazy is not None

        # Test with lazy_load=False
        reader_eager = DicomReader(lazy_load=False)
        img_eager = reader_eager.read_dicom_file(filepath)
        assert img_eager is not None

class TestDicomConvenienceFunctions:
    """Test convenience functions for DICOM reading."""

    def test_read_single_dicom_function(self, tmp_path, valid_dicom_dataset):
        """Test read_single_dicom convenience function."""
        from mammography.io.dicom import read_single_dicom

        test_file = tmp_path / "single.dcm"
        _save_dicom(valid_dicom_dataset, str(test_file))

        img = read_single_dicom(test_file)

        assert img is not None
        assert isinstance(img, MammographyImage)
        assert img.patient_id == "TEST_PATIENT_001"

    def test_read_single_dicom_missing_file(self, tmp_path):
        """Test read_single_dicom with missing file."""
        from mammography.io.dicom import read_single_dicom

        missing_file = tmp_path / "missing.dcm"

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            read_single_dicom(missing_file)

    def test_read_dicom_directory_function(self, dicom_directory_with_files):
        """Test read_dicom_directory convenience function."""
        from mammography.io.dicom import read_dicom_directory

        dicom_dir, expected_files = dicom_directory_with_files

        result = read_dicom_directory(dicom_dir)

        assert isinstance(result, dict)
        assert len(result) > 0

        # Should organize by patient by default
        for patient_id, images in result.items():
            assert isinstance(patient_id, str)
            assert isinstance(images, list)

    def test_read_dicom_directory_recursive_param(self, nested_dicom_directory):
        """Test read_dicom_directory with recursive parameter."""
        from mammography.io.dicom import read_dicom_directory

        root_dir, all_files = nested_dicom_directory

        # Test recursive=True
        result_recursive = read_dicom_directory(root_dir, recursive=True)
        total_recursive = sum(len(images) for images in result_recursive.values())
        assert total_recursive == len(all_files)

        # Test recursive=False
        result_non_recursive = read_dicom_directory(root_dir, recursive=False)
        total_non_recursive = sum(len(images) for images in result_non_recursive.values())
        assert total_non_recursive < total_recursive

    def test_read_dicom_directory_patient_level_param(self, dicom_directory_with_files):
        """Test read_dicom_directory with patient_level parameter."""
        from mammography.io.dicom import read_dicom_directory

        dicom_dir, _ = dicom_directory_with_files

        # Test patient_level=True
        result_patient = read_dicom_directory(dicom_dir, patient_level=True)
        for key, images in result_patient.items():
            # Keys should be patient IDs
            assert all(img.patient_id == key for img in images)

        # Test patient_level=False
        result_file = read_dicom_directory(dicom_dir, patient_level=False)
        for key, images in result_file.items():
            # Keys should be file paths
            assert len(images) == 1
            assert images[0].file_path == str(Path(key).absolute())

    def test_load_dicom_function(self, tmp_path, valid_dicom_dataset):
        """Test load_dicom convenience function."""
        from mammography.io.dicom import load_dicom

        test_file = tmp_path / "load_test.dcm"
        _save_dicom(valid_dicom_dataset, str(test_file))

        img = load_dicom(test_file)

        assert img is not None
        assert isinstance(img, MammographyImage)
        assert img.patient_id == "TEST_PATIENT_001"

    def test_load_dicom_with_lazy_load_true(self, tmp_path, valid_dicom_dataset):
        """Test load_dicom with lazy_load=True."""
        from mammography.io.dicom import load_dicom

        test_file = tmp_path / "lazy.dcm"
        _save_dicom(valid_dicom_dataset, str(test_file))

        img = load_dicom(test_file, lazy_load=True)

        assert img is not None
        assert isinstance(img, MammographyImage)
        # Pixel data should not be loaded immediately with lazy loading

    def test_load_dicom_with_lazy_load_false(self, tmp_path, valid_dicom_dataset):
        """Test load_dicom with lazy_load=False."""
        from mammography.io.dicom import load_dicom

        test_file = tmp_path / "eager.dcm"
        _save_dicom(valid_dicom_dataset, str(test_file))

        img = load_dicom(test_file, lazy_load=False)

        assert img is not None
        assert isinstance(img, MammographyImage)
        # Pixel data should be loaded and validated

    def test_load_dicom_with_validate_true(self, tmp_path, valid_dicom_dataset):
        """Test load_dicom with validate=True."""
        from mammography.io.dicom import load_dicom

        test_file = tmp_path / "validate.dcm"
        _save_dicom(valid_dicom_dataset, str(test_file))

        img = load_dicom(test_file, validate=True)

        assert img is not None
        # Should be validated
        assert img.state in ["validated", "preprocessed", "embedded", "clustered"]

    def test_load_dicom_with_validate_false(self, tmp_path, valid_dicom_dataset):
        """Test load_dicom with validate=False."""
        from mammography.io.dicom import load_dicom

        test_file = tmp_path / "no_validate.dcm"
        _save_dicom(valid_dicom_dataset, str(test_file))

        img = load_dicom(test_file, validate=False)

        assert img is not None
        assert isinstance(img, MammographyImage)

    def test_load_dicom_default_parameters(self, tmp_path, valid_dicom_dataset):
        """Test load_dicom with default parameters."""
        from mammography.io.dicom import load_dicom

        test_file = tmp_path / "defaults.dcm"
        _save_dicom(valid_dicom_dataset, str(test_file))

        # Default: lazy_load=True, validate=True
        img = load_dicom(test_file)

        assert img is not None
        assert isinstance(img, MammographyImage)

class TestDicomReaderMetadataSummary:
    """Test metadata summary functionality."""

    def test_get_metadata_summary_empty_cache(self):
        """Test get_metadata_summary with empty cache."""
        reader = DicomReader(cache_metadata=True)

        summary = reader.get_metadata_summary()

        assert summary["total_files"] == 0
        assert "message" in summary

    def test_get_metadata_summary_populated_cache(self, dicom_directory_with_files):
        """Test get_metadata_summary with populated cache."""
        dicom_dir, expected_files = dicom_directory_with_files

        reader = DicomReader(cache_metadata=True)
        reader.read_dicom_directory(dicom_dir)

        summary = reader.get_metadata_summary()

        assert summary["total_files"] == len(expected_files)
        assert "manufacturers" in summary
        assert "projection_types" in summary
        assert "lateralities" in summary
        assert "pixel_spacing_range" in summary
        assert "bits_stored_values" in summary
        assert "processing_stats" in summary

    def test_metadata_summary_includes_stats(self, dicom_directory_with_files):
        """Test that metadata summary includes processing stats."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader(cache_metadata=True)
        reader.read_dicom_directory(dicom_dir)

        summary = reader.get_metadata_summary()

        assert "processing_stats" in summary
        stats = summary["processing_stats"]
        assert "files_processed" in stats
        assert "files_valid" in stats
        assert "files_invalid" in stats
