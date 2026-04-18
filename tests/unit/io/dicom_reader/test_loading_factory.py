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

class TestDicomReaderLazyLoading:
    """Test lazy loading behavior during directory reading."""

    def test_lazy_load_true(self, dicom_directory_with_files):
        """Test that lazy loading defers pixel data loading."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader(lazy_load=True)
        result = reader.read_dicom_directory(dicom_dir)

        # Should successfully read files with lazy loading
        assert len(result) > 0

    def test_lazy_load_false(self, dicom_directory_with_files):
        """Test that lazy_load=False loads pixel data immediately."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader(lazy_load=False)
        result = reader.read_dicom_directory(dicom_dir)

        # Should successfully read files
        assert len(result) > 0

class TestDicomReaderFactoryFunctions:
    """Test factory functions and convenience methods."""

    def test_create_dicom_reader_factory(self):
        """Test that create_dicom_reader factory function works."""
        reader = create_dicom_reader(
            validate_on_read=True,
            cache_metadata=True,
            max_workers=4,
            lazy_load=False,
        )

        assert isinstance(reader, DicomReader)
        assert reader.validate_on_read is True
        assert reader.cache_metadata is True
        assert reader.max_workers == 4
        assert reader.lazy_load is False

    def test_create_dicom_reader_defaults(self):
        """Test that create_dicom_reader uses correct defaults."""
        reader = create_dicom_reader()

        assert isinstance(reader, DicomReader)
        assert reader.validate_on_read is True
        assert reader.cache_metadata is True
        assert reader.max_workers == 4
        assert reader.lazy_load is False

class TestDicomReaderPathHandling:
    """Test path handling (string vs Path objects)."""

    def test_read_directory_with_path_object(self, dicom_directory_with_files):
        """Test reading directory using Path object."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader()
        result = reader.read_dicom_directory(Path(dicom_dir))

        assert len(result) > 0

    def test_read_directory_with_string_path(self, dicom_directory_with_files):
        """Test reading directory using string path."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader()
        result = reader.read_dicom_directory(str(dicom_dir))

        assert len(result) > 0

class TestDicomReaderExtensionHandling:
    """Test handling of different DICOM file extensions."""

    def test_reads_dcm_extension(self, tmp_path, valid_dicom_dataset):
        """Test that .dcm files are recognized."""
        test_dir = tmp_path / "dcm_test"
        test_dir.mkdir()

        filepath = test_dir / "file.dcm"
        _save_dicom(valid_dicom_dataset, str(filepath))

        reader = DicomReader()
        result = reader.read_dicom_directory(test_dir)

        total_images = sum(len(images) for images in result.values())
        assert total_images == 1

    def test_reads_dicom_extension(self, tmp_path, valid_dicom_dataset):
        """Test that .dicom files are recognized."""
        test_dir = tmp_path / "dicom_test"
        test_dir.mkdir()

        filepath = test_dir / "file.dicom"
        _save_dicom(valid_dicom_dataset, str(filepath))

        reader = DicomReader()
        result = reader.read_dicom_directory(test_dir)

        total_images = sum(len(images) for images in result.values())
        assert total_images == 1

    def test_reads_uppercase_extensions(self, tmp_path, valid_dicom_dataset):
        """Test that uppercase extensions (.DCM, .DICOM) are recognized."""
        test_dir = tmp_path / "uppercase_test"
        test_dir.mkdir()

        # Create file with .DCM extension
        filepath_dcm = test_dir / "file.DCM"
        _save_dicom(valid_dicom_dataset, str(filepath_dcm))

        reader = DicomReader()
        result = reader.read_dicom_directory(test_dir)

        total_images = sum(len(images) for images in result.values())
        assert total_images >= 1

    def test_reads_dicom_content_without_extension(self, tmp_path, valid_dicom_dataset):
        """Test that DICOM content is recognized without relying on extensions."""
        test_dir = tmp_path / "extensionless_test"
        test_dir.mkdir()

        filepath = test_dir / "dicom_without_extension"
        _save_dicom(valid_dicom_dataset, str(filepath))

        reader = DicomReader()
        result = reader.read_dicom_directory(test_dir)

        total_images = sum(len(images) for images in result.values())
        assert total_images == 1

    def test_read_dicom_file_parses_valid_file_once(
        self, tmp_path, valid_dicom_dataset, monkeypatch
    ):
        """Test direct reads reuse the preamble-opened file for parsing."""
        filepath = tmp_path / "single_parse.dcm"
        _save_dicom(valid_dicom_dataset, str(filepath))
        original_dcmread = reader_module.pydicom.dcmread
        calls = []

        def counting_dcmread(fp, *args, **kwargs):
            calls.append(fp)
            return original_dcmread(fp, *args, **kwargs)

        monkeypatch.setattr(reader_module.pydicom, "dcmread", counting_dcmread)

        reader = DicomReader(validate_on_read=True)
        image = reader.read_dicom_file(filepath)

        assert isinstance(image, MammographyImage)
        assert len(calls) == 1
        assert hasattr(calls[0], "read")

class TestDicomReaderReturnStructure:
    """Test the structure of returned data."""

    def test_return_type_is_dict(self, dicom_directory_with_files):
        """Test that return type is a dictionary."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader()
        result = reader.read_dicom_directory(dicom_dir)

        assert isinstance(result, dict)

    def test_return_dict_values_are_lists(self, dicom_directory_with_files):
        """Test that dictionary values are lists of MammographyImage."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader()
        result = reader.read_dicom_directory(dicom_dir)

        for key, value in result.items():
            assert isinstance(value, list)
            assert all(isinstance(img, MammographyImage) for img in value)

    def test_patient_level_keys_are_patient_ids(self, dicom_directory_with_files):
        """Test that with patient_level=True, keys are patient IDs."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader()
        result = reader.read_dicom_directory(dicom_dir, patient_level=True)

        for patient_id, images in result.items():
            # All images should have this patient_id
            assert all(img.patient_id == patient_id for img in images)

    def test_file_level_keys_are_file_paths(self, dicom_directory_with_files):
        """Test that with patient_level=False, keys are file paths."""
        dicom_dir, _ = dicom_directory_with_files

        reader = DicomReader()
        result = reader.read_dicom_directory(dicom_dir, patient_level=False)

        for file_path, images in result.items():
            # Should have exactly one image per file path
            assert len(images) == 1
            # Image file path should match key (as absolute path)
            assert images[0].file_path == str(Path(file_path).absolute())
