"""
Unit tests for DICOM validation functionality.

These tests validate individual DICOM validation functions and rules.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import os
import tempfile

import numpy as np
import pydicom
import pytest

# Import the modules we'll be testing (these will be implemented later)
# from src.io_dicom.validator import DICOMValidator
# from src.io_dicom.metadata_extractor import MetadataExtractor
# from src.io_dicom.projection_inference import ProjectionInference


class TestDICOMValidation:
    """Unit tests for DICOM validation functions."""

    @pytest.fixture
    def valid_dicom_dataset(self):
        """Create a valid DICOM dataset for testing."""
        dataset = pydicom.Dataset()

        # Required fields for mammography
        dataset.PatientID = "TEST_PATIENT_001"
        dataset.StudyInstanceUID = "1.2.840.12345.123456789"
        dataset.SeriesInstanceUID = "1.2.840.12345.987654321"
        dataset.SOPInstanceUID = "1.2.840.12345.456789123"
        dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture

        # Mammography-specific fields
        dataset.Manufacturer = "SIEMENS"
        dataset.PixelSpacing = [0.1, 0.1]
        dataset.BitsStored = 16
        dataset.BitsAllocated = 16
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.Rows = 2048
        dataset.Columns = 1536

        # Projection information
        dataset.ViewPosition = "CC"
        dataset.ImageLaterality = "L"

        # Create pixel data
        dataset.PixelData = np.random.randint(
            0, 4095, (2048, 1536), dtype=np.uint16
        ).tobytes()

        return dataset

    @pytest.fixture
    def invalid_dicom_dataset(self):
        """Create an invalid DICOM dataset for testing."""
        dataset = pydicom.Dataset()

        # Missing required fields
        dataset.PatientID = "TEST_PATIENT_002"
        # Missing StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID

        # Invalid values
        dataset.PixelSpacing = [0.0, 0.0]  # Invalid pixel spacing
        dataset.BitsStored = 0  # Invalid bits stored
        dataset.Manufacturer = ""  # Empty manufacturer

        return dataset

    def test_validate_required_fields(self, valid_dicom_dataset, invalid_dicom_dataset):
        """Test validation of required DICOM fields."""
        required_fields = [
            "PatientID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SOPInstanceUID",
            "SOPClassUID",
        ]

        # Test valid dataset
        for field in required_fields:
            assert hasattr(valid_dicom_dataset, field)
            assert getattr(valid_dicom_dataset, field) is not None
            assert str(getattr(valid_dicom_dataset, field)).strip() != ""

        # Test invalid dataset
        for field in required_fields:
            if field == "PatientID":
                assert hasattr(invalid_dicom_dataset, field)
            else:
                assert not hasattr(invalid_dicom_dataset, field)

    def test_validate_pixel_spacing(self, valid_dicom_dataset, invalid_dicom_dataset):
        """Test validation of pixel spacing."""
        # Test valid pixel spacing
        pixel_spacing = valid_dicom_dataset.PixelSpacing
        assert isinstance(pixel_spacing, list)
        assert len(pixel_spacing) == 2
        assert all(isinstance(x, (int, float)) and x > 0 for x in pixel_spacing)

        # Test invalid pixel spacing
        invalid_pixel_spacing = invalid_dicom_dataset.PixelSpacing
        assert isinstance(invalid_pixel_spacing, list)
        assert len(invalid_pixel_spacing) == 2
        assert not all(
            isinstance(x, (int, float)) and x > 0 for x in invalid_pixel_spacing
        )

    def test_validate_bits_stored(self, valid_dicom_dataset, invalid_dicom_dataset):
        """Test validation of bits stored."""
        # Test valid bits stored
        bits_stored = valid_dicom_dataset.BitsStored
        assert isinstance(bits_stored, int)
        assert 8 <= bits_stored <= 16

        # Test invalid bits stored
        invalid_bits_stored = invalid_dicom_dataset.BitsStored
        assert isinstance(invalid_bits_stored, int)
        assert not (8 <= invalid_bits_stored <= 16)

    def test_validate_manufacturer(self, valid_dicom_dataset, invalid_dicom_dataset):
        """Test validation of manufacturer field."""
        # Test valid manufacturer
        manufacturer = valid_dicom_dataset.Manufacturer
        assert isinstance(manufacturer, str)
        assert len(manufacturer.strip()) > 0

        # Test invalid manufacturer
        invalid_manufacturer = invalid_dicom_dataset.Manufacturer
        assert isinstance(invalid_manufacturer, str)
        assert len(invalid_manufacturer.strip()) == 0

    def test_validate_projection_type(self, valid_dicom_dataset):
        """Test validation of projection type (CC/MLO)."""
        # Test valid projection type
        view_position = valid_dicom_dataset.ViewPosition
        assert view_position in ["CC", "MLO"]

        # Test valid laterality
        laterality = valid_dicom_dataset.ImageLaterality
        assert laterality in ["L", "R"]

    def test_validate_pixel_data(self, valid_dicom_dataset):
        """Test validation of pixel data."""
        # Test pixel data exists
        assert hasattr(valid_dicom_dataset, "PixelData")
        assert valid_dicom_dataset.PixelData is not None

        # Test pixel data dimensions
        rows = valid_dicom_dataset.Rows
        cols = valid_dicom_dataset.Columns
        assert rows > 0
        assert cols > 0

        # Test pixel data size
        expected_size = rows * cols * 2  # 2 bytes per pixel for 16-bit
        assert len(valid_dicom_dataset.PixelData) == expected_size

    def test_validate_photometric_interpretation(self, valid_dicom_dataset):
        """Test validation of photometric interpretation."""
        photometric = valid_dicom_dataset.PhotometricInterpretation
        assert photometric in ["MONOCHROME1", "MONOCHROME2"]

    def test_validate_samples_per_pixel(self, valid_dicom_dataset):
        """Test validation of samples per pixel."""
        samples_per_pixel = valid_dicom_dataset.SamplesPerPixel
        assert samples_per_pixel == 1  # Grayscale mammography

    def test_validate_bits_allocated(self, valid_dicom_dataset):
        """Test validation of bits allocated."""
        bits_allocated = valid_dicom_dataset.BitsAllocated
        assert isinstance(bits_allocated, int)
        assert bits_allocated in [8, 16, 32]
        assert bits_allocated >= valid_dicom_dataset.BitsStored

    def test_validate_pixel_array_extraction(self, valid_dicom_dataset):
        """Test extraction and validation of pixel array."""
        # Extract pixel array
        pixel_array = valid_dicom_dataset.pixel_array

        # Validate pixel array
        assert pixel_array is not None
        assert pixel_array.ndim == 2
        assert pixel_array.shape == (
            valid_dicom_dataset.Rows,
            valid_dicom_dataset.Columns,
        )
        assert pixel_array.dtype in ["uint8", "uint16", "int16"]

        # Validate pixel values
        assert pixel_array.min() >= 0
        assert pixel_array.max() <= (2**valid_dicom_dataset.BitsStored - 1)

    def test_validate_dicom_file_reading(self, valid_dicom_dataset):
        """Test reading DICOM files from disk."""
        # Create temporary DICOM file
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp_file:
            valid_dicom_dataset.save_as(tmp_file.name)
            tmp_file_path = tmp_file.name

        try:
            # Read DICOM file
            dataset = pydicom.dcmread(tmp_file_path)

            # Validate read dataset
            assert dataset.PatientID == valid_dicom_dataset.PatientID
            assert dataset.StudyInstanceUID == valid_dicom_dataset.StudyInstanceUID
            assert dataset.SeriesInstanceUID == valid_dicom_dataset.SeriesInstanceUID
            assert dataset.SOPInstanceUID == valid_dicom_dataset.SOPInstanceUID

            # Validate pixel data
            pixel_array = dataset.pixel_array
            assert pixel_array is not None
            assert pixel_array.shape == (
                valid_dicom_dataset.Rows,
                valid_dicom_dataset.Columns,
            )

        finally:
            os.unlink(tmp_file_path)

    def test_validate_error_handling(self, invalid_dicom_dataset):
        """Test error handling for invalid DICOM data."""
        # Test missing required fields
        with pytest.raises(AttributeError):
            invalid_dicom_dataset.StudyInstanceUID

        # Test invalid field values
        with pytest.raises(AssertionError):
            assert all(x > 0 for x in invalid_dicom_dataset.PixelSpacing)

        with pytest.raises(AssertionError):
            assert 8 <= invalid_dicom_dataset.BitsStored <= 16

        with pytest.raises(AssertionError):
            assert len(invalid_dicom_dataset.Manufacturer.strip()) > 0

    def test_validate_metadata_extraction(self, valid_dicom_dataset):
        """Test extraction of metadata from DICOM dataset."""
        # Extract metadata
        metadata = {
            "patient_id": getattr(valid_dicom_dataset, "PatientID", None),
            "study_id": getattr(valid_dicom_dataset, "StudyInstanceUID", None),
            "series_id": getattr(valid_dicom_dataset, "SeriesInstanceUID", None),
            "instance_id": getattr(valid_dicom_dataset, "SOPInstanceUID", None),
            "manufacturer": getattr(valid_dicom_dataset, "Manufacturer", None),
            "pixel_spacing": getattr(valid_dicom_dataset, "PixelSpacing", None),
            "bits_stored": getattr(valid_dicom_dataset, "BitsStored", None),
            "acquisition_date": getattr(valid_dicom_dataset, "AcquisitionDate", None),
            "projection_type": getattr(valid_dicom_dataset, "ViewPosition", None),
            "laterality": getattr(valid_dicom_dataset, "ImageLaterality", None),
        }

        # Validate extracted metadata
        assert metadata["patient_id"] is not None
        assert metadata["study_id"] is not None
        assert metadata["series_id"] is not None
        assert metadata["instance_id"] is not None
        assert metadata["manufacturer"] is not None
        assert metadata["pixel_spacing"] is not None
        assert metadata["bits_stored"] is not None
        assert metadata["projection_type"] is not None
        assert metadata["laterality"] is not None

        # Validate data types
        assert isinstance(metadata["patient_id"], str)
        assert isinstance(metadata["study_id"], str)
        assert isinstance(metadata["series_id"], str)
        assert isinstance(metadata["instance_id"], str)
        assert isinstance(metadata["manufacturer"], str)
        assert isinstance(metadata["pixel_spacing"], list)
        assert isinstance(metadata["bits_stored"], int)
        assert isinstance(metadata["projection_type"], str)
        assert isinstance(metadata["laterality"], str)

    def test_validate_projection_inference(self, valid_dicom_dataset):
        """Test inference of projection type from DICOM metadata."""
        # Test direct field access
        view_position = getattr(valid_dicom_dataset, "ViewPosition", None)
        if view_position:
            assert view_position in ["CC", "MLO"]

        # Test laterality inference
        laterality = getattr(valid_dicom_dataset, "ImageLaterality", None)
        if laterality:
            assert laterality in ["L", "R"]

        # Test combined projection inference
        if view_position and laterality:
            projection = f"{view_position}_{laterality}"
            assert projection in ["CC_L", "CC_R", "MLO_L", "MLO_R"]

    def test_validate_dicom_compliance(self, valid_dicom_dataset):
        """Test DICOM compliance validation."""
        # Test SOP Class UID
        sop_class_uid = valid_dicom_dataset.SOPClassUID
        assert sop_class_uid is not None
        assert isinstance(sop_class_uid, str)

        # Test UID format
        assert sop_class_uid.startswith("1.2.840.")
        assert "." in sop_class_uid

        # Test required UIDs are unique
        study_uid = valid_dicom_dataset.StudyInstanceUID
        series_uid = valid_dicom_dataset.SeriesInstanceUID
        instance_uid = valid_dicom_dataset.SOPInstanceUID

        assert study_uid != series_uid
        assert series_uid != instance_uid
        assert study_uid != instance_uid

    def test_validate_pixel_data_integrity(self, valid_dicom_dataset):
        """Test pixel data integrity validation."""
        # Extract pixel array
        pixel_array = valid_dicom_dataset.pixel_array

        # Test for NaN values
        assert not np.any(np.isnan(pixel_array))

        # Test for infinite values
        assert not np.any(np.isinf(pixel_array))

        # Test for negative values (should not occur in mammography)
        assert np.all(pixel_array >= 0)

        # Test for values within expected range
        max_value = 2**valid_dicom_dataset.BitsStored - 1
        assert np.all(pixel_array <= max_value)

    def test_validate_dicom_file_size(self, valid_dicom_dataset):
        """Test DICOM file size validation."""
        # Create temporary DICOM file
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp_file:
            valid_dicom_dataset.save_as(tmp_file.name)
            tmp_file_path = tmp_file.name

        try:
            # Check file size
            file_size = os.path.getsize(tmp_file_path)
            assert file_size > 0

            # Check file size is reasonable (adjust thresholds as needed)
            assert file_size < 100 * 1024 * 1024  # Less than 100MB
            assert file_size > 1024  # More than 1KB

        finally:
            os.unlink(tmp_file_path)

    def test_validate_dicom_encoding(self, valid_dicom_dataset):
        """Test DICOM encoding validation."""
        # Test character set handling
        if hasattr(valid_dicom_dataset, "SpecificCharacterSet"):
            char_set = valid_dicom_dataset.SpecificCharacterSet
            assert isinstance(char_set, (str, list))

        # Test patient name encoding
        if hasattr(valid_dicom_dataset, "PatientName"):
            patient_name = valid_dicom_dataset.PatientName
            assert isinstance(patient_name, str)

        # Test study description encoding
        if hasattr(valid_dicom_dataset, "StudyDescription"):
            study_desc = valid_dicom_dataset.StudyDescription
            assert isinstance(study_desc, str)

    def test_validate_dicom_metadata_consistency(self, valid_dicom_dataset):
        """Test consistency of DICOM metadata."""
        # Test pixel data dimensions match metadata
        rows = valid_dicom_dataset.Rows
        cols = valid_dicom_dataset.Columns
        pixel_array = valid_dicom_dataset.pixel_array

        assert pixel_array.shape == (rows, cols)

        # Test bits stored vs bits allocated
        bits_stored = valid_dicom_dataset.BitsStored
        bits_allocated = valid_dicom_dataset.BitsAllocated

        assert bits_allocated >= bits_stored

        # Test samples per pixel vs photometric interpretation
        samples_per_pixel = valid_dicom_dataset.SamplesPerPixel
        photometric = valid_dicom_dataset.PhotometricInterpretation

        if photometric in ["MONOCHROME1", "MONOCHROME2"]:
            assert samples_per_pixel == 1


if __name__ == "__main__":
    pytest.main([__file__])
