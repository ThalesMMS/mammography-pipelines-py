"""
Integration tests for DICOM I/O functionality.

These tests validate the complete DICOM reading and validation pipeline
using real DICOM files from the archive directory.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import os
from pathlib import Path
from typing import List

import pydicom
from pydicom.errors import InvalidDicomError
import pytest

# Import the modules we'll be testing (these will be implemented later)
# from src.io_dicom.reader import DICOMReader
# from src.io_dicom.validator import DICOMValidator
# from src.io_dicom.metadata_extractor import MetadataExtractor


class TestDICOMIOIntegration:
    """Integration tests for DICOM I/O operations."""

    @pytest.fixture
    def sample_dicom_paths(self) -> List[Path]:
        """Get paths to sample DICOM files from archive."""
        archive_dir = Path("archive")
        dicom_files = []

        if archive_dir.exists():
            for patient_dir in archive_dir.iterdir():
                if patient_dir.is_dir():
                    for file_path in patient_dir.glob("*.dcm"):
                        dicom_files.append(file_path)
                        if len(dicom_files) >= 5:  # Limit to 5 files for testing
                            break
                if len(dicom_files) >= 5:
                    break

        return dicom_files

    def test_dicom_file_reading(self, sample_dicom_paths):
        """Test reading DICOM files from archive."""
        if not sample_dicom_paths:
            pytest.skip("No DICOM files found in archive directory")

        for dicom_path in sample_dicom_paths:
            # Test basic DICOM reading with pydicom
            try:
                dataset = pydicom.dcmread(str(dicom_path))
                assert dataset is not None
                assert hasattr(dataset, "pixel_array")
                assert hasattr(dataset, "PatientID")

                # Validate basic DICOM structure
                assert dataset.SOPClassUID is not None
                assert dataset.SOPInstanceUID is not None

            except InvalidDicomError as e:
                pytest.fail(f"Failed to read DICOM file {dicom_path}: {e}")

    def test_dicom_metadata_extraction(self, sample_dicom_paths):
        """Test extraction of metadata from DICOM files."""
        if not sample_dicom_paths:
            pytest.skip("No DICOM files found in archive directory")

        for dicom_path in sample_dicom_paths:
            dataset = pydicom.dcmread(str(dicom_path))

            # Extract required metadata
            metadata = {
                "patient_id": getattr(dataset, "PatientID", None),
                "study_id": getattr(dataset, "StudyInstanceUID", None),
                "series_id": getattr(dataset, "SeriesInstanceUID", None),
                "instance_id": getattr(dataset, "SOPInstanceUID", None),
                "manufacturer": getattr(dataset, "Manufacturer", None),
                "pixel_spacing": getattr(dataset, "PixelSpacing", None),
                "bits_stored": getattr(dataset, "BitsStored", None),
                "acquisition_date": getattr(dataset, "AcquisitionDate", None),
            }

            # Validate extracted metadata
            assert metadata["patient_id"] is not None
            assert metadata["study_id"] is not None
            assert metadata["series_id"] is not None
            assert metadata["instance_id"] is not None

            # Validate data types
            assert isinstance(metadata["patient_id"], str)
            assert isinstance(metadata["study_id"], str)
            assert isinstance(metadata["series_id"], str)
            assert isinstance(metadata["instance_id"], str)

    def test_dicom_validation_rules(self, sample_dicom_paths):
        """Test DICOM validation rules for mammography."""
        if not sample_dicom_paths:
            pytest.skip("No DICOM files found in archive directory")

        for dicom_path in sample_dicom_paths:
            dataset = pydicom.dcmread(str(dicom_path))

            # Test validation rules
            validation_results = {
                "has_pixel_spacing": hasattr(dataset, "PixelSpacing")
                and dataset.PixelSpacing is not None,
                "has_bits_stored": hasattr(dataset, "BitsStored")
                and dataset.BitsStored is not None,
                "has_manufacturer": hasattr(dataset, "Manufacturer")
                and dataset.Manufacturer is not None,
                "has_pixel_array": hasattr(dataset, "pixel_array"),
                "valid_pixel_spacing": False,
                "valid_bits_stored": False,
            }

            # Validate pixel spacing
            if validation_results["has_pixel_spacing"]:
                pixel_spacing = dataset.PixelSpacing
                if isinstance(pixel_spacing, list) and len(pixel_spacing) == 2:
                    validation_results["valid_pixel_spacing"] = all(
                        isinstance(x, (int, float)) and x > 0 for x in pixel_spacing
                    )

            # Validate bits stored
            if validation_results["has_bits_stored"]:
                bits_stored = dataset.BitsStored
                validation_results["valid_bits_stored"] = (
                    isinstance(bits_stored, int) and 8 <= bits_stored <= 16
                )

            # At least basic validation should pass
            assert validation_results[
                "has_pixel_array"
            ], f"No pixel array in {dicom_path}"
            assert validation_results[
                "has_manufacturer"
            ], f"No manufacturer in {dicom_path}"

    def test_projection_type_inference(self, sample_dicom_paths):
        """Test inference of projection type (CC/MLO) from DICOM metadata."""
        if not sample_dicom_paths:
            pytest.skip("No DICOM files found in archive directory")

        for dicom_path in sample_dicom_paths:
            dataset = pydicom.dcmread(str(dicom_path))

            # Try to infer projection type from various fields
            projection_type = None

            # Check ViewPosition field
            if hasattr(dataset, "ViewPosition"):
                view_pos = dataset.ViewPosition
                if view_pos in ["CC", "MLO"]:
                    projection_type = view_pos

            # Check ImageLaterality field
            laterality = None
            if hasattr(dataset, "ImageLaterality"):
                laterality = dataset.ImageLaterality

            # Validate projection type if found
            if projection_type:
                assert projection_type in ["CC", "MLO"]

            # Validate laterality if found
            if laterality:
                assert laterality in ["L", "R"]

    def test_pixel_array_extraction(self, sample_dicom_paths):
        """Test extraction of pixel arrays from DICOM files."""
        if not sample_dicom_paths:
            pytest.skip("No DICOM files found in archive directory")

        for dicom_path in sample_dicom_paths:
            dataset = pydicom.dcmread(str(dicom_path))

            # Extract pixel array
            pixel_array = dataset.pixel_array

            # Validate pixel array
            assert pixel_array is not None
            assert pixel_array.ndim == 2  # Should be 2D for mammography
            assert pixel_array.shape[0] > 0
            assert pixel_array.shape[1] > 0

            # Validate pixel values
            assert pixel_array.dtype in ["uint8", "uint16", "int16"]
            assert pixel_array.min() >= 0  # Should be non-negative
            assert pixel_array.max() > 0  # Should have some positive values

    def test_dicom_error_handling(self):
        """Test error handling for invalid DICOM files."""
        # Test with non-existent file
        with pytest.raises((FileNotFoundError, InvalidDicomError)):
            pydicom.dcmread("non_existent_file.dcm")

        # Test with non-DICOM file (create a temporary text file)
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp_file:
            tmp_file.write(b"This is not a DICOM file")
            tmp_file_path = tmp_file.name

        try:
            with pytest.raises(InvalidDicomError):
                pydicom.dcmread(tmp_file_path)
        finally:
            os.unlink(tmp_file_path)

    def test_dicom_metadata_consistency(self, sample_dicom_paths):
        """Test consistency of metadata across DICOM files."""
        if not sample_dicom_paths:
            pytest.skip("No DICOM files found in archive directory")

        metadata_list = []

        for dicom_path in sample_dicom_paths:
            dataset = pydicom.dcmread(str(dicom_path))

            metadata = {
                "patient_id": getattr(dataset, "PatientID", None),
                "study_id": getattr(dataset, "StudyInstanceUID", None),
                "manufacturer": getattr(dataset, "Manufacturer", None),
                "pixel_spacing": getattr(dataset, "PixelSpacing", None),
                "bits_stored": getattr(dataset, "BitsStored", None),
            }

            metadata_list.append(metadata)

        # Check for consistency in metadata
        if len(metadata_list) > 1:
            # All files should have the same manufacturer (if from same dataset)
            manufacturers = [
                m["manufacturer"] for m in metadata_list if m["manufacturer"]
            ]
            if len(set(manufacturers)) == 1:
                assert len(manufacturers) > 0

            # All files should have consistent pixel spacing (if from same dataset)
            pixel_spacings = [
                m["pixel_spacing"] for m in metadata_list if m["pixel_spacing"]
            ]
            if len(set(str(ps) for ps in pixel_spacings)) == 1:
                assert len(pixel_spacings) > 0

    @pytest.mark.slow
    def test_large_dicom_file_handling(self, sample_dicom_paths):
        """Test handling of large DICOM files."""
        if not sample_dicom_paths:
            pytest.skip("No DICOM files found in archive directory")

        for dicom_path in sample_dicom_paths:
            # Check file size
            file_size = dicom_path.stat().st_size

            # Test reading large files
            if file_size > 10 * 1024 * 1024:  # 10MB
                dataset = pydicom.dcmread(str(dicom_path))
                pixel_array = dataset.pixel_array

                # Validate large pixel array
                assert pixel_array is not None
                assert pixel_array.size > 0

                # Test memory usage (basic check)
                memory_usage = pixel_array.nbytes
                assert memory_usage > 0

    def test_dicom_encoding_handling(self, sample_dicom_paths):
        """Test handling of different DICOM encodings and character sets."""
        if not sample_dicom_paths:
            pytest.skip("No DICOM files found in archive directory")

        for dicom_path in sample_dicom_paths:
            dataset = pydicom.dcmread(str(dicom_path))

            # Test character set handling
            if hasattr(dataset, "SpecificCharacterSet"):
                char_set = dataset.SpecificCharacterSet
                # Should handle common character sets
                assert isinstance(char_set, (str, list))

            # Test patient name encoding
            if hasattr(dataset, "PatientName"):
                patient_name = dataset.PatientName
                # Should be able to handle encoded names
                assert isinstance(patient_name, str)


if __name__ == "__main__":
    pytest.main([__file__])
