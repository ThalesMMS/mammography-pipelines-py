"""
DICOM I/O module for reading and processing mammography DICOM files.

This module provides comprehensive DICOM file reading capabilities with
metadata extraction, validation, and patient-level organization for
the breast density exploration pipeline.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- DICOM (Digital Imaging and Communications in Medicine) is the standard
  for medical imaging data storage and transmission
- This module handles the first step of our pipeline: raw data ingestion
- Patient-level organization ensures proper data splitting for ML
- Metadata validation ensures data quality and consistency

Author: Research Team
Version: 1.0.0
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pydicom

from .mammography_image import MammographyImage, create_mammography_image_from_dicom

# Configure logging for educational purposes
logger = logging.getLogger(__name__)


class DicomReader:
    """
    DICOM file reader for mammography images with comprehensive validation.

    This class provides methods for reading DICOM files, extracting metadata,
    validating mammography-specific requirements, and organizing data by
    patient for proper machine learning pipeline processing.

    Educational Notes:
    - DICOM files contain both pixel data and rich metadata
    - Mammography-specific tags ensure proper image classification
    - Patient-level organization prevents data leakage in ML
    - Validation ensures data quality and consistency

    Attributes:
        supported_extensions: List of supported DICOM file extensions
        validation_rules: Dictionary of validation rules for mammography
        metadata_cache: Cache for frequently accessed metadata
    """

    # Supported DICOM file extensions
    SUPPORTED_EXTENSIONS = [".dcm", ".dicom", ".DCM", ".DICOM"]

    # Required DICOM tags for mammography
    REQUIRED_TAGS = [
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "Manufacturer",
        "PixelSpacing",
        "BitsStored",
        "ViewPosition",  # CC or MLO
        "ImageLaterality",  # L or R
        "PixelData",
    ]

    # Optional but recommended tags
    OPTIONAL_TAGS = [
        "AcquisitionDate",
        "AcquisitionTime",
        "StudyDate",
        "StudyTime",
        "PatientAge",
        "PatientSex",
        "BodyPartExamined",
        "ImageType",
        "PhotometricInterpretation",
    ]

    def __init__(
        self,
        validate_on_read: bool = True,
        cache_metadata: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize DICOM reader with configuration options.

        Args:
            validate_on_read: Whether to validate DICOM files on reading
            cache_metadata: Whether to cache metadata for performance
            max_workers: Maximum number of worker threads for parallel processing

        Educational Note: These parameters control the balance between
        data validation, performance, and resource usage.
        """
        self.validate_on_read = validate_on_read
        self.cache_metadata = cache_metadata
        self.max_workers = max_workers

        # Initialize metadata cache
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}

        # Validation statistics
        self.stats = {
            "files_processed": 0,
            "files_valid": 0,
            "files_invalid": 0,
            "validation_errors": [],
        }

        logger.info(
            f"Initialized DicomReader with validate_on_read={validate_on_read}, max_workers={max_workers}"
        )

    def read_dicom_file(
        self, file_path: Union[str, Path]
    ) -> Optional[MammographyImage]:
        """
        Read a single DICOM file and create MammographyImage instance.

        Educational Note: This method demonstrates the complete DICOM
        reading process including file validation, metadata extraction,
        and data model creation.

        Args:
            file_path: Path to the DICOM file

        Returns:
            MammographyImage: Created instance if successful, None otherwise

        Raises:
            FileNotFoundError: If DICOM file doesn't exist
            InvalidDicomError: If DICOM file is corrupted
        """
        file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            error_msg = f"DICOM file not found: {file_path}"
            logger.error(error_msg)
            self.stats["validation_errors"].append(error_msg)
            raise FileNotFoundError(error_msg)

        # Check file extension
        if file_path.suffix.lower() not in [
            ext.lower() for ext in self.SUPPORTED_EXTENSIONS
        ]:
            error_msg = f"Unsupported file extension: {file_path.suffix}"
            logger.warning(error_msg)
            self.stats["validation_errors"].append(error_msg)
            return None

        try:
            # Read DICOM file
            logger.debug(f"Reading DICOM file: {file_path}")
            dataset = pydicom.dcmread(file_path)

            # Validate DICOM file if requested
            if self.validate_on_read:
                validation_result = self._validate_dicom_dataset(dataset, file_path)
                if not validation_result["valid"]:
                    logger.warning(
                        f"DICOM validation failed for {file_path}: {validation_result['errors']}"
                    )
                    self.stats["files_invalid"] += 1
                    self.stats["validation_errors"].extend(validation_result["errors"])
                    return None

            # Create MammographyImage instance
            mammography_image = create_mammography_image_from_dicom(str(file_path))

            # Cache metadata if enabled
            if self.cache_metadata:
                self.metadata_cache[str(file_path)] = (
                    mammography_image.get_metadata_dict()
                )

            # Update statistics
            self.stats["files_processed"] += 1
            self.stats["files_valid"] += 1

            logger.info(f"Successfully read DICOM file: {file_path}")
            return mammography_image

        except Exception as e:
            error_msg = f"Error reading DICOM file {file_path}: {e!s}"
            logger.error(error_msg)
            self.stats["files_invalid"] += 1
            self.stats["validation_errors"].append(error_msg)
            return None

    def read_dicom_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        patient_level: bool = True,
    ) -> Dict[str, List[MammographyImage]]:
        """
        Read all DICOM files from a directory with optional patient-level organization.

        Educational Note: This method demonstrates batch processing of DICOM
        files with patient-level organization to prevent data leakage in ML.

        Args:
            directory_path: Path to directory containing DICOM files
            recursive: Whether to search subdirectories recursively
            patient_level: Whether to organize results by patient ID

        Returns:
            Dict[str, List[MammographyImage]]: Dictionary mapping patient IDs to images
            or file paths to images if patient_level=False

        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        # Find all DICOM files
        dicom_files = self._find_dicom_files(directory_path, recursive)
        logger.info(f"Found {len(dicom_files)} DICOM files in {directory_path}")

        if not dicom_files:
            logger.warning(f"No DICOM files found in {directory_path}")
            return {}

        # Read DICOM files in parallel
        mammography_images = self._read_dicom_files_parallel(dicom_files)

        # Filter out None results (failed reads)
        valid_images = [img for img in mammography_images if img is not None]
        logger.info(
            f"Successfully read {len(valid_images)} out of {len(dicom_files)} DICOM files"
        )

        # Organize by patient ID if requested
        if patient_level:
            return self._organize_by_patient(valid_images)
        else:
            # Organize by file path
            return {img.file_path: [img] for img in valid_images}

    def _find_dicom_files(
        self, directory_path: Path, recursive: bool = True
    ) -> List[Path]:
        """
        Find all DICOM files in a directory.

        Educational Note: This method demonstrates file discovery with
        proper extension filtering and recursive search capabilities.

        Args:
            directory_path: Directory to search
            recursive: Whether to search subdirectories

        Returns:
            List[Path]: List of DICOM file paths
        """
        dicom_files = []

        if recursive:
            # Search recursively
            for ext in self.SUPPORTED_EXTENSIONS:
                pattern = f"**/*{ext}"
                dicom_files.extend(directory_path.glob(pattern))
        else:
            # Search only in current directory
            for ext in self.SUPPORTED_EXTENSIONS:
                pattern = f"*{ext}"
                dicom_files.extend(directory_path.glob(pattern))

        # Remove duplicates and sort
        dicom_files = sorted(set(dicom_files))

        logger.debug(f"Found {len(dicom_files)} DICOM files")
        return dicom_files

    def _read_dicom_files_parallel(
        self, dicom_files: List[Path]
    ) -> List[Optional[MammographyImage]]:
        """
        Read multiple DICOM files in parallel for improved performance.

        Educational Note: Parallel processing significantly improves
        performance when reading large numbers of DICOM files.

        Args:
            dicom_files: List of DICOM file paths to read

        Returns:
            List[Optional[MammographyImage]]: List of created instances
        """
        mammography_images = [None] * len(dicom_files)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.read_dicom_file, file_path): i
                for i, file_path in enumerate(dicom_files)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    mammography_images[index] = future.result()
                except Exception as e:
                    logger.error(
                        f"Error processing DICOM file {dicom_files[index]}: {e!s}"
                    )
                    mammography_images[index] = None

        return mammography_images

    def _organize_by_patient(
        self, mammography_images: List[MammographyImage]
    ) -> Dict[str, List[MammographyImage]]:
        """
        Organize mammography images by patient ID.

        Educational Note: Patient-level organization is crucial for
        preventing data leakage in machine learning pipelines.

        Args:
            mammography_images: List of MammographyImage instances

        Returns:
            Dict[str, List[MammographyImage]]: Dictionary mapping patient IDs to images
        """
        patient_organization = {}

        for image in mammography_images:
            patient_id = image.patient_id

            if patient_id not in patient_organization:
                patient_organization[patient_id] = []

            patient_organization[patient_id].append(image)

        # Sort images within each patient by acquisition date
        for patient_id in patient_organization:
            patient_organization[patient_id].sort(
                key=lambda x: x.acquisition_date or datetime.min
            )

        logger.info(
            f"Organized {len(mammography_images)} images into {len(patient_organization)} patients"
        )
        return patient_organization

    def _validate_dicom_dataset(
        self, dataset: pydicom.Dataset, file_path: Path
    ) -> Dict[str, Any]:
        """
        Validate DICOM dataset for mammography requirements.

        Educational Note: This validation ensures DICOM files meet
        mammography-specific requirements and contain necessary metadata.

        Args:
            dataset: DICOM dataset to validate
            file_path: Path to the DICOM file (for error reporting)

        Returns:
            Dict[str, Any]: Validation result with 'valid' flag and 'errors' list
        """
        validation_result = {"valid": True, "errors": [], "warnings": []}

        # Check required tags
        for tag in self.REQUIRED_TAGS:
            if not hasattr(dataset, tag):
                error_msg = f"Missing required DICOM tag '{tag}' in {file_path}"
                validation_result["errors"].append(error_msg)
                validation_result["valid"] = False

        # Validate mammography-specific requirements
        if hasattr(dataset, "ViewPosition"):
            if dataset.ViewPosition not in ["CC", "MLO"]:
                error_msg = f"Invalid ViewPosition '{dataset.ViewPosition}' in {file_path}. Must be 'CC' or 'MLO'"
                validation_result["errors"].append(error_msg)
                validation_result["valid"] = False

        if hasattr(dataset, "ImageLaterality"):
            if dataset.ImageLaterality not in ["L", "R"]:
                error_msg = f"Invalid ImageLaterality '{dataset.ImageLaterality}' in {file_path}. Must be 'L' or 'R'"
                validation_result["errors"].append(error_msg)
                validation_result["valid"] = False

        # Validate pixel data
        if hasattr(dataset, "PixelData"):
            try:
                pixel_array = dataset.pixel_array
                if pixel_array is None or pixel_array.size == 0:
                    error_msg = f"Empty or invalid PixelData in {file_path}"
                    validation_result["errors"].append(error_msg)
                    validation_result["valid"] = False
            except Exception as e:
                error_msg = f"Error reading PixelData from {file_path}: {e!s}"
                validation_result["errors"].append(error_msg)
                validation_result["valid"] = False

        # Validate pixel spacing
        if hasattr(dataset, "PixelSpacing"):
            pixel_spacing = dataset.PixelSpacing
            if not isinstance(pixel_spacing, (list, tuple)) or len(pixel_spacing) != 2:
                error_msg = f"Invalid PixelSpacing format in {file_path}"
                validation_result["errors"].append(error_msg)
                validation_result["valid"] = False
            elif any(spacing <= 0 for spacing in pixel_spacing):
                error_msg = (
                    f"Invalid PixelSpacing values in {file_path}: {pixel_spacing}"
                )
                validation_result["errors"].append(error_msg)
                validation_result["valid"] = False

        # Check for optional but recommended tags
        for tag in self.OPTIONAL_TAGS:
            if not hasattr(dataset, tag):
                warning_msg = f"Missing optional DICOM tag '{tag}' in {file_path}"
                validation_result["warnings"].append(warning_msg)

        return validation_result

    def get_metadata_summary(self) -> Dict[str, Any]:
        """
        Get summary of metadata from all processed DICOM files.

        Educational Note: This summary provides insights into the
        characteristics of the DICOM dataset for analysis and validation.

        Returns:
            Dict[str, Any]: Summary statistics and metadata
        """
        if not self.metadata_cache:
            return {"total_files": 0, "message": "No metadata available"}

        # Collect statistics
        manufacturers = set()
        projection_types = set()
        lateralities = set()
        pixel_spacings = []
        bits_stored_values = set()

        for metadata in self.metadata_cache.values():
            manufacturers.add(metadata.get("manufacturer", "Unknown"))
            projection_types.add(metadata.get("projection_type", "Unknown"))
            lateralities.add(metadata.get("laterality", "Unknown"))
            pixel_spacings.append(metadata.get("pixel_spacing", (0, 0)))
            bits_stored_values.add(metadata.get("bits_stored", 0))

        return {
            "total_files": len(self.metadata_cache),
            "manufacturers": list(manufacturers),
            "projection_types": list(projection_types),
            "lateralities": list(lateralities),
            "pixel_spacing_range": {
                "min": tuple(min(ps[i] for ps in pixel_spacings) for i in range(2)),
                "max": tuple(max(ps[i] for ps in pixel_spacings) for i in range(2)),
            },
            "bits_stored_values": list(bits_stored_values),
            "processing_stats": self.stats,
        }

    def clear_cache(self) -> None:
        """Clear the metadata cache to free memory."""
        self.metadata_cache.clear()
        logger.info("Cleared DICOM metadata cache")

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Educational Note: These statistics help monitor the DICOM
        reading process and identify potential issues.

        Returns:
            Dict[str, Any]: Processing statistics
        """
        return {
            "files_processed": self.stats["files_processed"],
            "files_valid": self.stats["files_valid"],
            "files_invalid": self.stats["files_invalid"],
            "success_rate": (
                self.stats["files_valid"] / max(self.stats["files_processed"], 1)
            )
            * 100,
            "total_validation_errors": len(self.stats["validation_errors"]),
            "recent_errors": self.stats["validation_errors"][-10:],  # Last 10 errors
        }


def create_dicom_reader(
    validate_on_read: bool = True, cache_metadata: bool = True, max_workers: int = 4
) -> DicomReader:
    """
    Factory function to create a DicomReader instance.

    Educational Note: This factory function provides a convenient way
    to create DicomReader instances with default configurations.

    Args:
        validate_on_read: Whether to validate DICOM files on reading
        cache_metadata: Whether to cache metadata for performance
        max_workers: Maximum number of worker threads

    Returns:
        DicomReader: Configured DicomReader instance
    """
    return DicomReader(
        validate_on_read=validate_on_read,
        cache_metadata=cache_metadata,
        max_workers=max_workers,
    )


def read_single_dicom(file_path: Union[str, Path]) -> Optional[MammographyImage]:
    """
    Convenience function to read a single DICOM file.

    Educational Note: This function provides a simple interface for
    reading individual DICOM files without creating a DicomReader instance.

    Args:
        file_path: Path to the DICOM file

    Returns:
        MammographyImage: Created instance if successful, None otherwise
    """
    reader = create_dicom_reader()
    return reader.read_dicom_file(file_path)


def read_dicom_directory(
    directory_path: Union[str, Path], recursive: bool = True, patient_level: bool = True
) -> Dict[str, List[MammographyImage]]:
    """
    Convenience function to read all DICOM files from a directory.

    Educational Note: This function provides a simple interface for
    batch DICOM reading without creating a DicomReader instance.

    Args:
        directory_path: Path to directory containing DICOM files
        recursive: Whether to search subdirectories recursively
        patient_level: Whether to organize results by patient ID

    Returns:
        Dict[str, List[MammographyImage]]: Organized mammography images
    """
    reader = create_dicom_reader()
    return reader.read_dicom_directory(directory_path, recursive, patient_level)
