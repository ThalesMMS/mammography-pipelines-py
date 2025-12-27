"""
MammographyImage model for DICOM mammography data representation.

This module defines the core data structure for representing individual
mammography images from DICOM files, including metadata extraction,
validation, and state management.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- This model represents the first stage in our unsupervised learning pipeline
- It captures essential DICOM metadata needed for proper data splitting
- State transitions ensure data integrity throughout the pipeline
- Validation rules enforce medical imaging standards

Author: Research Team
Version: 1.0.0
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pydicom
from pydicom.errors import InvalidDicomError

# Configure logging for educational purposes
logger = logging.getLogger(__name__)


class MammographyImage:
    """
    Represents a single DICOM mammography file with associated metadata.

    This class serves as the foundation of our data model, capturing all
    essential information needed for breast density analysis. It implements
    a state machine to track the image through different processing stages.

    Educational Notes:
    - State transitions: raw → validated → preprocessed → embedded → clustered
    - Each state represents a different stage in our ML pipeline
    - Validation ensures data quality and prevents data leakage
    - Metadata extraction follows DICOM standards for mammography

    Attributes:
        patient_id (str): Unique patient identifier for data splitting
        study_id (str): DICOM Study Instance UID
        series_id (str): DICOM Series Instance UID
        instance_id (str): DICOM SOP Instance UID
        projection_type (str): "CC" or "MLO" projection
        laterality (str): "L" or "R" breast
        manufacturer (str): DICOM Manufacturer tag
        pixel_spacing (tuple[float, float]): Physical pixel spacing in mm
        bits_stored (int): Number of bits per pixel
        file_path (str): Path to DICOM file
        acquisition_date (datetime): When image was acquired
        state (str): Current processing state
        validation_errors (List[str]): List of validation issues
    """

    # Define valid states in our processing pipeline
    VALID_STATES = ["raw", "validated", "preprocessed", "embedded", "clustered"]

    # Define valid projection types for mammography
    VALID_PROJECTION_TYPES = [
        "CC",
        "MLO",
    ]  # CC = Craniocaudal, MLO = Mediolateral Oblique

    # Define valid laterality options
    VALID_LATERALITY = ["L", "R"]  # L = Left, R = Right

    def __init__(
        self,
        patient_id: str,
        study_id: str,
        series_id: str,
        instance_id: str,
        projection_type: str,
        laterality: str,
        manufacturer: str,
        pixel_spacing: tuple[float, float],
        bits_stored: int,
        file_path: str,
        acquisition_date: Optional[datetime] = None,
        state: str = "raw",
    ):
        """
        Initialize a MammographyImage instance.

        Args:
            patient_id: Unique patient identifier (required for data splitting)
            study_id: DICOM Study Instance UID
            series_id: DICOM Series Instance UID
            instance_id: DICOM SOP Instance UID (unique identifier)
            projection_type: "CC" or "MLO" projection type
            laterality: "L" or "R" breast laterality
            manufacturer: DICOM Manufacturer tag
            pixel_spacing: Physical pixel spacing in mm (row, column)
            bits_stored: Number of bits per pixel
            file_path: Path to the DICOM file
            acquisition_date: When the image was acquired (optional)
            state: Current processing state (default: "raw")

        Raises:
            ValueError: If validation rules are violated
            TypeError: If data types are incorrect
        """
        # Initialize core attributes with validation
        self.patient_id = self._validate_patient_id(patient_id)
        self.study_id = self._validate_uid(study_id, "study_id")
        self.series_id = self._validate_uid(series_id, "series_id")
        self.instance_id = self._validate_uid(instance_id, "instance_id")
        self.projection_type = self._validate_projection_type(projection_type)
        self.laterality = self._validate_laterality(laterality)
        self.manufacturer = self._validate_manufacturer(manufacturer)
        self.pixel_spacing = self._validate_pixel_spacing(pixel_spacing)
        self.bits_stored = self._validate_bits_stored(bits_stored)
        self.file_path = self._validate_file_path(file_path)
        self.acquisition_date = acquisition_date or datetime.now()
        self.state = self._validate_state(state)

        # Initialize tracking attributes
        self.validation_errors: List[str] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        # Log creation for educational purposes
        logger.info(
            f"Created MammographyImage: {self.instance_id} for patient {self.patient_id}"
        )

    def _validate_patient_id(self, patient_id: str) -> str:
        """
        Validate patient ID.

        Educational Note: Patient ID is critical for data splitting to prevent
        data leakage between train/validation/test sets.

        Args:
            patient_id: Patient identifier to validate

        Returns:
            str: Validated patient ID

        Raises:
            ValueError: If patient ID is invalid
            TypeError: If patient ID is not a string
        """
        if not isinstance(patient_id, str):
            raise TypeError(f"patient_id must be a string, got {type(patient_id)}")

        if not patient_id.strip():
            raise ValueError("patient_id cannot be empty or whitespace")

        return patient_id.strip()

    def _validate_uid(self, uid: str, field_name: str) -> str:
        """
        Validate DICOM UID.

        Educational Note: DICOM UIDs must follow specific format requirements
        and are used for uniquely identifying studies, series, and instances.

        Args:
            uid: UID to validate
            field_name: Name of the field for error messages

        Returns:
            str: Validated UID

        Raises:
            ValueError: If UID is invalid
            TypeError: If UID is not a string
        """
        if not isinstance(uid, str):
            raise TypeError(f"{field_name} must be a string, got {type(uid)}")

        if not uid.strip():
            raise ValueError(f"{field_name} cannot be empty or whitespace")

        # Basic UID format validation (starts with number, contains dots and numbers)
        if not uid.replace(".", "").replace(" ", "").isdigit():
            raise ValueError(f"{field_name} must be a valid DICOM UID format")

        return uid.strip()

    def _validate_projection_type(self, projection_type: str) -> str:
        """
        Validate projection type.

        Educational Note: Mammography uses specific projection types:
        - CC (Craniocaudal): Top-down view
        - MLO (Mediolateral Oblique): Side view at an angle

        Args:
            projection_type: Projection type to validate

        Returns:
            str: Validated projection type

        Raises:
            ValueError: If projection type is invalid
            TypeError: If projection type is not a string
        """
        if not isinstance(projection_type, str):
            raise TypeError(
                f"projection_type must be a string, got {type(projection_type)}"
            )

        if projection_type not in self.VALID_PROJECTION_TYPES:
            raise ValueError(
                f"projection_type must be one of {self.VALID_PROJECTION_TYPES}, got {projection_type}"
            )

        return projection_type

    def _validate_laterality(self, laterality: str) -> str:
        """
        Validate laterality.

        Educational Note: Laterality indicates which breast the image shows:
        - L: Left breast
        - R: Right breast

        Args:
            laterality: Laterality to validate

        Returns:
            str: Validated laterality

        Raises:
            ValueError: If laterality is invalid
            TypeError: If laterality is not a string
        """
        if not isinstance(laterality, str):
            raise TypeError(f"laterality must be a string, got {type(laterality)}")

        if laterality not in self.VALID_LATERALITY:
            raise ValueError(
                f"laterality must be one of {self.VALID_LATERALITY}, got {laterality}"
            )

        return laterality

    def _validate_manufacturer(self, manufacturer: str) -> str:
        """
        Validate manufacturer.

        Educational Note: Manufacturer information is important for
        understanding potential differences in image acquisition protocols.

        Args:
            manufacturer: Manufacturer to validate

        Returns:
            str: Validated manufacturer

        Raises:
            ValueError: If manufacturer is invalid
            TypeError: If manufacturer is not a string
        """
        if not isinstance(manufacturer, str):
            raise TypeError(f"manufacturer must be a string, got {type(manufacturer)}")

        if not manufacturer.strip():
            raise ValueError("manufacturer cannot be empty or whitespace")

        return manufacturer.strip()

    def _validate_pixel_spacing(
        self, pixel_spacing: tuple[float, float]
    ) -> tuple[float, float]:
        """
        Validate pixel spacing.

        Educational Note: Pixel spacing is crucial for understanding the
        physical dimensions of the image and ensuring proper preprocessing.

        Args:
            pixel_spacing: Pixel spacing to validate (row, column)

        Returns:
            tuple[float, float]: Validated pixel spacing

        Raises:
            ValueError: If pixel spacing is invalid
            TypeError: If pixel spacing is not a tuple
        """
        if not isinstance(pixel_spacing, tuple):
            raise TypeError(f"pixel_spacing must be a tuple, got {type(pixel_spacing)}")

        if len(pixel_spacing) != 2:
            raise ValueError(
                f"pixel_spacing must have exactly 2 elements, got {len(pixel_spacing)}"
            )

        for i, spacing in enumerate(pixel_spacing):
            if not isinstance(spacing, (int, float)):
                raise TypeError(
                    f"pixel_spacing[{i}] must be a number, got {type(spacing)}"
                )

            if spacing <= 0:
                raise ValueError(f"pixel_spacing[{i}] must be positive, got {spacing}")

        return tuple(float(spacing) for spacing in pixel_spacing)

    def _validate_bits_stored(self, bits_stored: int) -> int:
        """
        Validate bits stored.

        Educational Note: Bits stored indicates the precision of pixel values,
        typically 8, 12, or 16 bits for mammography images.

        Args:
            bits_stored: Number of bits to validate

        Returns:
            int: Validated bits stored

        Raises:
            ValueError: If bits stored is invalid
            TypeError: If bits stored is not an integer
        """
        if not isinstance(bits_stored, int):
            raise TypeError(f"bits_stored must be an integer, got {type(bits_stored)}")

        if bits_stored <= 0:
            raise ValueError(f"bits_stored must be positive, got {bits_stored}")

        if bits_stored > 32:
            raise ValueError(f"bits_stored must be <= 32, got {bits_stored}")

        return bits_stored

    def _validate_file_path(self, file_path: str) -> str:
        """
        Validate file path.

        Educational Note: File path validation ensures the DICOM file
        exists and is accessible for processing.

        Args:
            file_path: File path to validate

        Returns:
            str: Validated file path

        Raises:
            ValueError: If file path is invalid
            TypeError: If file path is not a string
        """
        if not isinstance(file_path, str):
            raise TypeError(f"file_path must be a string, got {type(file_path)}")

        if not file_path.strip():
            raise ValueError("file_path cannot be empty or whitespace")

        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"DICOM file does not exist: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        return str(path.absolute())

    def _validate_state(self, state: str) -> str:
        """
        Validate processing state.

        Educational Note: State management ensures proper pipeline flow
        and prevents invalid state transitions.

        Args:
            state: State to validate

        Returns:
            str: Validated state

        Raises:
            ValueError: If state is invalid
            TypeError: If state is not a string
        """
        if not isinstance(state, str):
            raise TypeError(f"state must be a string, got {type(state)}")

        if state not in self.VALID_STATES:
            raise ValueError(f"state must be one of {self.VALID_STATES}, got {state}")

        return state

    def transition_to(self, new_state: str) -> bool:
        """
        Transition to a new processing state.

        Educational Note: State transitions ensure data integrity and
        proper pipeline flow. Each state represents a processing stage.

        Args:
            new_state: Target state for transition

        Returns:
            bool: True if transition successful, False otherwise

        Raises:
            ValueError: If new state is invalid
        """
        if new_state not in self.VALID_STATES:
            raise ValueError(
                f"Invalid state: {new_state}. Must be one of {self.VALID_STATES}"
            )

        # Define valid state transitions
        valid_transitions = {
            "raw": ["validated"],
            "validated": ["preprocessed"],
            "preprocessed": ["embedded"],
            "embedded": ["clustered"],
            "clustered": [],  # Terminal state
        }

        if new_state not in valid_transitions.get(self.state, []):
            error_msg = f"Invalid transition from {self.state} to {new_state}"
            logger.warning(error_msg)
            self.validation_errors.append(error_msg)
            return False

        old_state = self.state
        self.state = new_state
        self.updated_at = datetime.now()

        logger.info(f"State transition: {self.instance_id} {old_state} → {new_state}")
        return True

    def validate_dicom_file(self) -> bool:
        """
        Validate the DICOM file and extract additional metadata.

        Educational Note: This method performs comprehensive DICOM validation
        to ensure the file meets mammography standards and our requirements.

        Returns:
            bool: True if validation successful, False otherwise
        """
        try:
            # Read DICOM file
            dataset = pydicom.dcmread(self.file_path)

            # Validate required DICOM tags
            required_tags = [
                ("PatientID", self.patient_id),
                ("StudyInstanceUID", self.study_id),
                ("SeriesInstanceUID", self.series_id),
                ("SOPInstanceUID", self.instance_id),
                ("Manufacturer", self.manufacturer),
                ("PixelSpacing", self.pixel_spacing),
                ("BitsStored", self.bits_stored),
            ]

            for tag_name, expected_value in required_tags:
                if not hasattr(dataset, tag_name):
                    error_msg = f"Missing required DICOM tag: {tag_name}"
                    self.validation_errors.append(error_msg)
                    continue

                actual_value = getattr(dataset, tag_name)
                if actual_value != expected_value:
                    error_msg = f"DICOM tag {tag_name} mismatch: expected {expected_value}, got {actual_value}"
                    self.validation_errors.append(error_msg)

            # Validate mammography-specific tags
            if hasattr(dataset, "ViewPosition"):
                if dataset.ViewPosition != self.projection_type:
                    error_msg = f"ViewPosition mismatch: expected {self.projection_type}, got {dataset.ViewPosition}"
                    self.validation_errors.append(error_msg)

            if hasattr(dataset, "ImageLaterality"):
                if dataset.ImageLaterality != self.laterality:
                    error_msg = f"ImageLaterality mismatch: expected {self.laterality}, got {dataset.ImageLaterality}"
                    self.validation_errors.append(error_msg)

            # Validate pixel data
            if not hasattr(dataset, "PixelData"):
                self.validation_errors.append("Missing PixelData in DICOM file")
            else:
                # Check if we can extract pixel array
                try:
                    pixel_array = dataset.pixel_array
                    if pixel_array is None:
                        self.validation_errors.append("PixelData is empty")
                    elif pixel_array.size == 0:
                        self.validation_errors.append("PixelData has zero size")
                except Exception as e:
                    self.validation_errors.append(f"Error reading PixelData: {e!s}")

            # Transition to validated state if no errors
            if not self.validation_errors:
                self.transition_to("validated")
                logger.info(f"DICOM validation successful: {self.instance_id}")
                return True
            else:
                logger.warning(
                    f"DICOM validation failed: {self.instance_id}, errors: {self.validation_errors}"
                )
                return False

        except InvalidDicomError as e:
            error_msg = f"Invalid DICOM file: {e!s}"
            self.validation_errors.append(error_msg)
            logger.error(error_msg)
            return False
        except Exception as e:
            error_msg = f"Error validating DICOM file: {e!s}"
            self.validation_errors.append(error_msg)
            logger.error(error_msg)
            return False

    def get_metadata_dict(self) -> Dict[str, Any]:
        """
        Get metadata as a dictionary.

        Educational Note: This method provides a standardized way to access
        all metadata for logging, serialization, and analysis.

        Returns:
            Dict[str, Any]: Dictionary containing all metadata
        """
        return {
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "series_id": self.series_id,
            "instance_id": self.instance_id,
            "projection_type": self.projection_type,
            "laterality": self.laterality,
            "manufacturer": self.manufacturer,
            "pixel_spacing": self.pixel_spacing,
            "bits_stored": self.bits_stored,
            "file_path": self.file_path,
            "acquisition_date": (
                self.acquisition_date.isoformat() if self.acquisition_date else None
            ),
            "state": self.state,
            "validation_errors": self.validation_errors,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def __repr__(self) -> str:
        """String representation for debugging and logging."""
        return (
            f"MammographyImage("
            f"patient_id='{self.patient_id}', "
            f"instance_id='{self.instance_id}', "
            f"projection_type='{self.projection_type}', "
            f"laterality='{self.laterality}', "
            f"state='{self.state}')"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Mammography Image: {self.instance_id}\n"
            f"Patient: {self.patient_id}\n"
            f"Projection: {self.projection_type}\n"
            f"Laterality: {self.laterality}\n"
            f"State: {self.state}\n"
            f"File: {self.file_path}"
        )


def create_mammography_image_from_dicom(file_path: str) -> MammographyImage:
    """
    Create a MammographyImage instance from a DICOM file.

    Educational Note: This factory function demonstrates how to extract
    metadata from DICOM files and create our data model instances.

    Args:
        file_path: Path to the DICOM file

    Returns:
        MammographyImage: Created instance

    Raises:
        InvalidDicomError: If DICOM file is invalid
        ValueError: If required metadata is missing
    """
    try:
        # Read DICOM file
        dataset = pydicom.dcmread(file_path)

        # Extract required metadata
        patient_id = getattr(dataset, "PatientID", "")
        study_id = getattr(dataset, "StudyInstanceUID", "")
        series_id = getattr(dataset, "SeriesInstanceUID", "")
        instance_id = getattr(dataset, "SOPInstanceUID", "")
        projection_type = getattr(dataset, "ViewPosition", "")
        laterality = getattr(dataset, "ImageLaterality", "")
        manufacturer = getattr(dataset, "Manufacturer", "")
        pixel_spacing = getattr(dataset, "PixelSpacing", (0.0, 0.0))
        bits_stored = getattr(dataset, "BitsStored", 0)
        acquisition_date = getattr(dataset, "AcquisitionDate", None)

        # Parse acquisition date if available
        if acquisition_date:
            try:
                acquisition_date = datetime.strptime(str(acquisition_date), "%Y%m%d")
            except ValueError:
                acquisition_date = None

        # Create MammographyImage instance
        mammography_image = MammographyImage(
            patient_id=patient_id,
            study_id=study_id,
            series_id=series_id,
            instance_id=instance_id,
            projection_type=projection_type,
            laterality=laterality,
            manufacturer=manufacturer,
            pixel_spacing=pixel_spacing,
            bits_stored=bits_stored,
            file_path=file_path,
            acquisition_date=acquisition_date,
        )

        # Validate the DICOM file
        mammography_image.validate_dicom_file()

        return mammography_image

    except InvalidDicomError as e:
        raise InvalidDicomError(f"Invalid DICOM file {file_path}: {e!s}")
    except Exception as e:
        raise ValueError(f"Error creating MammographyImage from {file_path}: {e!s}")
