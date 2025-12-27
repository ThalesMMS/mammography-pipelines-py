"""
Patient model for patient-level data management and split assignment.

This module defines the data structure for representing individual patients
and managing their data splits to prevent data leakage in machine learning
experiments.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- This model ensures proper data splitting at the patient level
- Prevents data leakage by keeping patient data isolated across splits
- Manages patient metadata and image collections
- Enables reproducible train/validation/test splits

Author: Research Team
Version: 1.0.0
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Configure logging for educational purposes
logger = logging.getLogger(__name__)


class Patient:
    """
    Represents an individual patient with mammography images and split assignment.

    This class manages patient-level data to ensure proper data splitting
    and prevent data leakage in machine learning experiments. It tracks
    all images belonging to a patient and their assigned data split.

    Educational Notes:
    - Patient-level splitting prevents data leakage
    - All images from the same patient stay in the same split
    - Metadata tracking enables demographic analysis
    - Split validation ensures proper isolation

    Attributes:
        patient_id (str): Unique patient identifier
        image_count (int): Number of mammography images
        projections (list[str]): Available projection types
        laterality (list[str]): Available laterality options
        split_assignment (str): "train", "validation", or "test"
        created_at (datetime): First image processing timestamp
    """

    # Define valid data splits
    VALID_SPLITS = ["train", "validation", "test"]

    # Define valid projection types
    VALID_PROJECTIONS = ["CC", "MLO"]

    # Define valid laterality options
    VALID_LATERALITY = ["L", "R"]

    def __init__(
        self,
        patient_id: str,
        image_count: int = 0,
        projections: Optional[List[str]] = None,
        laterality: Optional[List[str]] = None,
        split_assignment: str = "train",
        created_at: Optional[datetime] = None,
    ):
        """
        Initialize a Patient instance.

        Args:
            patient_id: Unique patient identifier
            image_count: Number of mammography images
            projections: Available projection types
            laterality: Available laterality options
            split_assignment: Data split assignment
            created_at: First image processing timestamp (default: now)

        Raises:
            ValueError: If validation rules are violated
            TypeError: If data types are incorrect
        """
        # Initialize core attributes with validation
        self.patient_id = self._validate_patient_id(patient_id)
        self.image_count = self._validate_image_count(image_count)
        self.projections = self._validate_projections(projections or [])
        self.laterality = self._validate_laterality(laterality or [])
        self.split_assignment = self._validate_split_assignment(split_assignment)
        self.created_at = created_at or datetime.now()

        # Initialize tracking attributes
        self.image_ids: Set[str] = set()
        self.validation_errors: List[str] = []
        self.updated_at = datetime.now()

        # Log creation for educational purposes
        logger.info(
            f"Created Patient: {self.patient_id} with {self.image_count} images"
        )

    def _validate_patient_id(self, patient_id: str) -> str:
        """
        Validate patient ID.

        Educational Note: Patient ID validation ensures proper identification
        and prevents data leakage by maintaining patient-level isolation.

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

    def _validate_image_count(self, image_count: int) -> int:
        """
        Validate image count.

        Educational Note: Image count validation ensures reasonable
        data distribution and prevents empty patient records.

        Args:
            image_count: Number of images to validate

        Returns:
            int: Validated image count

        Raises:
            ValueError: If image count is invalid
            TypeError: If image count is not an integer
        """
        if not isinstance(image_count, int):
            raise TypeError(f"image_count must be an integer, got {type(image_count)}")

        if image_count < 0:
            raise ValueError(f"image_count must be non-negative, got {image_count}")

        return image_count

    def _validate_projections(self, projections: List[str]) -> List[str]:
        """
        Validate projection types.

        Educational Note: Projection validation ensures only valid
        mammography projection types are recorded.

        Args:
            projections: List of projection types to validate

        Returns:
            List[str]: Validated projection types

        Raises:
            ValueError: If projections are invalid
            TypeError: If projections are not a list
        """
        if not isinstance(projections, list):
            raise TypeError(f"projections must be a list, got {type(projections)}")

        for i, projection in enumerate(projections):
            if not isinstance(projection, str):
                raise TypeError(
                    f"projections[{i}] must be a string, got {type(projection)}"
                )

            if projection not in self.VALID_PROJECTIONS:
                raise ValueError(
                    f"projections[{i}] must be one of {self.VALID_PROJECTIONS}, got {projection}"
                )

        return list(set(projections))  # Remove duplicates

    def _validate_laterality(self, laterality: List[str]) -> List[str]:
        """
        Validate laterality options.

        Educational Note: Laterality validation ensures only valid
        breast laterality options are recorded.

        Args:
            laterality: List of laterality options to validate

        Returns:
            List[str]: Validated laterality options

        Raises:
            ValueError: If laterality is invalid
            TypeError: If laterality is not a list
        """
        if not isinstance(laterality, list):
            raise TypeError(f"laterality must be a list, got {type(laterality)}")

        for i, lat in enumerate(laterality):
            if not isinstance(lat, str):
                raise TypeError(f"laterality[{i}] must be a string, got {type(lat)}")

            if lat not in self.VALID_LATERALITY:
                raise ValueError(
                    f"laterality[{i}] must be one of {self.VALID_LATERALITY}, got {lat}"
                )

        return list(set(laterality))  # Remove duplicates

    def _validate_split_assignment(self, split_assignment: str) -> str:
        """
        Validate split assignment.

        Educational Note: Split validation ensures proper data
        distribution and prevents invalid split assignments.

        Args:
            split_assignment: Split assignment to validate

        Returns:
            str: Validated split assignment

        Raises:
            ValueError: If split assignment is invalid
            TypeError: If split assignment is not a string
        """
        if not isinstance(split_assignment, str):
            raise TypeError(
                f"split_assignment must be a string, got {type(split_assignment)}"
            )

        if split_assignment not in self.VALID_SPLITS:
            raise ValueError(
                f"split_assignment must be one of {self.VALID_SPLITS}, got {split_assignment}"
            )

        return split_assignment

    def add_image(self, image_id: str, projection_type: str, laterality: str) -> bool:
        """
        Add an image to this patient's collection.

        Educational Note: This method maintains patient-level data
        integrity and updates metadata automatically.

        Args:
            image_id: Unique image identifier
            projection_type: Image projection type
            laterality: Image laterality

        Returns:
            bool: True if image added successfully, False otherwise

        Raises:
            ValueError: If image parameters are invalid
        """
        try:
            # Validate image parameters
            if not isinstance(image_id, str) or not image_id.strip():
                raise ValueError("image_id must be a non-empty string")

            if projection_type not in self.VALID_PROJECTIONS:
                raise ValueError(
                    f"projection_type must be one of {self.VALID_PROJECTIONS}, got {projection_type}"
                )

            if laterality not in self.VALID_LATERALITY:
                raise ValueError(
                    f"laterality must be one of {self.VALID_LATERALITY}, got {laterality}"
                )

            # Add image ID to collection
            self.image_ids.add(image_id.strip())

            # Update projections and laterality if new
            if projection_type not in self.projections:
                self.projections.append(projection_type)

            if laterality not in self.laterality:
                self.laterality.append(laterality)

            # Update image count
            self.image_count = len(self.image_ids)
            self.updated_at = datetime.now()

            logger.debug(f"Added image {image_id} to patient {self.patient_id}")
            return True

        except Exception as e:
            error_msg = (
                f"Error adding image {image_id} to patient {self.patient_id}: {e!s}"
            )
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)
            return False

    def remove_image(self, image_id: str) -> bool:
        """
        Remove an image from this patient's collection.

        Educational Note: This method maintains data consistency
        when images are removed from the dataset.

        Args:
            image_id: Image identifier to remove

        Returns:
            bool: True if image removed successfully, False otherwise
        """
        try:
            if image_id in self.image_ids:
                self.image_ids.remove(image_id)
                self.image_count = len(self.image_ids)
                self.updated_at = datetime.now()

                # Update projections and laterality if needed
                self._update_metadata_from_images()

                logger.debug(f"Removed image {image_id} from patient {self.patient_id}")
                return True
            else:
                logger.warning(
                    f"Image {image_id} not found in patient {self.patient_id}"
                )
                return False

        except Exception as e:
            error_msg = (
                f"Error removing image {image_id} from patient {self.patient_id}: {e!s}"
            )
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)
            return False

    def _update_metadata_from_images(self) -> None:
        """
        Update projections and laterality metadata from current images.

        Educational Note: This method ensures metadata consistency
        when images are added or removed from the patient collection.
        """
        # This would typically query the image database to get current metadata
        # For now, we'll keep the existing metadata as it represents historical data
        pass

    def change_split_assignment(self, new_split: str) -> bool:
        """
        Change the patient's data split assignment.

        Educational Note: Split changes should be done carefully to
        maintain data integrity and prevent data leakage.

        Args:
            new_split: New split assignment

        Returns:
            bool: True if split changed successfully, False otherwise
        """
        try:
            if new_split not in self.VALID_SPLITS:
                raise ValueError(
                    f"new_split must be one of {self.VALID_SPLITS}, got {new_split}"
                )

            old_split = self.split_assignment
            self.split_assignment = new_split
            self.updated_at = datetime.now()

            logger.info(
                f"Changed patient {self.patient_id} split from {old_split} to {new_split}"
            )
            return True

        except Exception as e:
            error_msg = f"Error changing split for patient {self.patient_id}: {e!s}"
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)
            return False

    def get_patient_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of patient data.

        Educational Note: This summary provides a complete overview
        of the patient's data for analysis and reporting.

        Returns:
            Dict[str, Any]: Dictionary containing patient summary
        """
        return {
            "patient_id": self.patient_id,
            "image_count": self.image_count,
            "projections": sorted(self.projections),
            "laterality": sorted(self.laterality),
            "split_assignment": self.split_assignment,
            "image_ids": sorted(list(self.image_ids)),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "validation_errors": self.validation_errors,
        }

    def validate_split_isolation(self, other_patients: List["Patient"]) -> bool:
        """
        Validate that this patient's split assignment is properly isolated.

        Educational Note: This validation ensures no data leakage
        between train/validation/test splits at the patient level.

        Args:
            other_patients: List of other patients to check against

        Returns:
            bool: True if split is properly isolated, False otherwise
        """
        try:
            # Check for duplicate patient IDs in different splits
            for other_patient in other_patients:
                if other_patient.patient_id == self.patient_id:
                    if other_patient.split_assignment != self.split_assignment:
                        error_msg = f"Patient {self.patient_id} appears in multiple splits: {self.split_assignment} and {other_patient.split_assignment}"
                        self.validation_errors.append(error_msg)
                        logger.error(error_msg)
                        return False

            # Check for shared image IDs across splits
            for other_patient in other_patients:
                if other_patient.split_assignment != self.split_assignment:
                    shared_images = self.image_ids.intersection(other_patient.image_ids)
                    if shared_images:
                        error_msg = f"Patient {self.patient_id} shares images with patient {other_patient.patient_id} in different splits: {shared_images}"
                        self.validation_errors.append(error_msg)
                        logger.error(error_msg)
                        return False

            return True

        except Exception as e:
            error_msg = (
                f"Error validating split isolation for patient {self.patient_id}: {e!s}"
            )
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)
            return False

    def save_patient(self, file_path: str) -> bool:
        """
        Save patient data to file.

        Educational Note: Patient saving enables persistence of
        patient metadata and split assignments.

        Args:
            file_path: Path where to save the patient data

        Returns:
            bool: True if saving successful, False otherwise
        """
        try:
            import json

            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert sets to lists for JSON serialization
            patient_data = self.get_patient_summary()
            patient_data["image_ids"] = sorted(list(self.image_ids))

            with open(file_path, "w") as f:
                json.dump(patient_data, f, indent=2)

            logger.info(f"Saved Patient data to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving Patient data to {file_path}: {e!s}")
            return False

    @classmethod
    def load_patient(cls, file_path: str) -> "Patient":
        """
        Load patient data from file.

        Educational Note: This class method enables loading of
        previously saved patient data.

        Args:
            file_path: Path to the saved patient file

        Returns:
            Patient: Loaded instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is corrupted or invalid
        """
        try:
            import json

            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Patient file not found: {file_path}")

            with open(file_path, "r") as f:
                patient_data = json.load(f)

            # Parse creation timestamp
            created_at = (
                datetime.fromisoformat(patient_data["created_at"])
                if patient_data.get("created_at")
                else None
            )

            # Create Patient instance
            patient = cls(
                patient_id=patient_data["patient_id"],
                image_count=patient_data.get("image_count", 0),
                projections=patient_data.get("projections", []),
                laterality=patient_data.get("laterality", []),
                split_assignment=patient_data.get("split_assignment", "train"),
                created_at=created_at,
            )

            # Restore image IDs
            patient.image_ids = set(patient_data.get("image_ids", []))

            # Restore validation errors if any
            if patient_data.get("validation_errors"):
                patient.validation_errors = patient_data["validation_errors"]

            logger.info(f"Loaded Patient data from {file_path}")
            return patient

        except Exception as e:
            raise ValueError(f"Error loading Patient data from {file_path}: {e!s}")

    def __repr__(self) -> str:
        """String representation for debugging and logging."""
        return (
            f"Patient("
            f"patient_id='{self.patient_id}', "
            f"image_count={self.image_count}, "
            f"split='{self.split_assignment}')"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Patient: {self.patient_id}\n"
            f"Images: {self.image_count}\n"
            f"Projections: {', '.join(sorted(self.projections))}\n"
            f"Laterality: {', '.join(sorted(self.laterality))}\n"
            f"Split: {self.split_assignment}\n"
            f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )


def create_patient_from_images(
    patient_id: str, image_data: List[Dict[str, Any]], split_assignment: str = "train"
) -> Patient:
    """
    Create a Patient instance from image data.

    Educational Note: This factory function demonstrates how to create
    Patient instances from image metadata, enabling standardized
    patient creation.

    Args:
        patient_id: Unique patient identifier
        image_data: List of image metadata dictionaries
        split_assignment: Data split assignment

    Returns:
        Patient: Created instance

    Raises:
        ValueError: If image data is invalid
    """
    # Create Patient instance
    patient = Patient(
        patient_id=patient_id,
        image_count=len(image_data),
        split_assignment=split_assignment,
    )

    # Add images to patient
    for image_info in image_data:
        image_id = image_info.get("image_id", "")
        projection_type = image_info.get("projection_type", "")
        laterality = image_info.get("laterality", "")

        patient.add_image(image_id, projection_type, laterality)

    return patient


def validate_patient_splits(patients: List[Patient]) -> Dict[str, Any]:
    """
    Validate patient split assignments across all patients.

    Educational Note: This function ensures proper data splitting
    and prevents data leakage in machine learning experiments.

    Args:
        patients: List of Patient instances to validate

    Returns:
        Dict[str, Any]: Validation results and statistics
    """
    validation_results = {
        "is_valid": True,
        "errors": [],
        "split_counts": {"train": 0, "validation": 0, "test": 0},
        "total_patients": len(patients),
        "total_images": 0,
    }

    # Count patients and images per split
    for patient in patients:
        validation_results["split_counts"][patient.split_assignment] += 1
        validation_results["total_images"] += patient.image_count

        # Validate split isolation
        if not patient.validate_split_isolation(patients):
            validation_results["is_valid"] = False
            validation_results["errors"].extend(patient.validation_errors)

    # Check for reasonable split distribution
    total_patients = validation_results["total_patients"]
    if total_patients > 0:
        train_ratio = validation_results["split_counts"]["train"] / total_patients
        val_ratio = validation_results["split_counts"]["validation"] / total_patients
        test_ratio = validation_results["split_counts"]["test"] / total_patients

        if train_ratio < 0.5:
            validation_results["errors"].append(
                f"Training split too small: {train_ratio:.2%}"
            )

        if val_ratio < 0.1:
            validation_results["errors"].append(
                f"Validation split too small: {val_ratio:.2%}"
            )

        if test_ratio < 0.1:
            validation_results["errors"].append(
                f"Test split too small: {test_ratio:.2%}"
            )

    return validation_results
