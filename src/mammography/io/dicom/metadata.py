# DISCLAIMER: Educational project only - NOT for clinical or medical diagnostic purposes.
#
"""Metadata model for mammography DICOM images."""

from __future__ import annotations

from datetime import datetime
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pydicom
from pydicom.errors import InvalidDicomError

from .pixel_processing import allow_invalid_decimal_strings_context

logger = logging.getLogger(__name__)

VALID_STATES = ["raw", "validated", "preprocessed", "embedded", "clustered"]
VALID_PROJECTION_TYPES = ["CC", "MLO", "ML", "LM", "XCCL", "XCCM", "FB", "SIO"]
VALID_LATERALITY = ["L", "R"]
SENSITIVE_DICOM_TAGS = {"PATIENTID", "PATIENTNAME", "ACCESSIONNUMBER"}
VALID_TRANSITIONS = {
    "raw": ["validated"],
    "validated": ["preprocessed"],
    "preprocessed": ["embedded"],
    "embedded": ["clustered"],
    "clustered": [],
}


def _is_sensitive_dicom_tag(tag_name: str) -> bool:
    normalized = tag_name.upper()
    return (
        normalized in SENSITIVE_DICOM_TAGS
        or "UID" in normalized
        or normalized.endswith("ID")
    )


def _format_validation_value(tag_name: str, value: Any) -> Any:
    if _is_sensitive_dicom_tag(tag_name):
        return "<REDACTED>"
    return value


def fingerprint_uid(uid: str) -> str:
    return hashlib.sha256(uid.encode("utf-8", errors="replace")).hexdigest()


def _validation_exception_message(exc: Exception, description: str) -> str:
    return f"{exc.__class__.__name__}: {description}"


class DicomValidationError(ValueError):
    """Raised when a DICOM file is readable but fails metadata validation."""

    def __init__(self, errors: List[str]):
        self.errors = list(errors)
        details = "; ".join(self.errors)
        super().__init__(f"DICOM validation failed: {details}")


class MammographyImage:
    """
    Represents a single DICOM mammography file with associated metadata.

    This class captures metadata needed for density analysis and implements a
    simple state machine to track processing stages.
    """

    VALID_STATES = VALID_STATES
    VALID_PROJECTION_TYPES = VALID_PROJECTION_TYPES
    VALID_LATERALITY = VALID_LATERALITY

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
        dataset: Optional["pydicom.dataset.FileDataset"] = None,
        allow_nonstandard_uids: bool = False,
    ):
        """
        Initialize a MammographyImage instance.

        Args:
            patient_id: Patient identifier from DICOM
            study_id: Study instance UID
            series_id: Series instance UID
            instance_id: SOP instance UID
            projection_type: View position (CC or MLO)
            laterality: Image laterality (L or R)
            manufacturer: Equipment manufacturer
            pixel_spacing: Tuple of (row_spacing, col_spacing) in mm
            bits_stored: Number of bits used to store pixel data
            file_path: Path to the DICOM file
            acquisition_date: Optional date of image acquisition
            state: Processing state (defaults to "raw")
            dataset: Optional pre-loaded DICOM dataset to avoid re-reading the file
        """
        self.allow_nonstandard_uids = allow_nonstandard_uids
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
        self.acquisition_date = acquisition_date
        self.state = self._validate_state(state)
        self.dataset = dataset

        self.validation_errors: List[str] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        logger.info("Created MammographyImage: %s", fingerprint_uid(self.instance_id))

    def _validate_patient_id(self, patient_id: str) -> str:
        if not isinstance(patient_id, str):
            raise TypeError(f"patient_id must be a string, got {type(patient_id)}")
        if not patient_id.strip():
            raise ValueError("patient_id cannot be empty or whitespace")
        return patient_id.strip()

    def _validate_uid(self, uid: str, field_name: str) -> str:
        if not isinstance(uid, str):
            raise TypeError(f"{field_name} must be a string, got {type(uid)}")
        if not uid.strip():
            raise ValueError(f"{field_name} cannot be empty or whitespace")
        uid = uid.strip()
        if not self.allow_nonstandard_uids and not pydicom.uid.UID(uid).is_valid:
            raise ValueError(f"{field_name} must be a valid DICOM UID format")
        return uid

    def _validate_projection_type(self, projection_type: str) -> str:
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
        if not isinstance(laterality, str):
            raise TypeError(f"laterality must be a string, got {type(laterality)}")
        if laterality not in self.VALID_LATERALITY:
            raise ValueError(
                f"laterality must be one of {self.VALID_LATERALITY}, got {laterality}"
            )
        return laterality

    def _validate_manufacturer(self, manufacturer: str) -> str:
        if not isinstance(manufacturer, str):
            raise TypeError(f"manufacturer must be a string, got {type(manufacturer)}")
        if not manufacturer.strip():
            raise ValueError("manufacturer cannot be empty or whitespace")
        return manufacturer.strip()

    def _validate_pixel_spacing(
        self, pixel_spacing: tuple[float, float]
    ) -> tuple[float, float]:
        if not isinstance(pixel_spacing, tuple):
            raise TypeError(
                f"pixel_spacing must be a tuple, got {type(pixel_spacing)}"
            )
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
                raise ValueError(
                    f"pixel_spacing[{i}] must be positive, got {spacing}"
                )
        return tuple(float(spacing) for spacing in pixel_spacing)

    def _validate_bits_stored(self, bits_stored: int) -> int:
        if not isinstance(bits_stored, int):
            raise TypeError(f"bits_stored must be an integer, got {type(bits_stored)}")
        if bits_stored <= 0:
            raise ValueError(f"bits_stored must be positive, got {bits_stored}")
        if bits_stored > 32:
            raise ValueError(f"bits_stored must be <= 32, got {bits_stored}")
        return bits_stored

    def _validate_file_path(self, file_path: str) -> str:
        if not isinstance(file_path, str):
            raise TypeError(f"file_path must be a string, got {type(file_path)}")
        if not file_path.strip():
            raise ValueError("file_path cannot be empty or whitespace")
        path = Path(file_path)
        if not path.exists():
            raise ValueError("DICOM file does not exist")
        if not path.is_file():
            raise ValueError("Path is not a file")
        return str(path.absolute())

    def _validate_state(self, state: str) -> str:
        if not isinstance(state, str):
            raise TypeError(f"state must be a string, got {type(state)}")
        if state not in self.VALID_STATES:
            raise ValueError(f"state must be one of {self.VALID_STATES}, got {state}")
        return state

    @property
    def view_position(self) -> str:
        """Backward-compatible alias for ``projection_type``."""
        return self.projection_type

    @property
    def pixel_array(self):
        """Backward-compatible pixel data accessor."""
        try:
            if self.dataset is None or not hasattr(self.dataset, "PixelData"):
                with allow_invalid_decimal_strings_context():
                    self.dataset = pydicom.dcmread(
                        self.file_path,
                        force=True,
                        stop_before_pixels=False,
                    )
            return self.dataset.pixel_array
        except Exception as exc:
            logger.error(
                "Error reading pixel_array for %s (%s)",
                fingerprint_uid(self.instance_id),
                exc.__class__.__name__,
            )
            raise DicomValidationError(
                [_validation_exception_message(exc, "failed to read DICOM pixel data")]
            ) from exc

    def transition_to(self, new_state: str) -> bool:
        if new_state not in self.VALID_STATES:
            raise ValueError(
                f"Invalid state: {new_state}. Must be one of {self.VALID_STATES}"
            )

        if new_state not in VALID_TRANSITIONS.get(self.state, []):
            error_msg = f"Invalid transition from {self.state} to {new_state}"
            logger.warning(error_msg)
            self.validation_errors.append(error_msg)
            return False

        old_state = self.state
        self.state = new_state
        self.updated_at = datetime.now()

        logger.info(
            "State transition: %s %s -> %s",
            fingerprint_uid(self.instance_id),
            old_state,
            new_state,
        )
        return True

    def validate_dicom_file(
        self,
        dataset: Optional["pydicom.dataset.FileDataset"] = None,
        validate_pixel_data: bool = True,
    ) -> bool:
        """
        Validate DICOM file against stored metadata.

        Args:
            dataset: Optional pre-loaded DICOM dataset to avoid re-reading the file.
                    If None, the file will be read from self.file_path.

        Returns:
            True if validation passes, False otherwise. Validation errors are stored
            in self.validation_errors.
        """
        self.validation_errors = []
        try:
            if dataset is None:
                with allow_invalid_decimal_strings_context():
                    dataset = pydicom.dcmread(
                        self.file_path,
                        force=True,
                        stop_before_pixels=not validate_pixel_data,
                    )

            required_tags = [
                ("PatientID", self.patient_id),
                ("StudyInstanceUID", self.study_id),
                ("SeriesInstanceUID", self.series_id),
                ("SOPInstanceUID", self.instance_id),
                ("Manufacturer", self.manufacturer),
                ("PixelSpacing", self.pixel_spacing),
                ("BitsStored", self.bits_stored),
                ("ViewPosition", self.projection_type),
                ("ImageLaterality", self.laterality),
            ]

            for tag_name, expected_value in required_tags:
                if not hasattr(dataset, tag_name):
                    error_msg = f"Missing required DICOM tag: {tag_name}"
                    self.validation_errors.append(error_msg)
                    continue

                actual_value = getattr(dataset, tag_name)
                if tag_name == "PixelSpacing":
                    try:
                        actual_value = tuple(float(value) for value in actual_value)
                        expected_value = tuple(float(value) for value in expected_value)
                    except (TypeError, ValueError):
                        pass

                if actual_value != expected_value:
                    expected_display = _format_validation_value(
                        tag_name, expected_value
                    )
                    actual_display = _format_validation_value(tag_name, actual_value)
                    error_msg = (
                        f"DICOM tag {tag_name} mismatch: expected {expected_display}, got {actual_display}"
                    )
                    self.validation_errors.append(error_msg)

            if not validate_pixel_data:
                pass
            elif not hasattr(dataset, "PixelData"):
                self.validation_errors.append("Missing PixelData in DICOM file")
            else:
                try:
                    if len(dataset.PixelData) == 0:
                        self.validation_errors.append("PixelData has zero size")
                    else:
                        pixel_array = dataset.pixel_array
                        if pixel_array is None:
                            self.validation_errors.append("PixelData is empty")
                        elif pixel_array.size == 0:
                            self.validation_errors.append("PixelData has zero size")
                except Exception as exc:
                    self.validation_errors.append(
                        _validation_exception_message(exc, "error reading PixelData")
                    )
                    logger.error(
                        "Error reading PixelData for %s (%s)",
                        fingerprint_uid(self.instance_id),
                        exc.__class__.__name__,
                    )

            if not self.validation_errors:
                if self.state != "validated" and not self.transition_to("validated"):
                    return False
                logger.info(
                    "DICOM validation successful: %s",
                    fingerprint_uid(self.instance_id),
                )
                return True

            logger.warning(
                "DICOM validation failed: %s, errors: %s",
                fingerprint_uid(self.instance_id),
                self.validation_errors,
            )
            return False

        except InvalidDicomError as exc:
            error_msg = _validation_exception_message(exc, "invalid DICOM file")
            self.validation_errors.append(error_msg)
            logger.error(
                "Invalid DICOM file while validating %s (%s)",
                fingerprint_uid(self.instance_id),
                exc.__class__.__name__,
            )
            return False
        except Exception as exc:
            error_msg = _validation_exception_message(exc, "validation error")
            self.validation_errors.append(error_msg)
            logger.error(
                "Error validating DICOM file for %s (%s)",
                fingerprint_uid(self.instance_id),
                exc.__class__.__name__,
            )
            return False

    def get_metadata_dict(self) -> Dict[str, Any]:
        """Serialize metadata for internal caching.

        Unlike ``__repr__`` and ``__str__``, this intentionally includes PHI
        fields such as ``patient_id`` and ``file_path`` for internal
        serialization. Callers must redact or protect this data before exposing
        it outside trusted storage.
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
            "validation_errors": self.validation_errors.copy(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            "MammographyImage("  # noqa: E501
            f"instance_id='{fingerprint_uid(self.instance_id)}', "
            f"projection_type='{self.projection_type}', "
            f"laterality='{self.laterality}', "
            f"state='{self.state}')"
        )

    def __str__(self) -> str:
        return (
            f"Mammography Image: {fingerprint_uid(self.instance_id)}\n"
            f"Projection: {self.projection_type}\n"
            f"Laterality: {self.laterality}\n"
            f"State: {self.state}"
        )


def create_mammography_image_from_dicom(
    file_path: str,
    dataset: Optional["pydicom.dataset.FileDataset"] = None,
    validate_pixel_data: bool = True,
    validate: bool = True,
) -> MammographyImage:
    """
    Create a MammographyImage instance from a DICOM file.

    Args:
        file_path: Path to the DICOM file
        dataset: Optional pre-loaded DICOM dataset to avoid re-reading the file
        validate_pixel_data: Whether validation should decode/check PixelData
        validate: Whether to run MammographyImage.validate_dicom_file

    Raises InvalidDicomError when the file is not a valid DICOM.
    """
    try:
        if dataset is None:
            with allow_invalid_decimal_strings_context():
                dataset = pydicom.dcmread(
                    file_path,
                    force=True,
                    stop_before_pixels=not validate_pixel_data,
                )

        patient_id = getattr(dataset, "PatientID", None) or "<MISSING_PATIENT_ID>"
        study_id = (
            getattr(dataset, "StudyInstanceUID", None)
            or "<MISSING_STUDY_INSTANCE_UID>"
        )
        series_id = (
            getattr(dataset, "SeriesInstanceUID", None)
            or "<MISSING_SERIES_INSTANCE_UID>"
        )
        instance_id = (
            getattr(dataset, "SOPInstanceUID", None)
            or "<MISSING_SOP_INSTANCE_UID>"
        )
        raw_projection_type = str(getattr(dataset, "ViewPosition", "CC"))
        projection_type = (
            raw_projection_type
            if raw_projection_type in VALID_PROJECTION_TYPES
            else "CC"
        )
        raw_laterality = str(getattr(dataset, "ImageLaterality", "L"))
        laterality = raw_laterality if raw_laterality in VALID_LATERALITY else "L"
        manufacturer = getattr(dataset, "Manufacturer", "UNKNOWN")
        try:
            pixel_spacing = tuple(
                float(spacing)
                for spacing in getattr(dataset, "PixelSpacing", (1.0, 1.0))
            )
        except (TypeError, ValueError):
            pixel_spacing = (1.0, 1.0)
        if len(pixel_spacing) != 2 or any(spacing <= 0 for spacing in pixel_spacing):
            pixel_spacing = (1.0, 1.0)
        if not hasattr(dataset, "BitsStored"):
            raise ValueError("Missing required DICOM tag 'BitsStored'")
        bits_stored = dataset.BitsStored
        acquisition_date = getattr(dataset, "AcquisitionDate", None)

        if acquisition_date:
            try:
                acquisition_date = datetime.strptime(str(acquisition_date), "%Y%m%d")
            except ValueError:
                acquisition_date = None

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
            dataset=dataset,
            allow_nonstandard_uids=not validate,
        )

        if validate:
            validation_ok = mammography_image.validate_dicom_file(
                dataset=dataset,
                validate_pixel_data=validate_pixel_data,
            )
            if not validation_ok:
                raise DicomValidationError(mammography_image.validation_errors)

        return mammography_image

    except InvalidDicomError as exc:
        raise InvalidDicomError("Invalid DICOM file") from exc
    except DicomValidationError:
        raise
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Error creating MammographyImage") from exc
