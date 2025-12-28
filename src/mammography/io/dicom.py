"""
DICOM I/O utilities and data models for the mammography pipelines.

WARNING: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.pixels import apply_modality_lut
from PIL import Image

logger = logging.getLogger(__name__)

DICOM_EXTS = (".dcm", ".dicom")

RESEARCH_DISCLAIMER = (
    "WARNING: This is an EDUCATIONAL RESEARCH project.\n"
    "It must NOT be used for clinical or medical diagnostic purposes.\n"
    "No medical decision should be based on these results.\n\n"
    "This project is intended exclusively for research and education purposes\n"
    "in medical imaging processing and machine learning.\n"
)


def get_disclaimer() -> str:
    """Return the mandatory research disclaimer."""
    return RESEARCH_DISCLAIMER


def is_dicom_path(path: str) -> bool:
    """Cheap extension-based check against DICOM_EXTS."""
    return str(path).lower().endswith(DICOM_EXTS)


def _is_mono1(ds: "pydicom.dataset.FileDataset") -> bool:
    """Detect MONOCHROME1 photometry (black=high, white=low)."""
    return getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1"


def _to_float32(arr: np.ndarray) -> np.ndarray:
    """Ensure the array is float32 to avoid dtype surprises later."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _apply_rescale(ds: "pydicom.dataset.FileDataset", arr: np.ndarray) -> np.ndarray:
    """Apply RescaleSlope/Intercept or fall back to the modality LUT."""
    slope = getattr(ds, "RescaleSlope", 1.0)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    try:
        arr = arr * float(slope) + float(intercept)
    except Exception:
        try:
            arr = apply_modality_lut(arr, ds)
        except Exception:
            pass
    return arr


def robust_window(
    arr: np.ndarray, p_low: float = 0.5, p_high: float = 99.5
) -> np.ndarray:
    """Percentile-based windowing that is robust to stray high-intensity pixels."""
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return arr


def dicom_to_pil_rgb(
    dcm_path: str, window_low: float = 0.5, window_high: float = 99.5
) -> Image.Image:
    """Read a DICOM and convert it to 8-bit RGB (channel-replicated grayscale)."""
    try:
        ds = pydicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array
    except Exception as exc:
        raise RuntimeError(
            f"Falha ao ler pixel data de {dcm_path}. Erro: {exc!r}"
        ) from exc
    arr = _to_float32(arr)
    arr = _apply_rescale(ds, arr)
    if _is_mono1(ds):
        arr = arr.max() - arr
    arr = robust_window(arr, window_low, window_high)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    return Image.merge("RGB", (pil, pil, pil))


class MammographyImage:
    """
    Represents a single DICOM mammography file with associated metadata.

    This class captures metadata needed for density analysis and implements a
    simple state machine to track processing stages.
    """

    VALID_STATES = ["raw", "validated", "preprocessed", "embedded", "clustered"]
    VALID_PROJECTION_TYPES = ["CC", "MLO"]
    VALID_LATERALITY = ["L", "R"]

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

        self.validation_errors: List[str] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        logger.info(
            "Created MammographyImage: %s for patient %s",
            self.instance_id,
            self.patient_id,
        )

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
        if not uid.replace(".", "").replace(" ", "").isdigit():
            raise ValueError(f"{field_name} must be a valid DICOM UID format")
        return uid.strip()

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
            raise ValueError(f"DICOM file does not exist: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        return str(path.absolute())

    def _validate_state(self, state: str) -> str:
        if not isinstance(state, str):
            raise TypeError(f"state must be a string, got {type(state)}")
        if state not in self.VALID_STATES:
            raise ValueError(f"state must be one of {self.VALID_STATES}, got {state}")
        return state

    def transition_to(self, new_state: str) -> bool:
        if new_state not in self.VALID_STATES:
            raise ValueError(
                f"Invalid state: {new_state}. Must be one of {self.VALID_STATES}"
            )

        valid_transitions = {
            "raw": ["validated"],
            "validated": ["preprocessed"],
            "preprocessed": ["embedded"],
            "embedded": ["clustered"],
            "clustered": [],
        }

        if new_state not in valid_transitions.get(self.state, []):
            error_msg = f"Invalid transition from {self.state} to {new_state}"
            logger.warning(error_msg)
            self.validation_errors.append(error_msg)
            return False

        old_state = self.state
        self.state = new_state
        self.updated_at = datetime.now()

        logger.info("State transition: %s %s -> %s", self.instance_id, old_state, new_state)
        return True

    def validate_dicom_file(self) -> bool:
        try:
            dataset = pydicom.dcmread(self.file_path)

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
                    error_msg = (
                        f"DICOM tag {tag_name} mismatch: expected {expected_value}, got {actual_value}"
                    )
                    self.validation_errors.append(error_msg)

            if hasattr(dataset, "ViewPosition"):
                if dataset.ViewPosition != self.projection_type:
                    error_msg = (
                        f"ViewPosition mismatch: expected {self.projection_type}, got {dataset.ViewPosition}"
                    )
                    self.validation_errors.append(error_msg)

            if hasattr(dataset, "ImageLaterality"):
                if dataset.ImageLaterality != self.laterality:
                    error_msg = (
                        f"ImageLaterality mismatch: expected {self.laterality}, got {dataset.ImageLaterality}"
                    )
                    self.validation_errors.append(error_msg)

            if not hasattr(dataset, "PixelData"):
                self.validation_errors.append("Missing PixelData in DICOM file")
            else:
                try:
                    pixel_array = dataset.pixel_array
                    if pixel_array is None:
                        self.validation_errors.append("PixelData is empty")
                    elif pixel_array.size == 0:
                        self.validation_errors.append("PixelData has zero size")
                except Exception as exc:
                    self.validation_errors.append(f"Error reading PixelData: {exc!s}")

            if not self.validation_errors:
                self.transition_to("validated")
                logger.info("DICOM validation successful: %s", self.instance_id)
                return True

            logger.warning(
                "DICOM validation failed: %s, errors: %s",
                self.instance_id,
                self.validation_errors,
            )
            return False

        except InvalidDicomError as exc:
            error_msg = f"Invalid DICOM file: {exc!s}"
            self.validation_errors.append(error_msg)
            logger.error(error_msg)
            return False
        except Exception as exc:
            error_msg = f"Error validating DICOM file: {exc!s}"
            self.validation_errors.append(error_msg)
            logger.error(error_msg)
            return False

    def get_metadata_dict(self) -> Dict[str, Any]:
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
        return (
            "MammographyImage("  # noqa: E501
            f"patient_id='{self.patient_id}', "
            f"instance_id='{self.instance_id}', "
            f"projection_type='{self.projection_type}', "
            f"laterality='{self.laterality}', "
            f"state='{self.state}')"
        )

    def __str__(self) -> str:
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

    Raises InvalidDicomError when the file is not a valid DICOM.
    """
    try:
        dataset = pydicom.dcmread(file_path)

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
        )

        mammography_image.validate_dicom_file()

        return mammography_image

    except InvalidDicomError as exc:
        raise InvalidDicomError(f"Invalid DICOM file {file_path}: {exc!s}") from exc
    except Exception as exc:
        raise ValueError(f"Error creating MammographyImage from {file_path}: {exc!s}") from exc


class DicomReader:
    """
    DICOM file reader for mammography images with validation and metadata caching.
    """

    SUPPORTED_EXTENSIONS = [".dcm", ".dicom", ".DCM", ".DICOM"]

    REQUIRED_TAGS = [
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "Manufacturer",
        "PixelSpacing",
        "BitsStored",
        "ViewPosition",
        "ImageLaterality",
        "PixelData",
    ]

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
        self.validate_on_read = validate_on_read
        self.cache_metadata = cache_metadata
        self.max_workers = max_workers

        self.metadata_cache: Dict[str, Dict[str, Any]] = {}

        self.stats = {
            "files_processed": 0,
            "files_valid": 0,
            "files_invalid": 0,
            "validation_errors": [],
        }

        logger.info(
            "Initialized DicomReader with validate_on_read=%s, max_workers=%s",
            validate_on_read,
            max_workers,
        )

    def read_dicom_file(
        self, file_path: Union[str, Path]
    ) -> Optional[MammographyImage]:
        file_path = Path(file_path)

        if not file_path.exists():
            error_msg = f"DICOM file not found: {file_path}"
            logger.error(error_msg)
            self.stats["validation_errors"].append(error_msg)
            raise FileNotFoundError(error_msg)

        if file_path.suffix.lower() not in [ext.lower() for ext in self.SUPPORTED_EXTENSIONS]:
            error_msg = f"Unsupported file extension: {file_path.suffix}"
            logger.warning(error_msg)
            self.stats["validation_errors"].append(error_msg)
            return None

        try:
            dataset = pydicom.dcmread(file_path)

            if self.validate_on_read:
                validation_result = self._validate_dicom_dataset(dataset, file_path)
                if not validation_result["valid"]:
                    logger.warning(
                        "DICOM validation failed for %s: %s",
                        file_path,
                        validation_result["errors"],
                    )
                    self.stats["files_invalid"] += 1
                    self.stats["validation_errors"].extend(validation_result["errors"])
                    return None

            mammography_image = create_mammography_image_from_dicom(str(file_path))

            if self.cache_metadata:
                self.metadata_cache[str(file_path)] = (
                    mammography_image.get_metadata_dict()
                )

            self.stats["files_processed"] += 1
            self.stats["files_valid"] += 1

            logger.info("Successfully read DICOM file: %s", file_path)
            return mammography_image

        except Exception as exc:
            error_msg = f"Error reading DICOM file {file_path}: {exc!s}"
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
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        dicom_files = self._find_dicom_files(directory_path, recursive)
        logger.info("Found %s DICOM files in %s", len(dicom_files), directory_path)

        if not dicom_files:
            logger.warning("No DICOM files found in %s", directory_path)
            return {}

        mammography_images = self._read_dicom_files_parallel(dicom_files)
        valid_images = [img for img in mammography_images if img is not None]
        logger.info(
            "Successfully read %s out of %s DICOM files",
            len(valid_images),
            len(dicom_files),
        )

        if patient_level:
            return self._organize_by_patient(valid_images)

        return {img.file_path: [img] for img in valid_images}

    def _find_dicom_files(
        self, directory_path: Path, recursive: bool = True
    ) -> List[Path]:
        dicom_files: List[Path] = []

        if recursive:
            for ext in self.SUPPORTED_EXTENSIONS:
                dicom_files.extend(directory_path.glob(f"**/*{ext}"))
        else:
            for ext in self.SUPPORTED_EXTENSIONS:
                dicom_files.extend(directory_path.glob(f"*{ext}"))

        dicom_files = sorted(set(dicom_files))
        logger.debug("Found %s DICOM files", len(dicom_files))
        return dicom_files

    def _read_dicom_files_parallel(
        self, dicom_files: List[Path]
    ) -> List[Optional[MammographyImage]]:
        mammography_images: List[Optional[MammographyImage]] = [None] * len(dicom_files)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.read_dicom_file, file_path): i
                for i, file_path in enumerate(dicom_files)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    mammography_images[index] = future.result()
                except Exception as exc:
                    logger.error(
                        "Error processing DICOM file %s: %s",
                        dicom_files[index],
                        exc,
                    )
                    mammography_images[index] = None

        return mammography_images

    def _organize_by_patient(
        self, mammography_images: List[MammographyImage]
    ) -> Dict[str, List[MammographyImage]]:
        patient_organization: Dict[str, List[MammographyImage]] = {}

        for image in mammography_images:
            patient_id = image.patient_id
            patient_organization.setdefault(patient_id, []).append(image)

        for patient_id in patient_organization:
            patient_organization[patient_id].sort(
                key=lambda x: x.acquisition_date or datetime.min
            )

        logger.info(
            "Organized %s images into %s patients",
            len(mammography_images),
            len(patient_organization),
        )
        return patient_organization

    def _validate_dicom_dataset(
        self, dataset: pydicom.Dataset, file_path: Path
    ) -> Dict[str, Any]:
        validation_result: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        for tag in self.REQUIRED_TAGS:
            if not hasattr(dataset, tag):
                error_msg = f"Missing required DICOM tag '{tag}' in {file_path}"
                validation_result["errors"].append(error_msg)
                validation_result["valid"] = False

        if hasattr(dataset, "ViewPosition"):
            if dataset.ViewPosition not in ["CC", "MLO"]:
                error_msg = (
                    f"Invalid ViewPosition '{dataset.ViewPosition}' in {file_path}. "
                    "Must be 'CC' or 'MLO'"
                )
                validation_result["errors"].append(error_msg)
                validation_result["valid"] = False

        if hasattr(dataset, "ImageLaterality"):
            if dataset.ImageLaterality not in ["L", "R"]:
                error_msg = (
                    f"Invalid ImageLaterality '{dataset.ImageLaterality}' in {file_path}. "
                    "Must be 'L' or 'R'"
                )
                validation_result["errors"].append(error_msg)
                validation_result["valid"] = False

        if hasattr(dataset, "PixelData"):
            try:
                pixel_array = dataset.pixel_array
                if pixel_array is None or pixel_array.size == 0:
                    error_msg = f"Empty or invalid PixelData in {file_path}"
                    validation_result["errors"].append(error_msg)
                    validation_result["valid"] = False
            except Exception as exc:
                error_msg = f"Error reading PixelData from {file_path}: {exc!s}"
                validation_result["errors"].append(error_msg)
                validation_result["valid"] = False

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

        for tag in self.OPTIONAL_TAGS:
            if not hasattr(dataset, tag):
                warning_msg = f"Missing optional DICOM tag '{tag}' in {file_path}"
                validation_result["warnings"].append(warning_msg)

        return validation_result

    def get_metadata_summary(self) -> Dict[str, Any]:
        if not self.metadata_cache:
            return {"total_files": 0, "message": "No metadata available"}

        manufacturers = set()
        projection_types = set()
        lateralities = set()
        pixel_spacings: List[Tuple[float, float]] = []
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
        self.metadata_cache.clear()
        logger.info("Cleared DICOM metadata cache")

    def get_processing_stats(self) -> Dict[str, Any]:
        return {
            "files_processed": self.stats["files_processed"],
            "files_valid": self.stats["files_valid"],
            "files_invalid": self.stats["files_invalid"],
            "success_rate": (
                self.stats["files_valid"] / max(self.stats["files_processed"], 1)
            )
            * 100,
            "total_validation_errors": len(self.stats["validation_errors"]),
            "recent_errors": self.stats["validation_errors"][-10:],
        }


def create_dicom_reader(
    validate_on_read: bool = True, cache_metadata: bool = True, max_workers: int = 4
) -> DicomReader:
    """Factory function to create a DicomReader instance."""
    return DicomReader(
        validate_on_read=validate_on_read,
        cache_metadata=cache_metadata,
        max_workers=max_workers,
    )


def read_single_dicom(file_path: Union[str, Path]) -> Optional[MammographyImage]:
    """Convenience function to read a single DICOM file."""
    reader = create_dicom_reader()
    return reader.read_dicom_file(file_path)


def read_dicom_directory(
    directory_path: Union[str, Path],
    recursive: bool = True,
    patient_level: bool = True,
) -> Dict[str, List[MammographyImage]]:
    """Convenience function to read all DICOM files from a directory."""
    reader = create_dicom_reader()
    return reader.read_dicom_directory(directory_path, recursive, patient_level)
