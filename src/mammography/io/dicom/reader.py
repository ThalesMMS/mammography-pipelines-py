# DISCLAIMER: Educational project only - NOT for clinical or medical diagnostic purposes.
#
"""DICOM reader and metadata-cache orchestration."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import pydicom

from .metadata import (
    DicomValidationError,
    MammographyImage,
    create_mammography_image_from_dicom,
)
from .pixel_processing import allow_invalid_decimal_strings_context

logger = logging.getLogger(__name__)


def _redact_path(path: Union[str, Path]) -> str:
    """Return a stable non-identifying token for a potentially PHI-bearing path."""
    path_text = str(path)
    digest = hashlib.sha256(path_text.encode("utf-8", errors="replace")).hexdigest()[
        :12
    ]
    suffix = Path(path).suffix.lower()
    return f"<dicom-path:{digest}{suffix}>"


def _redact_path_in_message(message: str, path: Union[str, Path]) -> str:
    """Replace known renderings of a path with its redacted token."""
    path_obj = Path(path)
    redacted = _redact_path(path_obj)
    variants = {str(path_obj), path_obj.as_posix(), path_obj.name}
    try:
        resolved = path_obj.resolve()
    except OSError:
        resolved = None
    if resolved is not None:
        variants.update({str(resolved), resolved.as_posix()})

    sanitized = message
    for variant in sorted((value for value in variants if value), key=len, reverse=True):
        sanitized = sanitized.replace(variant, redacted)
    return sanitized


class DicomReader:
    """
    DICOM file reader for mammography images with validation and metadata caching.

    cache_size=0 disables retention: _cache_metadata clears metadata_cache and
    skips adding entries whenever cache_size <= 0.
    """

    VALIDATION_ERROR_HISTORY_LIMIT = 1000

    # Deprecated: DICOM discovery is content-based; extensions are not used.
    SUPPORTED_EXTENSIONS: Tuple[str, ...] = ()

    REQUIRED_TAGS = (
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
    )

    OPTIONAL_TAGS = (
        "AcquisitionDate",
        "AcquisitionTime",
        "StudyDate",
        "StudyTime",
        "PatientAge",
        "PatientSex",
        "BodyPartExamined",
        "ImageType",
        "PhotometricInterpretation",
    )

    def __init__(
        self,
        validate_on_read: bool = True,
        cache_metadata: bool = True,
        max_workers: int = 4,
        lazy_load: bool = False,
        cache_size: Optional[int] = None,
    ):
        if (
            not isinstance(max_workers, int)
            or isinstance(max_workers, bool)
            or max_workers < 1
        ):
            raise ValueError("max_workers must be an integer greater than or equal to 1")
        if (
            cache_size is not None
            and (
                not isinstance(cache_size, int)
                or isinstance(cache_size, bool)
                or cache_size < 0
            )
        ):
            raise ValueError("cache_size must be None or a non-negative integer")

        self.validate_on_read = validate_on_read
        self.cache_metadata = cache_metadata
        self.max_workers = max_workers
        self.lazy_load = lazy_load
        self.cache_size = cache_size

        self.metadata_cache: Dict[str, Dict[str, Any]] = {}

        self.stats = {
            "files_processed": 0,
            "files_valid": 0,
            "files_invalid": 0,
            "validation_errors": deque(
                maxlen=self.VALIDATION_ERROR_HISTORY_LIMIT
            ),
        }
        self._lock = Lock()

        logger.info(
            "Initialized DicomReader with validate_on_read=%s, max_workers=%s, lazy_load=%s",
            validate_on_read,
            max_workers,
            lazy_load,
        )

    def _record_attempt(self) -> None:
        with self._lock:
            self.stats["files_processed"] += 1

    def _record_valid(self) -> None:
        with self._lock:
            self.stats["files_valid"] += 1

    def _record_invalid(self, errors: Union[str, List[str]]) -> None:
        with self._lock:
            self.stats["files_invalid"] += 1
            if isinstance(errors, str):
                self.stats["validation_errors"].append(errors)
            else:
                self.stats["validation_errors"].extend(errors)

    def _cache_metadata(self, file_path: Path, metadata: Dict[str, Any]) -> None:
        with self._lock:
            if self.cache_size is not None and self.cache_size <= 0:
                self.metadata_cache.clear()
                return
            self.metadata_cache[str(file_path)] = metadata
            if self.cache_size is not None:
                while len(self.metadata_cache) > self.cache_size:
                    oldest_key = next(iter(self.metadata_cache))
                    self.metadata_cache.pop(oldest_key, None)

    def read_dicom_file(
        self, file_path: Union[str, Path]
    ) -> Optional[MammographyImage]:
        file_path = Path(file_path)
        redacted_path = _redact_path(file_path)
        self._record_attempt()

        if not file_path.exists():
            error_msg = f"DICOM file not found: {redacted_path}"
            logger.error(error_msg)
            self._record_invalid(error_msg)
            # Missing files are caller errors; non-DICOM content below is
            # batch-filtering noise recorded via _record_invalid.
            raise FileNotFoundError(error_msg)

        try:
            with file_path.open("rb") as file_obj:
                if not self._is_dicom_file(file_obj):
                    error_msg = f"File is not DICOM content: {redacted_path}"
                    logger.warning(error_msg)
                    self._record_invalid(error_msg)
                    return None

                file_obj.seek(0)
                with allow_invalid_decimal_strings_context():
                    dataset = pydicom.dcmread(
                        file_obj,
                        stop_before_pixels=self.lazy_load,
                        force=True,
                    )

            mammography_image = create_mammography_image_from_dicom(
                str(file_path),
                dataset=dataset,
                validate_pixel_data=not self.lazy_load,
                validate=self.validate_on_read,
            )

            if self.cache_metadata:
                self._cache_metadata(file_path, mammography_image.get_metadata_dict())

            self._record_valid()

            logger.info("Successfully read DICOM file: %s", redacted_path)
            return mammography_image

        except DicomValidationError as exc:
            safe_errors = [
                _redact_path_in_message(str(error), file_path)
                for error in exc.errors
            ]
            logger.warning(
                "DICOM validation failed for %s: %s",
                redacted_path,
                safe_errors,
            )
            self._record_invalid(safe_errors)
            return None
        except Exception as exc:
            error_msg = (
                f"Error reading DICOM file {redacted_path}: "
                f"{_redact_path_in_message(str(exc), file_path)}"
            )
            logger.error(error_msg)
            self._record_invalid(error_msg)
            return None

    def read(self, file_path: Union[str, Path]) -> Optional[MammographyImage]:
        """Backward-compatible alias for ``read_dicom_file``."""
        return self.read_dicom_file(file_path)

    def read_dicom_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        patient_level: bool = True,
    ) -> Dict[str, List[MammographyImage]]:
        directory_path = Path(directory_path)
        redacted_directory_path = _redact_path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {redacted_directory_path}")

        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {redacted_directory_path}")

        dicom_files = self._find_dicom_files(directory_path, recursive)
        logger.info(
            "Found %s candidate files in %s",
            len(dicom_files),
            redacted_directory_path,
        )

        if not dicom_files:
            logger.warning("No files found in %s", redacted_directory_path)
            return {}

        mammography_images = self._read_dicom_files_parallel(dicom_files)
        valid_images = [img for img in mammography_images if img is not None]
        logger.info(
            "Successfully read %s out of %s candidate files as DICOM",
            len(valid_images),
            len(dicom_files),
        )

        if patient_level:
            return self._organize_by_patient(valid_images)

        return {img.file_path: [img] for img in valid_images}

    def _is_dicom_file(self, source: Any) -> bool:
        try:
            if isinstance(source, (bytes, bytearray)):
                header = bytes(source[:132])
            elif hasattr(source, "read"):
                original_position = source.tell()
                header = source.read(132)
                source.seek(original_position)
            else:
                file_path = Path(source)
                with file_path.open("rb") as file_obj:
                    header = file_obj.read(132)
        except Exception as exc:
            if isinstance(source, (str, Path)):
                file_path = Path(source)
                logger.debug(
                    "DICOM preamble check failed for %s: %s",
                    _redact_path(file_path),
                    _redact_path_in_message(str(exc), file_path),
                )
            else:
                logger.debug("DICOM preamble check failed: %s", exc)
            return False

        return len(header) >= 132 and header[128:132] == b"DICM"

    def _find_dicom_files(
        self, directory_path: Path, recursive: bool = True
    ) -> List[Path]:
        candidates = directory_path.rglob("*") if recursive else directory_path.glob("*")
        dicom_files = sorted(path for path in candidates if path.is_file())
        logger.debug(
            "Found %s candidate files for DICOM content checks",
            len(dicom_files),
        )
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
                    redacted_path = _redact_path(dicom_files[index])
                    logger.error(
                        "Error processing DICOM file %s: %s",
                        redacted_path,
                        _redact_path_in_message(str(exc), dicom_files[index]),
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

    def get_metadata_summary(self) -> Dict[str, Any]:
        with self._lock:
            metadata_values = list(self.metadata_cache.values())
            cache_size = len(self.metadata_cache)
            stats = {
                "files_processed": self.stats["files_processed"],
                "files_valid": self.stats["files_valid"],
                "files_invalid": self.stats["files_invalid"],
                "validation_errors": list(self.stats["validation_errors"]),
            }

        if not metadata_values:
            return {"total_files": 0, "message": "No metadata available"}

        manufacturers = set()
        projection_types = set()
        lateralities = set()
        pixel_spacings: List[Tuple[float, float]] = []
        bits_stored_values = set()

        for metadata in metadata_values:
            manufacturers.add(metadata.get("manufacturer", "Unknown"))
            projection_types.add(metadata.get("projection_type", "Unknown"))
            lateralities.add(metadata.get("laterality", "Unknown"))
            pixel_spacings.append(metadata.get("pixel_spacing", (0, 0)))
            bits_stored_values.add(metadata.get("bits_stored", 0))

        return {
            "total_files": cache_size,
            "manufacturers": list(manufacturers),
            "projection_types": list(projection_types),
            "lateralities": list(lateralities),
            "pixel_spacing_range": {
                "min": tuple(min(ps[i] for ps in pixel_spacings) for i in range(2)),
                "max": tuple(max(ps[i] for ps in pixel_spacings) for i in range(2)),
            },
            "bits_stored_values": list(bits_stored_values),
            "processing_stats": stats,
        }

    def clear_cache(self) -> None:
        with self._lock:
            self.metadata_cache.clear()
        logger.info("Cleared DICOM metadata cache")

    def get_processing_stats(self) -> Dict[str, Any]:
        with self._lock:
            files_processed = self.stats["files_processed"]
            files_valid = self.stats["files_valid"]
            files_invalid = self.stats["files_invalid"]
            validation_errors = list(self.stats["validation_errors"])

        return {
            "files_processed": files_processed,
            "files_valid": files_valid,
            "files_invalid": files_invalid,
            "success_rate": (
                files_valid / max(files_processed, 1)
            )
            * 100,
            "total_validation_errors": len(validation_errors),
            "recent_errors": validation_errors[-10:],
        }


def create_dicom_reader(
    validate_on_read: bool = True,
    cache_metadata: bool = True,
    max_workers: int = 4,
    lazy_load: bool = False,
    cache_size: Optional[int] = None,
) -> DicomReader:
    """Factory function to create a DicomReader instance."""
    return DicomReader(
        validate_on_read=validate_on_read,
        cache_metadata=cache_metadata,
        max_workers=max_workers,
        lazy_load=lazy_load,
        cache_size=cache_size,
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


def load_dicom(
    file_path: Union[str, Path],
    lazy_load: bool = True,
    validate: bool = True,
) -> Optional[MammographyImage]:
    """
    Load a DICOM file with optional lazy pixel access.

    This is the main entry point for DICOM loading with support for lazy pixel
    access, which defers loading pixel data until it's actually accessed,
    providing significant memory savings (50%+ reduction).

    Args:
        file_path: Path to the DICOM file
        lazy_load: If True, defer loading pixel data until accessed (default: True)
        validate: If True, validate DICOM metadata on read (default: True)

    Returns:
        MammographyImage instance or None if loading/validation fails

    Example:
        >>> img = load_dicom("path/to/mammo.dcm", lazy_load=True)
        >>> # Pixel data is not loaded yet, saving memory
        >>> if img:
        >>>     # Access pixel data when needed
        >>>     pixels = img.pixel_array
    """
    reader = create_dicom_reader(
        validate_on_read=validate,
        lazy_load=lazy_load,
    )
    return reader.read_dicom_file(file_path)
