"""DICOM I/O utilities and data models for the mammography pipelines.

WARNING: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.
"""

from __future__ import annotations

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


from .metadata import (  # noqa: E402
    VALID_LATERALITY,
    VALID_PROJECTION_TYPES,
    VALID_STATES,
    VALID_TRANSITIONS,
    MammographyImage,
    create_mammography_image_from_dicom,
)
from .pixel_processing import (  # noqa: E402
    _apply_rescale,
    _is_mono1,
    _to_float32,
    allow_invalid_decimal_strings_context,
    apply_windowing,
    dicom_to_pil_rgb,
    extract_window_parameters,
    robust_window,
)
from .reader import (  # noqa: E402
    DicomReader,
    create_dicom_reader,
    load_dicom,
    read_dicom_directory,
    read_single_dicom,
)

__all__ = [
    "DICOM_EXTS",
    "RESEARCH_DISCLAIMER",
    "VALID_LATERALITY",
    "VALID_PROJECTION_TYPES",
    "VALID_STATES",
    "VALID_TRANSITIONS",
    "DicomReader",
    "MammographyImage",
    "_apply_rescale",
    "_is_mono1",
    "_to_float32",
    "allow_invalid_decimal_strings_context",
    "apply_windowing",
    "create_dicom_reader",
    "create_mammography_image_from_dicom",
    "dicom_to_pil_rgb",
    "extract_window_parameters",
    "get_disclaimer",
    "is_dicom_path",
    "load_dicom",
    "read_dicom_directory",
    "read_single_dicom",
    "robust_window",
]
