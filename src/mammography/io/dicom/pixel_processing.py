# DISCLAIMER: Educational project only - NOT for clinical or medical diagnostic purposes.
#
"""Pixel processing helpers for mammography DICOM images."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from threading import Lock
from typing import Iterator, Tuple

import numpy as np
import pydicom
try:
    from pydicom.pixels import apply_modality_lut
except ImportError:  # pragma: no cover - pydicom compatibility
    from pydicom.pixel_data_handlers.util import apply_modality_lut
from PIL import Image

logger = logging.getLogger(__name__)

_ds_patch_lock = Lock()
_ds_patch_count = 0
_ds_patch_original = None
_ds_patch_applied = False


def _make_tolerant_ds(original_ds):
    def tolerant_ds(val, auto_format=False, validation_mode=None):
        try:
            return original_ds(val, auto_format, validation_mode)
        except ValueError:
            if isinstance(val, str):
                return val
            raise

    tolerant_ds._mammography_tolerant = True
    return tolerant_ds


@contextmanager
def allow_invalid_decimal_strings_context() -> Iterator[None]:
    """Temporarily keep pydicom tolerant of malformed decimal strings."""
    global _ds_patch_count, _ds_patch_original, _ds_patch_applied

    with _ds_patch_lock:
        if _ds_patch_count == 0:
            _ds_patch_original = pydicom.valuerep.DS
            _ds_patch_applied = not getattr(
                _ds_patch_original, "_mammography_tolerant", False
            )
            if _ds_patch_applied:
                pydicom.valuerep.DS = _make_tolerant_ds(_ds_patch_original)
        _ds_patch_count += 1

    try:
        yield
    finally:
        with _ds_patch_lock:
            _ds_patch_count -= 1
            if _ds_patch_count == 0:
                if _ds_patch_applied and _ds_patch_original is not None:
                    pydicom.valuerep.DS = _ds_patch_original
                _ds_patch_original = None
                _ds_patch_applied = False


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
        return arr * float(slope) + float(intercept)
    except Exception as direct_exc:
        try:
            return apply_modality_lut(arr, ds)
        except Exception as lut_exc:
            ds_id = getattr(ds, "SOPInstanceUID", getattr(ds, "filename", "<unknown>"))
            logger.error(
                "Failed to apply DICOM rescale for %s (slope=%r, intercept=%r): "
                "direct=%r; modality_lut=%r",
                ds_id,
                slope,
                intercept,
                direct_exc,
                lut_exc,
            )
            raise ValueError(
                f"Failed to apply DICOM rescale for {ds_id} "
                f"(slope={slope!r}, intercept={intercept!r})."
            ) from lut_exc


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

def extract_window_parameters(
    ds: "pydicom.dataset.FileDataset", pixel_array: np.ndarray
) -> Tuple[float, float, str]:
    """
    Extract windowing parameters from DICOM dataset or calculate from pixel data.

    Args:
        ds: DICOM dataset object
        pixel_array: Pixel data array (after rescale slope/intercept if applicable)

    Returns:
        Tuple of (window_center, window_width, photometric_interpretation)
    """
    window_center = None
    window_width = None

    # Try to extract WindowCenter from DICOM tags
    if hasattr(ds, 'WindowCenter'):
        wc_val = ds.WindowCenter
        try:
            # Use the first value if multi-valued
            window_center = float(wc_val[0]) if isinstance(wc_val, pydicom.multival.MultiValue) else float(wc_val)
        except (ValueError, TypeError):
            logger.warning("Could not convert WindowCenter to float, will calculate from pixel data")
            window_center = None

    # Try to extract WindowWidth from DICOM tags
    if hasattr(ds, 'WindowWidth'):
        ww_val = ds.WindowWidth
        try:
            # Use the first value if multi-valued
            window_width = float(ww_val[0]) if isinstance(ww_val, pydicom.multival.MultiValue) else float(ww_val)
        except (ValueError, TypeError):
            logger.warning("Could not convert WindowWidth to float, will calculate from pixel data")
            window_width = None

    # If no windowing parameters are available, derive them from the pixel data
    if window_center is None or window_width is None:
        min_val = np.min(pixel_array)
        max_val = np.max(pixel_array)
        window_center = (max_val + min_val) / 2.0
        window_width = max_val - min_val
        if window_width <= 0:
            window_width = 1  # Avoid zero or negative WW

    # Extract photometric interpretation with default
    photometric_interpretation = "MONOCHROME2"
    if hasattr(ds, 'PhotometricInterpretation'):
        photometric_interpretation = ds.PhotometricInterpretation

    return window_center, window_width, photometric_interpretation

def apply_windowing(image: np.ndarray, wc: float, ww: float, photometric: str) -> np.ndarray:
    """Apply windowing and return a uint8 image."""
    img_min = wc - ww / 2.0
    img_max = wc + ww / 2.0

    windowed_image = np.clip(image, img_min, img_max)

    if img_max > img_min:
        windowed_image = (windowed_image - img_min) / (img_max - img_min)
    else:
        windowed_image = np.zeros_like(windowed_image)

    photometric_normalized = str(photometric or "").upper()
    if photometric_normalized == "MONOCHROME1":
        windowed_image = 1.0 - windowed_image

    return (windowed_image * 255.0).astype(np.uint8)

def dicom_to_pil_rgb(
    dcm_path: str, window_low: float = 0.5, window_high: float = 99.5
) -> Image.Image:
    """
    Read a DICOM and convert it to 8-bit RGB (channel-replicated grayscale).

    Args:
        dcm_path: Path to DICOM file
        window_low: Lower percentile for robust windowing (0-100)
        window_high: Upper percentile for robust windowing (0-100)

    Returns:
        PIL Image in RGB mode

    Raises:
        RuntimeError: If DICOM cannot be read or pixel data cannot be accessed

    Note:
        Requires NumPy for pixel_array access. For compressed DICOM files
        (JPEG, JPEG2000), additional codec libraries are required:
        - pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
    """
    try:
        with allow_invalid_decimal_strings_context():
            ds = pydicom.dcmread(dcm_path, force=True)
    except Exception as exc:
        raise RuntimeError(
            f"Falha ao ler pixel data de {dcm_path}. "
            f"Verifique se o arquivo é um DICOM válido. Erro: {exc!r}"
        ) from exc

    try:
        with allow_invalid_decimal_strings_context():
            arr = ds.pixel_array
    except AttributeError as exc:
        raise RuntimeError(
            f"Falha ao ler pixel data de {dcm_path}. "
            f"DICOM {dcm_path} não contém dados de pixel (PixelData ausente). "
            f"Erro: {exc!r}"
        ) from exc
    except ImportError as exc:
        # NumPy or codec library missing
        error_msg = str(exc).lower()
        if "numpy" in error_msg:
            raise RuntimeError(
                f"NumPy é necessário para acessar pixel_array. "
                f"Instale com: pip install numpy. "
                f"Arquivo: {dcm_path}. Erro: {exc!r}"
            ) from exc
        elif "jpeg" in error_msg or "pillow" in error_msg or "gdcm" in error_msg:
            raise RuntimeError(
                f"Codec ausente para DICOM comprimido {dcm_path}. "
                f"Para JPEG/JPEG2000, instale: "
                f"pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg. "
                f"Erro: {exc!r}"
            ) from exc
        else:
            raise RuntimeError(
                f"Falha ao acessar pixel_array de {dcm_path}. "
                f"Pode ser necessário instalar bibliotecas de codec adicionais. "
                f"Erro: {exc!r}"
            ) from exc
    except Exception as exc:
        # Catch-all for other pixel_array errors
        raise RuntimeError(
            f"Falha ao ler pixel data de {dcm_path}. "
            f"Verifique a sintaxe de transferência (Transfer Syntax) e "
            f"se os codecs necessários estão instalados. "
            f"Erro: {exc!r}"
        ) from exc

    arr = _to_float32(arr)
    arr = _apply_rescale(ds, arr)
    if _is_mono1(ds):
        arr = arr.max() - arr
    arr = robust_window(arr, window_low, window_high)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    return Image.merge("RGB", (pil, pil, pil))
