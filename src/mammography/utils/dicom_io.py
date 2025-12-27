#
# dicom_io.py
# mammography-pipelines
#
# DICOM utilities for detecting file types, applying rescale/windowing, and returning RGB PIL images.
#
# Thales Matheus Mendonça Santos - November 2025
#
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from PIL import Image

DICOM_EXTS = (".dcm", ".dicom")

def is_dicom_path(path: str) -> bool:
    """Cheap extension-based check against `DICOM_EXTS`."""
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

def robust_window(arr: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Percentile-based windowing that is robust to stray high-intensity pixels."""
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return arr

def dicom_to_pil_rgb(dcm_path: str, window_low: float = 0.5, window_high: float = 99.5) -> Image.Image:
    """Read a DICOM and convert it to 8-bit RGB (channel-replicated grayscale)."""
    try:
        ds = pydicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array
    except Exception as e:
        raise RuntimeError(
            f"Falha ao ler pixel data de {dcm_path}. "
            f"Erro: {repr(e)}"
        )
    arr = _to_float32(arr)
    arr = _apply_rescale(ds, arr)
    if _is_mono1(ds):
        arr = arr.max() - arr
    arr = robust_window(arr, window_low, window_high)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    # Replica canais para RGB
    pil_rgb = Image.merge("RGB", (pil, pil, pil))
    return pil_rgb
