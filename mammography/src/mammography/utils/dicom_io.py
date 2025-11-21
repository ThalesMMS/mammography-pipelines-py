import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from PIL import Image

DICOM_EXTS = (".dcm", ".dicom")

def is_dicom_path(path: str) -> bool:
    """Checa extensão contra `DICOM_EXTS`."""
    return str(path).lower().endswith(DICOM_EXTS)

def _is_mono1(ds: "pydicom.dataset.FileDataset") -> bool:
    """Detecta fotometria MONOCHROME1 (preto=alto, branco=baixo)."""
    return getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1"

def _to_float32(arr: np.ndarray) -> np.ndarray:
    """Garante dtype float32."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr

def _apply_rescale(ds: "pydicom.dataset.FileDataset", arr: np.ndarray) -> np.ndarray:
    """Aplica RescaleSlope/Intercept."""
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
    """Windowing por percentis."""
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return arr

def dicom_to_pil_rgb(dcm_path: str, window_low: float = 0.5, window_high: float = 99.5) -> Image.Image:
    """Lê DICOM e converte para RGB 8-bit (3 canais replicados)."""
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
