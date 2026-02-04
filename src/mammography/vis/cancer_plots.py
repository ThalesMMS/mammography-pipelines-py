#
# cancer_plots.py
# mammography-pipelines
#
# Provides DICOM preprocessing visualization helpers for cancer detection pipeline.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Visualization utilities for DICOM preprocessing and model debugging.

WARNING: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

This module provides comprehensive visualization tools for inspecting DICOM
preprocessing steps, debugging image transformations, and previewing dataset
samples. It includes utilities for both matplotlib and lets-plot visualization.

Components:
    - HP: Hyperparameters for image preprocessing and visualization
    - dicom_debug_preprocess: Detailed DICOM preprocessing with intermediate outputs
    - get_dicom_plt: Create lets-plot figure from DICOM file
    - preview_transformed_samples: Display grid of dataset samples
    - get_transforms: Get model and visualization transform pipelines
    - robust_window: Percentile-based contrast windowing

Example usage:
    >>> from mammography.vis.cancer_plots import get_dicom_plt, preview_transformed_samples
    >>> from mammography.data.cancer_dataset import MammoDicomDataset
    >>>
    >>> # Visualize a single DICOM file
    >>> plot = get_dicom_plt(
    ...     dcm_path="path/to/mammogram.dcm",
    ...     title="Mammography Image"
    ... )
    >>> plot.show()
    >>>
    >>> # Preview dataset samples
    >>> dataset = MammoDicomDataset(...)
    >>> preview_transformed_samples(
    ...     dataset=dataset,
    ...     num_samples=8,
    ...     seed=42
    ... )
    >>>
    >>> # Debug preprocessing pipeline
    >>> from mammography.vis.cancer_plots import dicom_debug_preprocess
    >>> debug_info = dicom_debug_preprocess("path/to/mammogram.dcm")
    >>> print(debug_info["safe_header"])
    >>> print(debug_info["shape_raw"])
"""

import random
from typing import Dict, Optional, Tuple

# Optional import: lets-plot is only available for Python < 3.14
try:
    from lets_plot import (
        element_blank,
        element_text,
        geom_imshow,
        ggplot,
        labs,
        theme,
    )
    HAS_LETS_PLOT = True
except ImportError:
    HAS_LETS_PLOT = False

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid


class HP:
    """Core hyperparameters for DICOM preprocessing and visualization."""

    IMG_RESIZE: int = 256
    IMG_CROP: int = 224
    WINDOW_P_LOW: float = 0.5
    WINDOW_P_HIGH: float = 99.5


def _is_mono1(ds: "pydicom.dataset.FileDataset") -> bool:
    """Check whether the image uses MONOCHROME1 (inverted black/white)."""
    photometric = getattr(ds, "PhotometricInterpretation", "").upper()
    return photometric == "MONOCHROME1"


def _to_float32(arr: np.ndarray) -> np.ndarray:
    """Ensure float32 for numerically stable downstream math."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _apply_rescale(ds: "pydicom.dataset.FileDataset", arr: np.ndarray) -> np.ndarray:
    """Apply RescaleSlope/Intercept (or modality LUT) when present."""
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
    """Percentile-based windowing to normalize contrast while being outlier-robust."""
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return arr


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Return model/visualization transforms aligned with ResNet50 expectations."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_model = transforms.Compose([
        transforms.Resize(HP.IMG_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(HP.IMG_CROP),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    transform_vis = transforms.Compose([
        transforms.Resize(HP.IMG_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(HP.IMG_CROP),
    ])
    return transform_model, transform_vis


def dicom_debug_preprocess(dcm_path: str) -> Dict[str, object]:
    """Detailed preprocessing pipeline for debugging/visualization."""
    ds = pydicom.dcmread(dcm_path, force=True)
    arr = ds.pixel_array
    arr = _to_float32(arr)
    arr = _apply_rescale(ds, arr)
    if _is_mono1(ds):
        arr = arr.max() - arr
    arr_raw = arr.copy()

    lo_raw, hi_raw = float(arr_raw.min()), float(arr_raw.max())
    eps = 1e-6 if hi_raw - lo_raw == 0 else 0.0
    arr_raw_mm = (arr_raw - lo_raw) / (hi_raw - lo_raw + eps)
    raw_uint8 = (arr_raw_mm * 255.0).clip(0, 255).astype(np.uint8)

    arr_win = robust_window(arr, HP.WINDOW_P_LOW, HP.WINDOW_P_HIGH)
    win_uint8 = (arr_win * 255.0).clip(0, 255).astype(np.uint8)

    pil_raw = Image.fromarray(raw_uint8, mode="L")
    pil_win = Image.fromarray(win_uint8, mode="L")
    pil_raw_rgb = Image.merge("RGB", (pil_raw, pil_raw, pil_raw))
    pil_win_rgb = Image.merge("RGB", (pil_win, pil_win, pil_win))

    vis_tf = transforms.Compose([
        transforms.Resize(HP.IMG_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(HP.IMG_CROP),
    ])
    pil_resized = vis_tf(pil_win_rgb)

    safe_header = {}
    safe_keys = [
        "Manufacturer",
        "ManufacturerModelName",
        "PhotometricInterpretation",
        "Rows",
        "Columns",
        "BitsStored",
        "BitsAllocated",
        "HighBit",
        "PixelRepresentation",
        "RescaleIntercept",
        "RescaleSlope",
        "ViewPosition",
        "Laterality",
        "BodyPartExamined",
        "SeriesDescription",
        "SOPClassUID",
        "Modality",
    ]
    for k in safe_keys:
        if hasattr(ds, k):
            v = getattr(ds, k)
            try:
                safe_header[k] = str(v)
            except Exception:
                pass

    return {
        "raw_uint8": raw_uint8,
        "win_uint8": win_uint8,
        "pil_raw_rgb": pil_raw_rgb,
        "pil_win_rgb": pil_win_rgb,
        "pil_resized_rgb": pil_resized,
        "safe_header": safe_header,
        "shape_raw": [int(arr_raw.shape[0]), int(arr_raw.shape[1])],
    }


def get_dicom_plt(dcm_path: str, title: str):
    """Create a Lets-Plot figure mirroring the full ResNet50 preprocessing."""
    if not HAS_LETS_PLOT:
        raise ImportError(
            "lets-plot is required for get_dicom_plt() but not installed. "
            "Install it with: pip install lets-plot (Python < 3.14 only)"
        )

    dbg = dicom_debug_preprocess(dcm_path)
    img_rgb = np.array(dbg["pil_win_rgb"])

    plt = \
        ggplot() + \
        geom_imshow(img_rgb) + \
        theme(
            legend_position="none",
            panel_grid=element_blank(),
            axis=element_blank(),
            plot_title=element_text(hjust=0.5, face="bold")) + \
        labs(title=title)

    return plt


def preview_transformed_samples(dataset: Dataset, num_samples: int = 8, seed: Optional[int] = None) -> None:
    """Render a grid with `train_dataset` samples after applying `transform`."""

    if len(dataset) == 0:
        return

    rng = random.Random(seed) if seed is not None else random
    sample_indices = rng.sample(range(len(dataset)), k=min(num_samples, len(dataset)))

    imgs, labels = [], []
    for idx in sample_indices:
        img, label = dataset[idx]

        if img.ndim == 2:
            img = img.unsqueeze(0)

        imgs.append(img)
        labels.append(int(label.item()))

    grid_tensor = make_grid(torch.stack(imgs), nrow=min(4, len(imgs)), normalize=True)
    grid_np = grid_tensor.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 8))
    if grid_np.shape[-1] == 1:
        plt.imshow(grid_np[..., 0], cmap="gray")
    else:
        plt.imshow(grid_np)
    plt.title(f"Amostras transformadas (rótulos: {labels})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
