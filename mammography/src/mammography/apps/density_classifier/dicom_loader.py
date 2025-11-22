#
# dicom_loader.py
# mammography-pipelines-py
#
# Background-friendly DICOM loader that prepares uint8 windowed images for the density classifier UI.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Background-friendly DICOM loader used by the density classifier UI."""

import pydicom
import numpy as np
import os

def load_dicom_task(dicom_path, accession_number, shared_buffer):
    """
    Worker function that fully loads and processes a DICOM, placing a ready-to-display
    uint8 array in the shared buffer.
    """
    final_image = None
    try:
        ds = pydicom.dcmread(dicom_path, force=True)
        
        if hasattr(ds, 'PixelData'):
            pixel_array = ds.pixel_array.astype(np.float32)

            # Apply Rescale Slope and Intercept if present
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                if slope != 1.0 or intercept != 0.0:
                    pixel_array = pixel_array * slope + intercept

            # Determine windowing parameters
            window_center, window_width = None, None
            if hasattr(ds, 'WindowCenter'):
                wc_val = ds.WindowCenter
                window_center = float(wc_val[0]) if isinstance(wc_val, pydicom.multival.MultiValue) else float(wc_val)
            if hasattr(ds, 'WindowWidth'):
                ww_val = ds.WindowWidth
                window_width = float(ww_val[0]) if isinstance(ww_val, pydicom.multival.MultiValue) else float(ww_val)

            # If no windowing parameters are available, derive them from the pixel data
            if window_center is None or window_width is None:
                min_val, max_val = np.min(pixel_array), np.max(pixel_array)
                window_center = (max_val + min_val) / 2.0
                window_width = max_val - min_val
                if window_width <= 0: window_width = 1

            photometric = ds.PhotometricInterpretation if hasattr(ds, 'PhotometricInterpretation') else "MONOCHROME2"
            
            # Perform final image processing in the worker process
            final_image = apply_windowing(pixel_array, window_center, window_width, photometric)

    except Exception as e:
        print(f"[Process-{os.getpid()}] Error loading {os.path.basename(dicom_path)}: {e}")
        final_image = None

    # Store the final uint8 image (or None) in the shared dictionary
    shared_buffer[accession_number] = final_image

def apply_windowing(image: np.ndarray, wc: float, ww: float, photometric: str) -> np.ndarray:
    """Apply windowing and return a uint8 image."""
    img_min = wc - ww / 2.0
    img_max = wc + ww / 2.0
    
    windowed_image = np.clip(image, img_min, img_max)

    if img_max > img_min:
        windowed_image = (windowed_image - img_min) / (img_max - img_min)
    else:
        windowed_image = np.zeros_like(windowed_image)

    if photometric == "MONOCHROME1":
        windowed_image = 1.0 - windowed_image
        
    return (windowed_image * 255.0).astype(np.uint8)
