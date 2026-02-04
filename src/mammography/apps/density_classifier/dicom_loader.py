#
# dicom_loader.py
# mammography-pipelines
#
# Background-friendly DICOM loader that prepares uint8 windowed images for the density classifier UI.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Background-friendly DICOM loader used by the density classifier UI."""

import os

import numpy as np
import pydicom

from mammography.io.dicom import apply_windowing, extract_window_parameters

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

            # Extract windowing parameters using centralized function
            window_center, window_width, photometric = extract_window_parameters(ds, pixel_array)

            # Perform final image processing in the worker process
            final_image = apply_windowing(pixel_array, window_center, window_width, photometric)

    except Exception as e:
        print(f"[Process-{os.getpid()}] Error loading {os.path.basename(dicom_path)}: {e}")
        final_image = None

    # Store the final uint8 image (or None) in the shared dictionary
    shared_buffer[accession_number] = final_image
