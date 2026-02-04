#
# dicom_loader.py
# mammography-pipelines
#
# Lightweight DICOM loader that extracts windowing hints and photometric metadata for the patch UI.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Lightweight DICOM loader that extracts windowing hints for the patch UI."""

import pydicom
import numpy as np
import os

# Import centralized DICOM utilities
from mammography.io.dicom import apply_windowing, extract_window_parameters

class DicomImageLoader:
    """Load DICOM pixel data and derive window/photometric metadata for display."""
    def __init__(self):
        pass  # No complex initialization required for now

    def load_dicom_data(self, dicom_path: str) -> tuple[np.ndarray | None, dict | None]:
        """
        Load a DICOM file, apply rescale slope/intercept, and determine
        windowing parameters.

        Args:
            dicom_path (str): Path to the .dcm file.

        Returns:
            tuple[np.ndarray | None, dict | None]: 
             - Pixel array after rescale (float32), or None on failure.
             - A dictionary with visualization parameters {'wc', 'ww', 'photometric', 'source'},
               or None on failure. 'source' indicates whether WC/WW came from the 'DICOM' or were 'Calculated'.
        """
        try:
            ds = pydicom.dcmread(dicom_path, force=True) 
            
            if not hasattr(ds, 'PixelData'):
                print(f"Error: DICOM file '{os.path.basename(dicom_path)}' was read with force=True but does not contain PixelData.")
                return None, None
                 
            pixel_array = ds.pixel_array.astype(np.float32) 

            # 1. Apply Rescale Slope and Intercept
            rescale_slope = 1.0
            rescale_intercept = 0.0
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                # Ensure values are treated as floats
                try:
                    rescale_slope = float(ds.RescaleSlope)
                    rescale_intercept = float(ds.RescaleIntercept)
                    pixel_array = pixel_array * rescale_slope + rescale_intercept
                except (ValueError, TypeError):
                    print(
                        f"Warning: Could not convert RescaleSlope/Intercept to float for '{os.path.basename(dicom_path)}'. Using default 1.0/0.0."
                    )
                    # Keep pixel_array unchanged if conversion fails

            # 2. Determine windowing parameters using centralized utility
            # Determine source before extraction for proper tracking
            has_dicom_window = hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth')

            # Use centralized extraction function
            window_center, window_width, photometric_interpretation = extract_window_parameters(ds, pixel_array)

            # Determine if values came from DICOM tags or were calculated
            source = "DICOM" if has_dicom_window else "Calculated"
            if not has_dicom_window:
                print(f"Info: Calculating WindowCenter/WindowWidth from min/max for '{os.path.basename(dicom_path)}'.")

            view_params = {
                "wc": window_center,
                "ww": window_width,
                "photometric": photometric_interpretation,
                "source": source 
            }
            
            return pixel_array, view_params

        except FileNotFoundError:
            print(f"Error: DICOM file not found at '{dicom_path}'")
            return None, None
        except Exception as e:
            print(f"Error loading or processing DICOM file '{os.path.basename(dicom_path)}': {e}")
            # import traceback
            # traceback.print_exc() 
            return None, None


# Example usage (for manual testing only):
if __name__ == '__main__':
    project_root = os.path.dirname(os.getcwd()) 
    archive_folder_path = os.path.join(project_root, "archive")
    
    test_dicom_path = None
    # Block to locate a test file (same logic as before)
    try:
        from data_manager import DataManager 
        dm = DataManager(archive_folder_path)
        valid_folders = dm._all_valid_patient_folders 
        if valid_folders:
            first_folder = valid_folders[0]
            dicoms_in_first_folder = dm.get_dicom_files(first_folder)
            if dicoms_in_first_folder:
                test_dicom_path = os.path.join(archive_folder_path, first_folder, dicoms_in_first_folder[0])
                print(f"Using DICOM file for testing: {test_dicom_path}")
            else:
                print(f"No DICOM files found in the test folder: {first_folder}")
        else:
            print("No valid folders found by DataManager to obtain a test DICOM file.")
    except Exception as e:
        print(f"Unable to use DataManager to locate a test DICOM file: {e}")

    if test_dicom_path and os.path.exists(test_dicom_path):
        loader = DicomImageLoader()
        pixel_data_float, view_params = loader.load_dicom_data(test_dicom_path)

        if pixel_data_float is not None and view_params is not None:
            print(f"DICOM data loaded. Shape: {pixel_data_float.shape}, Type: {pixel_data_float.dtype}")
            print(f"Raw data min/max (after rescale): {np.min(pixel_data_float):.2f}, {np.max(pixel_data_float):.2f}")
            print("Determined visualization parameters:")
            print(f"  Window Center (WC): {view_params['wc']:.2f}")
            print(f"  Window Width (WW): {view_params['ww']:.2f}")
            print(f"  Photometric Int.: {view_params['photometric']}")
            print(f"  WC/WW Source: {view_params['source']}") 

            # --- Display the float image directly ---
            print("\nDisplaying float image directly with matplotlib (using vmin/vmax)...")

            # Compute vmin/vmax from WC/WW
            wc = view_params['wc']
            ww = view_params['ww']
            vmin = wc - ww / 2.0
            vmax = wc + ww / 2.0

            # Pick the appropriate colormap (inverted for MONOCHROME1)
            cmap_to_use = 'gray'
            if view_params['photometric'] == 'MONOCHROME1':
                cmap_to_use = 'gray_r'  # '_r' denotes reversed colormap

            try:
                import matplotlib.pyplot as plt
                # Pass the float array directly along with vmin, vmax, and cmap
                plt.imshow(pixel_data_float, cmap=cmap_to_use, vmin=vmin, vmax=vmax) 
                plt.title(f"DICOM Float: {os.path.basename(test_dicom_path)} (WC/WW: {view_params['source']})")
                plt.colorbar()  # Colorbar now shows the original float values (post-rescale)
                plt.show()
            except ImportError:
                print("Matplotlib is not installed. Unable to display the test image.")
            except Exception as e:
                print(f"Error displaying image with Matplotlib: {e}")
            # --- End of float display section ---

        else:
            print("Failed to load DICOM data.")
    # Remaining error-handling block follows the same pattern as above
    elif test_dicom_path:
        print(f"Specified test path but DICOM file not found: {test_dicom_path}")
    else:
        print("Test DICOM path not defined or DataManager could not provide one.")
