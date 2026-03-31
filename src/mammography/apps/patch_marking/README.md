# apps/patch_marking

## Purpose
ROI annotation interface for patch-based workflows. It lets contributors browse studies, place
square regions of interest, and save the resulting metadata and image crops.

## Entry Points and Key Modules
- `main.py` launches the interactive desktop flow for ROI review and housekeeping.
- `streamlit_app.py` provides a browser-based annotation UI.
- `mammography label-patches` in `commands/label_patches.py` is the CLI wrapper that sends users
into this package.

### Key Files
- `data_manager.py`: Convenience wrapper around the archive folder to navigate studies and metadata.
- `dicom_loader.py`: Lightweight DICOM loader that extracts windowing hints for the patch UI.
- `main.py`: Entry point for the patch marking UI; guides backups, cleanup, and browsing.
- `roi_selector.py`: Utility class to compute square ROIs and keep them within image boundaries.
- `streamlit_app.py`: Streamlit UI to browse DICOMs and save ROI crops as PNGs.
- `test.py`: Tiny smoke test that prints the dimensions of a sample DICOM file.
- `ui_viewer.py`: Matplotlib-based viewer to browse DICOMs and save ROI crops as PNGs.
- `utils.py`: Backup and cleanup helpers for saved PNG patch exports and interactive patch-marking
workflows.

## How It Fits into the Pipeline
- Enables creation of patch datasets and localized annotations that can feed patch-level
experiments.
- Wraps study navigation, DICOM loading, ROI boundary logic, and export utilities into one operator
workflow.
- Sits upstream of patch preprocessing or downstream patch-model experiments rather than the core
density classifier loop.

## Inputs and Outputs
- Inputs: archive-style study folders, DICOM slices, annotation CSV state, and operator-selected ROI
centers.
- Outputs: saved ROI metadata, PNG patch crops, backup copies, and cleanup actions for previously
exported images.

## Dependencies
- Internal: [`apps`](../README.md), [`commands`](../../commands/README.md),
[`io`](../../io/README.md), [`preprocess`](../../preprocess/README.md).
- External: `pydicom`, `Pillow`, `matplotlib`, `streamlit`, `numpy`.

## Extension and Maintenance Notes
- `roi_selector.py` is the place to keep geometric boundary rules consistent; do not duplicate ROI
math across UIs.
- Because this package can remove or overwrite exported PNGs, be explicit about backup and cleanup
flows when changing file-management behavior.
- Keep annotation schema changes coordinated with any downstream code that reads generated patch
metadata.

## Related Directories
- [`apps`](../README.md): Umbrella package for operator-facing applications.
- [`commands`](../../commands/README.md): Internal command handlers behind the top-level
`mammography` CLI.
- [`io`](../../io/README.md): Low-level image I/O helpers, especially for DICOM handling.
- [`preprocess`](../../preprocess/README.md): Image preprocessing abstractions for mammography data.
