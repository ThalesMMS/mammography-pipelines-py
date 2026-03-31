# apps/density_classifier

## Purpose
Manual breast-density labeling and review UI. The package contains both a traditional desktop viewer
and a Streamlit implementation for browsing studies and updating classification data.

## Entry Points and Key Modules
- `main.py` launches the desktop-oriented flow with backup and filtering prompts.
- `streamlit_app.py` exposes the browser-based labeling experience.
- `mammography label-density` in `commands/label_density.py` is the CLI wrapper that points users
here.

### Key Files
- `data_manager.py`: Data and caching helpers for the density classifier desktop UI.
- `dicom_loader.py`: Background-friendly DICOM loader used by the density classifier UI.
- `main.py`: Launch the density classification UI with optional backup and filtering steps.
- `streamlit_app.py`: Streamlit UI to label breast density without OpenCV or subprocesses.
- `ui_viewer.py`: OpenCV-based viewer to quickly label breast density for each accession.
- `utils.py`: Small utility helpers used by the density classifier UI.

## How It Fits into the Pipeline
- Supports creation or correction of density labels before model training.
- Loads study metadata and DICOM images in a UI-friendly form while trying to keep image loading
responsive.
- Acts as a human validation layer upstream of `data/` and `training/` workflows.

## Inputs and Outputs
- Inputs: classification CSV data, study folders, DICOM files, and optional train-accession lists
used to focus review.
- Outputs: updated label CSV rows, backup copies of the source CSV, and reviewed study selections
for downstream training.

## Dependencies
- Internal: [`apps`](../README.md), [`commands`](../../commands/README.md),
[`data`](../../data/README.md), [`io`](../../io/README.md).
- External: `pandas`, `pydicom`, `streamlit`, `cv2`, `numpy`.

## Extension and Maintenance Notes
- Preserve CSV column expectations when changing save behavior so downstream loaders keep
recognizing the same labels and accession identifiers.
- Background or asynchronous DICOM loading belongs in the loader helpers; avoid moving heavy file
I/O into UI callback code.
- Always keep backup behavior intact before introducing bulk edit or cleanup features because this
app edits source labeling data.

## Related Directories
- [`apps`](../README.md): Umbrella package for operator-facing applications.
- [`commands`](../../commands/README.md): Internal command handlers behind the top-level
`mammography` CLI.
- [`data`](../../data/README.md): Source of truth for dataset ingestion.
- [`io`](../../io/README.md): Low-level image I/O helpers, especially for DICOM handling.
