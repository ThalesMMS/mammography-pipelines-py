#
# streamlit_app.py
# mammography-pipelines
#
# Streamlit UI for patch marking without desktop UI dependencies.
#
# Thales Matheus Mendonca Santos - November 2025
#
"""Streamlit UI to browse DICOMs and save ROI crops as PNGs."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None

from .data_manager import DataManager
from .dicom_loader import DicomImageLoader, apply_windowing
from .roi_selector import RoiSelector
from .utils import backup_pngs, cleanup_pngs


def _require_streamlit() -> None:
    if st is None:
        raise ImportError(
            "Streamlit is required to run the patch marking UI."
        ) from _STREAMLIT_IMPORT_ERROR


def _guess_project_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    for base in candidates:
        for parent in [base, *base.parents]:
            if (parent / "archive").is_dir():
                return parent
    return Path.cwd()


def _ensure_annotations_csv(project_root: Path) -> Path:
    path = project_root / "annotations.csv"
    if not path.exists():
        header = [
            "AccessionNumber",
            "DCM_Filename",
            "Adjusted_ROI_Center_X",
            "Adjusted_ROI_Center_Y",
            "ROI_Size",
            "Saved_PNG_Filename",
        ]
        path.write_text(",".join(header) + "\n", encoding="utf-8")
    return path


def _draw_roi_overlay(image: np.ndarray, bounds: tuple[int, int, int, int] | None) -> np.ndarray:
    if bounds is None:
        return image
    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    xmin, ymin, xmax, ymax = bounds
    draw.rectangle([xmin, ymin, xmax, ymax], outline="lime", width=3)
    return np.array(img)


def _save_roi(
    project_root: Path,
    folder_path: Path,
    dicom_filename: str,
    pixel_data: np.ndarray,
    view_params: dict,
    roi_selector: RoiSelector,
) -> tuple[bool, str]:
    if not roi_selector.is_defined():
        return False, "ROI is not defined."
    roi_float = roi_selector.extract_roi_from_image(pixel_data)
    if roi_float is None:
        return False, "Failed to extract ROI."
    roi_uint8 = apply_windowing(
        roi_float,
        view_params["wc"],
        view_params["ww"],
        view_params["photometric"],
    )
    base_name = Path(dicom_filename).stem
    output_path = None
    png_filename = None
    for idx in range(0, 1001):
        suffix = "" if idx == 0 else f"_{idx}"
        candidate = f"{base_name}{suffix}.png"
        candidate_path = folder_path / candidate
        if not candidate_path.exists():
            output_path = candidate_path
            png_filename = candidate
            break
    if output_path is None or png_filename is None:
        return False, "Too many PNG files for this DICOM."
    Image.fromarray(roi_uint8).save(output_path)
    annotations_path = _ensure_annotations_csv(project_root)
    xmin, ymin, xmax, ymax = roi_selector.current_roi_bounds or (0, 0, 0, 0)
    adj_center_x = (xmin + xmax) / 2.0
    adj_center_y = (ymin + ymax) / 2.0
    row = [
        folder_path.name,
        dicom_filename,
        adj_center_x,
        adj_center_y,
        roi_selector.roi_size,
        png_filename,
    ]
    with open(annotations_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)
    return True, f"Saved {png_filename}"


def _ensure_session_defaults() -> None:
    if "project_root" not in st.session_state:
        st.session_state.project_root = str(_guess_project_root())
    if "target_filter" not in st.session_state:
        st.session_state.target_filter = "all"
    if "skip_pngs" not in st.session_state:
        st.session_state.skip_pngs = False
    if "roi_size" not in st.session_state:
        st.session_state.roi_size = 448
    if "dm" not in st.session_state:
        st.session_state.dm = None
    if "current_accession" not in st.session_state:
        st.session_state.current_accession = None
    if "dicom_file" not in st.session_state:
        st.session_state.dicom_file = None
    if "roi_image_shape" not in st.session_state:
        st.session_state.roi_image_shape = None


def _load_studies() -> None:
    project_root = Path(st.session_state.project_root)
    archive_dir = project_root / "archive"
    if not archive_dir.is_dir():
        st.error(f"Archive directory not found: {archive_dir}")
        return
    try:
        dm = DataManager(str(archive_dir))
    except Exception as exc:
        st.error(str(exc))
        return
    target_filter = st.session_state.target_filter
    target_value = 0 if target_filter == "target0" else None
    dm.filter_folders(
        only_folders_without_pngs=st.session_state.skip_pngs,
        target_value_filter=target_value,
    )
    st.session_state.dm = dm
    st.session_state.current_accession = None
    st.session_state.dicom_file = None
    st.session_state.roi_image_shape = None
    st.session_state.index_input = dm.get_current_folder_index_display() or 1


def _reset_roi_center(height: int, width: int) -> None:
    st.session_state.roi_center_x = width // 2
    st.session_state.roi_center_y = height // 2
    st.session_state.roi_image_shape = (height, width)


def main() -> None:
    _require_streamlit()
    st.set_page_config(page_title="Patch Marking", layout="wide")
    st.title("Patch Marking")

    _ensure_session_defaults()

    with st.sidebar:
        st.header("Dataset")
        st.text_input("Project root", key="project_root")
        st.radio(
            "Target filter",
            options=["all", "target0"],
            format_func=lambda v: "All targets" if v == "all" else "Only target = 0",
            key="target_filter",
        )
        st.checkbox("Skip folders with existing PNGs", key="skip_pngs")
        if st.button("Load studies"):
            _load_studies()
        st.divider()
        st.header("Backup & Cleanup")
        if st.button("Backup PNGs"):
            project_root = Path(st.session_state.project_root)
            archive_dir = project_root / "archive"
            try:
                dm = st.session_state.dm or DataManager(str(archive_dir))
                backup_pngs(str(archive_dir), dm._all_valid_patient_folders, str(project_root))
                st.success("Backup completed.")
            except Exception as exc:
                st.error(str(exc))
        confirm_cleanup = st.checkbox("Confirm cleanup (destructive)")
        if st.button("Cleanup PNGs") and confirm_cleanup:
            project_root = Path(st.session_state.project_root)
            archive_dir = project_root / "archive"
            try:
                dm = st.session_state.dm or DataManager(str(archive_dir))
                cleanup_pngs(
                    str(archive_dir),
                    dm._all_valid_patient_folders,
                    str(project_root),
                    confirm=True,
                )
                st.success("Cleanup finished.")
            except Exception as exc:
                st.error(str(exc))

    dm: DataManager | None = st.session_state.dm
    if dm is None or dm.get_total_navigable_folders() == 0:
        st.info("Load studies to begin.")
        return

    total = dm.get_total_navigable_folders()
    current_index = dm.get_current_folder_index_display()
    if current_index == 0:
        st.warning("No folders match the selected filters.")
        return

    col_prev, col_next, col_jump = st.columns([1, 1, 2])
    if col_prev.button("Previous") and dm.move_to_previous_folder():
        st.session_state.current_accession = None
        st.session_state.dicom_file = None
        st.session_state.roi_image_shape = None
        st.rerun()
    if col_next.button("Next") and dm.move_to_next_folder():
        st.session_state.current_accession = None
        st.session_state.dicom_file = None
        st.session_state.roi_image_shape = None
        st.rerun()
    jump_value = col_jump.number_input(
        "Study index",
        min_value=1,
        max_value=total,
        value=current_index,
        step=1,
        key="index_input",
    )
    if jump_value != current_index:
        dm.current_folder_index = int(jump_value) - 1
        st.session_state.current_accession = None
        st.session_state.dicom_file = None
        st.session_state.roi_image_shape = None
        st.rerun()

    details = dm.get_current_folder_details()
    if not details:
        st.warning("No folder details available.")
        return

    accession = details["accession_number"]
    st.subheader(f"Study {current_index} of {total}")
    st.caption(
        f"Accession: {accession} | Target: {details.get('target')} | "
        f"Laterality: {details.get('laterality')} | Patient: {details.get('patient_id')}"
    )

    dicom_files = details.get("dicom_files", [])
    if not dicom_files:
        st.warning("No DICOM files found in this folder.")
        return

    if st.session_state.current_accession != accession:
        st.session_state.current_accession = accession
        st.session_state.dicom_file = dicom_files[0]
        st.session_state.roi_image_shape = None

    dicom_file = st.selectbox(
        "DICOM file",
        dicom_files,
        index=dicom_files.index(st.session_state.dicom_file)
        if st.session_state.dicom_file in dicom_files
        else 0,
        key="dicom_file",
    )

    folder_path = Path(details["folder_path"])
    dicom_path = folder_path / dicom_file

    if "dicom_loader" not in st.session_state:
        st.session_state.dicom_loader = DicomImageLoader()
    loader: DicomImageLoader = st.session_state.dicom_loader
    pixel_data, view_params = loader.load_dicom_data(str(dicom_path))
    if pixel_data is None or view_params is None:
        st.error("Failed to load DICOM data.")
        return

    image_uint8 = apply_windowing(
        pixel_data,
        view_params["wc"],
        view_params["ww"],
        view_params["photometric"],
    )

    height, width = image_uint8.shape[:2]
    if st.session_state.roi_image_shape != (height, width):
        _reset_roi_center(height, width)

    col_image, col_controls = st.columns([3, 2])
    roi_size = col_controls.number_input(
        "ROI size",
        min_value=64,
        max_value=1024,
        value=int(st.session_state.roi_size),
        step=16,
        key="roi_size",
    )
    center_x = col_controls.number_input(
        "Center X",
        min_value=0,
        max_value=width - 1,
        value=int(st.session_state.roi_center_x),
        step=1,
        key="roi_center_x",
    )
    center_y = col_controls.number_input(
        "Center Y",
        min_value=0,
        max_value=height - 1,
        value=int(st.session_state.roi_center_y),
        step=1,
        key="roi_center_y",
    )

    roi_selector = RoiSelector(roi_size=int(roi_size))
    roi_selector.set_center(int(center_x), int(center_y), height, width)
    overlay = _draw_roi_overlay(image_uint8, roi_selector.current_roi_bounds)
    col_image.image(overlay, caption="DICOM with ROI overlay", clamp=True, use_container_width=True)

    roi_preview = roi_selector.extract_roi_from_image(pixel_data)
    if roi_preview is not None:
        roi_preview_uint8 = apply_windowing(
            roi_preview,
            view_params["wc"],
            view_params["ww"],
            view_params["photometric"],
        )
        col_controls.image(roi_preview_uint8, caption="ROI preview", clamp=True)

    if col_controls.button("Save ROI"):
        success, message = _save_roi(
            Path(st.session_state.project_root),
            folder_path,
            dicom_file,
            pixel_data,
            view_params,
            roi_selector,
        )
        if success:
            st.success(message)
        else:
            st.error(message)

    png_files = dm.get_png_files(accession)
    if png_files:
        with st.expander("Existing PNGs", expanded=False):
            st.write(png_files)


def run(argv: Sequence[str] | None = None) -> int:
    _require_streamlit()
    script_path = Path(__file__).resolve()
    args = list(argv) if argv else []
    try:
        from streamlit.web import cli as stcli
    except Exception:
        try:
            from streamlit.web import bootstrap
        except Exception as exc:  # pragma: no cover - optional UI dependency
            raise ImportError(
                "Streamlit CLI is required to launch the patch marking UI."
            ) from exc
        try:
            bootstrap.run(str(script_path), "", args, {})
        except SystemExit as exc:
            return int(exc.code) if exc.code else 0
        return 0
    saved_argv = sys.argv[:]
    sys.argv = ["streamlit", "run", str(script_path), *args]
    try:
        stcli.main()
    except SystemExit as exc:
        return int(exc.code) if exc.code else 0
    finally:
        sys.argv = saved_argv
    return 0


if __name__ == "__main__":
    main()
