#
# streamlit_app.py
# mammography-pipelines
#
# Streamlit UI for labeling breast density with in-process navigation and CSV persistence.
#
# Thales Matheus Mendonca Santos - November 2025
#
"""Streamlit UI to label breast density without OpenCV or subprocesses."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Sequence
from functools import lru_cache

import numpy as np
import pandas as pd
import pydicom

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None

from .dicom_loader import apply_windowing
from .utils import backup_classification_csv


LABELS = {
    1: "1 - Fatty",
    2: "2 - Mostly Fatty",
    3: "3 - Mostly Dense",
    4: "4 - Dense",
    5: "5 - Issue/Skip",
}


def _require_streamlit() -> None:
    if st is None:
        raise ImportError(
            "Streamlit is required to run the density labeling UI."
        ) from _STREAMLIT_IMPORT_ERROR


def _guess_project_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    for base in candidates:
        for parent in [base, *base.parents]:
            if (parent / "archive").is_dir():
                return parent
    return Path.cwd()


def _resolve_classification_path(project_root: Path) -> Path:
    primary = project_root / "classification.csv"
    legacy = project_root / "classificacao.csv"
    if primary.exists():
        return primary
    if legacy.exists():
        return legacy
    return primary


def _load_classification_df(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, dtype={"AccessionNumber": str})
        if "AccessionNumber" in df.columns:
            df.set_index("AccessionNumber", inplace=True)
        return df
    df = pd.DataFrame(columns=["Classification", "ClassificationDate"])
    df.index.name = "AccessionNumber"
    return df


def _save_classification(df: pd.DataFrame, path: Path, accession: str, label: int) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[accession] = {"Classification": label, "ClassificationDate": timestamp}
    df.to_csv(path)


def _load_train_accessions(archive_dir: Path) -> list[str]:
    train_csv = archive_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found at {train_csv}")
    df = pd.read_csv(train_csv, dtype={"AccessionNumber": str})
    if "AccessionNumber" not in df.columns:
        raise ValueError("train.csv missing AccessionNumber column")
    index = set(df["AccessionNumber"].astype(str))
    folders = [
        name
        for name in sorted(os.listdir(archive_dir))
        if (archive_dir / name).is_dir() and name in index
    ]
    return folders


def _get_dicom_files(archive_dir: Path, accession: str) -> list[str]:
    folder = archive_dir / accession
    if not folder.is_dir():
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".dcm")])


@lru_cache(maxsize=128)
def _load_dicom_image(path: str) -> tuple[np.ndarray | None, str | None]:
    try:
        ds = pydicom.dcmread(path, force=True)
        if not hasattr(ds, "PixelData"):
            return None, "DICOM missing PixelData"
        pixel_array = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            try:
                pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            except (TypeError, ValueError):
                pass
        window_center = None
        window_width = None
        if hasattr(ds, "WindowCenter"):
            wc_val = ds.WindowCenter
            window_center = float(wc_val[0]) if isinstance(wc_val, pydicom.multival.MultiValue) else float(wc_val)
        if hasattr(ds, "WindowWidth"):
            ww_val = ds.WindowWidth
            window_width = float(ww_val[0]) if isinstance(ww_val, pydicom.multival.MultiValue) else float(ww_val)
        if window_center is None or window_width is None:
            min_val, max_val = np.min(pixel_array), np.max(pixel_array)
            window_center = (max_val + min_val) / 2.0
            window_width = max_val - min_val
            if window_width <= 0:
                window_width = 1.0
        photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        image = apply_windowing(pixel_array, window_center, window_width, photometric)
        return image, None
    except Exception as exc:
        return None, str(exc)


def _ensure_session_defaults() -> None:
    if "project_root" not in st.session_state:
        st.session_state.project_root = str(_guess_project_root())
    if "show_only_unclassified" not in st.session_state:
        st.session_state.show_only_unclassified = True
    if "accessions" not in st.session_state:
        st.session_state.accessions = []
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "classification_df" not in st.session_state:
        st.session_state.classification_df = None
    if "classification_path" not in st.session_state:
        st.session_state.classification_path = None
    if "current_accession" not in st.session_state:
        st.session_state.current_accession = None


def _load_exams() -> None:
    project_root = Path(st.session_state.project_root)
    archive_dir = project_root / "archive"
    if not archive_dir.is_dir():
        st.error(f"Archive directory not found: {archive_dir}")
        return
    try:
        folders = _load_train_accessions(archive_dir)
    except Exception as exc:
        st.error(str(exc))
        return
    classification_path = _resolve_classification_path(project_root)
    classification_df = _load_classification_df(classification_path)
    if st.session_state.show_only_unclassified and not classification_df.empty:
        classified = set(classification_df.index.astype(str))
        folders = [acc for acc in folders if acc not in classified]
    st.session_state.accessions = folders
    st.session_state.current_index = 0
    st.session_state.classification_df = classification_df
    st.session_state.classification_path = str(classification_path)
    st.session_state.current_accession = None
    if folders:
        st.session_state.index_input = 1


def _set_index(new_index: int) -> None:
    st.session_state.current_index = new_index
    st.session_state.index_input = new_index + 1
    st.session_state.current_accession = None
    st.rerun()


def _apply_label(label: int) -> None:
    accessions: list[str] = st.session_state.accessions
    if not accessions:
        return
    index = st.session_state.current_index
    if index >= len(accessions):
        return
    accession = accessions[index]
    df = st.session_state.classification_df
    path = st.session_state.classification_path
    if df is None or path is None:
        st.error("Dataset not loaded.")
        return
    _save_classification(df, Path(path), accession, label)
    if st.session_state.show_only_unclassified:
        accessions.pop(index)
        if accessions:
            st.session_state.accessions = accessions
            if index >= len(accessions):
                index = len(accessions) - 1
            _set_index(index)
        else:
            st.session_state.accessions = accessions
            st.session_state.current_index = 0
            st.session_state.index_input = 0
            st.success("All exams have been labeled.")
    else:
        next_index = min(index + 1, len(accessions) - 1)
        _set_index(next_index)


def _render_exam(archive_dir: Path, accession: str) -> None:
    dicom_files = _get_dicom_files(archive_dir, accession)
    if not dicom_files:
        st.warning("No DICOM files found for this exam.")
        return
    if st.session_state.current_accession != accession:
        st.session_state.current_accession = accession
        st.session_state.dicom_file = dicom_files[0]
    dicom_file = st.selectbox(
        "DICOM file",
        dicom_files,
        index=dicom_files.index(st.session_state.dicom_file)
        if st.session_state.dicom_file in dicom_files
        else 0,
        key="dicom_file",
    )
    image, error = _load_dicom_image(str(archive_dir / accession / dicom_file))
    if error:
        st.error(f"Failed to load DICOM: {error}")
        return
    if image is None:
        st.warning("DICOM could not be rendered.")
        return
    st.image(image, caption=f"{accession} ({dicom_file})", clamp=True, use_container_width=True)


def main() -> None:
    _require_streamlit()
    st.set_page_config(page_title="Density Labeling", layout="wide")
    st.title("Breast Density Labeling")

    _ensure_session_defaults()

    with st.sidebar:
        st.header("Dataset")
        st.text_input("Project root", key="project_root")
        st.checkbox("Show only unclassified", key="show_only_unclassified")
        if st.button("Load exams"):
            _load_exams()
        if st.button("Backup classification CSV"):
            try:
                backup_classification_csv(st.session_state.project_root)
                st.success("Backup completed.")
            except Exception as exc:
                st.error(str(exc))

    accessions: list[str] = st.session_state.accessions
    if not accessions:
        st.info("Load exams to begin.")
        return

    total = len(accessions)
    index = st.session_state.current_index
    if index >= total:
        index = max(0, total - 1)
        st.session_state.current_index = index

    col_prev, col_next, col_jump = st.columns([1, 1, 2])
    if col_prev.button("Previous") and index > 0:
        _set_index(index - 1)
    if col_next.button("Next") and index < total - 1:
        _set_index(index + 1)
    jump_value = col_jump.number_input(
        "Exam index",
        min_value=1,
        max_value=total,
        value=index + 1,
        step=1,
        key="index_input",
    )
    if jump_value - 1 != index:
        _set_index(int(jump_value) - 1)

    accession = accessions[index]
    st.subheader(f"Exam {index + 1} of {total}")
    st.caption(f"Accession: {accession}")

    df = st.session_state.classification_df
    current_label = None
    if df is not None and accession in df.index:
        try:
            current_label = int(df.loc[accession, "Classification"])
        except Exception:
            current_label = None
    if current_label:
        st.info(f"Already labeled: {LABELS.get(current_label, str(current_label))}")

    class_cols = st.columns(5)
    for idx, label in enumerate(sorted(LABELS.keys())):
        if class_cols[idx].button(LABELS[label], key=f"label_{label}"):
            _apply_label(label)
            return

    project_root = Path(st.session_state.project_root)
    archive_dir = project_root / "archive"
    _render_exam(archive_dir, accession)


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
                "Streamlit CLI is required to launch the density labeling UI."
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
