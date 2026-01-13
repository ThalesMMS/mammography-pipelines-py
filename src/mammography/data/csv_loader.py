#
# csv_loader.py
# mammography-pipelines
#
# Loads dataset rows from CSVs, featureS.txt directories, or presets and normalizes labels/paths.
#
# Thales Matheus Mendonça Santos - November 2025
#
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pandera as pa

from ..io.dicom import DICOM_EXTS

CACHE_AUTO_DISK_MAX = 6000
CACHE_AUTO_MEMORY_MAX = 1000
DATASET_PRESETS: Dict[str, Dict[str, Optional[str]]] = {
    "archive": {"csv": "classificacao.csv", "dicom_root": "archive"},
    "mamografias": {"csv": "mamografias", "dicom_root": None},
    "patches_completo": {"csv": "patches_completo", "dicom_root": None},
}
ALLOWED_DENSITY_LABELS = (1, 2, 3, 4, 5)
VALID_IMAGE_EXTS = DICOM_EXTS + (".png", ".jpg", ".jpeg")

def _find_first_dicom(folder: str) -> Optional[str]:
    """Return the first DICOM path found under the given folder (depth-first)."""
    exts = (".dcm", ".dicom", ".DCM", ".DICOM")
    dicoms: List[str] = []
    for curr, _, files in os.walk(folder):
        for f in files:
            fp = os.path.join(curr, f)
            lower = f.lower()
            if fp.endswith(exts) or lower.endswith(".dcm") or lower.endswith(".dicom"):
                dicoms.append(fp)
    dicoms.sort()
    return dicoms[0] if dicoms else None

def _coerce_density_label(val: Any) -> Optional[int]:
    """Normalize label inputs to integers in {1, 2, 3, 4, 5} when possible."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, str):
        s = val.strip().upper()
        if s in {"A", "B", "C", "D"}:
            return {"A": 1, "B": 2, "C": 3, "D": 4}[s]
        try:
            ival = int(s)
            if ival in {0, 1, 2, 3}:
                return ival + 1
            return ival
        except Exception:
            return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    try:
        return int(val)
    except Exception:
        return None

def _normalize_accession(value: Any) -> Optional[str]:
    """Trim and normalize accession strings, returning None for empty values."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None

def _has_valid_image_ext(series: pd.Series) -> pd.Series:
    """Validate file extensions against the allowed imaging formats."""
    return series.astype(str).str.lower().str.endswith(VALID_IMAGE_EXTS)

def _accession_is_valid(series: pd.Series) -> pd.Series:
    """Ensure accession values are non-empty when provided."""
    return series.isna() | series.astype(str).str.strip().str.len().gt(0)

def _label_is_valid(value: Any) -> bool:
    """Check if a label can be coerced into an allowed density class."""
    if value is None or pd.isna(value):
        return True
    return _coerce_density_label(value) in ALLOWED_DENSITY_LABELS

def _derive_accession_from_path(path: str) -> str:
    """Fallback accession derived from the parent directory or filename."""
    path_obj = Path(path)
    parent = path_obj.parent.name
    return parent or path_obj.stem

IMAGE_PATH_CHECK = pa.Check(_has_valid_image_ext, name="image_path_ext")
ACCESSION_CHECK = pa.Check(_accession_is_valid, name="accession_non_empty")
LABEL_CHECK = pa.Check(_label_is_valid, element_wise=True, name="density_label_valid")

def _label_column(required: bool = False) -> pa.Column:
    return pa.Column(object, nullable=True, required=required, checks=LABEL_CHECK)

def _accession_column(required: bool = False, nullable: Optional[bool] = None) -> pa.Column:
    if nullable is None:
        nullable = not required
    return pa.Column(str, nullable=nullable, required=required, coerce=True, checks=ACCESSION_CHECK)

CLASSIFICATION_SCHEMA = pa.DataFrameSchema(
    {
        "AccessionNumber": _accession_column(required=True, nullable=False),
        "Classification": _label_column(required=True),
    },
    strict=False,
)

RAW_PATH_SCHEMA = pa.DataFrameSchema(
    {
        "image_path": pa.Column(
            str,
            nullable=False,
            coerce=True,
            checks=[
                pa.Check.str_length(min_value=1),
                IMAGE_PATH_CHECK,
            ],
        ),
        "professional_label": _label_column(),
        "density_label": _label_column(),
        "label": _label_column(),
        "y": _label_column(),
        "AccessionNumber": _accession_column(),
        "accession": _accession_column(),
    },
    strict=False,
)

DATASET_SCHEMA = pa.DataFrameSchema(
    {
        "image_path": pa.Column(
            str,
            nullable=False,
            coerce=True,
            checks=[
                pa.Check.str_length(min_value=1),
                IMAGE_PATH_CHECK,
            ],
        ),
        "professional_label": pa.Column(
            "Int64",
            nullable=True,
            coerce=True,
            checks=pa.Check.isin(ALLOWED_DENSITY_LABELS, ignore_na=True),
        ),
        "accession": _accession_column(required=True, nullable=False),
    },
    strict=False,
)

PANDERA_ERRORS = (pa.errors.SchemaError, pa.errors.SchemaErrors)

def _validate_schema(schema: pa.DataFrameSchema, df: pd.DataFrame, context: str) -> pd.DataFrame:
    try:
        return schema.validate(df, lazy=True)
    except PANDERA_ERRORS as exc:
        raise ValueError(f"Falha de validacao ({context}): {exc}") from exc

def _try_schema(schema: pa.DataFrameSchema, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Exception]]:
    try:
        return schema.validate(df, lazy=True), None
    except PANDERA_ERRORS as exc:
        return None, exc

def _find_best_data_dir(pref: Optional[str]) -> Optional[str]:
    """Try common typos so the CLI is more forgiving for frequently used paths."""
    if not pref:
        return pref
    if os.path.isdir(pref):
        return pref
    alt = pref.replace("archieve", "archive")
    if os.path.isdir(alt):
        return alt
    return pref

def _rows_from_features_dir(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    has_subfolders = any(p.is_dir() for p in root.iterdir() if not p.name.startswith("."))
    search_dirs = [p for p in root.iterdir() if p.is_dir()] if has_subfolders else [root]
    for folder in search_dirs:
        feat_path = folder / "featureS.txt"
        if not feat_path.exists():
            continue
        lines = [l.strip() for l in feat_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                break
            fname, cls_raw = lines[i], lines[i + 1]
            try:
                birads = int(cls_raw) + 1
            except Exception:
                continue
            if "(" in fname and " (" not in fname:
                fname = fname.replace("(", " (")
            if not fname.lower().endswith(".png"):
                fname = f"{fname}.png"
            full_path = folder / fname
            if not full_path.exists():
                continue
            rows.append(
                {
                    "image_path": str(full_path),
                    "professional_label": birads,
                    "accession": folder.name,
                }
            )
    if not rows:
        raise ValueError(f"Nenhuma imagem encontrada via featureS.txt em {root}")
    return rows

def resolve_paths_from_preset(csv_path: Optional[str], dataset: Optional[str], dicom_root: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Fill `csv_path` and `dicom_root` using the named preset when the user did not pass explicit paths."""
    if dataset and dataset in DATASET_PRESETS:
        preset = DATASET_PRESETS[dataset]
        csv_path = csv_path or preset.get("csv")
        if preset.get("dicom_root") and not dicom_root:
            dicom_root = preset.get("dicom_root")
    return csv_path, dicom_root

def load_dataset_dataframe(csv_path: Optional[str], dicom_root: Optional[str] = None, exclude_class_5: bool = True, dataset: Optional[str] = None) -> pd.DataFrame:
    """Load a canonical DataFrame from CSVs, featureS.txt directories, or known presets."""
    csv_path, dicom_root = resolve_paths_from_preset(csv_path, dataset, dicom_root)
    if not csv_path:
        raise ValueError("csv_path não definido; use --csv ou --dataset com preset válido.")

    # Directory with featureS.txt (mammograms/patches)
    if os.path.isdir(csv_path):
        rows = _rows_from_features_dir(Path(csv_path))
        return _validate_schema(DATASET_SCHEMA, pd.DataFrame(rows), "featureS.txt")

    df = pd.read_csv(csv_path)

    classification_df, classification_error = _try_schema(CLASSIFICATION_SCHEMA, df)
    if classification_df is not None:
        if not dicom_root:
            raise ValueError("dicom_root é obrigatório para CSV com AccessionNumber/Classification.")
        dicom_root = _find_best_data_dir(dicom_root)
        rows = []
        classification_df["AccessionNumber"] = classification_df["AccessionNumber"].apply(_normalize_accession)
        classification_df["Classification"] = classification_df["Classification"].apply(_coerce_density_label)
        for _, r in classification_df.iterrows():
            lab = r["Classification"]
            if lab == 5 and exclude_class_5:
                continue
            acc = r.get("AccessionNumber")
            if not acc:
                continue
            folder = os.path.join(dicom_root, acc)
            if not os.path.isdir(folder) and str(acc).isdigit():
                padded = str(acc).zfill(6)
                padded_folder = os.path.join(dicom_root, padded)
                if os.path.isdir(padded_folder):
                    acc = padded
                    folder = padded_folder
            if not os.path.isdir(folder):
                continue
            dcm = _find_first_dicom(folder)
            if dcm is None:
                continue
            rows.append({"accession": acc, "image_path": dcm, "professional_label": lab})
        return _validate_schema(DATASET_SCHEMA, pd.DataFrame(rows), "classificacao.csv")

    path_df, path_error = _try_schema(RAW_PATH_SCHEMA, df)
    if path_df is not None:
        label_col_candidates = ["density_label", "label", "y", "professional_label"]
        lab_col = next((c for c in label_col_candidates if c in path_df.columns), None)
        if lab_col is None:
            path_df["professional_label"] = None
        else:
            path_df["professional_label"] = path_df[lab_col].apply(_coerce_density_label)

        accession_source = None
        if "AccessionNumber" in path_df.columns:
            accession_source = path_df["AccessionNumber"].apply(_normalize_accession)
        elif "accession" in path_df.columns:
            accession_source = path_df["accession"].apply(_normalize_accession)
        fallback_accession = path_df["image_path"].apply(_derive_accession_from_path)
        if accession_source is None:
            accession_source = fallback_accession
        path_df["accession"] = accession_source.fillna(fallback_accession)
        keep_cols = ["image_path", "professional_label", "accession"]
        for extra_col in ("patient_id", "PatientID"):
            if extra_col in path_df.columns:
                keep_cols.append(extra_col)
        return _validate_schema(DATASET_SCHEMA, path_df[keep_cols], "image_path")

    detail = []
    if classification_error:
        detail.append(f"AccessionNumber/Classification: {classification_error}")
    if path_error:
        detail.append(f"image_path: {path_error}")
    detail_text = " | ".join(detail) if detail else "schema desconhecido."
    raise ValueError(
        "CSV formato desconhecido ou faltando --dicom-root para CSV de classificação. "
        f"Detalhes: {detail_text}"
    )

def resolve_dataset_cache_mode(requested_mode: str, rows_or_df: Sequence[Any]) -> str:
    """Pick a cache strategy based on dataset size and whether paths point to DICOM files."""
    mode = (requested_mode or "none").lower()
    if mode != "auto":
        return mode

    rows: List[Dict[str, Any]]
    if isinstance(rows_or_df, pd.DataFrame):
        rows = rows_or_df.to_dict("records")
    else:
        rows = list(rows_or_df)  # type: ignore

    if not rows:
        return "none"

    paths = [str(r.get("image_path")) for r in rows if r.get("image_path")]
    total = len(paths)
    if not paths:
        return "none"

    has_dicom = any(str(p).lower().endswith(DICOM_EXTS) for p in paths)
    if total <= CACHE_AUTO_MEMORY_MAX:
        return "memory"
    if has_dicom and total <= CACHE_AUTO_DISK_MAX:
        return "disk"
    if has_dicom and total > CACHE_AUTO_DISK_MAX:
        return "tensor-disk"
    return "none"
