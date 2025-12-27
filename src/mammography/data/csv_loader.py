#
# csv_loader.py
# mammography-pipelines
#
# Loads dataset rows from CSVs, featureS.txt directories, or presets and normalizes labels/paths.
#
# Thales Matheus Mendonça Santos - November 2025
#
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Any, List, Sequence, Tuple, Dict
from ..utils.dicom_io import is_dicom_path, DICOM_EXTS

CACHE_AUTO_DISK_MAX = 6000
CACHE_AUTO_MEMORY_MAX = 1000
DATASET_PRESETS: Dict[str, Dict[str, Optional[str]]] = {
    "archive": {"csv": "classificacao.csv", "dicom_root": "archive"},
    "mamografias": {"csv": "mamografias", "dicom_root": None},
    "patches_completo": {"csv": "patches_completo", "dicom_root": None},
}

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
    """Normalize label inputs to integers in {1, 2, 3, 4} when possible."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
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
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return text or None

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
        return pd.DataFrame(rows)

    df = pd.read_csv(csv_path)

    # AccessionNumber + Classification format (DICOM root)
    if {"AccessionNumber", "Classification"}.issubset(df.columns):
        if not dicom_root:
            raise ValueError("dicom_root é obrigatório para CSV com AccessionNumber/Classification.")
        dicom_root = _find_best_data_dir(dicom_root)
        rows = []
        df["AccessionNumber"] = df["AccessionNumber"].astype(str).str.strip()
        for _, r in df.iterrows():
            lab = int(r["Classification"]) if pd.notna(r["Classification"]) else None
            if lab == 5 and exclude_class_5:
                continue
            acc = str(r.get("AccessionNumber", "")).strip()
            folder = os.path.join(dicom_root, acc)
            if not os.path.isdir(folder) and acc.isdigit():
                padded = acc.zfill(6)
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
        return pd.DataFrame(rows)

    # Generic format with direct path
    if "image_path" in df.columns:
        label_col_candidates = ["density_label", "label", "y", "professional_label"]
        lab_col = next((c for c in label_col_candidates if c in df.columns), None)
        if lab_col is None:
            df["professional_label"] = None
        else:
            df["professional_label"] = df[lab_col].apply(_coerce_density_label)

        if "AccessionNumber" in df.columns:
            df["accession"] = df["AccessionNumber"].astype(str)
        elif "accession" not in df.columns:
            df["accession"] = [os.path.basename(os.path.dirname(p)) for p in df["image_path"]]
        return df[["image_path", "professional_label", "accession"]]

    raise ValueError("CSV formato desconhecido ou faltando --dicom-root para CSV de classificação.")

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
