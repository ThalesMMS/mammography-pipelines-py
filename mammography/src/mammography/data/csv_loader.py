import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Any, List
from ..utils.dicom_io import is_dicom_path

CACHE_AUTO_DISK_MAX = 6000
CACHE_AUTO_MEMORY_MAX = 1000

def _find_first_dicom(folder: str) -> Optional[str]:
    """Retorna o primeiro DICOM encontrado na pasta."""
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
    """Converte rótulos para {1,2,3,4}."""
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
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return text or None

def load_dataset_dataframe(csv_path: str, dicom_root: Optional[str] = None, exclude_class_5: bool = True) -> pd.DataFrame:
    """Carrega DataFrame a partir de CSV de classificação ou caminhos."""
    df = pd.read_csv(csv_path)
    
    # Verifica formato
    if "Classification" in df.columns and dicom_root:
        # Formato 1: AccessionNumber, Classification
        # Requer dicom_root para encontrar imagens
        rows = []
        for _, r in df.iterrows():
            lab = int(r["Classification"]) if pd.notna(r["Classification"]) else None
            if lab == 5 and exclude_class_5:
                continue
            
            acc = str(r.get("AccessionNumber", "")).strip()
            folder = os.path.join(dicom_root, acc)
            if not os.path.isdir(folder):
                continue
            dcm = _find_first_dicom(folder)
            if dcm is None:
                continue
            
            rows.append({
                "accession": acc,
                "image_path": dcm,
                "professional_label": lab
            })
        return pd.DataFrame(rows)
    
    elif "image_path" in df.columns:
        # Formato 2: image_path, density_label/label
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
    
    else:
        raise ValueError("CSV formato desconhecido ou faltando --dicom-root para CSV de classificação.")

def resolve_dataset_cache_mode(requested_mode: str, df: pd.DataFrame) -> str:
    mode = (requested_mode or "none").lower()
    if mode != "auto":
        return mode

    if "image_path" not in df.columns:
        return "none"
    paths = [str(p) for p in df["image_path"].tolist() if pd.notna(p)]
    if not paths:
        return "none"

    total = len(paths)
    dicom_mask = [is_dicom_path(p) for p in paths]
    if all(dicom_mask):
        if total > CACHE_AUTO_DISK_MAX:
            return "none"
        return "disk"

    if total <= CACHE_AUTO_MEMORY_MAX:
        return "memory"
    return "none"
