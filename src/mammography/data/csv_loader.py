#
# csv_loader.py
# mammography-pipelines
#
# Loads dataset rows from CSVs, featureS.txt directories, or presets and normalizes labels/paths.
#
# Thales Matheus Mendonça Santos - November 2025
#
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pandera as pa
import pydicom

from ..io.dicom import DICOM_EXTS
from .format_detection import detect_dataset_format, validate_format, suggest_preprocessing

logger = logging.getLogger(__name__)

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

def _extract_view_from_dicom(dcm_path: str) -> Optional[str]:
    """Extract ViewPosition from DICOM file, returning None if missing or invalid.

    Args:
        dcm_path: Path to DICOM file

    Returns:
        ViewPosition ('CC' or 'MLO') if found, None otherwise
    """
    try:
        dataset = pydicom.dcmread(dcm_path, force=True)
        view_position = getattr(dataset, "ViewPosition", "")
        if view_position and isinstance(view_position, str):
            view_position = view_position.strip().upper()
            if view_position in ("CC", "MLO"):
                return view_position
        logger.debug(f"ViewPosition not found or invalid in {dcm_path}")
        return None
    except Exception as exc:
        logger.debug(f"Failed to extract view from {dcm_path}: {exc}")
        return None

def _coerce_density_label(val: Any, strict: bool = False, warn: bool = True) -> Optional[int]:
    """Normalize label inputs to integers in {1, 2, 3, 4, 5} when possible.

    Args:
        val: Label value to coerce
        strict: If True, raise ValueError on coercion failure instead of returning None
        warn: If True, log warning when coercion fails (ignored if strict=True)

    Returns:
        Coerced integer label or None if coercion fails (when strict=False)

    Raises:
        ValueError: If strict=True and coercion fails
    """
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
            if strict:
                raise ValueError(f"Invalid density label: {val!r}")
            if warn:
                logger.warning(f"Could not coerce label {val!r} to valid density class, returning None")
            return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    try:
        return int(val)
    except Exception:
        if strict:
            raise ValueError(f"Invalid density label: {val!r}")
        if warn:
            logger.warning(f"Could not coerce label {val!r} to valid density class, returning None")
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
    return pa.Column(object, nullable=True, required=required, coerce=True, checks=LABEL_CHECK)

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
        "view": pa.Column(
            str,
            nullable=True,
            coerce=True,
        ),
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

def _read_csv_with_encoding(csv_path: str) -> pd.DataFrame:
    """Read CSV with automatic encoding detection and fallback.

    Tries common encodings in order: utf-8, latin-1, windows-1252, iso-8859-1.
    This handles international datasets with non-ASCII characters.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame loaded from CSV

    Raises:
        ValueError: If CSV cannot be decoded with any common encoding
    """
    encodings = ["utf-8", "latin-1", "windows-1252", "iso-8859-1"]

    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            if encoding != "utf-8":
                logger.info(f"Successfully read {csv_path} with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            # Other errors (file not found, etc.) should propagate immediately
            raise ValueError(f"Failed to read CSV {csv_path}: {exc}") from exc

    raise ValueError(
        f"Could not decode CSV {csv_path} with any common encoding: {encodings}. "
        f"Try converting to UTF-8 or specify encoding explicitly."
    )

def _rows_from_features_dir(root: Path) -> List[Dict[str, Any]]:
    """Load dataset rows from directories containing featureS.txt files.

    Robustly parses featureS.txt files with format:
        filename
        class_label (0-3)
        filename
        class_label
        ...

    Args:
        root: Root directory containing subdirectories with featureS.txt files

    Returns:
        List of row dictionaries with image_path, professional_label, accession, view

    Raises:
        ValueError: If no valid images found
    """
    rows: List[Dict[str, Any]] = []

    # Determine search strategy:
    # 1. If featureS.txt exists in root, use root only
    # 2. Otherwise, search in immediate subdirectories
    root_features = root / "featureS.txt"
    if root_features.exists():
        search_dirs = [root]
    else:
        # Look for subdirectories with featureS.txt
        search_dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]

    for folder in search_dirs:
        feat_path = folder / "featureS.txt"
        if not feat_path.exists():
            continue

        # Read and clean lines (remove comments and empty lines)
        raw_lines = feat_path.read_text(encoding="utf-8").splitlines()
        lines = [l.strip() for l in raw_lines if l.strip() and not l.strip().startswith("#")]

        # Warn if odd number of lines
        if len(lines) % 2 != 0:
            logger.warning(
                f"{feat_path} has odd number of lines ({len(lines)}). "
                f"Last entry may be incomplete and will be skipped."
            )

        parsed_count = 0
        skipped_count = 0

        for i in range(0, len(lines) - 1, 2):
            fname, cls_raw = lines[i], lines[i + 1]

            # Try to parse label
            try:
                birads = int(cls_raw) + 1
                if birads not in ALLOWED_DENSITY_LABELS:
                    logger.warning(f"Skipping {fname}: invalid label {cls_raw} (birads={birads})")
                    skipped_count += 1
                    continue
            except Exception as exc:
                logger.warning(f"Skipping {fname}: could not parse label '{cls_raw}': {exc}")
                skipped_count += 1
                continue

            # Normalize filename - add .png extension if missing
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".dcm", ".dicom")):
                fname = f"{fname}.png"

            # Construct full path - try both with and without space normalization
            full_path = folder / fname
            if not full_path.exists():
                # Try adding space before parenthesis (old format)
                fname_with_space = fname
                if "(" in fname and " (" not in fname:
                    fname_with_space = fname.replace("(", " (")
                    full_path = folder / fname_with_space

                if not full_path.exists():
                    logger.warning(f"Skipping {fname}: file not found at {folder / fname}")
                    skipped_count += 1
                    continue

            rows.append(
                {
                    "image_path": str(full_path),
                    "professional_label": birads,
                    "accession": folder.name,
                    "view": None,
                }
            )
            parsed_count += 1

        if skipped_count > 0:
            logger.info(f"{feat_path}: parsed {parsed_count} entries, skipped {skipped_count}")

    if not rows:
        raise ValueError(
            f"Nenhuma imagem encontrada via featureS.txt em {root}. "
            f"Verifique o formato do arquivo (filename\\nclass\\nfilename\\nclass\\n...)"
        )
    return rows

def resolve_paths_from_preset(csv_path: Optional[str], dataset: Optional[str], dicom_root: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Fill `csv_path` and `dicom_root` using the named preset when the user did not pass explicit paths."""
    if dataset and dataset in DATASET_PRESETS:
        preset = DATASET_PRESETS[dataset]
        csv_path = csv_path or preset.get("csv")
        if preset.get("dicom_root") and not dicom_root:
            dicom_root = preset.get("dicom_root")
    return csv_path, dicom_root

def load_dataset_dataframe(csv_path: Optional[str], dicom_root: Optional[str] = None, exclude_class_5: bool = True, dataset: Optional[str] = None, auto_detect: bool = True) -> pd.DataFrame:
    """Load a canonical DataFrame from CSVs, featureS.txt directories, or known presets.

    Args:
        csv_path: Path to CSV file or directory with image data
        dicom_root: Root directory for DICOM files (for classification CSV format)
        exclude_class_5: Whether to exclude class 5 samples
        dataset: Optional dataset preset name (archive, mamografias, patches_completo)
        auto_detect: Enable automatic format detection for directories (default: True)

    Returns:
        DataFrame with standardized schema (image_path, professional_label, accession, view)

    Raises:
        ValueError: If csv_path not provided, or invalid format detected
    """
    csv_path, dicom_root = resolve_paths_from_preset(csv_path, dataset, dicom_root)
    if not csv_path:
        raise ValueError("csv_path não definido; use --csv ou --dataset com preset válido.")

    # Auto-detect format when csv_path is a directory
    if auto_detect and os.path.isdir(csv_path):
        try:
            detected_format = detect_dataset_format(csv_path)
            logger.info(f"Formato detectado: {detected_format.dataset_type}")
            logger.info(f"Formato de imagem: {detected_format.image_format}")
            logger.info(f"Total de imagens: {detected_format.image_count}")

            # Validate detected format and get warnings
            validation_warnings = validate_format(detected_format)
            if validation_warnings:
                logger.warning("=" * 60)
                logger.warning("AVISOS DE VALIDACAO:")
                for warning in validation_warnings:
                    logger.warning(f"  • {warning}")
                logger.warning("=" * 60)

            # Get preprocessing suggestions
            suggestions = suggest_preprocessing(detected_format)
            if suggestions:
                logger.info("=" * 60)
                logger.info("SUGESTOES DE PRE-PROCESSAMENTO:")
                for suggestion in suggestions:
                    logger.info(f"  • {suggestion}")
                logger.info("=" * 60)

            # Apply detected paths if they were found and not explicitly provided
            if detected_format.csv_path and not dataset:
                csv_path = detected_format.csv_path
                logger.info(f"Usando CSV detectado: {csv_path}")

            if detected_format.dicom_root and not dicom_root:
                dicom_root = detected_format.dicom_root
                logger.info(f"Usando dicom_root detectado: {dicom_root}")
        except Exception as exc:
            logger.warning(f"Auto-detecção falhou: {exc!r}. Usando lógica padrão.")

    # Directory with featureS.txt (mammograms/patches)
    if os.path.isdir(csv_path):
        rows = _rows_from_features_dir(Path(csv_path))
        return _validate_schema(DATASET_SCHEMA, pd.DataFrame(rows), "featureS.txt")

    # Read CSV with encoding detection
    df = _read_csv_with_encoding(csv_path)

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
            view = _extract_view_from_dicom(dcm)
            rows.append({"accession": acc, "image_path": dcm, "professional_label": lab, "view": view})
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

        # Extract view from DICOM files if applicable
        def extract_view_safe(image_path):
            if pd.isna(image_path):
                return None
            path_str = str(image_path).lower()
            if path_str.endswith(DICOM_EXTS):
                return _extract_view_from_dicom(str(image_path))
            return None

        path_df["view"] = path_df["image_path"].apply(extract_view_safe)

        keep_cols = ["image_path", "professional_label", "accession", "view"]
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

def validate_split_overlap(splits: Dict[str, pd.DataFrame], key: str = "accession") -> None:
    """Ensure train/val/test splits don't share samples.

    Args:
        splits: Dictionary mapping split names to DataFrames
        key: Column name to check for uniqueness across splits (default: "accession")

    Raises:
        ValueError: If any samples appear in multiple splits
    """
    if not splits or len(splits) < 2:
        return

    split_names = list(splits.keys())
    for i, name_a in enumerate(split_names):
        df_a = splits[name_a]
        if key not in df_a.columns:
            continue
        set_a = set(df_a[key].dropna().unique())

        for name_b in split_names[i + 1:]:
            df_b = splits[name_b]
            if key not in df_b.columns:
                continue
            set_b = set(df_b[key].dropna().unique())

            overlap = set_a & set_b
            if overlap:
                sample_list = ", ".join(str(x) for x in sorted(list(overlap))[:5])
                if len(overlap) > 5:
                    sample_list += f", ... ({len(overlap)} total)"
                raise ValueError(
                    f"Sobreposição detectada entre '{name_a}' e '{name_b}': "
                    f"{len(overlap)} amostras compartilhadas ({key}): {sample_list}"
                )

def load_multiple_csvs(csv_paths: Dict[str, str], dicom_root: Optional[str] = None, exclude_class_5: bool = True, dataset: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Load multiple CSV files for train/val/test splits and validate them.

    Args:
        csv_paths: Dictionary mapping split names (e.g., 'train', 'val', 'test') to CSV file paths
        dicom_root: Optional root directory for DICOM files
        exclude_class_5: Whether to exclude class 5 samples
        dataset: Optional dataset preset name

    Returns:
        Dictionary mapping split names to validated DataFrames

    Raises:
        ValueError: If any CSV file is invalid or if splits have overlapping samples
    """
    if not csv_paths:
        raise ValueError("csv_paths não pode estar vazio")

    result: Dict[str, pd.DataFrame] = {}
    for split_name, csv_path in csv_paths.items():
        if not csv_path:
            continue
        df = load_dataset_dataframe(csv_path, dicom_root=dicom_root, exclude_class_5=exclude_class_5, dataset=dataset)
        result[split_name] = df

    return result

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
