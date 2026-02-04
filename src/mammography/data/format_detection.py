"""
Dataset format auto-detection for mammography pipelines.

Automatically detects dataset structure and image formats from directory analysis.
Supports DICOM, PNG, JPEG formats and various CSV/metadata structures.

WARNING: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..io.dicom import DICOM_EXTS

logger = logging.getLogger(__name__)

# Valid image extensions (from csv_loader.py pattern)
VALID_IMAGE_EXTS = DICOM_EXTS + (".png", ".jpg", ".jpeg")

# Cache thresholds for preprocessing suggestions
CACHE_AUTO_DISK_MAX = 6000  # Threshold for suggesting disk caching

# Image format signatures (magic bytes) for robust detection
IMAGE_SIGNATURES = {
    "png": b"\x89PNG\r\n\x1a\n",
    "jpg": b"\xff\xd8\xff",
    "dicom": b"DICM",  # At offset 128
}


def detect_image_format(file_path: str, check_signature: bool = True) -> str:
    """
    Detect image format (DICOM vs PNG/JPG) from file path or content.

    Performs two-level detection:
    1. Extension-based check (fast)
    2. Optional signature-based check (robust against misnamed files)

    Args:
        file_path: Path to image file to detect
        check_signature: If True, verify format using file signature (magic bytes)

    Returns:
        Detected format: 'dicom', 'png', 'jpg', or 'unknown'

    Examples:
        >>> detect_image_format("image.dcm")
        'dicom'
        >>> detect_image_format("scan.png")
        'png'
        >>> detect_image_format("photo.jpeg")
        'jpg'

    Note:
        Extension check is case-insensitive. Signature check reads first 132 bytes
        of file to verify format, providing protection against misnamed files.
    """
    if not file_path:
        return "unknown"

    path_obj = Path(file_path)
    ext_lower = path_obj.suffix.lower()

    # Fast extension-based detection
    format_from_ext = "unknown"
    if ext_lower in (".dcm", ".dicom"):
        format_from_ext = "dicom"
    elif ext_lower == ".png":
        format_from_ext = "png"
    elif ext_lower in (".jpg", ".jpeg"):
        format_from_ext = "jpg"

    # Return extension-based result if signature check disabled
    if not check_signature:
        return format_from_ext

    # Verify with signature check if file exists
    if not path_obj.exists() or not path_obj.is_file():
        logger.warning(f"File does not exist for signature check: {file_path}")
        return format_from_ext

    try:
        with open(file_path, "rb") as f:
            # Read enough bytes for all signatures (132 for DICOM)
            header = f.read(132)

            if not header:
                return "unknown"

            # Check PNG signature (first 8 bytes)
            if header.startswith(IMAGE_SIGNATURES["png"]):
                return "png"

            # Check JPEG signature (first 3 bytes)
            if header.startswith(IMAGE_SIGNATURES["jpg"]):
                return "jpg"

            # Check DICOM signature (bytes 128-131)
            if len(header) >= 132 and header[128:132] == IMAGE_SIGNATURES["dicom"]:
                return "dicom"

            # No signature matched - return extension-based guess
            logger.debug(
                f"No matching signature found for {file_path}, using extension-based detection"
            )
            return format_from_ext

    except Exception as exc:
        logger.warning(f"Error reading file signature for {file_path}: {exc!r}")
        return format_from_ext


@dataclass
class CSVSchemaInfo:
    """
    Information about detected CSV/TSV schema.

    Attributes:
        delimiter: Detected delimiter (comma, tab, or unknown)
        schema_type: Type of schema (classification, raw_path, dataset, custom, unknown)
        columns: List of column names found in the file
        row_count: Number of data rows (excluding header)
        has_header: Whether a header row was detected
        warnings: List of schema validation warnings
        encoding: Detected file encoding
    """

    delimiter: str
    schema_type: str
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    has_header: bool = True
    warnings: List[str] = field(default_factory=list)
    encoding: str = "utf-8"


@dataclass
class DatasetFormat:
    """
    Represents the detected dataset format and structure.

    Attributes:
        dataset_type: Type of dataset (archive, mamografias, patches_completo, custom)
        image_format: Predominant image format (dicom, png, jpg, mixed)
        csv_path: Path to CSV/metadata file if detected
        dicom_root: Root directory for DICOM files if applicable
        has_features_txt: Whether featureS.txt format detected
        has_csv: Whether CSV metadata file detected
        image_count: Total number of image files found
        format_counts: Dictionary of format counts {ext: count}
        warnings: List of validation warnings
        suggestions: List of preprocessing suggestions
    """

    dataset_type: str
    image_format: str
    csv_path: Optional[str] = None
    dicom_root: Optional[str] = None
    has_features_txt: bool = False
    has_csv: bool = False
    image_count: int = 0
    format_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


def _count_files_by_extension(
    directory: str, extensions: Tuple[str, ...], max_depth: int = 5
) -> Dict[str, int]:
    """
    Count files by extension in directory tree up to max_depth.

    Args:
        directory: Root directory to search
        extensions: Tuple of file extensions to count (with dots, e.g., '.png')
        max_depth: Maximum depth to traverse (prevents excessive scanning)

    Returns:
        Dictionary mapping extension to count
    """
    counts: Dict[str, int] = {}
    root_path = Path(directory).resolve()

    try:
        for ext in extensions:
            # Use rglob for efficient recursive search with limited depth
            pattern = f"**/*{ext}"
            for path in root_path.rglob(pattern):
                # Check depth
                try:
                    relative = path.relative_to(root_path)
                    depth = len(relative.parts)
                    if depth > max_depth:
                        continue
                except ValueError:
                    continue

                if path.is_file():
                    counts[ext] = counts.get(ext, 0) + 1
    except Exception as exc:
        logger.warning(f"Error counting files in {directory}: {exc!r}")

    return counts


def _find_metadata_files(directory: str) -> Dict[str, Optional[str]]:
    """
    Search for common metadata files in directory.

    Args:
        directory: Root directory to search

    Returns:
        Dictionary with keys: 'csv', 'features_txt', 'classification_csv'
    """
    root_path = Path(directory)
    result: Dict[str, Optional[str]] = {
        "csv": None,
        "features_txt": None,
        "classification_csv": None,
    }

    if not root_path.exists():
        return result

    # Check for classificacao.csv (archive preset)
    class_csv = root_path / "classificacao.csv"
    if class_csv.is_file():
        result["classification_csv"] = str(class_csv)
        result["csv"] = str(class_csv)

    # Check for featureS.txt in subdirectories (mamografias/patches_completo)
    features_files: List[Path] = []
    try:
        for features_path in root_path.rglob("featureS.txt"):
            if features_path.is_file():
                features_files.append(features_path)
    except Exception as exc:
        logger.warning(f"Error searching for featureS.txt: {exc!r}")

    if features_files:
        # If multiple featureS.txt files in subdirectories, use root directory
        # Otherwise use the parent directory of the found file
        if len(features_files) > 1:
            # Check if they're in different subdirectories under root
            parents = set(f.parent for f in features_files)
            if len(parents) > 1 and all(p.parent == root_path for p in parents):
                # Multiple subdirectories each with featureS.txt -> use root
                result["features_txt"] = str(root_path)
            else:
                # Use first found
                result["features_txt"] = str(features_files[0].parent)
        else:
            # Single featureS.txt - check if it's at root or in subdirectory
            if features_files[0].parent == root_path:
                result["features_txt"] = str(root_path)
            else:
                result["features_txt"] = str(features_files[0].parent)

    # Check for any CSV files in root
    if not result["csv"]:
        try:
            for csv_path in root_path.glob("*.csv"):
                if csv_path.is_file():
                    result["csv"] = str(csv_path)
                    break
        except Exception as exc:
            logger.warning(f"Error searching for CSV files: {exc!r}")

    return result


def _detect_dicom_structure(directory: str) -> bool:
    """
    Check if directory has DICOM structure with AccessionNumber subdirectories.

    Args:
        directory: Root directory to check

    Returns:
        True if DICOM structure detected, False otherwise
    """
    root_path = Path(directory)

    if not root_path.exists():
        return False

    # Look for 'archive' subdirectory (common pattern)
    archive_path = root_path / "archive"
    if archive_path.is_dir():
        # Check if archive has subdirectories (potential AccessionNumbers)
        try:
            subdirs = [d for d in archive_path.iterdir() if d.is_dir()]
            if len(subdirs) > 0:
                # Sample first subdirectory for DICOM files
                for subdir in subdirs[:3]:  # Check up to 3 subdirs
                    dicom_files = list(subdir.glob("*.dcm")) + list(
                        subdir.glob("*.dicom")
                    )
                    if dicom_files:
                        return True
        except Exception as exc:
            logger.warning(f"Error checking DICOM structure: {exc!r}")

    return False


def _detect_delimiter(file_path: str, sample_size: int = 5) -> str:
    """
    Detect delimiter (comma or tab) from CSV/TSV file.

    Args:
        file_path: Path to the CSV/TSV file
        sample_size: Number of lines to sample for detection

    Returns:
        Detected delimiter: 'comma', 'tab', or 'unknown'
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for _ in range(sample_size):
                line = f.readline()
                if line:
                    lines.append(line)

        if not lines:
            return "unknown"

        # Count delimiters in first few lines
        comma_counts = [line.count(",") for line in lines]
        tab_counts = [line.count("\t") for line in lines]

        avg_commas = sum(comma_counts) / len(comma_counts) if comma_counts else 0
        avg_tabs = sum(tab_counts) / len(tab_counts) if tab_counts else 0

        # Delimiter should be consistent across lines
        if avg_tabs > avg_commas and avg_tabs >= 1:
            return "tab"
        elif avg_commas >= 1:
            return "comma"
        else:
            return "unknown"
    except Exception as exc:
        logger.warning(f"Error detecting delimiter in {file_path}: {exc!r}")
        return "unknown"


def _infer_schema_type(df: pd.DataFrame) -> Tuple[str, List[str]]:
    """
    Infer schema type from DataFrame columns.

    Tries to match against known schemas from csv_loader.py:
    - classification: AccessionNumber + Classification
    - raw_path: image_path + label columns
    - dataset: image_path + professional_label + accession + view
    - custom: anything else

    Args:
        df: DataFrame to analyze

    Returns:
        Tuple of (schema_type, warnings)
    """
    columns = set(df.columns)
    warnings: List[str] = []

    # Check for classification schema (AccessionNumber + Classification)
    if "AccessionNumber" in columns and "Classification" in columns:
        if len(columns) == 2:
            return "classification", warnings
        else:
            warnings.append(
                f"Matched classification schema but has extra columns: {columns - {'AccessionNumber', 'Classification'}}"
            )
            return "classification", warnings

    # Check for dataset schema (image_path + professional_label + accession + view)
    dataset_required = {"image_path", "professional_label", "accession", "view"}
    if dataset_required.issubset(columns):
        return "dataset", warnings

    # Check for raw_path schema variants (image_path + label column)
    if "image_path" in columns:
        label_columns = {
            "professional_label",
            "density_label",
            "label",
            "y",
        }
        has_label = bool(columns & label_columns)
        if has_label:
            return "raw_path", warnings
        else:
            warnings.append(
                "Has image_path but missing label columns (professional_label, density_label, label, or y)"
            )
            return "custom", warnings

    # Check for common column patterns
    if "path" in columns or "file" in columns or "filename" in columns:
        warnings.append(
            "Has path-like column but not standard 'image_path' - may need column renaming"
        )
        return "custom", warnings

    # No recognizable pattern
    warnings.append(f"No recognizable schema pattern. Columns: {list(columns)}")
    return "unknown", warnings


def infer_csv_schema(file_path: str, max_rows: int = 100) -> CSVSchemaInfo:
    """
    Detect and infer schema from CSV/TSV file.

    Analyzes file to determine:
    - Delimiter (comma vs tab)
    - Schema type (classification, raw_path, dataset, custom)
    - Column structure and basic validation
    - Encoding and format issues

    Args:
        file_path: Path to CSV/TSV file to analyze
        max_rows: Maximum rows to read for analysis (for performance)

    Returns:
        CSVSchemaInfo object with detection results

    Raises:
        ValueError: If file does not exist or cannot be read
        OSError: If file cannot be opened

    Examples:
        >>> schema = infer_csv_schema("classificacao.csv")
        >>> print(schema.schema_type)
        'classification'
        >>> print(schema.delimiter)
        'comma'
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    logger.info(f"Inferring CSV schema for: {file_path}")

    # Detect delimiter
    delimiter_type = _detect_delimiter(file_path)
    delimiter_char = "," if delimiter_type == "comma" else "\t"

    if delimiter_type == "unknown":
        return CSVSchemaInfo(
            delimiter="unknown",
            schema_type="unknown",
            warnings=["Could not detect valid delimiter (comma or tab)"],
        )

    # Try to read the CSV
    try:
        # Try UTF-8 first
        df = pd.read_csv(file_path, sep=delimiter_char, nrows=max_rows, encoding="utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        # Fallback to latin-1
        try:
            df = pd.read_csv(
                file_path, sep=delimiter_char, nrows=max_rows, encoding="latin-1"
            )
            encoding = "latin-1"
        except Exception as exc:
            return CSVSchemaInfo(
                delimiter=delimiter_type,
                schema_type="unknown",
                warnings=[f"Failed to read file with common encodings: {exc!r}"],
                encoding="unknown",
            )
    except Exception as exc:
        return CSVSchemaInfo(
            delimiter=delimiter_type,
            schema_type="unknown",
            warnings=[f"Failed to parse CSV: {exc!r}"],
        )

    # Infer schema type
    schema_type, warnings = _infer_schema_type(df)

    # Check for header
    has_header = True
    if df.columns[0].startswith("Unnamed"):
        has_header = False
        warnings.append("No header row detected - using default column names")

    # Build result
    result = CSVSchemaInfo(
        delimiter=delimiter_type,
        schema_type=schema_type,
        columns=list(df.columns),
        row_count=len(df),
        has_header=has_header,
        warnings=warnings,
        encoding=encoding,
    )

    logger.info(
        f"Inferred schema: type={result.schema_type}, "
        f"delimiter={result.delimiter}, "
        f"columns={len(result.columns)}, "
        f"rows={result.row_count}"
    )

    return result


def detect_dataset_format(path: str) -> DatasetFormat:
    """
    Auto-detect dataset format from directory structure and file analysis.

    Analyzes directory structure to identify:
    - Dataset type (archive, mamografias, patches_completo, custom)
    - Image format (DICOM, PNG, JPEG, or mixed)
    - Metadata files (CSV, featureS.txt)
    - Potential issues and suggestions

    Args:
        path: Root directory path to analyze

    Returns:
        DatasetFormat object with detection results

    Raises:
        ValueError: If path does not exist or is not a directory
        OSError: If directory cannot be read

    Examples:
        >>> fmt = detect_dataset_format("./mamografias")
        >>> print(fmt.dataset_type)
        'mamografias'
        >>> print(fmt.image_format)
        'png'
    """
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")

    if not os.path.isdir(path):
        raise ValueError(f"Path is not a directory: {path}")

    logger.info(f"Detecting dataset format for: {path}")

    # Initialize result
    fmt = DatasetFormat(dataset_type="custom", image_format="unknown")

    # Count image files by extension
    format_counts = _count_files_by_extension(path, VALID_IMAGE_EXTS)
    fmt.format_counts = format_counts
    fmt.image_count = sum(format_counts.values())

    # Find metadata files
    metadata = _find_metadata_files(path)

    # Determine predominant image format
    if format_counts:
        # Get format with highest count
        max_count = max(format_counts.values())
        dominant_formats = [
            ext for ext, count in format_counts.items() if count == max_count
        ]

        if len(dominant_formats) == 1:
            ext = dominant_formats[0].lower()
            if ext in (".dcm", ".dicom"):
                fmt.image_format = "dicom"
            elif ext == ".png":
                fmt.image_format = "png"
            elif ext in (".jpg", ".jpeg"):
                fmt.image_format = "jpg"
        else:
            fmt.image_format = "mixed"

        # Check for mixed formats (>10% minority format)
        for ext, count in format_counts.items():
            percentage = (count / fmt.image_count) * 100
            if 10 < percentage < 90:
                fmt.image_format = "mixed"
                fmt.warnings.append(
                    f"Mixed image formats detected: {ext} ({percentage:.1f}%)"
                )

    # Detect dataset type
    path_lower = path.lower()
    if "mamografias" in path_lower and metadata["features_txt"]:
        fmt.dataset_type = "mamografias"
        fmt.csv_path = metadata["features_txt"]
        fmt.has_features_txt = True
    elif "patches_completo" in path_lower or "patches-completo" in path_lower:
        if metadata["features_txt"]:
            fmt.dataset_type = "patches_completo"
            fmt.csv_path = metadata["features_txt"]
            fmt.has_features_txt = True
    elif _detect_dicom_structure(path):
        fmt.dataset_type = "archive"
        fmt.dicom_root = str(Path(path) / "archive")
        if metadata["classification_csv"]:
            fmt.csv_path = metadata["classification_csv"]
            fmt.has_csv = True
    else:
        # Custom dataset - try to find metadata
        fmt.dataset_type = "custom"
        if metadata["csv"]:
            fmt.csv_path = metadata["csv"]
            fmt.has_csv = True
        elif metadata["features_txt"]:
            fmt.csv_path = metadata["features_txt"]
            fmt.has_features_txt = True

    # Validation warnings
    if fmt.image_count == 0:
        fmt.warnings.append(
            f"No image files found in {path}. Expected extensions: {VALID_IMAGE_EXTS}"
        )

    if not fmt.csv_path:
        fmt.warnings.append(
            "No metadata file detected. Expected CSV or featureS.txt structure."
        )

    logger.info(
        f"Detected format: type={fmt.dataset_type}, "
        f"image_format={fmt.image_format}, "
        f"image_count={fmt.image_count}"
    )

    return fmt


def validate_format(format_info: DatasetFormat) -> List[str]:
    """
    Validate detected dataset format and return list of warnings.

    Performs comprehensive validation checks including:
    - Missing or empty metadata
    - Image count validation
    - Format consistency checks
    - Required field validation

    Args:
        format_info: DatasetFormat object from detect_dataset_format()

    Returns:
        List of validation warning messages

    Examples:
        >>> fmt = detect_dataset_format("./data")
        >>> warnings = validate_format(fmt)
        >>> for warn in warnings:
        ...     logger.warning(warn)
    """
    warnings: List[str] = []

    # Copy existing warnings from format detection
    warnings.extend(format_info.warnings)

    # Check for empty directories
    if format_info.image_count == 0:
        msg = "No images found. Dataset appears to be empty or in unsupported format."
        warnings.append(msg)
        logger.warning(f"Empty dataset detected: {format_info.dataset_type}")

    # Check for missing metadata
    if not format_info.csv_path and not format_info.has_features_txt:
        msg = (
            "No metadata file found. Dataset may be missing labels or classification data."
        )
        warnings.append(msg)
        logger.warning("Missing metadata files for dataset")

    # Check for mixed image formats
    if format_info.image_format == "mixed":
        format_list = ", ".join(
            f"{ext}: {count}" for ext, count in format_info.format_counts.items()
        )
        msg = (
            f"Mixed image formats detected: {format_list}. "
            "Consider converting to a single format for consistency."
        )
        warnings.append(msg)
        logger.warning(f"Mixed formats in dataset: {format_list}")

    # Check for unknown format
    if format_info.image_format == "unknown":
        msg = (
            "Could not determine image format. "
            f"Expected formats: {VALID_IMAGE_EXTS}"
        )
        warnings.append(msg)
        logger.warning("Unknown image format detected")

    # Check for DICOM-specific issues
    if format_info.image_format == "dicom" and not format_info.dicom_root:
        msg = (
            "DICOM format detected but no DICOM root directory identified. "
            "Expected 'archive/' directory structure."
        )
        warnings.append(msg)
        logger.warning("DICOM format without proper directory structure")

    # Check for low image count
    if 0 < format_info.image_count < 10:
        msg = (
            f"Very small dataset: only {format_info.image_count} images found. "
            "This may not be sufficient for training or evaluation."
        )
        warnings.append(msg)
        logger.warning(f"Small dataset detected: {format_info.image_count} images")

    # Check for custom dataset type
    if format_info.dataset_type == "custom":
        msg = (
            "Custom dataset format detected. Ensure metadata follows expected schema "
            "(image_path, professional_label, accession, view columns)."
        )
        warnings.append(msg)

    return warnings


def suggest_preprocessing(format_info: DatasetFormat) -> List[str]:
    """
    Generate preprocessing suggestions based on detected dataset format.

    Provides format-specific recommendations for preprocessing pipeline including:
    - Image format-specific normalization strategies
    - DICOM vs PNG/JPEG handling
    - Mixed format preprocessing
    - Dataset-specific best practices

    Args:
        format_info: DatasetFormat object from detect_dataset_format()

    Returns:
        List of preprocessing suggestion strings

    Examples:
        >>> fmt = detect_dataset_format("./archive")
        >>> suggestions = suggest_preprocessing(fmt)
        >>> for suggestion in suggestions:
        ...     print(f"SUGGESTION: {suggestion}")

    Note:
        Suggestions are advisory and should be reviewed in context of the
        specific research or educational use case. Not for clinical use.
    """
    suggestions: List[str] = []

    # DICOM-specific preprocessing suggestions
    if format_info.image_format == "dicom":
        suggestions.append(
            "DICOM format detected: Apply DICOM normalization with window/level adjustment "
            "(e.g., Breast window: C=2000, W=4000) for optimal contrast"
        )
        suggestions.append(
            "DICOM preprocessing: Extract ViewPosition metadata (CC/MLO) for view-specific models"
        )
        suggestions.append(
            "DICOM caching: Use 'tensor-disk' or 'tensor-memmap' caching mode for large datasets "
            "(saves preprocessing time on subsequent runs)"
        )

        # Archive-specific suggestions
        if format_info.dataset_type == "archive":
            suggestions.append(
                "Archive dataset: Ensure AccessionNumber-based grouping for patient-level splits "
                "(prevents data leakage between train/val/test)"
            )
            suggestions.append(
                "Archive dataset: Consider bilateral pairing (left/right breast) for ensemble predictions"
            )

    # PNG/JPEG preprocessing suggestions
    elif format_info.image_format in ("png", "jpg"):
        suggestions.append(
            f"{format_info.image_format.upper()} format detected: Verify image normalization "
            f"(check if pre-normalized or needs [0,1] or [-1,1] scaling)"
        )
        suggestions.append(
            f"{format_info.image_format.upper()} preprocessing: Check bit depth (8-bit vs 16-bit) "
            f"and apply appropriate normalization"
        )

        # Dataset-specific suggestions
        if format_info.dataset_type in ("mamografias", "patches_completo"):
            suggestions.append(
                f"{format_info.dataset_type}: Verify featureS.txt labels match image filenames "
                f"(common issue with case sensitivity or extensions)"
            )
            suggestions.append(
                f"{format_info.dataset_type}: Consider data augmentation (rotation, flip, zoom) "
                f"for improved model generalization"
            )

    # Mixed format handling
    elif format_info.image_format == "mixed":
        suggestions.append(
            "Mixed formats detected: Apply format-specific normalization in data loader "
            "(separate DICOM and PNG/JPEG pipelines)"
        )
        suggestions.append(
            "Mixed formats: Consider converting to single format (PNG recommended) "
            "for consistent preprocessing and faster loading"
        )
        format_breakdown = ", ".join(
            f"{ext}: {count}" for ext, count in format_info.format_counts.items()
        )
        suggestions.append(
            f"Mixed format distribution: {format_breakdown}. "
            f"Prioritize format with highest count for conversion target"
        )

    # Unknown format fallback
    elif format_info.image_format == "unknown":
        suggestions.append(
            f"Unknown format: Verify image files have valid extensions {VALID_IMAGE_EXTS}"
        )
        suggestions.append(
            "Unknown format: Check for corrupted files or unsupported formats in dataset"
        )

    # General preprocessing suggestions based on dataset size
    if format_info.image_count > 0:
        if format_info.image_count < 100:
            suggestions.append(
                f"Small dataset ({format_info.image_count} images): "
                f"Use aggressive data augmentation and consider transfer learning "
                f"(e.g., ImageNet-pretrained models)"
            )
        elif format_info.image_count < 1000:
            suggestions.append(
                f"Medium dataset ({format_info.image_count} images): "
                f"Use moderate augmentation and monitor for overfitting with early stopping"
            )
        else:
            suggestions.append(
                f"Large dataset ({format_info.image_count} images): "
                f"Consider distributed training and efficient caching strategies"
            )

    # Metadata-specific suggestions
    if not format_info.csv_path:
        suggestions.append(
            "No metadata detected: If labels exist, create CSV with columns "
            "(image_path, professional_label, accession, view) for compatibility"
        )
    elif format_info.has_features_txt:
        suggestions.append(
            "featureS.txt format: Verify tab-delimited structure and label encoding "
            "(A/B/C/D or 1/2/3/4)"
        )

    # View-specific training suggestion
    if format_info.image_format == "dicom" or format_info.dataset_type == "archive":
        suggestions.append(
            "View-specific training: Consider separate models for CC and MLO views, "
            "then ensemble predictions for improved accuracy"
        )

    # Memory optimization for large datasets
    if format_info.image_count > CACHE_AUTO_DISK_MAX:
        suggestions.append(
            f"Large dataset (>{CACHE_AUTO_DISK_MAX} images): "
            f"Use lazy loading with 'auto' or 'disk' caching to prevent memory issues"
        )

    # Dataset validation suggestion
    suggestions.append(
        "Pre-training validation: Run dataset validation checks to identify "
        "missing files, invalid labels, or corrupted images before training"
    )

    logger.info(f"Generated {len(suggestions)} preprocessing suggestions")
    return suggestions
