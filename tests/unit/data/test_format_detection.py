# ruff: noqa
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

Image = pytest.importorskip("PIL.Image")
pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pydicom = pytest.importorskip("pydicom")

from mammography.data.csv_loader import (
    _coerce_density_label,
    _extract_view_from_dicom,
    _normalize_accession,
    load_dataset_dataframe,
    load_multiple_csvs,
    resolve_dataset_cache_mode,
    resolve_paths_from_preset,
    validate_split_overlap,
    _read_csv_with_encoding,
)
from mammography.data.splits import (
    create_kfold_splits,
    create_splits,
    create_three_way_split,
    filter_by_view,
    load_splits_from_csvs,
)
from mammography.data.format_detection import (
    detect_dataset_format,
    detect_image_format,
    infer_csv_schema,
    suggest_preprocessing,
    validate_format,
)
from mammography.data.dataset import (
    EmbeddingStore,
    load_embedding_store,
    MammoDensityDataset,
    robust_collate,
)

def _write_sample_image(path: Path) -> None:
    img = Image.new("RGB", (16, 16), color=(120, 30, 60))
    img.save(path)

def test_detect_image_format_png(tmp_path: Path) -> None:
    """Test PNG format detection."""
    png_path = tmp_path / "test.png"
    _write_sample_image(png_path)

    fmt = detect_image_format(str(png_path), check_signature=False)
    assert fmt == "png"

    fmt_sig = detect_image_format(str(png_path), check_signature=True)
    assert fmt_sig == "png"

def test_detect_image_format_unknown() -> None:
    """Test unknown format detection."""
    fmt = detect_image_format("test.xyz", check_signature=False)
    assert fmt == "unknown"

    fmt = detect_image_format("", check_signature=False)
    assert fmt == "unknown"

def test_infer_csv_schema_classification(tmp_path: Path) -> None:
    """Test CSV schema inference for classification format."""
    csv_path = tmp_path / "classificacao.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\n"
        "ACC001,1\n"
        "ACC002,2\n",
        encoding="utf-8",
    )

    schema = infer_csv_schema(str(csv_path))
    assert schema.schema_type == "classification"
    assert schema.delimiter == "comma"
    assert "AccessionNumber" in schema.columns
    assert "Classification" in schema.columns

def test_infer_csv_schema_raw_path(tmp_path: Path) -> None:
    """Test CSV schema inference for raw_path format."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "image_path,professional_label\n"
        "/path/to/img1.png,1\n"
        "/path/to/img2.png,2\n",
        encoding="utf-8",
    )

    schema = infer_csv_schema(str(csv_path))
    assert schema.schema_type == "raw_path"
    assert "image_path" in schema.columns
    assert "professional_label" in schema.columns

def test_detect_dataset_format_mamografias(tmp_path: Path) -> None:
    """Test dataset format detection for mamografias structure."""
    # Create mamografias structure
    case_dir = tmp_path / "mamografias" / "case_001"
    case_dir.mkdir(parents=True)

    img_path = case_dir / "img_001.png"
    _write_sample_image(img_path)

    features_txt = case_dir / "featureS.txt"
    features_txt.write_text("img_001\n1\n", encoding="utf-8")

    fmt = detect_dataset_format(str(tmp_path / "mamografias"))
    assert fmt.dataset_type == "mamografias"
    assert fmt.has_features_txt is True
    assert fmt.image_format in ("png", "unknown")

def test_validate_format_warnings(tmp_path: Path) -> None:
    """Test format validation warning generation."""
    # Create empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    from mammography.data.format_detection import DatasetFormat

    fmt = DatasetFormat(
        dataset_type="custom",
        image_format="unknown",
        image_count=0,
    )

    warnings = validate_format(fmt)
    assert len(warnings) > 0
    assert any("empty" in w.lower() or "no images" in w.lower() for w in warnings)

def test_suggest_preprocessing_dicom() -> None:
    """Test preprocessing suggestions for DICOM format."""
    from mammography.data.format_detection import DatasetFormat

    fmt = DatasetFormat(
        dataset_type="archive",
        image_format="dicom",
        image_count=1000,
    )

    suggestions = suggest_preprocessing(fmt)
    assert len(suggestions) > 0
    assert any("DICOM" in s for s in suggestions)

def test_suggest_preprocessing_png() -> None:
    """Test preprocessing suggestions for PNG format."""
    from mammography.data.format_detection import DatasetFormat

    fmt = DatasetFormat(
        dataset_type="mamografias",
        image_format="png",
        image_count=500,
    )

    suggestions = suggest_preprocessing(fmt)
    assert len(suggestions) > 0
    assert any("PNG" in s for s in suggestions)
