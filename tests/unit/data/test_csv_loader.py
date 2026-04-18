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

def test_coerce_density_label_valid_integer() -> None:
    """Test label coercion with valid integer inputs."""
    assert _coerce_density_label(1) == 1
    assert _coerce_density_label(2) == 2
    assert _coerce_density_label(3) == 3
    assert _coerce_density_label(4) == 4
    assert _coerce_density_label(5) == 5

def test_coerce_density_label_zero_indexed() -> None:
    """Test label coercion with 0-indexed string inputs (0-3 -> 1-4)."""
    assert _coerce_density_label("0") == 1
    assert _coerce_density_label("1") == 2
    assert _coerce_density_label("2") == 3
    assert _coerce_density_label("3") == 4

def test_coerce_density_label_birads_letters() -> None:
    """Test label coercion with BI-RADS letter grades."""
    assert _coerce_density_label("A") == 1
    assert _coerce_density_label("B") == 2
    assert _coerce_density_label("C") == 3
    assert _coerce_density_label("D") == 4
    assert _coerce_density_label("a") == 1  # Case insensitive
    assert _coerce_density_label(" B ") == 2  # With whitespace

def test_coerce_density_label_invalid() -> None:
    """Test label coercion with invalid inputs."""
    assert _coerce_density_label(None) is None
    assert _coerce_density_label(pd.NA) is None
    assert _coerce_density_label("invalid") is None
    assert _coerce_density_label(10) == 10  # Out of range but returns as-is
    assert _coerce_density_label(-1) == -1  # Negative integers returned as-is
    assert _coerce_density_label(0) == 0  # Integer 0 returns 0 (unlike string "0" -> 1)

def test_coerce_density_label_string_numeric_no_mapping() -> None:
    """Test string numeric values outside 0-3 range are not mapped (returned as-is)."""
    assert _coerce_density_label("4") == 4  # Not mapped to 5
    assert _coerce_density_label("5") == 5  # Returned as-is
    assert _coerce_density_label("10") == 10  # Out of range string returns as-is
    assert _coerce_density_label(" 4 ") == 4  # With whitespace

def test_coerce_density_label_strict_mode() -> None:
    """Test strict mode raises ValueError on invalid labels."""
    with pytest.raises(ValueError, match="Invalid density label"):
        _coerce_density_label("invalid", strict=True)

def test_normalize_accession() -> None:
    """Test accession string normalization."""
    assert _normalize_accession("ACC001") == "ACC001"
    assert _normalize_accession("  ACC002  ") == "ACC002"
    assert _normalize_accession(None) is None
    assert _normalize_accession(pd.NA) is None
    assert _normalize_accession("") is None
    assert _normalize_accession("   ") is None
    assert _normalize_accession(123) == "123"

def test_resolve_paths_from_preset() -> None:
    """Test preset path resolution."""
    csv, dicom = resolve_paths_from_preset(None, "archive", None)
    assert csv == "classificacao.csv"
    assert dicom == "archive"

    csv, dicom = resolve_paths_from_preset(None, "mamografias", None)
    assert csv == "mamografias"
    assert dicom is None

    csv, dicom = resolve_paths_from_preset("custom.csv", "archive", "custom_root")
    assert csv == "custom.csv"
    assert dicom == "custom_root"

def test_resolve_dataset_cache_mode_auto_small() -> None:
    """Test auto cache mode resolution for small datasets."""
    rows = [{"image_path": f"img_{i}.png"} for i in range(10)]
    mode = resolve_dataset_cache_mode("auto", rows)
    assert mode == "memory"

def test_resolve_dataset_cache_mode_auto_large() -> None:
    """Test auto cache mode resolution for large datasets."""
    rows = [{"image_path": f"img_{i}.dcm"} for i in range(2000)]
    mode = resolve_dataset_cache_mode("auto", rows)
    assert mode == "disk"

def test_resolve_dataset_cache_mode_explicit() -> None:
    """Test explicit cache mode is preserved."""
    rows = [{"image_path": "img.png"}]
    assert resolve_dataset_cache_mode("memory", rows) == "memory"
    assert resolve_dataset_cache_mode("disk", rows) == "disk"
    assert resolve_dataset_cache_mode("none", rows) == "none"

def test_extract_view_from_dicom_cc(valid_dicom_dataset, tmp_path: Path) -> None:
    """Test extraction of CC ViewPosition from DICOM."""
    valid_dicom_dataset.ViewPosition = "CC"
    dcm_path = tmp_path / "test_cc.dcm"
    valid_dicom_dataset.save_as(str(dcm_path), write_like_original=False)

    view = _extract_view_from_dicom(str(dcm_path))
    assert view == "CC"

def test_extract_view_from_dicom_mlo(valid_dicom_dataset, tmp_path: Path) -> None:
    """Test extraction of MLO ViewPosition from DICOM."""
    valid_dicom_dataset.ViewPosition = "MLO"
    dcm_path = tmp_path / "test_mlo.dcm"
    valid_dicom_dataset.save_as(str(dcm_path), write_like_original=False)

    view = _extract_view_from_dicom(str(dcm_path))
    assert view == "MLO"

def test_extract_view_from_dicom_lowercase(valid_dicom_dataset, tmp_path: Path) -> None:
    """Test extraction of ViewPosition with case normalization."""
    valid_dicom_dataset.ViewPosition = "cc"
    dcm_path = tmp_path / "test_lowercase.dcm"
    valid_dicom_dataset.save_as(str(dcm_path), write_like_original=False)

    view = _extract_view_from_dicom(str(dcm_path))
    assert view == "CC"

def test_extract_view_from_dicom_missing(valid_dicom_dataset, tmp_path: Path) -> None:
    """Test handling of missing ViewPosition."""
    # Remove ViewPosition attribute if present
    if hasattr(valid_dicom_dataset, "ViewPosition"):
        delattr(valid_dicom_dataset, "ViewPosition")
    dcm_path = tmp_path / "test_no_view.dcm"
    valid_dicom_dataset.save_as(str(dcm_path), write_like_original=False)

    view = _extract_view_from_dicom(str(dcm_path))
    assert view is None

def test_extract_view_from_dicom_invalid(valid_dicom_dataset, tmp_path: Path) -> None:
    """Test handling of invalid ViewPosition values."""
    valid_dicom_dataset.ViewPosition = "INVALID"
    dcm_path = tmp_path / "test_invalid.dcm"
    valid_dicom_dataset.save_as(str(dcm_path), write_like_original=False)

    view = _extract_view_from_dicom(str(dcm_path))
    assert view is None

def test_extract_view_from_dicom_nonexistent() -> None:
    """Test handling of nonexistent DICOM file."""
    view = _extract_view_from_dicom("/nonexistent/path/file.dcm")
    assert view is None

def test_validate_split_overlap_no_overlap() -> None:
    """Test split validation with no overlap."""
    train_df = pd.DataFrame({"accession": ["ACC001", "ACC002"]})
    val_df = pd.DataFrame({"accession": ["ACC003", "ACC004"]})
    splits = {"train": train_df, "val": val_df}

    # Should not raise
    validate_split_overlap(splits)

def test_validate_split_overlap_with_overlap() -> None:
    """Test split validation detects overlap."""
    train_df = pd.DataFrame({"accession": ["ACC001", "ACC002"]})
    val_df = pd.DataFrame({"accession": ["ACC002", "ACC003"]})
    splits = {"train": train_df, "val": val_df}

    with pytest.raises(ValueError, match="Sobreposição detectada"):
        validate_split_overlap(splits)

def test_load_dataset_dataframe_from_paths(tmp_path: Path) -> None:
    """Test loading dataset from CSV with image paths."""
    image_path = tmp_path / "sample.png"
    _write_sample_image(image_path)

    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "image_path,professional_label,accession\n"
        f"{image_path},2,ACC001\n",
        encoding="utf-8",
    )

    df = load_dataset_dataframe(str(csv_path), dicom_root=None)
    assert list(df.columns) == ["image_path", "professional_label", "accession", "view"]
    assert len(df) == 1
    assert df.iloc[0]["professional_label"] == 2

    cache_mode = resolve_dataset_cache_mode("auto", df)
    assert cache_mode in {"memory", "none"}

def test_load_dataset_dataframe_from_features_dir(tmp_path: Path) -> None:
    """Test loading dataset from featureS.txt directory structure."""
    folder = tmp_path / "case_001"
    folder.mkdir()

    img_path = folder / "img_001.png"
    _write_sample_image(img_path)

    (folder / "featureS.txt").write_text("img_001\n1\n", encoding="utf-8")

    df = load_dataset_dataframe(str(tmp_path), dicom_root=None)
    assert len(df) == 1
    assert df.iloc[0]["image_path"].endswith("img_001.png")
    assert df.iloc[0]["professional_label"] == 2

def test_load_dataset_dataframe_missing_csv() -> None:
    """Test error handling when CSV path not provided."""
    with pytest.raises(ValueError, match="csv_path não definido"):
        load_dataset_dataframe(None, dicom_root=None)

def test_load_dataset_dataframe_classification_csv_requires_dicom_root(tmp_path: Path) -> None:
    csv_path = tmp_path / "classificacao.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\n"
        "ACC001,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dicom_root is required"):
        load_dataset_dataframe(str(csv_path), dicom_root=None, auto_detect=False)

def test_load_dataset_dataframe_classification_csv_integer_labels(
    valid_dicom_dataset, tmp_path: Path
) -> None:
    """Test loading from classificacao.csv with integer Classification labels."""
    # Create archive directory structure
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    # Create DICOM files with AccessionNumbers
    acc001_dir = archive_dir / "ACC001"
    acc001_dir.mkdir()
    valid_dicom_dataset.ViewPosition = "CC"
    dcm1_path = acc001_dir / "image1.dcm"
    valid_dicom_dataset.save_as(str(dcm1_path), write_like_original=False)

    acc002_dir = archive_dir / "ACC002"
    acc002_dir.mkdir()
    valid_dicom_dataset.ViewPosition = "MLO"
    dcm2_path = acc002_dir / "image2.dcm"
    valid_dicom_dataset.save_as(str(dcm2_path), write_like_original=False)

    # Create classificacao.csv with integer labels
    csv_path = tmp_path / "classificacao.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\n"
        "ACC001,2\n"
        "ACC002,3\n",
        encoding="utf-8",
    )

    df = load_dataset_dataframe(str(csv_path), dicom_root=str(archive_dir))
    assert len(df) == 2
    assert "professional_label" in df.columns
    assert "accession" in df.columns
    assert "view" in df.columns
    assert "patient_id" in df.columns

    # Check labels were coerced correctly
    acc001_rows = df[df["accession"] == "ACC001"]
    assert len(acc001_rows) == 1
    assert acc001_rows.iloc[0]["professional_label"] == 2
    assert acc001_rows.iloc[0]["view"] == "CC"
    assert acc001_rows.iloc[0]["patient_id"] == valid_dicom_dataset.PatientID

    acc002_rows = df[df["accession"] == "ACC002"]
    assert len(acc002_rows) == 1
    assert acc002_rows.iloc[0]["professional_label"] == 3
    assert acc002_rows.iloc[0]["view"] == "MLO"
    assert acc002_rows.iloc[0]["patient_id"] == valid_dicom_dataset.PatientID

def test_load_dataset_dataframe_classification_csv_birads_labels(
    valid_dicom_dataset, tmp_path: Path
) -> None:
    """Test loading from classificacao.csv with BI-RADS letter Classification labels."""
    # Create archive directory structure
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    # Create DICOM file with AccessionNumber
    acc003_dir = archive_dir / "ACC003"
    acc003_dir.mkdir()
    valid_dicom_dataset.ViewPosition = "CC"
    dcm_path = acc003_dir / "image.dcm"
    valid_dicom_dataset.save_as(str(dcm_path), write_like_original=False)

    # Create classificacao.csv with BI-RADS letters
    csv_path = tmp_path / "classificacao.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\n"
        "ACC003,B\n",
        encoding="utf-8",
    )

    df = load_dataset_dataframe(str(csv_path), dicom_root=str(archive_dir))
    assert len(df) == 1
    assert df.iloc[0]["professional_label"] == 2  # B -> 2
    assert df.iloc[0]["accession"] == "ACC003"

def test_load_dataset_dataframe_classification_csv_mixed_labels(
    valid_dicom_dataset, tmp_path: Path
) -> None:
    """Test loading from classificacao.csv with mixed label formats."""
    # Create archive directory structure
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    # Create DICOM files
    for acc_num in ["ACC004", "ACC005", "ACC006"]:
        acc_dir = archive_dir / acc_num
        acc_dir.mkdir()
        dcm_path = acc_dir / "image.dcm"
        valid_dicom_dataset.save_as(str(dcm_path), write_like_original=False)

    # Create classificacao.csv with mixed formats
    csv_path = tmp_path / "classificacao.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\n"
        "ACC004,1\n"  # Integer
        "ACC005,C\n"  # BI-RADS letter
        "ACC006,2\n",  # String "2"
        encoding="utf-8",
    )

    df = load_dataset_dataframe(str(csv_path), dicom_root=str(archive_dir))
    assert len(df) == 3

    # Verify each label was coerced correctly
    assert df[df["accession"] == "ACC004"].iloc[0]["professional_label"] == 1
    assert df[df["accession"] == "ACC005"].iloc[0]["professional_label"] == 3  # C -> 3
    assert df[df["accession"] == "ACC006"].iloc[0]["professional_label"] == 2

def test_load_multiple_csvs(tmp_path: Path) -> None:
    """Test loading multiple CSV files for splits."""
    # Create train CSV
    train_img = tmp_path / "train.png"
    _write_sample_image(train_img)
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image_path,professional_label,accession\n"
        f"{train_img},1,TRAIN001\n",
        encoding="utf-8",
    )

    # Create val CSV
    val_img = tmp_path / "val.png"
    _write_sample_image(val_img)
    val_csv = tmp_path / "val.csv"
    val_csv.write_text(
        "image_path,professional_label,accession\n"
        f"{val_img},2,VAL001\n",
        encoding="utf-8",
    )

    csv_paths = {"train": str(train_csv), "val": str(val_csv)}
    result = load_multiple_csvs(csv_paths)

    assert "train" in result
    assert "val" in result
    assert len(result["train"]) == 1
    assert len(result["val"]) == 1

def test_read_csv_with_encoding_utf8(tmp_path: Path) -> None:
    """Test CSV reading with UTF-8 encoding."""
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("col1,col2\nvalue1,value2\n", encoding="utf-8")

    df = _read_csv_with_encoding(str(csv_path))
    assert len(df) == 1
    assert list(df.columns) == ["col1", "col2"]

def test_read_csv_with_encoding_latin1(tmp_path: Path) -> None:
    """Test CSV reading with latin-1 encoding fallback."""
    csv_path = tmp_path / "test.csv"
    # Write with latin-1 encoding
    csv_path.write_bytes(b"col1,col2\n\xe9\xe7\xe0,value\n")

    df = _read_csv_with_encoding(str(csv_path))
    assert len(df) == 1
