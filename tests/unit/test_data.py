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


# ==================== CSV Loader Tests ====================


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


# ==================== Splits Tests ====================


def test_create_splits_basic(tmp_path: Path) -> None:
    """Test basic train/val split creation."""
    # Create simple dataframe
    df = pd.DataFrame({
        "image_path": [f"img_{i}.png" for i in range(20)],
        "professional_label": [1, 2, 3, 4] * 5,
        "accession": [f"ACC{i:03d}" for i in range(20)],
    })

    train_df, val_df = create_splits(df, val_frac=0.2, seed=42)

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(train_df) + len(val_df) == len(df)


def test_create_splits_without_accession(tmp_path: Path) -> None:
    """Test split creation without accession column."""
    df = pd.DataFrame({
        "image_path": [f"img_{i}.png" for i in range(20)],
        "professional_label": [1, 2, 3, 4] * 5,
    })

    train_df, val_df = create_splits(df, val_frac=0.2, seed=42)

    assert len(train_df) > 0
    assert len(val_df) > 0


def test_create_splits_single_accession_falls_back_to_random() -> None:
    """Split should work even when all rows share the same accession."""
    df = pd.DataFrame({
        "image_path": [f"img_{i}.png" for i in range(20)],
        "professional_label": [1, 2, 3, 4] * 5,
        "accession": ["PATCHES001"] * 20,
    })

    train_df, val_df = create_splits(df, val_frac=0.2, seed=42)

    assert len(train_df) > 0
    assert len(val_df) > 0


def test_create_splits_supports_custom_group_column_without_accession() -> None:
    """Custom group columns should work even when accession is absent."""
    rows = []
    for patient_idx in range(12):
        patient_id = f"PAT{patient_idx:03d}"
        label = (patient_idx % 4) + 1
        for image_idx in range(2):
            rows.append(
                {
                    "image_path": f"img_{patient_idx}_{image_idx}.png",
                    "professional_label": label,
                    "patient_id": patient_id,
                }
            )
    df = pd.DataFrame(rows)

    train_df, val_df = create_splits(
        df,
        val_frac=0.25,
        seed=42,
        ensure_val_has_all_classes=False,
        group_col="patient_id",
    )

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert set(train_df["patient_id"]).isdisjoint(set(val_df["patient_id"]))


def test_create_three_way_split(tmp_path: Path) -> None:
    """Test train/val/test split creation."""
    df = pd.DataFrame({
        "image_path": [f"img_{i}.png" for i in range(30)],
        "professional_label": [1, 2, 3, 4] * 7 + [1, 2],
        "accession": [f"ACC{i:03d}" for i in range(30)],
    })

    train_df, val_df, test_df = create_three_way_split(
        df, val_frac=0.15, test_frac=0.15, seed=42
    )

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(df)


def test_create_three_way_split_supports_custom_group_column_without_accession() -> None:
    """Three-way splits should honor custom grouping columns."""
    rows = []
    for patient_idx in range(16):
        patient_id = f"PAT{patient_idx:03d}"
        label = (patient_idx % 4) + 1
        for image_idx in range(2):
            rows.append(
                {
                    "image_path": f"img_{patient_idx}_{image_idx}.png",
                    "professional_label": label,
                    "patient_id": patient_id,
                }
            )
    df = pd.DataFrame(rows)

    train_df, val_df, test_df = create_three_way_split(
        df,
        val_frac=0.2,
        test_frac=0.2,
        seed=42,
        ensure_all_splits_have_all_classes=False,
        group_col="patient_id",
    )

    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    test_patients = set(test_df["patient_id"])

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)
def test_create_kfold_splits(tmp_path: Path) -> None:
    """Test k-fold cross-validation splits."""
    df = pd.DataFrame({
        "image_path": [f"img_{i}.png" for i in range(25)],
        "professional_label": [1, 2, 3, 4] * 6 + [1],
        "accession": [f"ACC{i:03d}" for i in range(25)],
    })

    folds = create_kfold_splits(df, n_splits=5, seed=42)

    assert len(folds) == 5
    for train_df, val_df in folds:
        assert len(train_df) > 0
        assert len(val_df) > 0


def test_create_kfold_splits_none_input() -> None:
    """Test k-fold with None input returns empty list."""
    folds = create_kfold_splits(None, n_splits=5)
    assert folds == []


def test_filter_by_view() -> None:
    """Test view filtering."""
    df = pd.DataFrame({
        "image_path": [f"img_{i}.png" for i in range(6)],
        "professional_label": [1, 2, 3, 1, 2, 3],
        "view": ["CC", "MLO", "CC", "MLO", "CC", "MLO"],
        "accession": [f"ACC{i:03d}" for i in range(6)],
    })

    cc_df = filter_by_view(df, "CC")
    assert len(cc_df) == 3
    assert all(cc_df["view"] == "CC")

    mlo_df = filter_by_view(df, "MLO")
    assert len(mlo_df) == 3
    assert all(mlo_df["view"] == "MLO")


def test_filter_by_view_missing_column() -> None:
    """Test error when view column missing."""
    df = pd.DataFrame({
        "image_path": ["img.png"],
        "professional_label": [1],
    })

    with pytest.raises(ValueError, match="coluna 'view'"):
        filter_by_view(df, "CC")


def test_load_splits_from_csvs(tmp_path: Path) -> None:
    """Test loading pre-defined splits from CSV files."""
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

    train_df, val_df, test_df = load_splits_from_csvs(
        str(train_csv), str(val_csv), None
    )

    assert len(train_df) == 1
    assert len(val_df) == 1
    assert test_df is None


def test_load_splits_from_csvs_with_overlap(tmp_path: Path) -> None:
    """Test error detection for overlapping splits."""
    img = tmp_path / "shared.png"
    _write_sample_image(img)

    # Create CSVs with overlapping data
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image_path,professional_label,accession\n"
        f"{img},1,ACC001\n",
        encoding="utf-8",
    )

    val_csv = tmp_path / "val.csv"
    val_csv.write_text(
        "image_path,professional_label,accession\n"
        f"{img},2,ACC001\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Sobreposição detectada"):
        load_splits_from_csvs(str(train_csv), str(val_csv), None)


# ==================== Format Detection Tests ====================


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


# ==================== Dataset Tests ====================


def test_embedding_store_lookup() -> None:
    """Test embedding store lookup functionality."""
    embeddings_by_accession = {
        "ACC001": torch.randn(2048),
        "ACC002": torch.randn(2048),
    }
    embeddings_by_path = {
        "/path/to/img1.png": torch.randn(2048),
    }

    store = EmbeddingStore(
        embeddings_by_accession=embeddings_by_accession,
        embeddings_by_path=embeddings_by_path,
        feature_dim=2048,
    )

    # Test accession lookup
    row = {"accession": "ACC001", "image_path": "/other/path.png"}
    emb = store.lookup(row)
    assert emb is not None
    assert emb.shape == (2048,)

    # Test path lookup
    row = {"accession": "UNKNOWN", "image_path": "/path/to/img1.png"}
    emb = store.lookup(row)
    assert emb is not None

    # Test missing lookup
    row = {"accession": "UNKNOWN", "image_path": "/missing.png"}
    emb = store.lookup(row)
    assert emb is None


def test_load_embedding_store_valid(tmp_path: Path) -> None:
    """Test loading embedding store from valid directory."""
    # Create valid features.npy and metadata.csv
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    features = np.random.randn(3, 2048).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    metadata = pd.DataFrame({
        "accession": ["ACC001", "ACC002", "ACC003"],
        "path": ["img1.png", "img2.png", "img3.png"],
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    # Load embedding store
    store = load_embedding_store(str(embeddings_dir))

    assert store.feature_dim == 2048
    assert len(store.embeddings_by_accession) == 3
    assert len(store.embeddings_by_path) >= 3  # May have normalized paths too
    assert "ACC001" in store.embeddings_by_accession
    assert store.embeddings_by_accession["ACC001"].shape == (2048,)


def test_load_embedding_store_1d_features(tmp_path: Path) -> None:
    """Test loading embedding store with 1D features (should be reshaped to 2D)."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    # 1D features should be reshaped to (1, 2048)
    features = np.random.randn(2048).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    metadata = pd.DataFrame({
        "accession": ["ACC001"],
        "path": ["img1.png"],
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    # Load embedding store
    store = load_embedding_store(str(embeddings_dir))

    assert store.feature_dim == 2048
    assert len(store.embeddings_by_accession) == 1
    assert "ACC001" in store.embeddings_by_accession


def test_load_embedding_store_missing_features(tmp_path: Path) -> None:
    """Test error when features.npy is missing."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    # Only create metadata.csv
    metadata = pd.DataFrame({
        "accession": ["ACC001"],
        "path": ["img1.png"],
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    with pytest.raises(FileNotFoundError, match="features.npy nao encontrado"):
        load_embedding_store(str(embeddings_dir))


def test_load_embedding_store_missing_metadata(tmp_path: Path) -> None:
    """Test error when metadata.csv is missing."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    # Only create features.npy
    features = np.random.randn(3, 2048).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    with pytest.raises(FileNotFoundError, match="metadata.csv nao encontrado"):
        load_embedding_store(str(embeddings_dir))


def test_load_embedding_store_invalid_features_shape(tmp_path: Path) -> None:
    """Test error when features.npy has invalid shape (3D+)."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    # 3D features are invalid
    features = np.random.randn(3, 2048, 5).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    metadata = pd.DataFrame({
        "accession": ["ACC001", "ACC002", "ACC003"],
        "path": ["img1.png", "img2.png", "img3.png"],
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    with pytest.raises(ValueError, match="deve ter 2 dimensoes"):
        load_embedding_store(str(embeddings_dir))


def test_load_embedding_store_shape_mismatch(tmp_path: Path) -> None:
    """Test error when metadata rows don't match features rows."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    # 3 rows in features
    features = np.random.randn(3, 2048).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    # But only 2 rows in metadata
    metadata = pd.DataFrame({
        "accession": ["ACC001", "ACC002"],
        "path": ["img1.png", "img2.png"],
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    with pytest.raises(ValueError, match="nao bate com features.npy"):
        load_embedding_store(str(embeddings_dir))


def test_load_embedding_store_no_valid_columns(tmp_path: Path) -> None:
    """Test error when metadata has no valid accession/path columns."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    features = np.random.randn(2, 2048).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    # Metadata with neither accession nor path/image_path columns
    metadata = pd.DataFrame({
        "id": [1, 2],
        "label": [0, 1],
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    with pytest.raises(ValueError, match="nao possui colunas 'accession'/'path' validas"):
        load_embedding_store(str(embeddings_dir))


def test_load_embedding_store_with_image_path_column(tmp_path: Path) -> None:
    """Test loading with 'image_path' column instead of 'path'."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    features = np.random.randn(2, 2048).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    metadata = pd.DataFrame({
        "accession": ["ACC001", "ACC002"],
        "image_path": ["img1.png", "img2.png"],  # Use image_path instead of path
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    store = load_embedding_store(str(embeddings_dir))

    assert store.feature_dim == 2048
    assert len(store.embeddings_by_accession) == 2
    assert "img1.png" in store.embeddings_by_path


def test_load_embedding_store_nan_accessions(tmp_path: Path) -> None:
    """Test loading with NaN/empty accessions (should be skipped)."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    features = np.random.randn(4, 2048).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    metadata = pd.DataFrame({
        "accession": ["ACC001", None, np.nan, ""],
        "path": ["img1.png", "img2.png", "img3.png", "img4.png"],
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    store = load_embedding_store(str(embeddings_dir))

    # Only ACC001 should be in accession index (others are invalid)
    assert len(store.embeddings_by_accession) == 1
    assert "ACC001" in store.embeddings_by_accession
    # But all 4 paths should be indexed
    assert len(store.embeddings_by_path) >= 4


def test_mammo_density_dataset_creation(tmp_path: Path) -> None:
    """Test MammoDensityDataset instantiation."""
    img_path = tmp_path / "test.png"
    _write_sample_image(img_path)

    rows = [
        {
            "image_path": str(img_path),
            "professional_label": 2,
            "accession": "ACC001",
        }
    ]

    dataset = MammoDensityDataset(
        rows=rows,
        img_size=224,
        train=True,
        augment=False,
        cache_mode="none",
    )

    assert len(dataset) == 1

    # Test __getitem__
    item = dataset[0]
    assert item is not None
    img, label, meta, emb = item
    assert img.shape[0] == 3  # RGB channels
    assert label == 1  # 2 -> 1 (zero-indexed)
    assert meta["accession"] == "ACC001"


def test_mammo_density_dataset_cache_modes(tmp_path: Path) -> None:
    """Test different cache modes."""
    img_path = tmp_path / "test.png"
    _write_sample_image(img_path)

    rows = [{"image_path": str(img_path), "professional_label": 1, "accession": "ACC001"}]

    # Test memory cache
    dataset_mem = MammoDensityDataset(
        rows=rows, img_size=224, train=False, cache_mode="memory"
    )
    assert len(dataset_mem) == 1

    # Test none cache
    dataset_none = MammoDensityDataset(
        rows=rows, img_size=224, train=False, cache_mode="none"
    )
    assert len(dataset_none) == 1


def test_mammo_density_dataset_invalid_cache_mode() -> None:
    """Test error on invalid cache mode."""
    rows = [{"image_path": "test.png", "professional_label": 1}]

    with pytest.raises(ValueError, match="cache_mode inválido"):
        MammoDensityDataset(
            rows=rows, img_size=224, train=False, cache_mode="invalid"
        )


def test_mammo_density_dataset_cache_dir_required() -> None:
    """Test error when cache_dir missing for disk cache."""
    rows = [{"image_path": "test.png", "professional_label": 1}]

    with pytest.raises(ValueError, match="cache_dir é obrigatório"):
        MammoDensityDataset(
            rows=rows, img_size=224, train=False, cache_mode="disk"
        )


def test_robust_collate_filters_none() -> None:
    """Test robust_collate filters out None samples."""
    valid_sample = (
        torch.randn(3, 224, 224),
        0,
        {"path": "img1.png"},
        None,
    )

    batch = [valid_sample, None, valid_sample]
    result = robust_collate(batch)

    assert result is not None
    imgs, labels, meta, embs = result
    assert imgs.shape[0] == 2  # Only valid samples
    assert labels.shape[0] == 2


def test_robust_collate_all_none() -> None:
    """Test robust_collate returns None for all-None batch."""
    batch = [None, None, None]
    result = robust_collate(batch)
    assert result is None


# ==================== DICOM Loading Tests ====================


def test_mammo_density_dataset_dicom_loading(dicom_file_path: Path) -> None:
    """Test MammoDensityDataset loading DICOM files."""
    rows = [
        {
            "image_path": str(dicom_file_path),
            "professional_label": 3,
            "accession": "DICOM001",
        }
    ]

    dataset = MammoDensityDataset(
        rows=rows,
        img_size=128,
        train=False,
        augment=False,
        cache_mode="none",
    )

    assert len(dataset) == 1

    # Test DICOM loading
    item = dataset[0]
    assert item is not None
    img, label, meta, emb = item
    assert img.shape == (3, 128, 128)  # RGB channels, resized
    assert label == 2  # 3 -> 2 (zero-indexed)
    assert meta["accession"] == "DICOM001"


def test_mammo_density_dataset_dicom_with_cache(dicom_file_path: Path, tmp_path: Path) -> None:
    """Test MammoDensityDataset DICOM loading with memory cache."""
    rows = [
        {
            "image_path": str(dicom_file_path),
            "professional_label": 2,
            "accession": "DICOM002",
        }
    ]

    dataset = MammoDensityDataset(
        rows=rows,
        img_size=128,
        train=False,
        augment=False,
        cache_mode="memory",
    )

    # First access - loads DICOM
    item1 = dataset[0]
    assert item1 is not None
    img1, label1, meta1, _ = item1
    assert img1.shape == (3, 128, 128)

    # Second access - should use cache
    item2 = dataset[0]
    assert item2 is not None
    img2, label2, meta2, _ = item2
    assert img2.shape == (3, 128, 128)
    assert label1 == label2


def test_mammo_density_dataset_dicom_tensor_disk_cache(dicom_file_path: Path, tmp_path: Path) -> None:
    """Test MammoDensityDataset DICOM with tensor-disk cache."""
    cache_dir = tmp_path / "tensor_cache"
    cache_dir.mkdir()

    rows = [
        {
            "image_path": str(dicom_file_path),
            "professional_label": 4,
            "accession": "DICOM003",
        }
    ]

    dataset = MammoDensityDataset(
        rows=rows,
        img_size=128,
        train=False,
        augment=False,
        cache_mode="tensor-disk",
        cache_dir=str(cache_dir),
        split_name="test",
    )

    # First access - creates cache
    item = dataset[0]
    assert item is not None
    img, label, meta, _ = item
    assert img.shape == (3, 128, 128)
    assert label == 3  # 4 -> 3 (zero-indexed)

    # Verify cache file was created
    cache_files = list(cache_dir.glob("*.pt"))
    assert len(cache_files) > 0


def test_mammo_density_dataset_memory_cache_with_getitem(tmp_path: Path) -> None:
    """Test memory cache mode with __getitem__ to verify caching behavior."""
    # Create multiple test images
    img_paths = []
    for i in range(3):
        img_path = tmp_path / f"test_{i}.png"
        _write_sample_image(img_path)
        img_paths.append(img_path)

    rows = [
        {
            "image_path": str(img_paths[0]),
            "professional_label": 1,
            "accession": "MEM001",
        },
        {
            "image_path": str(img_paths[1]),
            "professional_label": 2,
            "accession": "MEM002",
        },
        {
            "image_path": str(img_paths[2]),
            "professional_label": 3,
            "accession": "MEM003",
        },
    ]

    dataset = MammoDensityDataset(
        rows=rows,
        img_size=128,
        train=False,
        augment=False,
        cache_mode="memory",
    )

    # Verify cache is initialized but empty
    assert dataset._image_cache is not None
    assert len(dataset._image_cache) == 0

    # First access - should populate cache
    item0 = dataset[0]
    assert item0 is not None
    img0, label0, meta0, _ = item0
    assert img0.shape == (3, 128, 128)
    assert label0 == 0  # 1 -> 0 (zero-indexed)
    assert meta0["accession"] == "MEM001"

    # Cache should now contain one entry
    assert len(dataset._image_cache) == 1

    # Access different item - should add to cache
    item1 = dataset[1]
    assert item1 is not None
    img1, label1, meta1, _ = item1
    assert img1.shape == (3, 128, 128)
    assert label1 == 1  # 2 -> 1 (zero-indexed)
    assert meta1["accession"] == "MEM002"

    # Cache should now contain two entries
    assert len(dataset._image_cache) == 2

    # Re-access first item - should use cached version
    item0_again = dataset[0]
    assert item0_again is not None
    img0_again, label0_again, meta0_again, _ = item0_again
    assert img0_again.shape == (3, 128, 128)
    assert label0_again == label0

    # Cache size should remain the same (no new entry)
    assert len(dataset._image_cache) == 2

    # Access all items to fully populate cache
    item2 = dataset[2]
    assert item2 is not None
    assert len(dataset._image_cache) == 3


def test_mammo_density_dataset_auto_normalize(tmp_path: Path) -> None:
    """Test auto-normalization computes and applies stats correctly."""
    # Create multiple test images to compute stats from
    img_paths = []
    for i in range(10):
        img_path = tmp_path / f"auto_norm_{i}.png"
        # Create images with varying pixel values
        img = Image.new("RGB", (32, 32), color=(100 + i * 10, 50 + i * 5, 150 - i * 5))
        img.save(img_path)
        img_paths.append(img_path)

    rows = [
        {
            "image_path": str(img_path),
            "professional_label": (i % 4) + 1,
            "accession": f"AUTO{i:03d}",
        }
        for i, img_path in enumerate(img_paths)
    ]

    # Create dataset with auto-normalization enabled
    dataset = MammoDensityDataset(
        rows=rows,
        img_size=64,
        train=False,
        augment=False,
        cache_mode="none",
        auto_normalize=True,
        auto_normalize_samples=5,  # Use subset for speed
    )

    # Verify normalization stats were computed
    assert dataset._norm_mean is not None
    assert dataset._norm_std is not None
    assert len(dataset._norm_mean) == 3  # RGB channels
    assert len(dataset._norm_std) == 3

    # Stats should be non-trivial (not passthrough values)
    # Mean should not be exactly [0.0, 0.0, 0.0]
    assert not all(m == 0.0 for m in dataset._norm_mean)
    # Std should not be exactly [1.0, 1.0, 1.0]
    assert not all(s == 1.0 for s in dataset._norm_std)

    # Verify dataset can load items with computed normalization
    item = dataset[0]
    assert item is not None
    img, label, meta, _ = item
    assert img.shape == (3, 64, 64)

    # Check that normalization is applied (tensor should be normalized)
    # Mean should be close to 0 and std close to 1 after normalization
    # (allowing some variance since we use a small subset)
    img_mean = img.mean(dim=[1, 2])
    img_std = img.std(dim=[1, 2])

    # Check values are in reasonable normalized range
    assert torch.all(img_mean.abs() < 3.0)  # Should be somewhat centered
    assert torch.all(img_std > 0.1)  # Should have some variance


def test_mammo_density_dataset_auto_normalize_disabled(tmp_path: Path) -> None:
    """Test that auto-normalization can be disabled with explicit mean/std."""
    img_path = tmp_path / "test.png"
    _write_sample_image(img_path)

    rows = [
        {
            "image_path": str(img_path),
            "professional_label": 2,
            "accession": "EXPLICIT001",
        }
    ]

    # Provide explicit normalization values
    explicit_mean = [0.5, 0.5, 0.5]
    explicit_std = [0.25, 0.25, 0.25]

    dataset = MammoDensityDataset(
        rows=rows,
        img_size=128,
        train=False,
        augment=False,
        cache_mode="none",
        auto_normalize=True,  # Even with auto enabled
        mean=explicit_mean,  # Explicit values should take precedence
        std=explicit_std,
    )

    # Verify explicit values were used (not auto-computed)
    assert dataset._norm_mean == explicit_mean
    assert dataset._norm_std == explicit_std

    # Verify dataset works correctly
    item = dataset[0]
    assert item is not None
    img, label, meta, _ = item
    assert img.shape == (3, 128, 128)


# ==================== Augmentation Tests ====================


def test_mammo_density_dataset_augmentation_enabled(tmp_path: Path) -> None:
    """Test MammoDensityDataset with augmentation enabled in train mode."""
    img_path = tmp_path / "test.png"
    _write_sample_image(img_path)

    rows = [
        {
            "image_path": str(img_path),
            "professional_label": 2,
            "accession": "AUG001",
        }
    ]

    dataset = MammoDensityDataset(
        rows=rows,
        img_size=224,
        train=True,
        augment=True,
        cache_mode="none",
    )

    assert dataset.augment is True
    assert dataset.train is True

    # Get multiple samples to verify augmentation varies
    item1 = dataset[0]
    item2 = dataset[0]
    assert item1 is not None
    assert item2 is not None

    img1, _, _, _ = item1
    img2, _, _, _ = item2

    # Images should potentially differ due to random augmentation
    assert img1.shape == img2.shape == (3, 224, 224)


def test_mammo_density_dataset_augmentation_disabled_in_eval(tmp_path: Path) -> None:
    """Test augmentation is disabled when train=False."""
    img_path = tmp_path / "test.png"
    _write_sample_image(img_path)

    rows = [
        {
            "image_path": str(img_path),
            "professional_label": 1,
            "accession": "AUG002",
        }
    ]

    dataset = MammoDensityDataset(
        rows=rows,
        img_size=224,
        train=False,
        augment=True,  # Requested but should be disabled
        cache_mode="none",
    )

    # Augmentation should be disabled in eval mode
    assert dataset.augment is False
    assert dataset.train is False


def test_mammo_density_dataset_vertical_augmentation(tmp_path: Path) -> None:
    """Test vertical flip augmentation."""
    img_path = tmp_path / "test.png"
    _write_sample_image(img_path)

    rows = [
        {
            "image_path": str(img_path),
            "professional_label": 3,
            "accession": "AUG003",
        }
    ]

    dataset = MammoDensityDataset(
        rows=rows,
        img_size=224,
        train=True,
        augment=True,
        augment_vertical=True,
        cache_mode="none",
    )

    assert dataset.augment_vertical is True

    # Test that dataset can be accessed (vertical flip is random)
    item = dataset[0]
    assert item is not None
    img, label, _, _ = item
    assert img.shape == (3, 224, 224)
    assert label == 2  # 3 -> 2 (zero-indexed)


def test_mammo_density_dataset_color_augmentation(tmp_path: Path) -> None:
    """Test color jitter augmentation."""
    img_path = tmp_path / "test.png"
    _write_sample_image(img_path)

    rows = [
        {
            "image_path": str(img_path),
            "professional_label": 4,
            "accession": "AUG004",
        }
    ]

    dataset = MammoDensityDataset(
        rows=rows,
        img_size=224,
        train=True,
        augment=True,
        augment_color=True,
        cache_mode="none",
    )

    assert dataset.augment_color is True

    # Test that dataset can be accessed (color jitter is random)
    item = dataset[0]
    assert item is not None
    img, label, _, _ = item
    assert img.shape == (3, 224, 224)
    assert label == 3  # 4 -> 3 (zero-indexed)


def test_mammo_density_dataset_rotation_augmentation(tmp_path: Path) -> None:
    """Test rotation augmentation with custom degrees."""
    img_path = tmp_path / "test.png"
    _write_sample_image(img_path)

    rows = [
        {
            "image_path": str(img_path),
            "professional_label": 2,
            "accession": "AUG005",
        }
    ]

    dataset = MammoDensityDataset(
        rows=rows,
        img_size=224,
        train=True,
        augment=True,
        rotation_deg=10.0,
        cache_mode="none",
    )

    assert dataset.rotation_deg == 10.0

    # Test that dataset can be accessed (rotation is random)
    item = dataset[0]
    assert item is not None
    img, label, _, _ = item
    assert img.shape == (3, 224, 224)
    assert label == 1  # 2 -> 1 (zero-indexed)


def test_mammo_density_dataset_all_augmentations(tmp_path: Path) -> None:
    """Test all augmentation options enabled together."""
    img_path = tmp_path / "test.png"
    _write_sample_image(img_path)

    rows = [
        {
            "image_path": str(img_path),
            "professional_label": 3,
            "accession": "AUG006",
        }
    ]

    dataset = MammoDensityDataset(
        rows=rows,
        img_size=224,
        train=True,
        augment=True,
        augment_vertical=True,
        augment_color=True,
        rotation_deg=15.0,
        cache_mode="none",
    )

    assert dataset.augment is True
    assert dataset.augment_vertical is True
    assert dataset.augment_color is True
    assert dataset.rotation_deg == 15.0

    # Test that all augmentations can be applied
    item = dataset[0]
    assert item is not None
    img, label, meta, _ = item
    assert img.shape == (3, 224, 224)
    assert label == 2  # 3 -> 2 (zero-indexed)
    assert meta["accession"] == "AUG006"


def test_mammo_density_dataset_dicom_with_augmentation(dicom_file_path: Path) -> None:
    """Test DICOM loading combined with augmentation."""
    rows = [
        {
            "image_path": str(dicom_file_path),
            "professional_label": 2,
            "accession": "DICOM_AUG001",
        }
    ]

    dataset = MammoDensityDataset(
        rows=rows,
        img_size=128,
        train=True,
        augment=True,
        augment_vertical=True,
        augment_color=True,
        rotation_deg=5.0,
        cache_mode="none",
    )

    # Test DICOM loading with augmentation
    item = dataset[0]
    assert item is not None
    img, label, meta, _ = item
    assert img.shape == (3, 128, 128)
    assert label == 1  # 2 -> 1 (zero-indexed)
    assert meta["accession"] == "DICOM_AUG001"


# ==================== Transform Pipeline Tests ====================


def test_mammo_density_dataset_transform_consistency(tmp_path: Path) -> None:
    """Test that transforms produce consistent output shapes."""
    img_path = tmp_path / "test.png"
    _write_sample_image(img_path)

    rows = [
        {
            "image_path": str(img_path),
            "professional_label": 1,
            "accession": "TRANS001",
        }
    ]

    # Test different image sizes
    for size in [128, 224, 256]:
        dataset = MammoDensityDataset(
            rows=rows,
            img_size=size,
            train=False,
            augment=False,
            cache_mode="none",
        )

        item = dataset[0]
        assert item is not None
        img, _, _, _ = item
        assert img.shape == (3, size, size)


def test_mammo_density_dataset_dicom_transform_pipeline(dicom_file_path: Path) -> None:
    """Test complete transform pipeline for DICOM files."""
    rows = [
        {
            "image_path": str(dicom_file_path),
            "professional_label": 4,
            "accession": "DICOM_TRANS001",
        }
    ]

    dataset = MammoDensityDataset(
        rows=rows,
        img_size=224,
        train=True,
        augment=True,
        rotation_deg=5.0,
        cache_mode="none",
    )

    # Test transform pipeline: DICOM -> PIL -> Tensor -> Augment -> Normalize
    item = dataset[0]
    assert item is not None
    img, label, meta, _ = item

    # Verify tensor properties
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert img.dtype == torch.float32

    # Verify normalization (values should be roughly in [-3, 3] range)
    assert img.min() >= -5.0
    assert img.max() <= 5.0

    # Verify label mapping
    assert label == 3  # 4 -> 3 (zero-indexed)
    assert meta["accession"] == "DICOM_TRANS001"
