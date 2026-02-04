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

from mammography.data.csv_loader import (
    _coerce_density_label,
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


def test_load_dataset_dataframe_with_preset(tmp_path: Path) -> None:
    """Test loading dataset with preset configuration."""
    folder = tmp_path / "case_001"
    folder.mkdir()

    img_path = folder / "img_001.png"
    _write_sample_image(img_path)

    (folder / "featureS.txt").write_text("img_001\n2\n", encoding="utf-8")

    # Should resolve paths from preset
    df = load_dataset_dataframe(
        csv_path=None,
        dicom_root=None,
        dataset="mamografias",
    )


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

    with pytest.raises(ValueError, match="Sobreposicao"):
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
