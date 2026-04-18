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

def test_create_splits_rejects_unstratified_fallback_when_ensuring_classes() -> None:
    """Stratification failures should raise unless fallback is explicitly allowed."""
    df = pd.DataFrame({
        "image_path": [f"img_{i}.png" for i in range(6)],
        "professional_label": [1, 1, 2, 2, 3, 3],
    })

    with pytest.raises(ValueError):
        create_splits(df, val_frac=0.2, seed=42, ensure_val_has_all_classes=True)

    train_df, val_df = create_splits(
        df,
        val_frac=0.2,
        seed=42,
        ensure_val_has_all_classes=False,
    )
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
