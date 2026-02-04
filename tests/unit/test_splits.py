from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pd = pytest.importorskip("pandas")

from mammography.data.csv_loader import validate_split_overlap
from mammography.data.splits import create_kfold_splits, create_three_way_split, load_splits_from_csvs


def _create_sample_dataframe(num_samples: int = 100, with_accession: bool = True, num_classes: int = 4) -> pd.DataFrame:
    """Create a sample DataFrame for testing splits."""
    data = {
        "image_path": [f"/path/to/image_{i}.png" for i in range(num_samples)],
        "professional_label": [(i % num_classes) + 1 for i in range(num_samples)],
    }
    if with_accession:
        # Create groups with multiple images per accession
        data["accession"] = [f"ACC{i // 3:03d}" for i in range(num_samples)]
    return pd.DataFrame(data)


def test_create_three_way_split() -> None:
    """Test basic three-way split - main verification test."""
    df = _create_sample_dataframe(num_samples=100, with_accession=False, num_classes=4)

    train_df, val_df, test_df = create_three_way_split(
        df, val_frac=0.15, test_frac=0.15, seed=42, num_classes=4
    )

    # Check sizes are reasonable
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # Check total adds up
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # Check no overlaps
    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])
    test_paths = set(test_df["image_path"])

    assert len(train_paths & val_paths) == 0
    assert len(train_paths & test_paths) == 0
    assert len(val_paths & test_paths) == 0

    # Check required columns are present
    assert "professional_label" in train_df.columns
    assert "professional_label" in val_df.columns
    assert "professional_label" in test_df.columns


def test_create_three_way_split_basic() -> None:
    """Test basic three-way split without groups."""
    df = _create_sample_dataframe(num_samples=100, with_accession=False, num_classes=4)

    train_df, val_df, test_df = create_three_way_split(
        df, val_frac=0.15, test_frac=0.15, seed=42, num_classes=4
    )

    # Check sizes are reasonable
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # Check total adds up
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # Check no overlaps
    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])
    test_paths = set(test_df["image_path"])

    assert len(train_paths & val_paths) == 0
    assert len(train_paths & test_paths) == 0
    assert len(val_paths & test_paths) == 0

    # Check required columns are present
    assert "professional_label" in train_df.columns
    assert "professional_label" in val_df.columns
    assert "professional_label" in test_df.columns


def test_create_three_way_split_with_groups() -> None:
    """Test three-way split with group-aware stratification."""
    df = _create_sample_dataframe(num_samples=120, with_accession=True, num_classes=4)

    train_df, val_df, test_df = create_three_way_split(
        df, val_frac=0.15, test_frac=0.15, seed=42, num_classes=4
    )

    # Check sizes
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # Check no group leakage (accessions should not overlap)
    train_acc = set(train_df["accession"])
    val_acc = set(val_df["accession"])
    test_acc = set(test_df["accession"])

    assert len(train_acc & val_acc) == 0
    assert len(train_acc & test_acc) == 0
    assert len(val_acc & test_acc) == 0

    # Check total adds up
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_create_three_way_split_reproducibility() -> None:
    """Test that splits are reproducible with same seed."""
    df = _create_sample_dataframe(num_samples=100, with_accession=True, num_classes=4)

    train1, val1, test1 = create_three_way_split(df, seed=42)
    train2, val2, test2 = create_three_way_split(df, seed=42)

    # Check same splits are produced
    assert set(train1["image_path"]) == set(train2["image_path"])
    assert set(val1["image_path"]) == set(val2["image_path"])
    assert set(test1["image_path"]) == set(test2["image_path"])


def test_create_three_way_split_different_seeds() -> None:
    """Test that different seeds produce different splits."""
    df = _create_sample_dataframe(num_samples=100, with_accession=True, num_classes=4)

    train1, val1, test1 = create_three_way_split(df, seed=42)
    train2, val2, test2 = create_three_way_split(df, seed=123)

    # Splits should be different
    assert set(train1["image_path"]) != set(train2["image_path"])


def test_create_three_way_split_two_classes() -> None:
    """Test split with binary classification (num_classes=2)."""
    df = _create_sample_dataframe(num_samples=100, with_accession=False, num_classes=4)

    train_df, val_df, test_df = create_three_way_split(
        df, val_frac=0.2, test_frac=0.2, seed=42, num_classes=2
    )

    # Check all splits have data
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # Total should equal original
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_create_three_way_split_custom_fractions() -> None:
    """Test split with custom validation and test fractions."""
    df = _create_sample_dataframe(num_samples=100, with_accession=False, num_classes=4)

    train_df, val_df, test_df = create_three_way_split(
        df, val_frac=0.25, test_frac=0.25, seed=42
    )

    # Check approximate fractions (allowing for rounding)
    total = len(df)
    assert len(test_df) / total > 0.20  # At least 20%
    assert len(test_df) / total < 0.30  # At most 30%
    assert len(val_df) / total > 0.20
    assert len(val_df) / total < 0.30


def test_create_three_way_split_ensure_all_classes() -> None:
    """Test that ensuring all classes works."""
    # Create unbalanced dataset
    data = {
        "image_path": [f"/path/to/image_{i}.png" for i in range(100)],
        "professional_label": [1] * 50 + [2] * 30 + [3] * 15 + [4] * 5,
        "accession": [f"ACC{i:03d}" for i in range(100)],
    }
    df = pd.DataFrame(data)

    train_df, val_df, test_df = create_three_way_split(
        df,
        val_frac=0.15,
        test_frac=0.15,
        seed=42,
        num_classes=4,
        ensure_all_splits_have_all_classes=True,
        max_tries=200,
    )

    # All splits should have data
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0


def test_create_three_way_split_missing_column() -> None:
    """Test error handling when professional_label column is missing."""
    df = pd.DataFrame({
        "image_path": [f"/path/to/image_{i}.png" for i in range(10)],
    })

    with pytest.raises(ValueError, match="professional_label"):
        create_three_way_split(df)


def test_create_three_way_split_invalid_labels() -> None:
    """Test handling of invalid labels."""
    df = pd.DataFrame({
        "image_path": [f"/path/to/image_{i}.png" for i in range(10)],
        "professional_label": [None] * 10,
    })

    with pytest.raises(RuntimeError, match="Nenhuma amostra valida"):
        create_three_way_split(df, num_classes=4)


def test_create_three_way_split_insufficient_groups() -> None:
    """Test error when there are insufficient groups for splitting."""
    df = pd.DataFrame({
        "image_path": ["/path/to/image_1.png", "/path/to/image_2.png"],
        "professional_label": [1, 2],
        "accession": ["ACC001", "ACC002"],
    })

    with pytest.raises(RuntimeError, match="Grupos insuficientes"):
        create_three_way_split(df, num_classes=4)


def test_create_three_way_split_label_mapping_4class() -> None:
    """Test correct label mapping for 4-class classification."""
    df = pd.DataFrame({
        "image_path": [f"/path/to/image_{i}.png" for i in range(100)],
        "professional_label": [1, 2, 3, 4] * 25,
        "accession": [f"ACC{i // 4:03d}" for i in range(100)],
    })

    train_df, val_df, test_df = create_three_way_split(
        df, val_frac=0.15, test_frac=0.15, seed=42, num_classes=4
    )

    # Check labels are still in original range
    all_labels = list(train_df["professional_label"]) + list(val_df["professional_label"]) + list(test_df["professional_label"])
    unique_labels = set(all_labels)
    assert unique_labels.issubset({1, 2, 3, 4})


def test_create_three_way_split_label_mapping_2class() -> None:
    """Test correct label mapping for 2-class classification."""
    df = pd.DataFrame({
        "image_path": [f"/path/to/image_{i}.png" for i in range(100)],
        "professional_label": [1, 2, 3, 4] * 25,
        "accession": [f"ACC{i // 4:03d}" for i in range(100)],
    })

    train_df, val_df, test_df = create_three_way_split(
        df, val_frac=0.15, test_frac=0.15, seed=42, num_classes=2
    )

    # Check all three splits exist
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0


def test_create_three_way_split_no_accession_column() -> None:
    """Test split works when accession column is missing."""
    df = pd.DataFrame({
        "image_path": [f"/path/to/image_{i}.png" for i in range(100)],
        "professional_label": [(i % 4) + 1 for i in range(100)],
    })

    train_df, val_df, test_df = create_three_way_split(
        df, val_frac=0.15, test_frac=0.15, seed=42
    )

    # Should still work without accession
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_create_three_way_split_all_na_accession() -> None:
    """Test split works when accession column has all NaN values."""
    df = pd.DataFrame({
        "image_path": [f"/path/to/image_{i}.png" for i in range(100)],
        "professional_label": [(i % 4) + 1 for i in range(100)],
        "accession": [None] * 100,
    })

    train_df, val_df, test_df = create_three_way_split(
        df, val_frac=0.15, test_frac=0.15, seed=42
    )

    # Should work with all NaN accessions
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0


def test_create_three_way_split_no_target_column_in_output() -> None:
    """Test that _target column is not present in output DataFrames."""
    df = _create_sample_dataframe(num_samples=100, with_accession=True, num_classes=4)

    train_df, val_df, test_df = create_three_way_split(df, seed=42)

    # _target should be internal only and removed from output
    assert "_target" not in train_df.columns
    assert "_target" not in val_df.columns
    assert "_target" not in test_df.columns


def test_load_splits_from_csvs() -> None:
    """Test loading train/val/test splits from independent CSV files."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create train CSV
        train_csv = tmp_path / "train.csv"
        with train_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "professional_label", "accession"])
            writer.writerow(["/path/to/train_1.png", "1", "ACC001"])
            writer.writerow(["/path/to/train_2.png", "2", "ACC002"])

        # Create val CSV
        val_csv = tmp_path / "val.csv"
        with val_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "professional_label", "accession"])
            writer.writerow(["/path/to/val_1.png", "3", "ACC003"])

        # Create test CSV
        test_csv = tmp_path / "test.csv"
        with test_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "professional_label", "accession"])
            writer.writerow(["/path/to/test_1.png", "4", "ACC004"])

        # Load splits
        train_df, val_df, test_df = load_splits_from_csvs(
            str(train_csv), str(val_csv), str(test_csv)
        )

        # Verify loaded data
        assert len(train_df) == 2
        assert len(val_df) == 1
        assert test_df is not None
        assert len(test_df) == 1

        # Check required columns exist
        assert "image_path" in train_df.columns
        assert "image_path" in val_df.columns
        assert "image_path" in test_df.columns


def test_load_splits_from_csvs_without_test() -> None:
    """Test loading train/val splits without test set."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create train CSV
        train_csv = tmp_path / "train.csv"
        with train_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "accession"])
            writer.writerow(["/path/to/train_1.png", "ACC001"])

        # Create val CSV
        val_csv = tmp_path / "val.csv"
        with val_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "accession"])
            writer.writerow(["/path/to/val_1.png", "ACC002"])

        # Load splits without test
        train_df, val_df, test_df = load_splits_from_csvs(
            str(train_csv), str(val_csv), test_csv=None
        )

        assert len(train_df) == 1
        assert len(val_df) == 1
        assert test_df is None


def test_load_splits_from_csvs_missing_train_file() -> None:
    """Test error when train CSV file is missing."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create only val CSV
        val_csv = tmp_path / "val.csv"
        with val_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path"])
            writer.writerow(["/path/to/val_1.png"])

        # Try to load with non-existent train file
        train_csv = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError, match="treino"):
            load_splits_from_csvs(str(train_csv), str(val_csv))


def test_load_splits_from_csvs_missing_val_file() -> None:
    """Test error when val CSV file is missing."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create only train CSV
        train_csv = tmp_path / "train.csv"
        with train_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path"])
            writer.writerow(["/path/to/train_1.png"])

        # Try to load with non-existent val file
        val_csv = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError, match="validacao"):
            load_splits_from_csvs(str(train_csv), str(val_csv))


def test_load_splits_from_csvs_missing_test_file() -> None:
    """Test error when test CSV file is specified but missing."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create train and val CSVs
        train_csv = tmp_path / "train.csv"
        with train_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path"])
            writer.writerow(["/path/to/train_1.png"])

        val_csv = tmp_path / "val.csv"
        with val_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path"])
            writer.writerow(["/path/to/val_1.png"])

        # Try to load with non-existent test file
        test_csv = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError, match="teste"):
            load_splits_from_csvs(str(train_csv), str(val_csv), str(test_csv))


def test_load_splits_from_csvs_missing_required_column() -> None:
    """Test error when CSV is missing required columns."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create train CSV without image_path column
        train_csv = tmp_path / "train.csv"
        with train_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["accession", "label"])
            writer.writerow(["ACC001", "1"])

        # Create val CSV with proper columns
        val_csv = tmp_path / "val.csv"
        with val_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path"])
            writer.writerow(["/path/to/val_1.png"])

        with pytest.raises(ValueError, match="faltando colunas obrigatorias"):
            load_splits_from_csvs(str(train_csv), str(val_csv))


def test_load_splits_from_csvs_train_val_overlap() -> None:
    """Test error when train and val splits have overlapping samples."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create train CSV
        train_csv = tmp_path / "train.csv"
        with train_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "accession"])
            writer.writerow(["/path/to/image_1.png", "ACC001"])
            writer.writerow(["/path/to/image_2.png", "ACC002"])

        # Create val CSV with overlapping image_path
        val_csv = tmp_path / "val.csv"
        with val_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "accession"])
            writer.writerow(["/path/to/image_1.png", "ACC003"])

        with pytest.raises(ValueError, match="Sobreposicao encontrada entre train e val"):
            load_splits_from_csvs(str(train_csv), str(val_csv))


def test_load_splits_from_csvs_train_test_overlap() -> None:
    """Test error when train and test splits have overlapping samples."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create train CSV
        train_csv = tmp_path / "train.csv"
        with train_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path"])
            writer.writerow(["/path/to/image_1.png"])

        # Create val CSV
        val_csv = tmp_path / "val.csv"
        with val_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path"])
            writer.writerow(["/path/to/image_2.png"])

        # Create test CSV with overlapping image_path
        test_csv = tmp_path / "test.csv"
        with test_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path"])
            writer.writerow(["/path/to/image_1.png"])

        with pytest.raises(ValueError, match="Sobreposicao encontrada entre train e test"):
            load_splits_from_csvs(str(train_csv), str(val_csv), str(test_csv))


def test_load_splits_from_csvs_val_test_overlap() -> None:
    """Test error when val and test splits have overlapping samples."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create train CSV
        train_csv = tmp_path / "train.csv"
        with train_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path"])
            writer.writerow(["/path/to/image_1.png"])

        # Create val CSV
        val_csv = tmp_path / "val.csv"
        with val_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path"])
            writer.writerow(["/path/to/image_2.png"])

        # Create test CSV with overlapping image_path from val
        test_csv = tmp_path / "test.csv"
        with test_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path"])
            writer.writerow(["/path/to/image_2.png"])

        with pytest.raises(ValueError, match="Sobreposicao encontrada entre val e test"):
            load_splits_from_csvs(str(train_csv), str(val_csv), str(test_csv))


def test_validate_split_overlap_no_overlap() -> None:
    """Test validate_split_overlap with valid non-overlapping splits."""
    train_df = pd.DataFrame({
        "image_path": ["/train/1.png", "/train/2.png"],
        "accession": ["ACC001", "ACC002"],
    })
    val_df = pd.DataFrame({
        "image_path": ["/val/1.png"],
        "accession": ["ACC003"],
    })
    test_df = pd.DataFrame({
        "image_path": ["/test/1.png"],
        "accession": ["ACC004"],
    })

    splits = {"train": train_df, "val": val_df, "test": test_df}

    # Should not raise any error
    validate_split_overlap(splits, key="accession")


def test_validate_split_overlap_with_overlap() -> None:
    """Test validate_split_overlap detects overlapping samples."""
    train_df = pd.DataFrame({
        "image_path": ["/train/1.png", "/train/2.png"],
        "accession": ["ACC001", "ACC002"],
    })
    val_df = pd.DataFrame({
        "image_path": ["/val/1.png"],
        "accession": ["ACC001"],  # Overlaps with train
    })

    splits = {"train": train_df, "val": val_df}

    with pytest.raises(ValueError, match="Sobreposição detectada entre 'train' e 'val'"):
        validate_split_overlap(splits, key="accession")


def test_validate_split_overlap_image_path_key() -> None:
    """Test validate_split_overlap using image_path as key."""
    train_df = pd.DataFrame({
        "image_path": ["/path/image_1.png", "/path/image_2.png"],
        "accession": ["ACC001", "ACC002"],
    })
    val_df = pd.DataFrame({
        "image_path": ["/path/image_1.png"],  # Overlaps with train
        "accession": ["ACC003"],
    })

    splits = {"train": train_df, "val": val_df}

    with pytest.raises(ValueError, match="Sobreposição detectada"):
        validate_split_overlap(splits, key="image_path")


def test_validate_split_overlap_empty_splits() -> None:
    """Test validate_split_overlap with empty splits dictionary."""
    splits = {}

    # Should not raise any error
    validate_split_overlap(splits, key="accession")


def test_validate_split_overlap_single_split() -> None:
    """Test validate_split_overlap with single split (no validation needed)."""
    train_df = pd.DataFrame({
        "image_path": ["/train/1.png"],
        "accession": ["ACC001"],
    })

    splits = {"train": train_df}

    # Should not raise any error (need at least 2 splits to check overlap)
    validate_split_overlap(splits, key="accession")


def test_validate_split_overlap_missing_key_column() -> None:
    """Test validate_split_overlap when key column is missing."""
    train_df = pd.DataFrame({
        "image_path": ["/train/1.png"],
    })
    val_df = pd.DataFrame({
        "image_path": ["/val/1.png"],
    })

    splits = {"train": train_df, "val": val_df}

    # Should not raise error when key column is missing (validation is skipped)
    validate_split_overlap(splits, key="accession")


def test_validate_split_overlap_with_nan_values() -> None:
    """Test validate_split_overlap handles NaN values correctly."""
    train_df = pd.DataFrame({
        "image_path": ["/train/1.png", "/train/2.png"],
        "accession": ["ACC001", None],
    })
    val_df = pd.DataFrame({
        "image_path": ["/val/1.png"],
        "accession": [None],
    })

    splits = {"train": train_df, "val": val_df}

    # Should not raise error (NaN values are dropped before comparison)
    validate_split_overlap(splits, key="accession")


def test_validate_split_overlap_multiple_overlaps() -> None:
    """Test validate_split_overlap error message with multiple overlapping samples."""
    train_df = pd.DataFrame({
        "accession": ["ACC001", "ACC002", "ACC003"],
    })
    val_df = pd.DataFrame({
        "accession": ["ACC001", "ACC002"],  # Two overlaps
    })

    splits = {"train": train_df, "val": val_df}

    with pytest.raises(ValueError) as exc_info:
        validate_split_overlap(splits, key="accession")

    # Check error message contains count
    assert "2 amostras compartilhadas" in str(exc_info.value)


# K-fold splits tests


def test_create_kfold_splits() -> None:
    """Test basic k-fold split without groups - main verification test."""
    df = _create_sample_dataframe(num_samples=100, with_accession=False, num_classes=4)

    folds = create_kfold_splits(df, n_splits=5, seed=42, num_classes=4)

    # Check we got 5 folds
    assert len(folds) == 5

    # Check each fold
    for i, (train_df, val_df) in enumerate(folds):
        # Both splits should have data
        assert len(train_df) > 0, f"Fold {i}: train split is empty"
        assert len(val_df) > 0, f"Fold {i}: val split is empty"

        # Check required columns are present
        assert "professional_label" in train_df.columns
        assert "professional_label" in val_df.columns
        assert "image_path" in train_df.columns
        assert "image_path" in val_df.columns

        # Check no overlaps within this fold
        train_paths = set(train_df["image_path"])
        val_paths = set(val_df["image_path"])
        assert len(train_paths & val_paths) == 0, f"Fold {i}: overlap between train and val"

        # Check _target column is not in output
        assert "_target" not in train_df.columns
        assert "_target" not in val_df.columns

    # Check all data is used across folds
    all_paths = set()
    for train_df, val_df in folds:
        all_paths.update(train_df["image_path"])
        all_paths.update(val_df["image_path"])

    # All original paths should appear at least once
    original_paths = set(df["image_path"])
    assert original_paths.issubset(all_paths)


def test_create_kfold_splits_basic() -> None:
    """Test basic k-fold split without groups."""
    df = _create_sample_dataframe(num_samples=100, with_accession=False, num_classes=4)

    folds = create_kfold_splits(df, n_splits=5, seed=42)

    # Check we got 5 folds
    assert len(folds) == 5

    # Each fold should have train and val
    for train_df, val_df in folds:
        assert len(train_df) > 0
        assert len(val_df) > 0


def test_create_kfold_splits_with_groups() -> None:
    """Test k-fold split with group-aware stratification."""
    df = _create_sample_dataframe(num_samples=150, with_accession=True, num_classes=4)

    folds = create_kfold_splits(df, n_splits=5, seed=42, num_classes=4)

    # Check we got 5 folds
    assert len(folds) == 5

    # Check each fold for group non-overlapping
    for i, (train_df, val_df) in enumerate(folds):
        assert len(train_df) > 0
        assert len(val_df) > 0

        # Check no group leakage (accessions should not overlap)
        train_acc = set(train_df["accession"])
        val_acc = set(val_df["accession"])
        assert len(train_acc & val_acc) == 0, f"Fold {i}: accession overlap between train and val"

        # Check columns
        assert "accession" in train_df.columns
        assert "accession" in val_df.columns


def test_create_kfold_splits_reproducibility() -> None:
    """Test that k-fold splits are reproducible with same seed."""
    df = _create_sample_dataframe(num_samples=100, with_accession=True, num_classes=4)

    folds1 = create_kfold_splits(df, n_splits=5, seed=42)
    folds2 = create_kfold_splits(df, n_splits=5, seed=42)

    # Check same number of folds
    assert len(folds1) == len(folds2)

    # Check each fold produces same splits
    for i, ((train1, val1), (train2, val2)) in enumerate(zip(folds1, folds2)):
        assert set(train1["image_path"]) == set(train2["image_path"]), f"Fold {i}: train splits differ"
        assert set(val1["image_path"]) == set(val2["image_path"]), f"Fold {i}: val splits differ"


def test_create_kfold_splits_different_seeds() -> None:
    """Test that different seeds produce different k-fold splits."""
    df = _create_sample_dataframe(num_samples=100, with_accession=True, num_classes=4)

    folds1 = create_kfold_splits(df, n_splits=5, seed=42)
    folds2 = create_kfold_splits(df, n_splits=5, seed=123)

    # Check we got same number of folds
    assert len(folds1) == len(folds2)

    # At least one fold should be different
    different = False
    for (train1, val1), (train2, val2) in zip(folds1, folds2):
        if set(train1["image_path"]) != set(train2["image_path"]):
            different = True
            break

    assert different, "Different seeds should produce different splits"


def test_create_kfold_splits_none_input() -> None:
    """Test that None input returns empty list."""
    folds = create_kfold_splits(None, n_splits=5, seed=42)

    assert folds == []
    assert isinstance(folds, list)


def test_create_kfold_splits_invalid_n_splits() -> None:
    """Test error when n_splits < 2."""
    df = _create_sample_dataframe(num_samples=100, with_accession=False, num_classes=4)

    with pytest.raises(ValueError) as exc_info:
        create_kfold_splits(df, n_splits=1, seed=42)

    assert "n_splits deve ser >= 2" in str(exc_info.value)


def test_create_kfold_splits_missing_label_column() -> None:
    """Test error when professional_label column is missing."""
    df = pd.DataFrame({
        "image_path": [f"/path/to/image_{i}.png" for i in range(100)],
        "accession": [f"ACC{i:03d}" for i in range(100)],
    })

    with pytest.raises(ValueError) as exc_info:
        create_kfold_splits(df, n_splits=5, seed=42)

    assert "professional_label" in str(exc_info.value)


def test_create_kfold_splits_insufficient_groups() -> None:
    """Test error when there are insufficient groups for k-fold."""
    # Create DataFrame with only 3 groups but want 5 folds
    df = pd.DataFrame({
        "image_path": [f"/path/to/image_{i}.png" for i in range(12)],
        "professional_label": [(i % 4) + 1 for i in range(12)],
        "accession": [f"ACC{i // 4:03d}" for i in range(12)],  # Only 3 groups
    })

    with pytest.raises(RuntimeError) as exc_info:
        create_kfold_splits(df, n_splits=5, seed=42)

    assert "Grupos insuficientes" in str(exc_info.value)


def test_create_kfold_splits_two_classes() -> None:
    """Test k-fold split with binary classification (num_classes=2)."""
    df = _create_sample_dataframe(num_samples=100, with_accession=False, num_classes=4)

    folds = create_kfold_splits(df, n_splits=5, seed=42, num_classes=2)

    # Check we got 5 folds
    assert len(folds) == 5

    # Each fold should have data
    for train_df, val_df in folds:
        assert len(train_df) > 0
        assert len(val_df) > 0


def test_create_kfold_splits_custom_n_splits() -> None:
    """Test k-fold split with custom number of folds."""
    df = _create_sample_dataframe(num_samples=100, with_accession=False, num_classes=4)

    # Test with 3 folds
    folds_3 = create_kfold_splits(df, n_splits=3, seed=42)
    assert len(folds_3) == 3

    # Test with 10 folds
    folds_10 = create_kfold_splits(df, n_splits=10, seed=42)
    assert len(folds_10) == 10


def test_create_kfold_splits_no_target_column_in_output() -> None:
    """Test that _target column is not present in output DataFrames."""
    df = _create_sample_dataframe(num_samples=100, with_accession=True, num_classes=4)

    folds = create_kfold_splits(df, n_splits=5, seed=42)

    for train_df, val_df in folds:
        # _target should be internal only and removed from output
        assert "_target" not in train_df.columns
        assert "_target" not in val_df.columns


def test_create_kfold_splits_all_data_used() -> None:
    """Test that all data points appear exactly once in validation across all folds."""
    df = _create_sample_dataframe(num_samples=100, with_accession=False, num_classes=4)

    folds = create_kfold_splits(df, n_splits=5, seed=42)

    # Collect all validation paths across folds
    all_val_paths = []
    for train_df, val_df in folds:
        all_val_paths.extend(val_df["image_path"].tolist())

    # Each sample should appear in validation exactly once
    assert len(all_val_paths) == len(df)
    assert len(set(all_val_paths)) == len(df)

    # All original paths should be present
    original_paths = set(df["image_path"])
    assert set(all_val_paths) == original_paths


def test_create_kfold_splits_with_groups_all_data_used() -> None:
    """Test that all groups appear exactly once in validation across all folds."""
    df = _create_sample_dataframe(num_samples=150, with_accession=True, num_classes=4)

    folds = create_kfold_splits(df, n_splits=5, seed=42)

    # Collect all validation groups across folds
    all_val_groups = []
    for train_df, val_df in folds:
        all_val_groups.extend(val_df["accession"].unique().tolist())

    # Each group should appear in validation exactly once
    original_groups = df["accession"].unique()
    assert len(all_val_groups) == len(original_groups)
    assert set(all_val_groups) == set(original_groups)
