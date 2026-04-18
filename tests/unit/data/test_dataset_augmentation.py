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
