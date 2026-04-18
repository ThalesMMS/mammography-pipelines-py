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
        base = np.linspace(0, 255, 32 * 32, dtype=np.float32).reshape(32, 32)
        img = np.stack(
            [
                np.clip(base + i * 5, 0, 255),
                np.clip(base / 2 + i * 3, 0, 255),
                np.clip(255 - base + i * 2, 0, 255),
            ],
            axis=-1,
        ).astype(np.uint8)
        Image.fromarray(img, mode="RGB").save(img_path)
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

def test_mammo_density_dataset_auto_normalize_degenerate_sample_keeps_pixels() -> None:
    dataset = MammoDensityDataset(
        rows=[],
        img_size=8,
        train=False,
        augment=False,
        cache_mode="none",
    )
    dataset._auto_normalize_enabled = True
    dataset._norm_mean = [0.5, 0.5, 0.5]
    dataset._norm_std = [0.2, 0.2, 0.2]
    tensor = torch.full((3, 8, 8), 0.5, dtype=torch.float32)

    result = dataset._apply_transforms(tensor)

    assert torch.allclose(result, tensor)
    assert result.std() == 0

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
