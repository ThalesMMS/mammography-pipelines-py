import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pydicom = pytest.importorskip("pydicom")

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from mammography.data.dataset import MammoDensityDataset
from mammography.data.csv_loader import resolve_dataset_cache_mode


def _write_dummy_dicom(path: Path) -> None:
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.PatientID = "TEST"
    ds.Modality = "MG"
    ds.Rows = 8
    ds.Columns = 8
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsStored = 12
    ds.BitsAllocated = 16
    ds.HighBit = 11
    ds.PixelRepresentation = 0

    arr = np.arange(64, dtype=np.uint16).reshape(8, 8)
    ds.PixelData = arr.tobytes()
    ds.save_as(path)


def _make_dataset(tmp_path: Path, cache_mode: str = "disk"):
    dicom_path = tmp_path / "example.dcm"
    _write_dummy_dicom(dicom_path)

    rows = [{"image_path": str(dicom_path), "professional_label": 1}]
    cache_dir = tmp_path / "cache"
    ds = MammoDensityDataset(
        rows,
        img_size=32,
        train=False,
        cache_mode=cache_mode,
        cache_dir=str(cache_dir),
        split_name="test",
    )
    return ds, dicom_path, cache_dir


def test_disk_cache_materialization(tmp_path: Path):
    ds, dicom_path, cache_dir = _make_dataset(tmp_path)

    png_files = list(cache_dir.glob("*.png"))
    assert len(png_files) == 1

    os.remove(dicom_path)

    img, label, meta, _ = ds[0]

    assert isinstance(img, torch.Tensor)
    assert tuple(img.shape) == (3, 32, 32)
    assert label == 0
    assert meta["path"].endswith("example.dcm")

    # cache deve continuar disponível para acessos subsequentes
    img2, label2, _, _ = ds[0]
    assert torch.allclose(img, img2)
    assert label2 == label


def test_tensor_disk_cache(tmp_path: Path):
    ds, dicom_path, cache_dir = _make_dataset(tmp_path, cache_mode="tensor-disk")

    pt_files = list(cache_dir.glob("*.pt"))
    assert len(pt_files) == 1

    os.remove(dicom_path)

    img, label, meta, _ = ds[0]

    assert isinstance(img, torch.Tensor)
    assert tuple(img.shape) == (3, 32, 32)
    assert label == 0
    assert meta["path"].endswith("example.dcm")

    img2, label2, _, _ = ds[0]
    assert torch.allclose(img, img2)
    assert label2 == label


def test_tensor_memmap_cache(tmp_path: Path):
    ds, dicom_path, cache_dir = _make_dataset(tmp_path, cache_mode="tensor-memmap")

    dat_files = list(cache_dir.glob("*.dat"))
    json_files = list(cache_dir.glob("*.json"))
    assert len(dat_files) == 1
    assert len(json_files) == 1

    os.remove(dicom_path)

    img, label, meta, _ = ds[0]

    assert isinstance(img, torch.Tensor)
    assert tuple(img.shape) == (3, 32, 32)
    assert label == 0
    assert meta["path"].endswith("example.dcm")

    img2, label2, _, _ = ds[0]
    assert torch.allclose(img, img2)
    assert label2 == label


def test_memory_cache(tmp_path: Path):
    """Test memory cache mode - caches in RAM without disk persistence."""
    dicom_path = tmp_path / "example.dcm"
    _write_dummy_dicom(dicom_path)

    rows = [{"image_path": str(dicom_path), "professional_label": 1}]
    # Memory cache doesn't require cache_dir
    ds = MammoDensityDataset(
        rows,
        img_size=32,
        train=False,
        cache_mode="memory",
        cache_dir=None,
        split_name="test",
    )

    # First access - loads from disk
    img, label, meta, _ = ds[0]

    assert isinstance(img, torch.Tensor)
    assert tuple(img.shape) == (3, 32, 32)
    assert label == 0
    assert meta["path"].endswith("example.dcm")

    # Remove original DICOM file
    os.remove(dicom_path)

    # Second access should still work from memory cache
    img2, label2, _, _ = ds[0]
    assert torch.allclose(img, img2)
    assert label2 == label

    # Verify no disk files were created (memory cache only)
    cache_dir = tmp_path / "cache"
    if cache_dir.exists():
        assert len(list(cache_dir.glob("*"))) == 0


def test_none_cache_mode(tmp_path: Path):
    """Test 'none' cache mode - no caching, always loads from disk."""
    dicom_path = tmp_path / "example.dcm"
    _write_dummy_dicom(dicom_path)

    rows = [{"image_path": str(dicom_path), "professional_label": 1}]
    ds = MammoDensityDataset(
        rows,
        img_size=32,
        train=False,
        cache_mode="none",
        cache_dir=None,
        split_name="test",
    )

    # First access
    img, label, meta, _ = ds[0]

    assert isinstance(img, torch.Tensor)
    assert tuple(img.shape) == (3, 32, 32)
    assert label == 0
    assert meta["path"].endswith("example.dcm")

    # Second access - should still work (file still exists)
    img2, label2, _, _ = ds[0]
    assert torch.allclose(img, img2)
    assert label2 == label

    # Verify no cache files were created
    cache_dir = tmp_path / "cache"
    if cache_dir.exists():
        assert len(list(cache_dir.glob("*"))) == 0


def test_multiple_images_memory_cache(tmp_path: Path):
    """Test memory cache with multiple images to verify cache consistency."""
    dicom_paths = [tmp_path / f"example_{i}.dcm" for i in range(3)]
    for path in dicom_paths:
        _write_dummy_dicom(path)

    rows = [{"image_path": str(path), "professional_label": i % 2} for i, path in enumerate(dicom_paths)]
    ds = MammoDensityDataset(
        rows,
        img_size=32,
        train=False,
        cache_mode="memory",
        cache_dir=None,
        split_name="test",
    )

    # Load all images
    cached_imgs = []
    for i in range(len(rows)):
        img, label, _, _ = ds[i]
        cached_imgs.append((img.clone(), label))
        assert isinstance(img, torch.Tensor)
        assert tuple(img.shape) == (3, 32, 32)

    # Access again and verify consistency
    for i in range(len(rows)):
        img, label, _, _ = ds[i]
        cached_img, cached_label = cached_imgs[i]
        assert torch.allclose(img, cached_img), f"Image {i} not consistent in memory cache"
        assert label == cached_label


def test_invalid_cache_mode(tmp_path: Path):
    """Test that invalid cache mode raises ValueError."""
    dicom_path = tmp_path / "example.dcm"
    _write_dummy_dicom(dicom_path)

    rows = [{"image_path": str(dicom_path), "professional_label": 1}]

    with pytest.raises(ValueError, match="cache_mode inválido"):
        MammoDensityDataset(
            rows,
            img_size=32,
            train=False,
            cache_mode="invalid-mode",
            cache_dir=None,
            split_name="test",
        )


def test_cache_dir_required_for_disk_modes(tmp_path: Path):
    """Test that cache_dir is required for disk-based cache modes."""
    dicom_path = tmp_path / "example.dcm"
    _write_dummy_dicom(dicom_path)

    rows = [{"image_path": str(dicom_path), "professional_label": 1}]

    # Test each disk-based mode
    for mode in ["disk", "tensor-disk", "tensor-memmap"]:
        with pytest.raises(ValueError, match="cache_dir é obrigatório"):
            MammoDensityDataset(
                rows,
                img_size=32,
                train=False,
                cache_mode=mode,
                cache_dir=None,
                split_name="test",
            )


def test_auto_cache_mode_small_dataset(tmp_path: Path):
    """Test auto mode with small dataset (<= 1000 images) - should use memory cache."""
    # Create a small dataset (10 images)
    dicom_paths = [tmp_path / f"img_{i}.dcm" for i in range(10)]
    for path in dicom_paths:
        _write_dummy_dicom(path)

    rows = [{"image_path": str(path), "professional_label": i % 2} for i, path in enumerate(dicom_paths)]

    # Test resolve function directly
    resolved_mode = resolve_dataset_cache_mode("auto", rows)
    assert resolved_mode == "memory", f"Expected 'memory' for small dataset, got '{resolved_mode}'"

    # Verify dataset works with resolved mode
    ds = MammoDensityDataset(
        rows,
        img_size=32,
        train=False,
        cache_mode=resolved_mode,
        cache_dir=None,
        split_name="test",
    )

    # Verify images load correctly
    for i in range(len(rows)):
        img, label, _, _ = ds[i]
        assert isinstance(img, torch.Tensor)
        assert tuple(img.shape) == (3, 32, 32)


def test_auto_cache_mode_non_dicom(tmp_path: Path):
    """Test auto mode with non-DICOM files - should use 'none' cache."""
    # Create PNG files instead of DICOM
    from PIL import Image as PILImage
    png_paths = [tmp_path / f"img_{i}.png" for i in range(10)]
    for path in png_paths:
        # Create a dummy PNG file
        img = PILImage.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(path)

    rows = [{"image_path": str(path), "professional_label": 0} for path in png_paths]

    # Test resolve function - should return "none" for non-DICOM files
    resolved_mode = resolve_dataset_cache_mode("auto", rows)
    assert resolved_mode == "none", f"Expected 'none' for non-DICOM files, got '{resolved_mode}'"


def test_auto_cache_mode_empty_dataset():
    """Test auto mode with empty dataset - should use 'none' cache."""
    rows = []
    resolved_mode = resolve_dataset_cache_mode("auto", rows)
    assert resolved_mode == "none", f"Expected 'none' for empty dataset, got '{resolved_mode}'"


def test_cache_modes_data_integrity(tmp_path: Path):
    """
    Test all cache modes for data integrity - verify no corruption occurs.
    This is a comprehensive test that accesses the same image multiple times
    across different cache modes to ensure consistency.
    """
    dicom_path = tmp_path / "integrity_test.dcm"
    _write_dummy_dicom(dicom_path)

    rows = [{"image_path": str(dicom_path), "professional_label": 1}]
    cache_dir = tmp_path / "cache"

    # Store reference image from first mode
    reference_img = None
    reference_label = None

    # Test each cache mode
    modes_to_test = [
        ("none", None),
        ("memory", None),
        ("disk", str(cache_dir / "disk")),
        ("tensor-disk", str(cache_dir / "tensor-disk")),
        ("tensor-memmap", str(cache_dir / "tensor-memmap")),
    ]

    for mode, cache_dir_for_mode in modes_to_test:
        ds = MammoDensityDataset(
            rows,
            img_size=32,
            train=False,
            cache_mode=mode,
            cache_dir=cache_dir_for_mode,
            split_name="test",
        )

        # Access image multiple times
        for access in range(3):
            img, label, meta, _ = ds[0]

            # Verify basic properties
            assert isinstance(img, torch.Tensor), f"Mode {mode}, access {access}: not a tensor"
            assert tuple(img.shape) == (3, 32, 32), f"Mode {mode}, access {access}: wrong shape"
            assert label == 0, f"Mode {mode}, access {access}: wrong label (expected 0, got {label})"
            assert meta["path"].endswith("integrity_test.dcm"), f"Mode {mode}, access {access}: wrong metadata"

            # Store reference on first iteration
            if reference_img is None:
                reference_img = img.clone()
                reference_label = label

            # Compare with reference to detect corruption
            assert torch.allclose(img, reference_img, rtol=1e-5, atol=1e-5), \
                f"Mode {mode}, access {access}: image data corrupted (doesn't match reference)"
            assert label == reference_label, \
                f"Mode {mode}, access {access}: label corrupted"
