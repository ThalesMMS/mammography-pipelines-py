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
