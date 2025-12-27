from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")
pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography.data.csv_loader import load_dataset_dataframe


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


def test_load_dataset_excludes_class_5(tmp_path: Path) -> None:
    csv_path = tmp_path / "classificacao.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["AccessionNumber", "Classification", "ClassificationDate"])
        writer.writerow(["000001", "1", "2025-01-01 00:00:00"])
        writer.writerow(["000002", "5", "2025-01-02 00:00:00"])

    data_dir = tmp_path / "archive"
    for accession in ["000001", "000002"]:
        folder = data_dir / accession
        folder.mkdir(parents=True, exist_ok=True)
        _write_dummy_dicom(folder / "image.dcm")

    df = load_dataset_dataframe(str(csv_path), dicom_root=str(data_dir), exclude_class_5=True)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["professional_label"] == 1
    assert row["image_path"].endswith("image.dcm")
