from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")
pytest.importorskip("pydicom")

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from mammography.commands.train import main as train_main


def _write_dummy_dicom(path: Path, patient_id: str, accession: str) -> None:
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ImplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.PatientID = patient_id
    ds.AccessionNumber = accession
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.Modality = "MG"

    arr = np.zeros((100, 100), dtype=np.uint16)
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    ds.save_as(str(path), write_like_original=False)


@pytest.fixture
def mock_data(tmp_path: Path):
    """Generate a minimal DICOM dataset and matching CSV for training."""
    dcm_root = tmp_path / "archive"
    for accession, patient in [("ACC123", "PAT_1"), ("ACC124", "PAT_2")]:
        dcm_dir = dcm_root / accession
        dcm_dir.mkdir(parents=True)
        dcm_path = dcm_dir / "test.dcm"
        _write_dummy_dicom(dcm_path, patient, accession)

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\nACC123,1\nACC124,1\n",
        encoding="utf-8",
    )

    return csv_path, dcm_root


def test_train_smoke(mock_data, tmp_path: Path):
    """Ensure the training command runs end-to-end on synthetic data."""
    csv_path, dcm_root = mock_data
    outdir = tmp_path / "output"

    argv = [
        "--csv",
        str(csv_path),
        "--dicom-root",
        str(dcm_root),
        "--outdir",
        str(outdir),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--img-size",
        "64",
        "--arch",
        "efficientnet_b0",
        "--cache-mode",
        "none",
        "--num-workers",
        "0",
        "--val-frac",
        "0.5",
        "--device",
        "cpu",
        "--no-pretrained",
        "--no-augment",
    ]

    train_main(argv)

    summary_files = list(outdir.rglob("summary.json"))
    assert summary_files, "summary.json was not created"
