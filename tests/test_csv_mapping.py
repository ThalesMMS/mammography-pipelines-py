from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from extract_mammo_resnet50 import MammoDicomDataset, load_labels_dict


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


def test_dataset_excludes_class_5(tmp_path):
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

    labels = load_labels_dict(str(csv_path))
    assert labels["000001"] == 1
    assert labels["000002"] == 5

    dataset = MammoDicomDataset(
        data_dir=str(data_dir),
        labels_by_accession=labels,
        exclude_class_5=True,
        include_unlabeled=False,
        transform=None,
    )
    assert len(dataset) == 1
    img, label, accession, _, _ = dataset[0]
    assert accession == "000001"
    assert label == 1
    assert img.mode == "RGB"
