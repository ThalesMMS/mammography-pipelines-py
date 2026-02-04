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
pd = pytest.importorskip("pandas")
pytest.importorskip("pydicom")

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from mammography.commands.train import main as train_main


def _write_dummy_dicom(path: Path, patient_id: str, accession: str) -> None:
    """Create a minimal DICOM file for testing."""
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
def mock_data_three_way(tmp_path: Path):
    """Generate DICOM dataset with enough samples for three-way split testing."""
    dcm_root = tmp_path / "archive"

    # Create 6 patients with different accession numbers and classifications
    samples = [
        ("ACC001", "PAT_1", 1),
        ("ACC002", "PAT_2", 2),
        ("ACC003", "PAT_3", 3),
        ("ACC004", "PAT_4", 4),
        ("ACC005", "PAT_5", 1),
        ("ACC006", "PAT_6", 2),
    ]

    dicom_paths = {}
    for accession, patient, _ in samples:
        dcm_dir = dcm_root / accession
        dcm_dir.mkdir(parents=True)
        dcm_path = dcm_dir / "test.dcm"
        _write_dummy_dicom(dcm_path, patient, accession)
        dicom_paths[accession] = str(dcm_path)

    # Use image_path format with letter labels (A,B,C,D) to ensure object dtype
    # _coerce_density_label will convert A->1, B->2, C->3, D->4
    label_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    csv_path = tmp_path / "labels.csv"
    df = pd.DataFrame({
        "image_path": [dicom_paths[acc] for acc, _, _ in samples],
        "accession": [acc for acc, _, _ in samples],
        "professional_label": [label_map[cls] for _, _, cls in samples]
    })
    df.to_csv(csv_path, index=False)

    return csv_path, dcm_root


@pytest.fixture
def mock_data_preset(tmp_path: Path):
    """Generate separate train/val/test CSV files and corresponding DICOMs."""
    dcm_root = tmp_path / "archive"

    # Create 9 patients split across train/val/test
    train_samples = [
        ("ACC001", "PAT_1", 1),
        ("ACC002", "PAT_2", 2),
        ("ACC003", "PAT_3", 3),
        ("ACC004", "PAT_4", 4),
    ]
    val_samples = [
        ("ACC005", "PAT_5", 1),
        ("ACC006", "PAT_6", 3),
    ]
    test_samples = [
        ("ACC007", "PAT_7", 2),
        ("ACC008", "PAT_8", 4),
    ]

    # Create DICOMs for all samples
    all_samples = train_samples + val_samples + test_samples
    dicom_paths = {}
    for accession, patient, _ in all_samples:
        dcm_dir = dcm_root / accession
        dcm_dir.mkdir(parents=True)
        dcm_path = dcm_dir / "test.dcm"
        _write_dummy_dicom(dcm_path, patient, accession)
        dicom_paths[accession] = str(dcm_path)

    # Create separate CSV files with image_path column
    def _write_csv(path: Path, samples):
        # Use letter labels (A,B,C,D) to ensure object dtype
        label_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        df = pd.DataFrame({
            "image_path": [dicom_paths[acc] for acc, _, _ in samples],
            "accession": [acc for acc, _, _ in samples],
            "professional_label": [label_map[cls] for _, _, cls in samples]
        })
        df.to_csv(path, index=False)

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"

    _write_csv(train_csv, train_samples)
    _write_csv(val_csv, val_samples)
    _write_csv(test_csv, test_samples)

    return train_csv, val_csv, test_csv, dcm_root


def test_three_way_split_smoke(mock_data_three_way, tmp_path: Path):
    """Test end-to-end training with automatic three-way split (train/val/test)."""
    csv_path, dcm_root = mock_data_three_way
    outdir = tmp_path / "output_three_way"

    argv = [
        "--csv",
        str(csv_path),
        "--outdir",
        str(outdir),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--img-size",
        "64",
        "--arch",
        "efficientnet_b0",
        "--cache-mode",
        "none",
        "--num-workers",
        "0",
        "--val-frac",
        "0.2",
        "--test-frac",
        "0.2",
        "--device",
        "cpu",
        "--no-pretrained",
        "--no-augment",
        "--split-ensure-all-classes",
        "--split-max-tries",
        "100",
    ]

    train_main(argv)

    # Verify outputs were created
    summary_files = list(outdir.rglob("summary.json"))
    assert summary_files, "summary.json was not created"


def test_preset_split_smoke(mock_data_preset, tmp_path: Path):
    """Test end-to-end training with pre-defined train/val/test CSV files."""
    train_csv, val_csv, test_csv, dcm_root = mock_data_preset
    outdir = tmp_path / "output_preset"

    # In preset mode, we still need --csv to satisfy initial loading,
    # but the splits come from the preset CSVs
    argv = [
        "--csv",
        str(train_csv),
        "--outdir",
        str(outdir),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--img-size",
        "64",
        "--arch",
        "efficientnet_b0",
        "--cache-mode",
        "none",
        "--num-workers",
        "0",
        "--split-mode",
        "preset",
        "--train-csv",
        str(train_csv),
        "--val-csv",
        str(val_csv),
        "--test-csv",
        str(test_csv),
        "--device",
        "cpu",
        "--no-pretrained",
        "--no-augment",
    ]

    train_main(argv)

    # Verify outputs were created
    summary_files = list(outdir.rglob("summary.json"))
    assert summary_files, "summary.json was not created"


def test_backward_compatibility_two_way_split(mock_data_three_way, tmp_path: Path):
    """Ensure default two-way split (train/val) still works for backward compatibility."""
    csv_path, dcm_root = mock_data_three_way
    outdir = tmp_path / "output_two_way"

    argv = [
        "--csv",
        str(csv_path),
        "--outdir",
        str(outdir),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--img-size",
        "64",
        "--arch",
        "efficientnet_b0",
        "--cache-mode",
        "none",
        "--num-workers",
        "0",
        "--val-frac",
        "0.3",
        # NOTE: No --test-frac, should default to 0.0 for two-way split
        "--device",
        "cpu",
        "--no-pretrained",
        "--no-augment",
    ]

    train_main(argv)

    # Verify outputs were created
    summary_files = list(outdir.rglob("summary.json"))
    assert summary_files, "summary.json was not created"
