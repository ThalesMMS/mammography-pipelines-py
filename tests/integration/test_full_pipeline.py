"""
Integration tests for complete workflows.

These tests validate end-to-end pipelines including training, embedding extraction,
inference, and classical ML baselines. They use synthetic data to ensure reproducibility
without requiring large datasets.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
from PIL import Image

from mammography.commands.extract_features import main as extract_main
from mammography.commands.inference import main as inference_main
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


def _write_dummy_png(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    """Generate a dummy PNG image with random grayscale pixels."""
    arr = np.random.randint(0, 256, size, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    img.save(path)


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
    # Use letter labels to ensure object dtype (A=1, B=2, C=3, D=4)
    csv_path.write_text(
        "AccessionNumber,Classification\nACC123,A\nACC124,A\n",
        encoding="utf-8",
    )

    return csv_path, dcm_root


@pytest.fixture
def mock_png_data(tmp_path: Path):
    """Generate a minimal PNG dataset for testing image-based workflows."""
    img_root = tmp_path / "images"
    img_root.mkdir()

    # Create dummy PNG images
    for i, label in enumerate(["A", "B", "C", "D"]):
        for j in range(2):
            img_path = img_root / f"img_{label}_{j}.png"
            _write_dummy_png(img_path)

    # Create CSV with image paths and labels
    csv_path = tmp_path / "labels_png.csv"
    rows = []
    for img in sorted(img_root.glob("*.png")):
        label = img.stem.split("_")[1]
        rows.append(f"{img.name},{label}\n")

    csv_path.write_text(
        "image_path,density_label\n" + "".join(rows),
        encoding="utf-8",
    )

    return csv_path, img_root


@pytest.fixture
def mock_varied_data(tmp_path: Path):
    """Generate dataset with varied labels for classification testing."""
    dcm_root = tmp_path / "archive"
    labels = []

    for i, (accession, patient, label) in enumerate(
        [
            ("ACC001", "PAT_1", "A"),
            ("ACC002", "PAT_2", "B"),
            ("ACC003", "PAT_3", "C"),
            ("ACC004", "PAT_4", "D"),
            ("ACC005", "PAT_5", "A"),
            ("ACC006", "PAT_6", "B"),
        ]
    ):
        dcm_dir = dcm_root / accession
        dcm_dir.mkdir(parents=True)
        dcm_path = dcm_dir / "test.dcm"
        _write_dummy_dicom(dcm_path, patient, accession)
        labels.append((accession, label))

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\n"
        + "\n".join(f"{acc},{lbl}" for acc, lbl in labels)
        + "\n",
        encoding="utf-8",
    )

    return csv_path, dcm_root


@pytest.mark.integration
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


@pytest.mark.integration
def test_train_resnet_architecture(mock_data, tmp_path: Path):
    """Test training with ResNet50 architecture."""
    csv_path, dcm_root = mock_data
    outdir = tmp_path / "output_resnet"

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
        "resnet50",
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
    assert summary_files, "summary.json was not created for resnet50"


@pytest.mark.integration
def test_extract_embeddings_workflow(mock_varied_data, tmp_path: Path):
    """Test embedding extraction workflow end-to-end."""
    csv_path, dcm_root = mock_varied_data
    outdir = tmp_path / "embeddings"

    argv = [
        "--csv",
        str(csv_path),
        "--dicom-root",
        str(dcm_root),
        "--outdir",
        str(outdir),
        "--arch",
        "efficientnet_b0",
        "--img-size",
        "64",
        "--batch-size",
        "2",
        "--device",
        "cpu",
        "--num-workers",
        "0",
        "--no-pretrained",
        "--cache-mode",
        "none",
        "--no-pca",
        "--no-tsne",
        "--no-umap",
        "--no-clustering",
    ]

    extract_main(argv)

    # Verify embeddings were saved
    embeddings_file = outdir / "embeddings.npz"
    assert embeddings_file.exists(), "embeddings.npz was not created"

    # Verify metadata was saved
    metadata_file = outdir / "metadata.csv"
    assert metadata_file.exists(), "metadata.csv was not created"

    # Load and verify embeddings shape
    data = np.load(embeddings_file)
    assert "embeddings" in data, "embeddings key not found in npz"
    embeddings = data["embeddings"]
    assert len(embeddings) == 6, f"Expected 6 embeddings, got {len(embeddings)}"


@pytest.mark.integration
def test_train_then_inference_workflow(mock_varied_data, tmp_path: Path):
    """Test complete train-then-inference workflow."""
    csv_path, dcm_root = mock_varied_data
    train_outdir = tmp_path / "trained_model"

    # Step 1: Train a model
    train_argv = [
        "--csv",
        str(csv_path),
        "--dicom-root",
        str(dcm_root),
        "--outdir",
        str(train_outdir),
        "--epochs",
        "2",
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
        "0.33",
        "--device",
        "cpu",
        "--no-pretrained",
        "--no-augment",
    ]

    train_main(train_argv)

    # Find checkpoint
    checkpoints = list(train_outdir.rglob("*.pt"))
    assert checkpoints, "No checkpoint files created"
    checkpoint = checkpoints[0]

    # Step 2: Run inference with trained checkpoint
    inference_outdir = tmp_path / "inference_results"
    inference_csv = inference_outdir / "predictions.csv"

    inference_argv = [
        "--checkpoint",
        str(checkpoint),
        "--input",
        str(dcm_root),
        "--arch",
        "efficientnet_b0",
        "--img-size",
        "64",
        "--batch-size",
        "2",
        "--device",
        "cpu",
        "--output",
        str(inference_csv),
    ]

    inference_main(inference_argv)

    # Verify predictions were saved
    assert inference_csv.exists(), "predictions.csv was not created"

    # Load and verify predictions
    df_pred = pd.read_csv(inference_csv)
    assert len(df_pred) == 6, f"Expected 6 predictions, got {len(df_pred)}"
    assert "predicted_class" in df_pred.columns, "predicted_class column missing"
    assert "confidence" in df_pred.columns, "confidence column missing"


@pytest.mark.integration
def test_embedding_with_clustering(mock_varied_data, tmp_path: Path):
    """Test embedding extraction with clustering analysis."""
    csv_path, dcm_root = mock_varied_data
    outdir = tmp_path / "embeddings_clustered"

    argv = [
        "--csv",
        str(csv_path),
        "--dicom-root",
        str(dcm_root),
        "--outdir",
        str(outdir),
        "--arch",
        "resnet50",
        "--img-size",
        "64",
        "--batch-size",
        "2",
        "--device",
        "cpu",
        "--num-workers",
        "0",
        "--no-pretrained",
        "--cache-mode",
        "none",
        "--run-clustering",
        "--no-pca",
        "--no-tsne",
        "--no-umap",
    ]

    extract_main(argv)

    # Verify clustering outputs
    embeddings_file = outdir / "embeddings.npz"
    assert embeddings_file.exists(), "embeddings.npz was not created"

    clustering_results = outdir / "clustering_kmeans_k4.json"
    assert clustering_results.exists(), "clustering results not created"


@pytest.mark.integration
def test_train_with_memory_cache(mock_varied_data, tmp_path: Path):
    """Test training with memory caching enabled."""
    csv_path, dcm_root = mock_varied_data
    outdir = tmp_path / "cached_training"

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
        "2",
        "--img-size",
        "64",
        "--arch",
        "efficientnet_b0",
        "--cache-mode",
        "memory",
        "--num-workers",
        "0",
        "--val-frac",
        "0.33",
        "--device",
        "cpu",
        "--no-pretrained",
        "--no-augment",
    ]

    train_main(argv)

    summary_files = list(outdir.rglob("summary.json"))
    assert summary_files, "summary.json was not created with memory cache"


@pytest.mark.integration
@pytest.mark.slow
def test_train_multiclass_all_labels(mock_varied_data, tmp_path: Path):
    """Test training with all 4 BI-RADS labels present."""
    csv_path, dcm_root = mock_varied_data
    outdir = tmp_path / "multiclass_training"

    argv = [
        "--csv",
        str(csv_path),
        "--dicom-root",
        str(dcm_root),
        "--outdir",
        str(outdir),
        "--epochs",
        "3",
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
        "0.33",
        "--device",
        "cpu",
        "--no-pretrained",
        "--no-augment",
    ]

    train_main(argv)

    # Verify summary.json exists and contains valid metrics
    summary_files = list(outdir.rglob("summary.json"))
    assert summary_files, "summary.json was not created"

    with open(summary_files[0], "r") as f:
        summary = json.load(f)

    # Verify key metrics are present
    assert "val_acc" in summary or "test_acc" in summary, "No accuracy metrics in summary"


@pytest.mark.integration
def test_png_image_workflow(mock_png_data, tmp_path: Path):
    """Test embedding extraction with PNG images instead of DICOM."""
    csv_path, img_root = mock_png_data
    outdir = tmp_path / "png_embeddings"

    # Update CSV paths to be absolute
    df = pd.read_csv(csv_path)
    df["image_path"] = df["image_path"].apply(lambda x: str(img_root / x))
    csv_with_paths = tmp_path / "labels_with_paths.csv"
    df.to_csv(csv_with_paths, index=False)

    argv = [
        "--csv",
        str(csv_with_paths),
        "--outdir",
        str(outdir),
        "--arch",
        "efficientnet_b0",
        "--img-size",
        "64",
        "--batch-size",
        "2",
        "--device",
        "cpu",
        "--num-workers",
        "0",
        "--no-pretrained",
        "--cache-mode",
        "none",
        "--no-pca",
        "--no-tsne",
        "--no-umap",
        "--no-clustering",
    ]

    extract_main(argv)

    # Verify embeddings were created
    embeddings_file = outdir / "embeddings.npz"
    assert embeddings_file.exists(), "embeddings.npz not created for PNG images"

    data = np.load(embeddings_file)
    assert "embeddings" in data, "embeddings key not found"
    assert len(data["embeddings"]) == 8, "Expected 8 embeddings from PNG images"
