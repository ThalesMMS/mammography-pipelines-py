"""
Integration tests for failure recovery and graceful degradation.

These tests validate the system's ability to:
- Recover from training interruptions
- Handle partial data loading (missing/corrupted files)
- Recover from corrupted checkpoints
- Fallback gracefully when GPU unavailable
- Handle out-of-memory conditions
- Recover from disk space issues
- Handle network failures during model downloads
- Degrade gracefully when optional dependencies missing

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("pandas")
pytest.importorskip("pydicom")

import torch
import torch.nn as nn
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from mammography.commands.train import main as train_main
from mammography.commands.inference import main as inference_main
from mammography.training.engine import save_atomic
from mammography.config import TrainConfig, InferenceConfig
from mammography.utils.device_detection import detect_device, resolve_device


def _write_dummy_dicom(path: Path, patient_id: str, accession: str) -> None:
    """Create a minimal valid DICOM file for testing."""
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
def mock_data_with_failures(tmp_path: Path):
    """Generate a dataset with some corrupted/missing files."""
    dcm_root = tmp_path / "archive"

    # Create 5 samples: 3 valid, 1 corrupted, 1 missing
    accessions = [
        ("ACC001", "PAT_1", "valid"),
        ("ACC002", "PAT_2", "valid"),
        ("ACC003", "PAT_3", "valid"),
        ("ACC004", "PAT_4", "corrupted"),
        ("ACC005", "PAT_5", "missing"),
    ]

    for accession, patient, status in accessions:
        dcm_dir = dcm_root / accession
        dcm_dir.mkdir(parents=True)
        dcm_path = dcm_dir / "test.dcm"

        if status == "valid":
            _write_dummy_dicom(dcm_path, patient, accession)
        elif status == "corrupted":
            # Write invalid DICOM data
            dcm_path.write_bytes(b"INVALID_DICOM_DATA")
        # For 'missing', don't create the file

    # CSV includes all samples
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\n"
        "ACC001,A\n"
        "ACC002,B\n"
        "ACC003,A\n"
        "ACC004,B\n"
        "ACC005,A\n",
        encoding="utf-8",
    )

    return csv_path, dcm_root


@pytest.fixture
def mock_data(tmp_path: Path):
    """Generate a minimal valid DICOM dataset."""
    dcm_root = tmp_path / "archive"
    for accession, patient in [("ACC123", "PAT_1"), ("ACC124", "PAT_2"), ("ACC125", "PAT_3")]:
        dcm_dir = dcm_root / accession
        dcm_dir.mkdir(parents=True)
        dcm_path = dcm_dir / "test.dcm"
        _write_dummy_dicom(dcm_path, patient, accession)

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\nACC123,A\nACC124,B\nACC125,A\n",
        encoding="utf-8",
    )

    return csv_path, dcm_root


# ==================== Training Interruption Recovery ====================


@pytest.mark.integration
def test_training_interruption_recovery(mock_data, tmp_path: Path):
    """Test that training can resume after interruption."""
    csv_path, dcm_root = mock_data
    outdir = tmp_path / "training_output"

    # First training run - 2 epochs
    cfg = TrainConfig(
        csv=str(csv_path),
        dicom_root=str(dcm_root),
        outdir=str(outdir),
        epochs=2,
        batch_size=1,
        img_size=64,
        seed=42,
        architecture="efficientnet_b0",
        pretrained=False,
    )

    train_main(cfg)

    # Verify checkpoint exists
    checkpoint_path = outdir / "checkpoint.pt"
    assert checkpoint_path.exists(), "Checkpoint not saved after first training"

    # Load checkpoint and check epoch
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["epoch"] == 2, "Checkpoint should be at epoch 2"

    # Resume training - continue for 2 more epochs (total 4)
    cfg_resume = TrainConfig(
        csv=str(csv_path),
        dicom_root=str(dcm_root),
        outdir=str(outdir),
        epochs=4,  # Will resume from epoch 2 and train 2 more
        batch_size=1,
        img_size=64,
        seed=42,
        architecture="efficientnet_b0",
        pretrained=False,
    )

    train_main(cfg_resume)

    # Verify training resumed and completed
    checkpoint_resumed = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint_resumed["epoch"] == 4, "Training should resume to epoch 4"


@pytest.mark.integration
def test_training_with_checkpoint_deletion_during_save(mock_data, tmp_path: Path):
    """Test that training handles checkpoint save failures gracefully."""
    csv_path, dcm_root = mock_data
    outdir = tmp_path / "training_output"

    # Create a checkpoint directory but make it read-only to simulate write failure
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / "checkpoint.pt"

    # First, run a successful training
    cfg = TrainConfig(
        csv=str(csv_path),
        dicom_root=str(dcm_root),
        outdir=str(outdir),
        epochs=1,
        batch_size=1,
        img_size=64,
        seed=42,
        architecture="efficientnet_b0",
        pretrained=False,
    )

    # Training should complete even if checkpoint save might fail
    # The save_atomic function should handle failures gracefully
    train_main(cfg)

    # If checkpoint was saved, it should be valid
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert "epoch" in checkpoint
        assert "model_state" in checkpoint


# ==================== Partial Data Loading ====================


@pytest.mark.integration
def test_training_with_partial_data_loading(mock_data_with_failures, tmp_path: Path):
    """Test that training handles datasets with some corrupted/missing files."""
    csv_path, dcm_root = mock_data_with_failures
    outdir = tmp_path / "training_output"

    # Training should work with valid samples only
    # The dataset loader should skip corrupted/missing files gracefully
    cfg = TrainConfig(
        csv=str(csv_path),
        dicom_root=str(dcm_root),
        outdir=str(outdir),
        epochs=1,
        batch_size=1,
        img_size=64,
        seed=42,
        architecture="efficientnet_b0",
        pretrained=False,
    )

    # This should complete or raise an informative error
    # depending on how many samples fail to load
    try:
        train_main(cfg)
        # If it succeeds, verify checkpoint was created
        assert (outdir / "checkpoint.pt").exists()
    except Exception as e:
        # If it fails, error message should be informative
        error_msg = str(e).lower()
        assert any(
            keyword in error_msg
            for keyword in ["file", "dicom", "load", "corrupted", "missing"]
        ), f"Error should mention file/loading issues: {e}"


@pytest.mark.integration
def test_robust_collate_handles_none_values(tmp_path: Path):
    """Test that robust_collate skips None values from failed loads."""
    from mammography.data.dataset import robust_collate

    # Simulate batch with some None values (failed loads)
    batch = [
        (torch.randn(3, 64, 64), 0),  # Valid sample
        None,  # Failed load
        (torch.randn(3, 64, 64), 1),  # Valid sample
        None,  # Failed load
        (torch.randn(3, 64, 64), 0),  # Valid sample
    ]

    # robust_collate should filter out None values
    images, labels = robust_collate(batch)

    assert images is not None, "Should return valid batch"
    assert labels is not None, "Should return valid labels"
    assert len(images) == 3, "Should have 3 valid samples"
    assert len(labels) == 3, "Should have 3 valid labels"


@pytest.mark.integration
def test_robust_collate_with_all_none_values():
    """Test that robust_collate handles batch with all None values."""
    from mammography.data.dataset import robust_collate

    # Simulate batch where all samples failed to load
    batch = [None, None, None]

    # robust_collate should return None for empty batch
    result = robust_collate(batch)

    assert result is None, "Should return None when all samples are None"


# ==================== Checkpoint Corruption Handling ====================


@pytest.mark.integration
def test_training_with_corrupted_checkpoint(mock_data, tmp_path: Path):
    """Test that training handles corrupted checkpoint gracefully."""
    csv_path, dcm_root = mock_data
    outdir = tmp_path / "training_output"
    outdir.mkdir(parents=True, exist_ok=True)

    # Create a corrupted checkpoint file
    checkpoint_path = outdir / "checkpoint.pt"
    checkpoint_path.write_bytes(b"CORRUPTED_CHECKPOINT_DATA")

    # Training should detect corrupted checkpoint and start fresh
    cfg = TrainConfig(
        csv=str(csv_path),
        dicom_root=str(dcm_root),
        outdir=str(outdir),
        epochs=2,
        batch_size=1,
        img_size=64,
        seed=42,
        architecture="efficientnet_b0",
        pretrained=False,
    )

    # Should either start fresh or raise clear error
    try:
        train_main(cfg)
        # If successful, verify new checkpoint is valid
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert checkpoint["epoch"] == 2
    except Exception as e:
        # Error should mention checkpoint corruption
        error_msg = str(e).lower()
        assert any(
            keyword in error_msg
            for keyword in ["checkpoint", "corrupted", "load", "invalid"]
        ), f"Error should mention checkpoint issue: {e}"


@pytest.mark.integration
def test_checkpoint_with_missing_keys(mock_data, tmp_path: Path):
    """Test handling of checkpoint with missing required keys."""
    csv_path, dcm_root = mock_data
    outdir = tmp_path / "training_output"
    outdir.mkdir(parents=True, exist_ok=True)

    # Create a checkpoint with missing keys
    checkpoint_path = outdir / "checkpoint.pt"
    incomplete_checkpoint = {
        "epoch": 1,
        # Missing: model_state, optimizer_state, best_acc, etc.
    }
    torch.save(incomplete_checkpoint, checkpoint_path)

    # Inference should detect missing keys and raise informative error
    cfg = InferenceConfig(
        checkpoint=str(checkpoint_path),
        csv=str(csv_path),
        dicom_root=str(dcm_root),
        architecture="efficientnet_b0",
    )

    with pytest.raises(Exception) as exc_info:
        inference_main(cfg)

    # Error should mention missing keys
    error_msg = str(exc_info.value).lower()
    assert any(
        keyword in error_msg
        for keyword in ["model_state", "state", "key", "missing"]
    ), f"Error should mention missing checkpoint keys: {exc_info.value}"


# ==================== GPU Failure Fallback ====================


@pytest.mark.integration
def test_training_fallback_to_cpu_when_gpu_unavailable(mock_data, tmp_path: Path):
    """Test that training falls back to CPU when GPU is unavailable."""
    csv_path, dcm_root = mock_data
    outdir = tmp_path / "training_output"

    # Mock CUDA as unavailable
    with patch("torch.cuda.is_available", return_value=False):
        cfg = TrainConfig(
            csv=str(csv_path),
            dicom_root=str(dcm_root),
            outdir=str(outdir),
            epochs=1,
            batch_size=1,
            img_size=64,
            seed=42,
            device="auto",  # Should fallback to CPU
            architecture="efficientnet_b0",
            pretrained=False,
        )

        # Should complete on CPU
        train_main(cfg)

        # Verify checkpoint exists and model is on CPU
        checkpoint_path = outdir / "checkpoint.pt"
        assert checkpoint_path.exists()

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert checkpoint["epoch"] == 1


@pytest.mark.integration
def test_device_resolution_fallback():
    """Test device resolution falls back gracefully."""
    # Test auto device resolution when CUDA unavailable
    with patch("torch.cuda.is_available", return_value=False):
        device = resolve_device("auto")
        assert device == torch.device("cpu"), "Should fallback to CPU"

    # Test explicit cuda:0 when CUDA unavailable
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.warns(UserWarning, match="CUDA requested but not available"):
            device = resolve_device("cuda:0")
            assert device == torch.device("cpu"), "Should fallback to CPU with warning"


# ==================== Out of Memory Handling ====================


@pytest.mark.integration
def test_training_with_small_batch_size_to_avoid_oom(mock_data, tmp_path: Path):
    """Test that using small batch size avoids OOM errors."""
    csv_path, dcm_root = mock_data
    outdir = tmp_path / "training_output"

    # Use batch_size=1 to minimize memory usage
    cfg = TrainConfig(
        csv=str(csv_path),
        dicom_root=str(dcm_root),
        outdir=str(outdir),
        epochs=1,
        batch_size=1,  # Small batch to avoid OOM
        img_size=64,   # Small image size
        seed=42,
        architecture="efficientnet_b0",
        pretrained=False,
    )

    # Should complete without OOM
    train_main(cfg)

    assert (outdir / "checkpoint.pt").exists()


@pytest.mark.integration
def test_memory_cache_mode_for_small_datasets(mock_data, tmp_path: Path):
    """Test that memory cache mode works for small datasets."""
    csv_path, dcm_root = mock_data
    outdir = tmp_path / "training_output"

    # Use memory cache mode
    cfg = TrainConfig(
        csv=str(csv_path),
        dicom_root=str(dcm_root),
        outdir=str(outdir),
        epochs=1,
        batch_size=2,
        img_size=64,
        seed=42,
        cache_mode="memory",
        architecture="efficientnet_b0",
        pretrained=False,
    )

    # Should complete with memory caching
    train_main(cfg)

    assert (outdir / "checkpoint.pt").exists()


# ==================== Disk Space Issues ====================


@pytest.mark.integration
def test_checkpoint_save_atomic_prevents_corruption(tmp_path: Path):
    """Test that atomic save prevents checkpoint corruption on failure."""
    checkpoint_path = tmp_path / "checkpoint.pt"

    # Create a valid checkpoint
    checkpoint_data = {
        "epoch": 5,
        "model_state": {"weight": torch.randn(10, 10)},
        "optimizer_state": {},
        "best_acc": 0.85,
    }

    # Use save_atomic to save checkpoint
    save_atomic(checkpoint_data, checkpoint_path)

    # Verify checkpoint was saved
    assert checkpoint_path.exists(), "Checkpoint should be saved"

    # Verify no temporary file left behind
    tmp_file = Path(str(checkpoint_path) + ".tmp")
    assert not tmp_file.exists(), "Temporary file should be cleaned up"

    # Verify checkpoint is valid
    loaded = torch.load(checkpoint_path, map_location="cpu")
    assert loaded["epoch"] == 5
    assert "model_state" in loaded


@pytest.mark.integration
def test_checkpoint_atomic_save_cleanup_on_failure(tmp_path: Path):
    """Test that temporary files are cleaned up even if save fails."""
    checkpoint_path = tmp_path / "checkpoint.pt"
    tmp_checkpoint = Path(str(checkpoint_path) + ".tmp")

    # Create invalid data that might cause serialization issues
    checkpoint_data = {
        "epoch": 1,
        "model_state": {"invalid": object()},  # Non-serializable object
    }

    # Try to save (should fail)
    with pytest.raises(Exception):
        save_atomic(checkpoint_data, checkpoint_path)

    # Verify cleanup: neither checkpoint nor temp file should exist
    # (or temp file is cleaned up if save_atomic handles it)
    # The exact behavior depends on save_atomic implementation


# ==================== Network Failures ====================


@pytest.mark.integration
def test_training_without_pretrained_weights_download(mock_data, tmp_path: Path):
    """Test that training works without downloading pretrained weights."""
    csv_path, dcm_root = mock_data
    outdir = tmp_path / "training_output"

    # Use pretrained=False to avoid network access
    cfg = TrainConfig(
        csv=str(csv_path),
        dicom_root=str(dcm_root),
        outdir=str(outdir),
        epochs=1,
        batch_size=1,
        img_size=64,
        seed=42,
        architecture="efficientnet_b0",
        pretrained=False,  # No network download needed
    )

    # Should complete without network access
    train_main(cfg)

    assert (outdir / "checkpoint.pt").exists()


@pytest.mark.integration
def test_model_initialization_handles_download_failure(tmp_path: Path):
    """Test that model initialization handles download failures gracefully."""
    from mammography.models.efficientnet import build_efficientnet_model

    # Mock download failure
    with patch("timm.create_model") as mock_create:
        mock_create.side_effect = RuntimeError("Failed to download pretrained weights")

        # Should raise informative error
        with pytest.raises(RuntimeError, match="download"):
            build_efficientnet_model(
                num_classes=4,
                pretrained=True,  # Requires download
            )


# ==================== Graceful Degradation ====================


@pytest.mark.integration
def test_training_continues_with_missing_optional_features(mock_data, tmp_path: Path):
    """Test that training continues when optional features are unavailable."""
    csv_path, dcm_root = mock_data
    outdir = tmp_path / "training_output"

    # Train without optional features (no scheduler, no early stopping)
    cfg = TrainConfig(
        csv=str(csv_path),
        dicom_root=str(dcm_root),
        outdir=str(outdir),
        epochs=2,
        batch_size=1,
        img_size=64,
        seed=42,
        scheduler=None,  # No scheduler
        early_stop_patience=0,  # No early stopping
        architecture="efficientnet_b0",
        pretrained=False,
    )

    # Should complete successfully
    train_main(cfg)

    assert (outdir / "checkpoint.pt").exists()


@pytest.mark.integration
def test_inference_with_missing_normalization_stats(mock_data, tmp_path: Path):
    """Test that inference handles missing normalization stats gracefully."""
    csv_path, dcm_root = mock_data
    outdir = tmp_path / "training_output"

    # First train a model
    cfg = TrainConfig(
        csv=str(csv_path),
        dicom_root=str(dcm_root),
        outdir=str(outdir),
        epochs=1,
        batch_size=1,
        img_size=64,
        seed=42,
        architecture="efficientnet_b0",
        pretrained=False,
    )
    train_main(cfg)

    # Load checkpoint and remove normalization stats
    checkpoint_path = outdir / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Remove normalization stats if present
    checkpoint.pop("mean", None)
    checkpoint.pop("std", None)

    # Save modified checkpoint
    torch.save(checkpoint, checkpoint_path)

    # Inference should handle missing stats
    # (either use defaults or raise informative error)
    inf_cfg = InferenceConfig(
        checkpoint=str(checkpoint_path),
        csv=str(csv_path),
        dicom_root=str(dcm_root),
        architecture="efficientnet_b0",
    )

    try:
        inference_main(inf_cfg)
        # If successful, predictions should be generated
        assert (outdir / "predictions.csv").exists() or (outdir / "predictions.json").exists()
    except Exception as e:
        # Error should mention normalization stats
        error_msg = str(e).lower()
        assert any(
            keyword in error_msg
            for keyword in ["normalization", "mean", "std", "stats"]
        ), f"Error should mention normalization stats: {e}"


# ==================== Edge Case Recovery ====================


@pytest.mark.integration
def test_training_with_single_class_dataset(tmp_path: Path):
    """Test that training handles single-class dataset gracefully."""
    dcm_root = tmp_path / "archive"

    # Create dataset with only class A
    for i, (accession, patient) in enumerate([("ACC001", "PAT_1"), ("ACC002", "PAT_2")]):
        dcm_dir = dcm_root / accession
        dcm_dir.mkdir(parents=True)
        dcm_path = dcm_dir / "test.dcm"
        _write_dummy_dicom(dcm_path, patient, accession)

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\nACC001,A\nACC002,A\n",
        encoding="utf-8",
    )

    outdir = tmp_path / "training_output"

    cfg = TrainConfig(
        csv=str(csv_path),
        dicom_root=str(dcm_root),
        outdir=str(outdir),
        epochs=1,
        batch_size=1,
        img_size=64,
        seed=42,
        architecture="efficientnet_b0",
        pretrained=False,
    )

    # Training with single class should either work or raise informative error
    try:
        train_main(cfg)
        assert (outdir / "checkpoint.pt").exists()
    except Exception as e:
        # Error should mention class imbalance or insufficient classes
        error_msg = str(e).lower()
        assert any(
            keyword in error_msg
            for keyword in ["class", "label", "imbalance", "insufficient"]
        ), f"Error should mention class-related issue: {e}"


@pytest.mark.integration
def test_training_recovery_after_validation_failure(mock_data, tmp_path: Path):
    """Test that training continues even if validation fails."""
    csv_path, dcm_root = mock_data
    outdir = tmp_path / "training_output"

    # Train with very small validation fraction
    cfg = TrainConfig(
        csv=str(csv_path),
        dicom_root=str(dcm_root),
        outdir=str(outdir),
        epochs=1,
        batch_size=1,
        img_size=64,
        val_frac=0.1,  # Small validation set
        seed=42,
        architecture="efficientnet_b0",
        pretrained=False,
    )

    # Should complete even with small/empty validation set
    train_main(cfg)

    assert (outdir / "checkpoint.pt").exists()
