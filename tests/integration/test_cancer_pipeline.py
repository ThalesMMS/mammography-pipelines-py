"""
Integration tests for cancer detection pipeline.

These tests validate the complete cancer detection pipeline from DICOM files
through dataset creation, model building, training, and evaluation.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pl = pytest.importorskip("polars")
Image = pytest.importorskip("PIL.Image")
pydicom = pytest.importorskip("pydicom")

from mammography.data.cancer_dataset import (
    MammoDicomDataset,
    MammographyDataset,
    dataset_summary,
    make_dataloader,
    split_dataset,
)
from mammography.models.cancer_models import (
    build_resnet50_classifier,
    resolve_device,
)
from mammography.training.cancer_trainer import (
    collect_predictions,
    evaluate,
    fit_classifier,
    get_sens_spec,
    train_one_epoch,
)
from mammography.vis.cancer_plots import get_transforms


def _create_mock_dicom(path: Path, rows: int = 256, cols: int = 256) -> None:
    """Create a minimal DICOM file for testing."""
    arr = np.random.randint(0, 4096, size=(rows, cols), dtype=np.uint16)
    ds = pydicom.Dataset()
    ds.Rows = rows
    ds.Columns = cols
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.save_as(path, write_like_original=False)


def _write_sample_png(path: Path, size: int = 224) -> None:
    """Create a sample grayscale PNG for testing (mammography is single-channel)."""
    img = Image.new("L", (size, size), color=120)
    img.save(path)


@pytest.fixture
def mock_dicom_data(tmp_path: Path):
    """Create mock DICOM dataset with labels."""
    data_dir = tmp_path / "data"

    # Create samples with different classifications
    labels = {}
    for i, (accession, label) in enumerate([
        ("ACC001", 1),
        ("ACC002", 1),
        ("ACC003", 2),
        ("ACC004", 2),
        ("ACC005", 3),
        ("ACC006", 3),
        ("ACC007", 4),
        ("ACC008", 4),
    ]):
        case_dir = data_dir / accession
        case_dir.mkdir(parents=True)
        dcm_path = case_dir / "image.dcm"
        _create_mock_dicom(dcm_path, rows=256, cols=256)
        labels[accession] = label

    return data_dir, labels


@pytest.fixture
def mock_png_data(tmp_path: Path):
    """Create mock PNG dataset with labels for MammographyDataset."""
    data_dir = tmp_path / "png_data"
    data_dir.mkdir(parents=True)

    # Create PNG files
    rows = []
    for i, label in enumerate([0, 0, 1, 1, 0, 1, 0, 1]):
        png_path = data_dir / f"image_{i:03d}.png"
        _write_sample_png(png_path, size=224)
        rows.append({
            "image_path": str(png_path),
            "cancer": label,
        })

    # Create Polars DataFrame
    df = pl.DataFrame(rows)
    return df


class TestCancerPipelineIntegration:
    """Integration tests for the complete cancer detection pipeline."""

    def test_dicom_dataset_to_model_pipeline(self, mock_dicom_data):
        """Test complete pipeline: DICOM dataset → model → training."""
        data_dir, labels = mock_dicom_data

        # Step 1: Create dataset
        dataset = MammoDicomDataset(
            data_dir=str(data_dir),
            labels_by_accession=labels,
            exclude_class_5=True,
            include_unlabeled=False,
        )

        assert len(dataset) == 8, "Should have 8 labeled samples"

        # Verify dataset summary
        summary = dataset_summary(dataset)
        assert summary is not None

        # Step 2: Get a sample
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "image" in sample
        assert "label" in sample

        # Step 3: Build model
        model = build_resnet50_classifier(num_classes=1, pretrained=False)
        assert model is not None

        # Step 4: Verify forward pass
        device = resolve_device()
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            img_tensor = sample["image"].unsqueeze(0).to(device)
            output = model(img_tensor)
            assert output.shape == (1, 1), "Binary classification should output (batch, 1)"

    def test_png_dataset_training_loop(self, mock_png_data):
        """Test training loop with PNG-based MammographyDataset."""
        df = mock_png_data

        # Step 1: Create dataset with transforms
        model_transform, _ = get_transforms()
        dataset = MammographyDataset(
            dataframe=df,
            transform=model_transform,
        )

        assert len(dataset) > 0, "Dataset should not be empty"

        # Step 2: Split dataset
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = split_dataset(dataset, [train_size, val_size])

        assert len(train_ds) == train_size
        assert len(val_ds) == val_size

        # Step 3: Create dataloaders
        train_loader = make_dataloader(train_ds, batch_size=2, shuffle=True, num_workers=0)
        val_loader = make_dataloader(val_ds, batch_size=2, shuffle=False, num_workers=0)

        # Step 4: Build model
        model = build_resnet50_classifier(num_classes=1, pretrained=False)
        device = resolve_device()
        model = model.to(device)

        # Step 5: Setup training
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Step 6: Train one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp_enabled=False,
            scaler=None,
        )

        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert 0.0 <= train_acc <= 1.0

        # Step 7: Evaluate
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert 0.0 <= val_acc <= 1.0

    def test_full_training_pipeline_with_fit_classifier(self, mock_png_data):
        """Test the complete fit_classifier workflow."""
        df = mock_png_data

        # Create dataset
        model_transform, _ = get_transforms()
        dataset = MammographyDataset(
            dataframe=df,
            transform=model_transform,
        )

        # Split dataset
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = split_dataset(dataset, [train_size, val_size])

        # Create dataloaders
        train_loader = make_dataloader(train_ds, batch_size=2, shuffle=True, num_workers=0)
        val_loader = make_dataloader(val_ds, batch_size=2, shuffle=False, num_workers=0)

        # Build model
        model = build_resnet50_classifier(num_classes=1, pretrained=False)
        device = resolve_device()
        model = model.to(device)

        # Fit classifier
        history = fit_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,  # Just 2 epochs for testing
            lr=0.001,
            device=device,
            amp_enabled=False,
        )

        # Verify history
        assert len(history) == 2, "Should have 2 epochs in history"
        for entry in history:
            assert hasattr(entry, "epoch")
            assert hasattr(entry, "train_loss")
            assert hasattr(entry, "train_acc")
            assert hasattr(entry, "val_loss")
            assert hasattr(entry, "val_acc")

    def test_prediction_pipeline(self, mock_png_data):
        """Test prediction collection from trained model."""
        df = mock_png_data

        # Create dataset
        model_transform, _ = get_transforms()
        dataset = MammographyDataset(
            dataframe=df,
            transform=model_transform,
        )

        # Create dataloader
        loader = make_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)

        # Build model
        model = build_resnet50_classifier(num_classes=1, pretrained=False)
        device = resolve_device()
        model = model.to(device)
        model.eval()

        # Collect predictions
        preds, labels = collect_predictions(
            model=model,
            loader=loader,
            device=device,
        )

        # Verify predictions
        assert len(preds) == len(dataset), "Should have predictions for all samples"
        assert len(labels) == len(dataset), "Should have labels for all samples"
        assert isinstance(preds, np.ndarray)
        assert isinstance(labels, np.ndarray)

        # Check prediction range (probabilities after sigmoid)
        assert np.all((preds >= 0) & (preds <= 1)), "Predictions should be in [0, 1]"

    def test_sensitivity_specificity_metrics(self, mock_png_data):
        """Test sensitivity and specificity computation."""
        # Create some dummy predictions
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])

        sensitivity, specificity = get_sens_spec(y_true, y_pred)

        assert isinstance(sensitivity, (float, np.floating))
        assert isinstance(specificity, (float, np.floating))
        assert 0.0 <= sensitivity <= 1.0
        assert 0.0 <= specificity <= 1.0

    def test_end_to_end_pipeline_smoke_test(self, mock_dicom_data):
        """Smoke test: Run the entire pipeline end-to-end."""
        data_dir, labels = mock_dicom_data

        # Create dataset
        dataset = MammoDicomDataset(
            data_dir=str(data_dir),
            labels_by_accession=labels,
            exclude_class_5=True,
            include_unlabeled=False,
        )

        # Split
        train_size = int(0.75 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = split_dataset(dataset, [train_size, val_size])

        # Create loaders
        train_loader = make_dataloader(train_ds, batch_size=2, shuffle=True, num_workers=0)
        val_loader = make_dataloader(val_ds, batch_size=2, shuffle=False, num_workers=0)

        # Build model
        model = build_resnet50_classifier(num_classes=1, pretrained=False)
        device = resolve_device()
        model = model.to(device)

        # Train for 1 epoch
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp_enabled=False,
            scaler=None,
        )

        # Evaluate
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Collect predictions
        preds, true_labels = collect_predictions(
            model=model,
            loader=val_loader,
            device=device,
        )

        # Compute metrics
        binary_preds = (preds > 0.5).astype(int)
        sensitivity, specificity = get_sens_spec(true_labels, binary_preds)

        # Verify everything completed successfully
        assert train_loss is not None
        assert val_loss is not None
        assert len(preds) == len(val_ds)
        assert sensitivity is not None
        assert specificity is not None

        print(f"\n✓ Pipeline completed successfully!")
        print(f"  Training: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Validation: loss={val_loss:.4f}, acc={val_acc:.4f}")
        print(f"  Metrics: sens={sensitivity:.4f}, spec={specificity:.4f}")
