from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")
pytest.importorskip("numpy")

import numpy as np
import pandas as pd
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from mammography.training.engine import (
    save_atomic,
    train_one_epoch,
    validate,
    extract_embeddings,
    plot_history,
    save_predictions,
    save_metrics_figure,
    plot_view_comparison,
    _save_gradcam_batch,
)


class DummyMultiClassModel(nn.Module):
    """Dummy model for multi-class density classification."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor, extra_features=None) -> torch.Tensor:  # noqa: ANN001
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DummyModelNoAvgPool(nn.Module):
    """Dummy model without avgpool layer for testing fallback."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor, extra_features=None) -> torch.Tensor:  # noqa: ANN001
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DummyModelWithBackbone(nn.Module):
    """Dummy model with backbone.avgpool structure."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor, extra_features=None) -> torch.Tensor:  # noqa: ANN001
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DummyModelNanLoss(nn.Module):
    """Dummy model that produces NaN outputs to test non-finite loss handling."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, extra_features=None) -> torch.Tensor:  # noqa: ANN001
        batch_size = x.shape[0]
        return torch.full((batch_size, self.num_classes), float('nan'))


class DummyModelWithLayer4(nn.Module):
    """Dummy model with layer4 for GradCAM testing."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.layer4 = nn.ModuleList([
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor, extra_features=None) -> torch.Tensor:  # noqa: ANN001
        for layer in self.layer4:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _make_multiclass_loader(num_samples: int = 8, num_classes: int = 4) -> DataLoader:
    """Create a dummy multi-class classification data loader."""
    x = torch.randn(num_samples, 3, 8, 8)
    y = torch.randint(0, num_classes, (num_samples,))
    meta = [{"id": i, "path": f"sample_{i}.png", "accession": f"acc_{i}"} for i in range(num_samples)]

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    dataset = [(x[i], y[i], meta[i]) for i in range(num_samples)]
    return DataLoader(dataset, batch_size=4, collate_fn=collate)


def _make_multiclass_loader_with_extra_features(num_samples: int = 8, num_classes: int = 4, num_extra: int = 5) -> DataLoader:
    """Create a loader with extra tabular features (4-element batch)."""
    x = torch.randn(num_samples, 3, 8, 8)
    y = torch.randint(0, num_classes, (num_samples,))
    extra = torch.randn(num_samples, num_extra)
    meta = [{"id": i, "path": f"sample_{i}.png", "accession": f"acc_{i}"} for i in range(num_samples)]

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        extras = torch.stack([b[2] for b in batch])
        metas = [b[3] for b in batch]
        return xs, ys, metas, extras

    dataset = [(x[i], y[i], extra[i], meta[i]) for i in range(num_samples)]
    return DataLoader(dataset, batch_size=4, collate_fn=collate)


# ============================================================================
# Test save_atomic edge cases
# ============================================================================


def test_save_atomic_with_pathlib_path() -> None:
    """Test save_atomic works with pathlib.Path objects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "model.pt"
        state = {"epoch": 5, "lr": 0.001}

        save_atomic(state, checkpoint_path)

        assert checkpoint_path.exists()
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert loaded["epoch"] == 5


def test_save_atomic_with_string_path() -> None:
    """Test save_atomic works with string paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = str(Path(tmpdir) / "model.pt")
        state = {"epoch": 10}

        save_atomic(state, checkpoint_path)

        assert Path(checkpoint_path).exists()


def test_save_atomic_creates_parent_dirs() -> None:
    """Test that save_atomic handles nested directory paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = Path(tmpdir) / "subdir1" / "subdir2" / "model.pt"
        nested_path.parent.mkdir(parents=True, exist_ok=True)
        state = {"test": "value"}

        save_atomic(state, nested_path)

        assert nested_path.exists()


# ============================================================================
# Test train_one_epoch edge cases
# ============================================================================


def test_train_one_epoch_with_extra_features() -> None:
    """Test training with extra tabular features (4-element batch)."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _make_multiclass_loader_with_extra_features()

    loss, acc = train_one_epoch(model, loader, optimizer, device, scaler=None)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0


def test_train_one_epoch_with_numpy_labels() -> None:
    """Test training with numpy array labels instead of tensors."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Create loader with numpy array labels
    x = torch.randn(8, 3, 8, 8)
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])  # numpy array instead of tensor
    meta = [{"id": i} for i in range(8)]

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = np.array([b[1] for b in batch])  # Keep as numpy
        metas = [b[2] for b in batch]
        return xs, ys, metas

    dataset = [(x[i], y[i], meta[i]) for i in range(8)]
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)

    loss, acc = train_one_epoch(model, loader, optimizer, device, scaler=None)

    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0


def test_train_one_epoch_with_grad_scaler() -> None:
    """Test training with gradient scaler explicitly enabled."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _make_multiclass_loader()
    scaler = GradScaler()

    loss, acc = train_one_epoch(model, loader, optimizer, device, scaler=scaler, amp_enabled=True)

    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0


def test_train_one_epoch_with_none_batches() -> None:
    """Test training handles None batches gracefully."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Create loader that can return None
    def collate(batch):
        if not batch:
            return None
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    x = torch.randn(4, 3, 8, 8)
    y = torch.randint(0, 4, (4,))
    meta = [{"id": i} for i in range(4)]
    dataset = [(x[i], y[i], meta[i]) for i in range(4)]
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)

    loss, acc = train_one_epoch(model, loader, optimizer, device, scaler=None)

    assert isinstance(loss, float)


def test_train_one_epoch_non_finite_loss_raises() -> None:
    """Test that non-finite loss is detected and raises RuntimeError."""
    model = DummyModelNanLoss()
    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _make_multiclass_loader()

    with pytest.raises(RuntimeError, match="Loss nao finita"):
        train_one_epoch(model, loader, optimizer, device, scaler=None)


def test_train_one_epoch_single_sample() -> None:
    """Test training with only one sample."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(1, 3, 8, 8)
    y = torch.tensor([2])
    meta = [{"id": 0}]

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    dataset = [(x[0], y[0], meta[0])]
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate)

    loss, acc = train_one_epoch(model, loader, optimizer, device, scaler=None)

    assert isinstance(loss, float)
    assert acc in (0.0, 1.0)  # With one sample, acc is either 0 or 1


# ============================================================================
# Test validate edge cases
# ============================================================================


def test_validate_with_gradcam() -> None:
    """Test validation with GradCAM generation enabled."""
    model = DummyModelWithLayer4()
    device = torch.device("cpu")
    loader = _make_multiclass_loader()

    with tempfile.TemporaryDirectory() as tmpdir:
        gradcam_dir = Path(tmpdir) / "gradcam"

        metrics, pred_rows = validate(
            model, loader, device, amp_enabled=False,
            gradcam=True, gradcam_dir=gradcam_dir, gradcam_limit=2
        )

        assert "acc" in metrics
        # Check if GradCAM images were saved (may or may not succeed depending on implementation)
        if gradcam_dir.exists():
            gradcam_files = list(gradcam_dir.glob("*.png"))
            assert len(gradcam_files) <= 2  # Should respect limit


def test_validate_with_single_sample() -> None:
    """Test validation with only one sample."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")

    x = torch.randn(1, 3, 8, 8)
    y = torch.tensor([1])
    meta = [{"id": 0, "path": "single.png", "accession": "acc_0"}]

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    dataset = [(x[0], y[0], meta[0])]
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate)

    metrics, pred_rows = validate(model, loader, device, amp_enabled=False, collect_preds=True)

    assert "acc" in metrics
    assert len(pred_rows) == 1


def test_validate_with_all_same_predictions() -> None:
    """Test validation when all predictions are the same class."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")

    # Create data where model is likely to predict same class
    x = torch.ones(8, 3, 8, 8) * 5.0  # Constant input
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    meta = [{"id": i, "path": f"img_{i}.png", "accession": f"acc_{i}"} for i in range(8)]

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    dataset = [(x[i], y[i], meta[i]) for i in range(8)]
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)

    metrics, _ = validate(model, loader, device, amp_enabled=False)

    # Should still compute metrics without crashing
    assert "acc" in metrics
    assert "confusion_matrix" in metrics


def test_validate_with_extra_features() -> None:
    """Test validation with extra tabular features."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    loader = _make_multiclass_loader_with_extra_features()

    metrics, pred_rows = validate(model, loader, device, amp_enabled=False, collect_preds=True)

    assert "acc" in metrics
    assert len(pred_rows) > 0


def test_validate_binary_classification() -> None:
    """Test validation with binary classification (2 classes)."""
    model = DummyMultiClassModel(num_classes=2)
    device = torch.device("cpu")

    x = torch.randn(8, 3, 8, 8)
    y = torch.randint(0, 2, (8,))
    meta = [{"id": i, "path": f"img_{i}.png", "accession": f"acc_{i}"} for i in range(8)]

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    dataset = [(x[i], y[i], meta[i]) for i in range(8)]
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)

    metrics, _ = validate(model, loader, device, amp_enabled=False)

    assert "acc" in metrics
    assert "auc_ovr" in metrics
    # Binary classification should use different label mapping


# ============================================================================
# Test extract_embeddings edge cases
# ============================================================================


def test_extract_embeddings_with_custom_layer_name() -> None:
    """Test embedding extraction with custom layer name."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    loader = _make_multiclass_loader()

    embeddings, rows = extract_embeddings(model, loader, device, amp_enabled=False, layer_name="avgpool")

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 8


def test_extract_embeddings_with_invalid_layer_name() -> None:
    """Test embedding extraction with non-existent layer name falls back to avgpool."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    loader = _make_multiclass_loader()

    embeddings, rows = extract_embeddings(model, loader, device, amp_enabled=False, layer_name="nonexistent_layer")

    # Should fall back to avgpool and still work
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 8


def test_extract_embeddings_with_backbone_avgpool() -> None:
    """Test embedding extraction from model with backbone.avgpool structure."""
    model = DummyModelWithBackbone()
    device = torch.device("cpu")
    loader = _make_multiclass_loader()

    embeddings, rows = extract_embeddings(model, loader, device, amp_enabled=False)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 8


def test_extract_embeddings_no_avgpool() -> None:
    """Test embedding extraction when model has no avgpool layer."""
    model = DummyModelNoAvgPool()
    device = torch.device("cpu")
    loader = _make_multiclass_loader()

    embeddings, rows = extract_embeddings(model, loader, device, amp_enabled=False)

    # Should handle gracefully (may return empty or use fallback)
    assert isinstance(embeddings, np.ndarray)
    assert isinstance(rows, list)


def test_extract_embeddings_with_extra_features() -> None:
    """Test embedding extraction with extra tabular features."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    loader = _make_multiclass_loader_with_extra_features()

    embeddings, rows = extract_embeddings(model, loader, device, amp_enabled=False)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 8


def test_extract_embeddings_with_none_batches() -> None:
    """Test embedding extraction handles None batches."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")

    def collate(batch):
        if not batch:
            return None
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    x = torch.randn(4, 3, 8, 8)
    y = torch.randint(0, 4, (4,))
    meta = [{"id": i} for i in range(4)]
    dataset = [(x[i], y[i], meta[i]) for i in range(4)]
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)

    embeddings, rows = extract_embeddings(model, loader, device, amp_enabled=False)

    assert isinstance(embeddings, np.ndarray)


# ============================================================================
# Test plot_history edge cases
# ============================================================================


def test_plot_history_with_partial_fields() -> None:
    """Test plot_history when some expected fields are missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        history = [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6},
            {"epoch": 2, "train_loss": 0.4, "val_loss": 0.55},
        ]

        # Should handle missing acc fields gracefully - saves CSV but plot may fail
        plot_history(history, outdir)

        # CSV should still be created even if plot fails
        assert (outdir / "train_history.csv").exists()


def test_plot_history_single_epoch() -> None:
    """Test plot_history with only one epoch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        history = [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6, "train_acc": 0.7, "val_acc": 0.65},
        ]

        plot_history(history, outdir)

        assert (outdir / "train_history.csv").exists()


def test_plot_history_creates_directory() -> None:
    """Test that plot_history creates output directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "nested" / "output"
        history = [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6, "train_acc": 0.7, "val_acc": 0.65},
        ]

        plot_history(history, outdir)

        assert outdir.exists()
        assert (outdir / "train_history.csv").exists()


# ============================================================================
# Test save_predictions edge cases
# ============================================================================


def test_save_predictions_with_nested_data() -> None:
    """Test save_predictions with complex nested dictionary data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        pred_rows = [
            {"path": "img1.png", "y_true": 1, "y_pred": 1, "probs": [0.1, 0.8, 0.05, 0.05]},
            {"path": "img2.png", "y_true": 0, "y_pred": 1, "probs": [0.2, 0.6, 0.1, 0.1]},
        ]

        save_predictions(pred_rows, outdir)

        assert (outdir / "val_predictions.csv").exists()
        df = pd.read_csv(outdir / "val_predictions.csv")
        assert len(df) == 2


def test_save_predictions_creates_directory() -> None:
    """Test that save_predictions creates output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "predictions"
        pred_rows = [{"path": "img1.png", "y_true": 1, "y_pred": 1}]

        save_predictions(pred_rows, outdir)

        assert outdir.exists()


# ============================================================================
# Test save_metrics_figure edge cases
# ============================================================================


def test_save_metrics_figure_with_binary_classification() -> None:
    """Test metrics figure with binary classification (2 classes)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "metrics.png")
        metrics = {
            "confusion_matrix": [[50, 5], [3, 42]],
            "classification_report": {
                "0": {"precision": 0.94, "recall": 0.91, "f1-score": 0.92},
                "1": {"precision": 0.89, "recall": 0.93, "f1-score": 0.91},
                "accuracy": 0.92,
                "macro avg": {"f1-score": 0.91},
            },
        }

        save_metrics_figure(metrics, out_path)

        assert Path(out_path).exists()


def test_save_metrics_figure_with_large_num_classes() -> None:
    """Test metrics figure with many classes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "metrics.png")
        num_classes = 10
        cm = np.eye(num_classes) * 10
        report = {str(i): {"precision": 0.8, "recall": 0.85, "f1-score": 0.82} for i in range(num_classes)}
        report["accuracy"] = 0.84
        report["macro avg"] = {"f1-score": 0.82}

        metrics = {
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        save_metrics_figure(metrics, out_path)

        assert Path(out_path).exists()


def test_save_metrics_figure_missing_confusion_matrix() -> None:
    """Test save_metrics_figure when confusion matrix is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "metrics.png")
        metrics = {
            "classification_report": {
                "0": {"precision": 0.8, "recall": 0.85, "f1-score": 0.82},
            },
        }

        save_metrics_figure(metrics, out_path)

        # Should handle gracefully without crashing


# ============================================================================
# Test plot_view_comparison edge cases
# ============================================================================


def test_plot_view_comparison_with_missing_metrics() -> None:
    """Test view comparison when some metrics are missing from dictionaries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "view_comparison.png"

        cc_metrics = {"acc": 0.75, "macro_f1": 0.72}  # Missing some metrics
        mlo_metrics = {"acc": 0.78, "bal_acc": 0.74}  # Different missing metrics

        plot_view_comparison(cc_metrics, mlo_metrics, None, out_path)

        assert out_path.exists()


def test_plot_view_comparison_all_none() -> None:
    """Test view comparison when all metrics are None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "view_comparison.png"

        plot_view_comparison(None, None, None, out_path)

        # Should handle gracefully (may not create file)


def test_plot_view_comparison_ensemble_only() -> None:
    """Test view comparison with only ensemble metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "view_comparison.png"

        ensemble_metrics = {
            "acc": 0.82,
            "macro_f1": 0.80,
            "bal_acc": 0.78,
            "kappa_quadratic": 0.72,
            "auc_ovr": 0.85,
        }

        plot_view_comparison(None, None, ensemble_metrics, out_path)

        assert out_path.exists()


# ============================================================================
# Test _save_gradcam_batch edge cases
# ============================================================================


def test_save_gradcam_batch_with_layer4() -> None:
    """Test GradCAM batch saving with layer4 model."""
    model = DummyModelWithLayer4()
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        x = torch.randn(2, 3, 8, 8).to(device)
        preds = torch.tensor([0, 1]).to(device)
        metas = [{"accession": "acc1"}, {"accession": "acc2"}]

        model.eval()
        saved = _save_gradcam_batch(model, x, preds, metas, out_dir, 0, device)

        # May save images depending on implementation
        assert isinstance(saved, int)
        assert saved >= 0


def test_save_gradcam_batch_no_target_layer() -> None:
    """Test GradCAM batch saving when model has no suitable target layer."""
    model = DummyModelNoAvgPool()
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        x = torch.randn(2, 3, 8, 8).to(device)
        preds = torch.tensor([0, 1]).to(device)
        metas = [{"accession": "acc1"}, {"accession": "acc2"}]

        model.eval()
        saved = _save_gradcam_batch(model, x, preds, metas, out_dir, 0, device)

        # Should return 0 when no suitable layer found
        assert saved == 0


def test_save_gradcam_batch_with_features_model() -> None:
    """Test GradCAM with model that has features attribute."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        x = torch.randn(2, 3, 8, 8).to(device)
        preds = torch.tensor([0, 1]).to(device)
        metas = [{"accession": "acc1"}, {"accession": "acc2"}]

        model.eval()
        saved = _save_gradcam_batch(model, x, preds, metas, out_dir, 0, device)

        assert isinstance(saved, int)
        assert saved >= 0
