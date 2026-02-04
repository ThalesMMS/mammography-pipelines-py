from __future__ import annotations

import sys
import tempfile
from pathlib import Path

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
)
from mammography.training.cancer_trainer import (
    get_sens_spec,
    _prepare_targets,
    train_one_epoch as cancer_train_one_epoch,
    evaluate as cancer_evaluate,
    collect_predictions,
    fit_classifier,
    DensityHistoryEntry,
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


class DummyBinaryModel(nn.Module):
    """Dummy model for binary cancer classification."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def _make_binary_loader(num_samples: int = 8) -> DataLoader:
    """Create a dummy binary classification data loader."""
    x = torch.randn(num_samples, 3, 8, 8)
    y = torch.randint(0, 2, (num_samples,)).float()
    meta = [{"id": i} for i in range(num_samples)]

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    dataset = [(x[i], y[i], meta[i]) for i in range(num_samples)]
    return DataLoader(dataset, batch_size=4, collate_fn=collate)


# ============================================================================
# Test save_atomic (checkpoint persistence)
# ============================================================================


def test_save_atomic_basic() -> None:
    """Test basic checkpoint saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "model.pt"
        state = {"epoch": 1, "model_state": {"dummy": "value"}}

        save_atomic(state, checkpoint_path)

        assert checkpoint_path.exists()
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert loaded["epoch"] == 1
        assert loaded["model_state"]["dummy"] == "value"


def test_save_atomic_with_normalization_stats() -> None:
    """Test checkpoint saving with normalization statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "model.pt"
        state = {"epoch": 1}
        norm_stats = {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]}

        save_atomic(state, checkpoint_path, normalization_stats=norm_stats)

        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert "normalization_stats" in loaded
        assert loaded["normalization_stats"]["mean"] == [0.5, 0.5, 0.5]


def test_save_atomic_overwrite_existing() -> None:
    """Test that save_atomic correctly overwrites existing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "model.pt"

        # Save first checkpoint
        state1 = {"epoch": 1}
        save_atomic(state1, checkpoint_path)

        # Overwrite with second checkpoint
        state2 = {"epoch": 2}
        save_atomic(state2, checkpoint_path)

        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert loaded["epoch"] == 2


# ============================================================================
# Test train_one_epoch (engine.py) - comprehensive coverage
# ============================================================================


def test_train_one_epoch_with_invalid_labels() -> None:
    """Test training with some invalid labels (negative values)."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Create loader with some invalid labels (-1)
    x = torch.randn(8, 3, 8, 8)
    y = torch.tensor([0, 1, -1, 2, -1, 3, 0, 1])
    meta = [{"id": i} for i in range(8)]

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    dataset = [(x[i], y[i], meta[i]) for i in range(8)]
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)

    loss, acc = train_one_epoch(model, loader, optimizer, device, scaler=None)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0


def test_train_one_epoch_with_custom_loss() -> None:
    """Test training with custom loss function."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _make_multiclass_loader()
    custom_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    loss, acc = train_one_epoch(model, loader, optimizer, device, loss_fn=custom_loss, scaler=None)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0


def test_train_one_epoch_with_amp() -> None:
    """Test training with automatic mixed precision."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _make_multiclass_loader()

    loss, acc = train_one_epoch(model, loader, optimizer, device, scaler=None, amp_enabled=True)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0


def test_train_one_epoch_empty_batches() -> None:
    """Test training handles empty batches (all labels invalid)."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Create loader with all invalid labels
    x = torch.randn(8, 3, 8, 8)
    y = torch.full((8,), -1)
    meta = [{"id": i} for i in range(8)]

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    dataset = [(x[i], y[i], meta[i]) for i in range(8)]
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)

    loss, acc = train_one_epoch(model, loader, optimizer, device, scaler=None)
    assert loss == 0.0
    assert acc == 0.0


# ============================================================================
# Test validate (engine.py) - comprehensive coverage
# ============================================================================


def test_validate_with_collect_preds() -> None:
    """Test validation with prediction collection enabled."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    loader = _make_multiclass_loader()

    metrics, pred_rows = validate(model, loader, device, amp_enabled=False, collect_preds=True)

    assert "acc" in metrics
    assert isinstance(pred_rows, list)
    assert len(pred_rows) > 0
    assert "y_true" in pred_rows[0]
    assert "y_pred" in pred_rows[0]
    assert "probs" in pred_rows[0]


def test_validate_with_loss_fn() -> None:
    """Test validation with custom loss function."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    loader = _make_multiclass_loader()
    loss_fn = nn.CrossEntropyLoss()

    metrics, pred_rows = validate(model, loader, device, amp_enabled=False, loss_fn=loss_fn)

    assert "loss" in metrics
    assert metrics["loss"] is not None
    assert isinstance(metrics["loss"], float)


def test_validate_metrics_completeness() -> None:
    """Test that validate returns all expected metrics."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    loader = _make_multiclass_loader()

    metrics, _ = validate(model, loader, device, amp_enabled=False)

    expected_keys = [
        "acc",
        "kappa_quadratic",
        "auc_ovr",
        "macro_f1",
        "bal_acc",
        "bal_acc_adj",
        "confusion_matrix",
        "classification_report",
    ]

    for key in expected_keys:
        assert key in metrics


def test_validate_empty_loader() -> None:
    """Test validation with empty data loader."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")

    # Create empty loader
    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    dataset = []
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)

    metrics, pred_rows = validate(model, loader, device, amp_enabled=False)

    assert metrics["acc"] == 0.0
    assert isinstance(pred_rows, list)
    assert len(pred_rows) == 0


# ============================================================================
# Test extract_embeddings
# ============================================================================


def test_extract_embeddings_basic() -> None:
    """Test basic embedding extraction."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    loader = _make_multiclass_loader()

    embeddings, rows = extract_embeddings(model, loader, device, amp_enabled=False)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 8
    assert isinstance(rows, list)
    assert len(rows) == 8


def test_extract_embeddings_with_amp() -> None:
    """Test embedding extraction with AMP."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    loader = _make_multiclass_loader()

    embeddings, rows = extract_embeddings(model, loader, device, amp_enabled=True)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 8


def test_extract_embeddings_empty_loader() -> None:
    """Test embedding extraction with empty loader."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    dataset = []
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)

    embeddings, rows = extract_embeddings(model, loader, device, amp_enabled=False)

    assert embeddings.shape == (0, 0)
    assert rows == []


# ============================================================================
# Test plot_history
# ============================================================================


def test_plot_history_basic() -> None:
    """Test training history plotting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        history = [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6, "train_acc": 0.7, "val_acc": 0.65},
            {"epoch": 2, "train_loss": 0.4, "val_loss": 0.55, "train_acc": 0.75, "val_acc": 0.7},
        ]

        plot_history(history, outdir)

        assert (outdir / "train_history.csv").exists()
        assert (outdir / "train_history.png").exists()

        df = pd.read_csv(outdir / "train_history.csv")
        assert len(df) == 2
        assert "epoch" in df.columns
        assert "train_loss" in df.columns


def test_plot_history_empty() -> None:
    """Test plot_history with empty history."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        history = []

        plot_history(history, outdir)

        # Should not create files
        assert not (outdir / "train_history.csv").exists()


# ============================================================================
# Test save_predictions
# ============================================================================


def test_save_predictions_basic() -> None:
    """Test saving prediction rows to CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        pred_rows = [
            {"path": "img1.png", "y_true": 1, "y_pred": 1},
            {"path": "img2.png", "y_true": 0, "y_pred": 1},
        ]

        save_predictions(pred_rows, outdir)

        assert (outdir / "val_predictions.csv").exists()
        df = pd.read_csv(outdir / "val_predictions.csv")
        assert len(df) == 2
        assert "y_true" in df.columns


def test_save_predictions_empty() -> None:
    """Test save_predictions with empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        pred_rows = []

        save_predictions(pred_rows, outdir)

        # Should not create file
        assert not (outdir / "val_predictions.csv").exists()


# ============================================================================
# Test save_metrics_figure
# ============================================================================


def test_save_metrics_figure_basic() -> None:
    """Test saving metrics visualization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "metrics.png")
        metrics = {
            "confusion_matrix": [[10, 2], [3, 15]],
            "classification_report": {
                "0": {"precision": 0.8, "recall": 0.85, "f1-score": 0.82},
                "1": {"precision": 0.88, "recall": 0.83, "f1-score": 0.85},
                "accuracy": 0.84,
                "macro avg": {"f1-score": 0.83},
            },
        }

        save_metrics_figure(metrics, out_path)

        assert Path(out_path).exists()


def test_save_metrics_figure_empty() -> None:
    """Test save_metrics_figure with empty metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "metrics.png")
        metrics = {}

        save_metrics_figure(metrics, out_path)

        # Should not crash, file may or may not exist depending on implementation


# ============================================================================
# Test plot_view_comparison
# ============================================================================


def test_plot_view_comparison_all_views() -> None:
    """Test view comparison plot with all three views."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "view_comparison.png"

        cc_metrics = {
            "acc": 0.75,
            "macro_f1": 0.72,
            "bal_acc": 0.70,
            "kappa_quadratic": 0.65,
            "auc_ovr": 0.80,
        }
        mlo_metrics = {
            "acc": 0.78,
            "macro_f1": 0.76,
            "bal_acc": 0.74,
            "kappa_quadratic": 0.68,
            "auc_ovr": 0.82,
        }
        ensemble_metrics = {
            "acc": 0.82,
            "macro_f1": 0.80,
            "bal_acc": 0.78,
            "kappa_quadratic": 0.72,
            "auc_ovr": 0.85,
        }

        plot_view_comparison(cc_metrics, mlo_metrics, ensemble_metrics, out_path)

        assert out_path.exists()


def test_plot_view_comparison_partial() -> None:
    """Test view comparison with only some views."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "view_comparison.png"

        cc_metrics = {"acc": 0.75, "macro_f1": 0.72, "bal_acc": 0.70, "kappa_quadratic": 0.65}

        plot_view_comparison(cc_metrics, None, None, out_path)

        assert out_path.exists()


def test_plot_view_comparison_no_auc() -> None:
    """Test view comparison when AUC is not available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "view_comparison.png"

        cc_metrics = {"acc": 0.75, "macro_f1": 0.72, "bal_acc": 0.70, "kappa_quadratic": 0.65}
        mlo_metrics = {"acc": 0.78, "macro_f1": 0.76, "bal_acc": 0.74, "kappa_quadratic": 0.68}

        plot_view_comparison(cc_metrics, mlo_metrics, None, out_path)

        assert out_path.exists()


# ============================================================================
# Test cancer_trainer functions - additional edge cases
# ============================================================================


def test_get_sens_spec_edge_cases() -> None:
    """Test sensitivity and specificity with edge cases."""
    # All positive predictions
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    sens, spec = get_sens_spec(y_true, y_pred)
    assert sens == 1.0
    assert spec == 0.0

    # All negative predictions
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    sens, spec = get_sens_spec(y_true, y_pred)
    assert sens == 0.0
    assert spec == 1.0


def test_prepare_targets_edge_cases() -> None:
    """Test label preparation with various edge cases."""
    # All invalid labels
    labels = torch.tensor([0, 0, 0, 0])
    converted, mask = _prepare_targets(labels)
    assert len(converted) == 0
    assert not torch.any(mask)

    # Single valid label
    labels = torch.tensor([0, 0, 1, 0])
    converted, mask = _prepare_targets(labels)
    assert len(converted) == 1
    assert converted[0] == 0


def test_cancer_train_one_epoch_different_optimizers() -> None:
    """Test training with different optimizer types."""
    model = DummyBinaryModel()
    loader = _make_binary_loader()
    criterion = nn.BCELoss()
    device = torch.device("cpu")

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss, acc = cancer_train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, amp_enabled=False)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0

    # AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss, acc = cancer_train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, amp_enabled=False)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0


def test_cancer_evaluate_empty_loader() -> None:
    """Test cancer evaluation with empty loader."""
    model = DummyBinaryModel()
    criterion = nn.BCELoss()
    device = torch.device("cpu")

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    dataset = []
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)

    loss, acc = cancer_evaluate(model, loader, criterion, device, amp_enabled=False)
    assert loss == 0.0
    assert acc == 0.0


def test_collect_predictions_threshold() -> None:
    """Test that binary predictions use correct threshold."""
    model = DummyBinaryModel()
    device = torch.device("cpu")
    loader = _make_binary_loader()

    results = collect_predictions(model, loader, device, amp_enabled=False)

    # Check that binary predictions are properly thresholded at 0.5
    binary_preds = results["binary_predictions"]
    predictions = results["predictions"]

    for i in range(len(predictions)):
        expected_binary = 1.0 if predictions[i] > 0.5 else 0.0
        assert binary_preds[i] == expected_binary


def test_fit_classifier_history_structure() -> None:
    """Test that fit_classifier returns properly structured history."""
    model = DummyBinaryModel()
    train_loader = _make_binary_loader()
    val_loader = _make_binary_loader()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")
    num_epochs = 3

    history = fit_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        scaler=None,
        amp_enabled=False,
    )

    assert len(history) == num_epochs
    for i, entry in enumerate(history):
        assert entry.epoch == i + 1
        assert entry.train_loss >= 0.0
        assert entry.val_loss >= 0.0
        assert 0.0 <= entry.train_acc <= 1.0
        assert 0.0 <= entry.val_acc <= 1.0


def test_density_history_entry_equality() -> None:
    """Test DensityHistoryEntry equality and attributes."""
    entry1 = DensityHistoryEntry(
        epoch=1,
        train_loss=0.5,
        train_acc=0.8,
        val_loss=0.6,
        val_acc=0.75,
    )

    entry2 = DensityHistoryEntry(
        epoch=1,
        train_loss=0.5,
        train_acc=0.8,
        val_loss=0.6,
        val_acc=0.75,
    )

    # Dataclasses have automatic equality
    assert entry1 == entry2


# ============================================================================
# Integration tests
# ============================================================================


def test_full_training_workflow_multiclass() -> None:
    """Test complete training workflow for multi-class model."""
    model = DummyMultiClassModel()
    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _make_multiclass_loader()

    # Train for one epoch
    loss1, acc1 = train_one_epoch(model, loader, optimizer, device, scaler=None)
    assert isinstance(loss1, float)

    # Validate
    metrics, pred_rows = validate(model, loader, device, amp_enabled=False, collect_preds=True)
    assert "acc" in metrics
    assert len(pred_rows) > 0

    # Extract embeddings
    embeddings, rows = extract_embeddings(model, loader, device, amp_enabled=False)
    assert embeddings.shape[0] == 8


def test_full_training_workflow_binary() -> None:
    """Test complete training workflow for binary classification."""
    model = DummyBinaryModel()
    train_loader = _make_binary_loader()
    val_loader = _make_binary_loader()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cpu")

    # Full training loop
    history = fit_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
        scaler=None,
        amp_enabled=False,
    )

    assert len(history) == 2

    # Collect predictions
    results = collect_predictions(model, train_loader, device, amp_enabled=False)
    assert "predictions" in results
    assert "labels" in results

    # Evaluate
    loss, acc = cancer_evaluate(model, val_loader, criterion, device, amp_enabled=False)
    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0
