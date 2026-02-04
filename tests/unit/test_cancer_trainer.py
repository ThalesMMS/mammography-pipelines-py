from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("numpy")

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from mammography.training.cancer_trainer import (
    get_sens_spec,
    _prepare_targets,
    train_one_epoch,
    evaluate,
    collect_predictions,
    fit_classifier,
    DensityHistoryEntry,
)


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


def test_get_sens_spec() -> None:
    """Test sensitivity and specificity computation."""
    # Perfect predictions
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    sens, spec = get_sens_spec(y_true, y_pred)
    assert sens == 1.0
    assert spec == 1.0

    # All wrong predictions
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    sens, spec = get_sens_spec(y_true, y_pred)
    assert sens == 0.0
    assert spec == 0.0

    # Mixed predictions
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    sens, spec = get_sens_spec(y_true, y_pred)
    assert sens == 0.5
    assert spec == 0.5


def test_prepare_targets() -> None:
    """Test label conversion from 1-4 to 0-3 range."""
    # Valid labels (1-4)
    labels = torch.tensor([1, 2, 3, 4])
    converted, mask = _prepare_targets(labels)
    assert torch.all(mask)
    assert torch.equal(converted, torch.tensor([0, 1, 2, 3]))

    # Labels with zeros (invalid)
    labels = torch.tensor([0, 1, 2, 0, 3])
    converted, mask = _prepare_targets(labels)
    expected_mask = torch.tensor([False, True, True, False, True])
    assert torch.equal(mask, expected_mask)
    assert torch.equal(converted, torch.tensor([0, 1, 2]))

    # 2D labels should be flattened
    labels = torch.tensor([[1, 2], [3, 4]])
    converted, mask = _prepare_targets(labels)
    assert converted.ndim == 1
    assert torch.equal(converted, torch.tensor([0, 1, 2, 3]))


def test_train_one_epoch() -> None:
    """Test training for one epoch."""
    model = DummyBinaryModel()
    loader = _make_binary_loader()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    loss, acc = train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scaler=None,
        amp_enabled=False,
    )

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_train_one_epoch_with_amp() -> None:
    """Test training with automatic mixed precision (CPU fallback)."""
    model = DummyBinaryModel()
    loader = _make_binary_loader()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    loss, acc = train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scaler=None,
        amp_enabled=True,  # Should be disabled on CPU
    )

    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_evaluate() -> None:
    """Test model evaluation."""
    model = DummyBinaryModel()
    loader = _make_binary_loader()
    criterion = nn.BCELoss()
    device = torch.device("cpu")

    loss, acc = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        amp_enabled=False,
    )

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_collect_predictions() -> None:
    """Test prediction collection."""
    model = DummyBinaryModel()
    loader = _make_binary_loader()
    device = torch.device("cpu")

    results = collect_predictions(
        model=model,
        loader=loader,
        device=device,
        amp_enabled=False,
    )

    assert "predictions" in results
    assert "labels" in results
    assert "binary_predictions" in results

    assert isinstance(results["predictions"], np.ndarray)
    assert isinstance(results["labels"], np.ndarray)
    assert isinstance(results["binary_predictions"], np.ndarray)

    # Check shapes match
    assert results["predictions"].shape == results["labels"].shape
    assert results["binary_predictions"].shape == results["labels"].shape

    # Check binary predictions are 0 or 1
    assert np.all((results["binary_predictions"] == 0) | (results["binary_predictions"] == 1))

    # Check predictions are between 0 and 1
    assert np.all((results["predictions"] >= 0) & (results["predictions"] <= 1))


def test_fit_classifier() -> None:
    """Test full training loop with multiple epochs."""
    model = DummyBinaryModel()
    train_loader = _make_binary_loader(num_samples=8)
    val_loader = _make_binary_loader(num_samples=8)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")
    num_epochs = 2

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

    # Check history structure
    assert isinstance(history, list)
    assert len(history) == num_epochs

    # Check each entry
    for i, entry in enumerate(history):
        assert isinstance(entry, DensityHistoryEntry)
        assert entry.epoch == i + 1
        assert isinstance(entry.train_loss, float)
        assert isinstance(entry.train_acc, float)
        assert isinstance(entry.val_loss, float)
        assert isinstance(entry.val_acc, float)
        assert 0.0 <= entry.train_acc <= 1.0
        assert 0.0 <= entry.val_acc <= 1.0


def test_density_history_entry() -> None:
    """Test DensityHistoryEntry dataclass."""
    entry = DensityHistoryEntry(
        epoch=1,
        train_loss=0.5,
        train_acc=0.8,
        val_loss=0.6,
        val_acc=0.75,
    )

    assert entry.epoch == 1
    assert entry.train_loss == 0.5
    assert entry.train_acc == 0.8
    assert entry.val_loss == 0.6
    assert entry.val_acc == 0.75

    # Test with None values
    entry_none = DensityHistoryEntry(
        epoch=2,
        train_loss=0.4,
        train_acc=0.85,
        val_loss=None,
        val_acc=None,
    )

    assert entry_none.val_loss is None
    assert entry_none.val_acc is None
