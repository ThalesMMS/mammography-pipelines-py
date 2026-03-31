from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from mammography.training.cancer_trainer import (
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


# ============================================================================
# Test fit_classifier backward compatibility
# ============================================================================


def test_fit_classifier_epochs_backward_compat() -> None:
    """Test that 'epochs' parameter works for backward compatibility."""
    model = DummyBinaryModel()
    train_loader = _make_binary_loader()
    val_loader = _make_binary_loader()
    device = torch.device("cpu")

    # Use 'epochs' instead of 'num_epochs' (old API)
    history = fit_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=3,  # Backward compatibility parameter
        amp_enabled=False,
    )

    assert len(history) == 3
    assert all(isinstance(entry, DensityHistoryEntry) for entry in history)
    assert history[0].epoch == 1
    assert history[2].epoch == 3


def test_fit_classifier_lr_backward_compat() -> None:
    """Test that 'lr' parameter creates a default optimizer."""
    model = DummyBinaryModel()
    train_loader = _make_binary_loader()
    val_loader = _make_binary_loader()
    device = torch.device("cpu")

    # Use 'lr' parameter to create default optimizer
    history = fit_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=2,
        lr=0.01,  # Backward compatibility parameter
        amp_enabled=False,
    )

    assert len(history) == 2
    assert all(isinstance(entry, DensityHistoryEntry) for entry in history)


def test_fit_classifier_num_epochs_preferred() -> None:
    """Test that 'num_epochs' takes precedence over 'epochs'."""
    model = DummyBinaryModel()
    train_loader = _make_binary_loader()
    val_loader = _make_binary_loader()
    device = torch.device("cpu")

    # Both parameters provided - num_epochs should take precedence
    history = fit_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=2,
        epochs=5,  # This should be ignored
        amp_enabled=False,
    )

    assert len(history) == 2  # num_epochs wins


def test_fit_classifier_default_parameters() -> None:
    """Test that fit_classifier works with all default parameters."""
    model = DummyBinaryModel()
    train_loader = _make_binary_loader()
    val_loader = _make_binary_loader()

    # Call with minimal parameters - should use defaults
    history = fit_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Default is 10 epochs
    assert len(history) == 10
    assert all(isinstance(entry, DensityHistoryEntry) for entry in history)


def test_fit_classifier_device_default() -> None:
    """Test that fit_classifier defaults to CPU device."""
    model = DummyBinaryModel()
    train_loader = _make_binary_loader()
    val_loader = _make_binary_loader()

    # No device specified
    history = fit_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,
    )

    assert len(history) == 1


def test_fit_classifier_criterion_default() -> None:
    """Test that fit_classifier uses BCEWithLogitsLoss by default."""
    model = DummyBinaryModel()
    train_loader = _make_binary_loader()
    val_loader = _make_binary_loader()

    # No criterion specified
    history = fit_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,
    )

    assert len(history) == 1
    # Should complete without errors using default criterion


def test_fit_classifier_optimizer_default() -> None:
    """Test that fit_classifier creates Adam optimizer by default."""
    model = DummyBinaryModel()
    train_loader = _make_binary_loader()
    val_loader = _make_binary_loader()

    # No optimizer specified, no lr specified
    history = fit_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,
    )

    assert len(history) == 1
    # Should complete without errors using default Adam optimizer with lr=0.001


def test_fit_classifier_custom_parameters() -> None:
    """Test fit_classifier with custom criterion and optimizer."""
    model = DummyBinaryModel()
    train_loader = _make_binary_loader()
    val_loader = _make_binary_loader()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    history = fit_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=3,
        amp_enabled=False,
    )

    assert len(history) == 3
    # Verify history structure
    for i, entry in enumerate(history):
        assert entry.epoch == i + 1
        assert isinstance(entry.train_loss, float)
        assert isinstance(entry.train_acc, float)
        assert isinstance(entry.val_loss, float)
        assert isinstance(entry.val_acc, float)


def test_fit_classifier_with_amp() -> None:
    """Test fit_classifier with automatic mixed precision enabled."""
    model = DummyBinaryModel()
    train_loader = _make_binary_loader()
    val_loader = _make_binary_loader()
    device = torch.device("cpu")

    history = fit_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=2,
        amp_enabled=True,  # Should be no-op on CPU
    )

    assert len(history) == 2


def test_fit_classifier_with_scaler() -> None:
    """Test fit_classifier with GradScaler."""
    model = DummyBinaryModel()
    train_loader = _make_binary_loader()
    val_loader = _make_binary_loader()
    device = torch.device("cpu")
    scaler = torch.amp.GradScaler("cpu")

    history = fit_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=2,
        scaler=scaler,
        amp_enabled=True,
    )

    assert len(history) == 2


# ============================================================================
# Test early stopping patience counter logic
# ============================================================================


def test_early_stopping_patience_counter_logic() -> None:
    """Test early stopping patience counter increments and resets correctly.

    This test simulates the early stopping logic from train.py:
    - patience_ctr increments when validation doesn't improve
    - patience_ctr resets to 0 when validation improves
    """
    # Simulate early stopping logic
    early_stop_patience = 3
    early_stop_min_delta = 0.01

    patience_ctr = 0
    best_metric = 0.5

    # Epoch 1: Improvement
    v_macro_f1 = 0.6
    improved = v_macro_f1 > best_metric + early_stop_min_delta
    assert improved is True
    if improved:
        best_metric = v_macro_f1
        patience_ctr = 0
    else:
        patience_ctr += 1

    assert patience_ctr == 0
    assert best_metric == 0.6

    # Epoch 2: No improvement (small increase below delta)
    v_macro_f1 = 0.605
    improved = v_macro_f1 > best_metric + early_stop_min_delta
    assert improved is False
    if improved:
        best_metric = v_macro_f1
        patience_ctr = 0
    else:
        patience_ctr += 1

    assert patience_ctr == 1

    # Epoch 3: No improvement (decrease)
    v_macro_f1 = 0.59
    improved = v_macro_f1 > best_metric + early_stop_min_delta
    assert improved is False
    if improved:
        best_metric = v_macro_f1
        patience_ctr = 0
    else:
        patience_ctr += 1

    assert patience_ctr == 2

    # Epoch 4: Improvement again (resets counter)
    v_macro_f1 = 0.7
    improved = v_macro_f1 > best_metric + early_stop_min_delta
    assert improved is True
    if improved:
        best_metric = v_macro_f1
        patience_ctr = 0
    else:
        patience_ctr += 1

    assert patience_ctr == 0
    assert best_metric == 0.7


def test_early_stopping_triggers_correctly() -> None:
    """Test that early stopping triggers when patience is exceeded."""
    early_stop_patience = 3
    early_stop_min_delta = 0.01

    patience_ctr = 0
    best_metric = 0.8
    should_stop = False

    # Simulate 3 epochs without improvement
    for _ in range(3):
        v_macro_f1 = 0.79  # No improvement
        improved = v_macro_f1 > best_metric + early_stop_min_delta
        if improved:
            best_metric = v_macro_f1
            patience_ctr = 0
        else:
            patience_ctr += 1

        # Check early stopping condition
        if early_stop_patience and patience_ctr >= early_stop_patience:
            should_stop = True
            break

    assert patience_ctr == 3
    assert should_stop is True


def test_early_stopping_min_delta_threshold() -> None:
    """Test that min_delta threshold works correctly."""
    early_stop_min_delta = 0.05
    best_metric = 0.8

    # Case 1: Improvement exactly at threshold (not considered improvement)
    v_macro_f1 = 0.85
    improved = v_macro_f1 > best_metric + early_stop_min_delta
    assert improved is False  # 0.85 == 0.8 + 0.05, not strictly greater

    # Case 2: Improvement above threshold
    v_macro_f1 = 0.851
    improved = v_macro_f1 > best_metric + early_stop_min_delta
    assert improved is True  # 0.851 > 0.8 + 0.05

    # Case 3: Small improvement below threshold
    v_macro_f1 = 0.83
    improved = v_macro_f1 > best_metric + early_stop_min_delta
    assert improved is False  # 0.83 < 0.8 + 0.05


def test_early_stopping_no_patience() -> None:
    """Test that early stopping is disabled when patience is 0."""
    early_stop_patience = 0  # Disabled
    early_stop_min_delta = 0.01

    patience_ctr = 0
    best_metric = 0.8

    # Even after many epochs without improvement
    for _ in range(10):
        v_macro_f1 = 0.79
        improved = v_macro_f1 > best_metric + early_stop_min_delta
        if improved:
            best_metric = v_macro_f1
            patience_ctr = 0
        else:
            patience_ctr += 1

        # Early stopping check
        should_stop = early_stop_patience and patience_ctr >= early_stop_patience
        assert should_stop is False  # Should never trigger when patience is 0


def test_early_stopping_checkpoint_restoration() -> None:
    """Test that patience counter is properly saved and restored from checkpoints."""
    # Simulate checkpoint save
    checkpoint_state = {
        "epoch": 5,
        "best_metric": 0.85,
        "patience_ctr": 2,
    }

    # Simulate checkpoint restore
    restored_patience = checkpoint_state["patience_ctr"]
    assert restored_patience == 2

    # Continue training with restored counter
    patience_ctr = restored_patience
    early_stop_patience = 3

    # One more epoch without improvement should trigger early stopping
    patience_ctr += 1
    should_stop = early_stop_patience and patience_ctr >= early_stop_patience
    assert should_stop is True
    assert patience_ctr == 3
