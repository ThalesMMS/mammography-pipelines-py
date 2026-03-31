from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

torch = pytest.importorskip("torch")

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR


class DummyModel(nn.Module):
    """Minimal model for optimizer tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ============================================================================
# Test ReduceLROnPlateau (mode="max" as used in training pipeline)
# ============================================================================


def test_reduce_lr_on_plateau_initialization() -> None:
    """Test ReduceLROnPlateau initializes with correct parameters."""
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        cooldown=2,
    )

    assert scheduler.mode == "max"
    assert scheduler.factor == 0.5
    assert scheduler.patience == 3
    assert scheduler.min_lrs == [1e-7]
    assert scheduler.cooldown == 2


def test_reduce_lr_on_plateau_reduces_lr_on_no_improvement() -> None:
    """Test ReduceLROnPlateau reduces LR when metric doesn't improve."""
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )

    # Initial LR
    assert optimizer.param_groups[0]["lr"] == 0.1

    # No improvement for patience epochs
    scheduler.step(0.5)  # metric = 0.5
    assert optimizer.param_groups[0]["lr"] == 0.1  # no change yet

    scheduler.step(0.5)  # no improvement
    assert optimizer.param_groups[0]["lr"] == 0.1  # no change yet

    scheduler.step(0.5)  # patience exceeded
    assert optimizer.param_groups[0]["lr"] == 0.05  # reduced by factor


def test_reduce_lr_on_plateau_does_not_reduce_on_improvement() -> None:
    """Test ReduceLROnPlateau doesn't reduce LR when metric improves."""
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    # Improving metrics
    scheduler.step(0.5)
    assert optimizer.param_groups[0]["lr"] == 0.1

    scheduler.step(0.6)  # improvement
    assert optimizer.param_groups[0]["lr"] == 0.1

    scheduler.step(0.7)  # improvement
    assert optimizer.param_groups[0]["lr"] == 0.1


def test_reduce_lr_on_plateau_respects_min_lr() -> None:
    """Test ReduceLROnPlateau doesn't reduce below min_lr."""
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=1,
        min_lr=1e-4,
    )

    # Trigger multiple reductions
    for _ in range(10):
        scheduler.step(0.5)  # no improvement

    # Should not go below min_lr
    assert optimizer.param_groups[0]["lr"] >= 1e-4


def test_reduce_lr_on_plateau_cooldown() -> None:
    """Test ReduceLROnPlateau respects cooldown period."""
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        cooldown=2,
    )

    # Trigger first reduction
    scheduler.step(0.5)
    scheduler.step(0.5)  # patience exceeded, should reduce
    assert optimizer.param_groups[0]["lr"] == 0.05

    # During cooldown, should not reduce again
    scheduler.step(0.5)
    scheduler.step(0.5)
    assert optimizer.param_groups[0]["lr"] == 0.05  # still in cooldown


def test_reduce_lr_on_plateau_state_dict() -> None:
    """Test ReduceLROnPlateau state dict save/load."""
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler1 = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    # Run a few steps
    scheduler1.step(0.5)
    scheduler1.step(0.6)

    # Save state
    state_dict = scheduler1.state_dict()

    # Create new scheduler and load state
    optimizer2 = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler2 = ReduceLROnPlateau(
        optimizer2,
        mode="max",
        factor=0.5,
        patience=2,
    )
    scheduler2.load_state_dict(state_dict)

    # State should be preserved
    assert scheduler2.best == scheduler1.best
    assert scheduler2.num_bad_epochs == scheduler1.num_bad_epochs


# ============================================================================
# Test CosineAnnealingLR
# ============================================================================


def test_cosine_annealing_initialization() -> None:
    """Test CosineAnnealingLR initializes with correct parameters."""
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=10,
        eta_min=1e-7,
    )

    assert scheduler.T_max == 10
    assert scheduler.eta_min == 1e-7


def test_cosine_annealing_lr_decay() -> None:
    """Test CosineAnnealingLR decays learning rate according to cosine schedule."""
    model = DummyModel()
    initial_lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)

    T_max = 10
    eta_min = 0.0
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=eta_min,
    )

    # LR should start at initial value
    assert optimizer.param_groups[0]["lr"] == initial_lr

    # LR should decrease after first step
    scheduler.step()
    lr_after_1 = optimizer.param_groups[0]["lr"]
    assert lr_after_1 < initial_lr

    # LR should continue decreasing toward eta_min
    for _ in range(T_max // 2 - 1):
        scheduler.step()

    lr_at_half = optimizer.param_groups[0]["lr"]
    assert lr_at_half < lr_after_1

    # LR should reach minimum near T_max
    for _ in range(T_max // 2):
        scheduler.step()

    final_lr = optimizer.param_groups[0]["lr"]
    assert abs(final_lr - eta_min) < 1e-6


def test_cosine_annealing_restart() -> None:
    """Test CosineAnnealingLR restarts after T_max."""
    model = DummyModel()
    initial_lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)

    T_max = 5
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=0.0,
    )

    # Run through full cycle
    for _ in range(T_max):
        scheduler.step()

    # LR should restart back to initial value
    scheduler.step()
    restarted_lr = optimizer.param_groups[0]["lr"]
    assert abs(restarted_lr - initial_lr) < 1e-6


def test_cosine_annealing_respects_eta_min() -> None:
    """Test CosineAnnealingLR respects eta_min."""
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    eta_min = 1e-5
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=10,
        eta_min=eta_min,
    )

    # Run through full cycle
    for _ in range(11):
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        # LR should never go below eta_min
        assert current_lr >= eta_min - 1e-8


def test_cosine_annealing_state_dict() -> None:
    """Test CosineAnnealingLR state dict save/load."""
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler1 = CosineAnnealingLR(
        optimizer,
        T_max=10,
        eta_min=1e-7,
    )

    # Run a few steps
    for _ in range(3):
        scheduler1.step()

    # Save state
    state_dict = scheduler1.state_dict()

    # Create new scheduler and load state
    optimizer2 = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler2 = CosineAnnealingLR(
        optimizer2,
        T_max=10,
        eta_min=1e-7,
    )
    scheduler2.load_state_dict(state_dict)

    # Continue stepping both schedulers
    scheduler1.step()
    scheduler2.step()

    # LR should match
    lr1 = optimizer.param_groups[0]["lr"]
    lr2 = optimizer2.param_groups[0]["lr"]
    assert abs(lr1 - lr2) < 1e-8


# ============================================================================
# Test StepLR
# ============================================================================


def test_step_lr_initialization() -> None:
    """Test StepLR initializes with correct parameters."""
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler = StepLR(
        optimizer,
        step_size=5,
        gamma=0.5,
    )

    assert scheduler.step_size == 5
    assert scheduler.gamma == 0.5


def test_step_lr_reduces_at_intervals() -> None:
    """Test StepLR reduces LR at step_size intervals."""
    model = DummyModel()
    initial_lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)

    step_size = 3
    gamma = 0.5
    scheduler = StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )

    # LR should stay constant until step_size
    assert optimizer.param_groups[0]["lr"] == initial_lr

    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == initial_lr

    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == initial_lr

    # After step_size steps, LR should be reduced
    scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - initial_lr * gamma) < 1e-8

    # Should stay constant for next step_size steps
    scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - initial_lr * gamma) < 1e-8

    scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - initial_lr * gamma) < 1e-8

    # Next reduction
    scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - initial_lr * gamma * gamma) < 1e-8


def test_step_lr_multiple_reductions() -> None:
    """Test StepLR applies gamma multiple times."""
    model = DummyModel()
    initial_lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)

    step_size = 2
    gamma = 0.1
    scheduler = StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )

    # Run through multiple steps
    expected_lrs = [
        initial_lr,           # epoch 0
        initial_lr,           # epoch 1
        initial_lr * gamma,   # epoch 2
        initial_lr * gamma,   # epoch 3
        initial_lr * gamma ** 2,  # epoch 4
        initial_lr * gamma ** 2,  # epoch 5
        initial_lr * gamma ** 3,  # epoch 6
    ]

    for i, expected_lr in enumerate(expected_lrs):
        current_lr = optimizer.param_groups[0]["lr"]
        assert abs(current_lr - expected_lr) < 1e-8, f"Epoch {i}: expected {expected_lr}, got {current_lr}"
        if i < len(expected_lrs) - 1:
            scheduler.step()


def test_step_lr_state_dict() -> None:
    """Test StepLR state dict save/load."""
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler1 = StepLR(
        optimizer,
        step_size=3,
        gamma=0.5,
    )

    # Run a few steps
    for _ in range(5):
        scheduler1.step()

    # Save state
    state_dict = scheduler1.state_dict()

    # Create new scheduler and load state
    optimizer2 = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler2 = StepLR(
        optimizer2,
        step_size=3,
        gamma=0.5,
    )
    scheduler2.load_state_dict(state_dict)

    # Continue stepping both schedulers
    scheduler1.step()
    scheduler2.step()

    # LR should match
    lr1 = optimizer.param_groups[0]["lr"]
    lr2 = optimizer2.param_groups[0]["lr"]
    assert abs(lr1 - lr2) < 1e-8


def test_step_lr_with_minimum_step_size() -> None:
    """Test StepLR with step_size=1 (reduces every epoch)."""
    model = DummyModel()
    initial_lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)

    gamma = 0.9
    scheduler = StepLR(
        optimizer,
        step_size=1,
        gamma=gamma,
    )

    # LR should reduce every step
    expected_lr = initial_lr
    for i in range(5):
        current_lr = optimizer.param_groups[0]["lr"]
        assert abs(current_lr - expected_lr) < 1e-8
        scheduler.step()
        expected_lr *= gamma


# ============================================================================
# Test edge cases and multiple parameter groups
# ============================================================================


def test_schedulers_with_multiple_param_groups() -> None:
    """Test schedulers work correctly with multiple parameter groups."""
    model = DummyModel()

    # Create optimizer with two parameter groups (simulating backbone + head)
    optimizer = torch.optim.SGD([
        {"params": [model.fc.weight], "lr": 0.01},  # backbone
        {"params": [model.fc.bias], "lr": 0.1},     # head
    ])

    # Test with ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    # Both groups should start with their respective LRs
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[1]["lr"] == 0.1

    # Trigger reduction
    scheduler.step(0.5)
    scheduler.step(0.5)

    # Both groups should be reduced
    assert abs(optimizer.param_groups[0]["lr"] - 0.005) < 1e-8
    assert abs(optimizer.param_groups[1]["lr"] - 0.05) < 1e-8


def test_schedulers_with_zero_initial_lr() -> None:
    """Test schedulers handle edge case of zero initial LR gracefully."""
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    # CosineAnnealingLR with zero initial LR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=10,
        eta_min=0.0,
    )

    # Should not crash
    for _ in range(5):
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == 0.0


def test_reduce_lr_on_plateau_with_nan_metric() -> None:
    """Test ReduceLROnPlateau handles NaN metric gracefully."""
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    initial_lr = optimizer.param_groups[0]["lr"]

    # Step with NaN should not crash (behavior may vary by PyTorch version)
    try:
        scheduler.step(float("nan"))
        # If it doesn't crash, LR should remain unchanged or reduce
        # (PyTorch behavior may vary)
    except (ValueError, RuntimeError):
        # Some versions may raise an error, which is also acceptable
        pass
