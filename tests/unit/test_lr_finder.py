#
# test_lr_finder.py
# mammography-pipelines
#
# Tests for LR Finder implementation in tuning/lr_finder.py
#
# Medical Disclaimer: This is an educational research project.
# Not intended for clinical use or medical diagnosis.
#
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mammography.tuning.lr_finder import LRFinder


class DummyModel(nn.Module):
    """Dummy model for testing."""

    def __init__(self, input_size: int = 10, num_classes: int = 4):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x, extra_features=None):
        # Simple forward pass
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.fc(x_flat)


@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""
    return DummyModel(input_size=100, num_classes=4)


@pytest.fixture
def dummy_optimizer(dummy_model):
    """Create a dummy optimizer for testing."""
    return torch.optim.SGD(dummy_model.parameters(), lr=1e-3)


@pytest.fixture
def dummy_criterion():
    """Create a dummy loss function for testing."""
    return nn.CrossEntropyLoss()


@pytest.fixture
def dummy_dataloader():
    """Create a dummy dataloader for testing."""
    # Create simple tensor dataset
    x = torch.randn(32, 1, 10, 10)  # batch_size=32, channels=1, h=10, w=10
    y = torch.randint(0, 4, (32,))  # 4 classes
    dataset = TensorDataset(x, y, torch.zeros(32))  # Add dummy metadata
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def lr_finder(dummy_model, dummy_optimizer, dummy_criterion):
    """Create LRFinder instance for testing."""
    device = torch.device("cpu")
    return LRFinder(
        model=dummy_model,
        optimizer=dummy_optimizer,
        criterion=dummy_criterion,
        device=device,
        amp_enabled=False,
    )


class TestLRFinderInit:
    """Test LRFinder initialization."""

    def test_init_basic(self, dummy_model, dummy_optimizer, dummy_criterion):
        """Test basic initialization."""
        device = torch.device("cpu")
        lr_finder = LRFinder(
            model=dummy_model,
            optimizer=dummy_optimizer,
            criterion=dummy_criterion,
            device=device,
        )

        assert lr_finder.model is dummy_model
        assert lr_finder.optimizer is dummy_optimizer
        assert lr_finder.criterion is dummy_criterion
        assert lr_finder.device == device
        assert lr_finder.amp_enabled is False

    def test_init_with_amp(self, dummy_model, dummy_optimizer, dummy_criterion):
        """Test initialization with AMP enabled."""
        device = torch.device("cpu")
        lr_finder = LRFinder(
            model=dummy_model,
            optimizer=dummy_optimizer,
            criterion=dummy_criterion,
            device=device,
            amp_enabled=True,
        )

        assert lr_finder.amp_enabled is True

    def test_initial_state_saved(self, lr_finder):
        """Test that initial state is properly saved."""
        assert "model" in lr_finder.initial_state
        assert "optimizer" in lr_finder.initial_state
        assert isinstance(lr_finder.initial_state["model"], dict)
        assert isinstance(lr_finder.initial_state["optimizer"], dict)

    def test_results_initialized_empty(self, lr_finder):
        """Test that results are initialized as empty."""
        assert lr_finder.lrs == []
        assert lr_finder.losses == []
        assert lr_finder.best_lr is None


class TestLRFinderCopyStateDict:
    """Test _copy_state_dict helper method."""

    def test_copy_state_dict_tensors(self, lr_finder):
        """Test that tensors are cloned in state dict."""
        state_dict = {
            "weight": torch.randn(10, 10),
            "bias": torch.randn(10),
        }

        copied = lr_finder._copy_state_dict(state_dict)

        # Check that tensors are cloned (not same reference)
        assert copied["weight"] is not state_dict["weight"]
        assert copied["bias"] is not state_dict["bias"]

        # Check that values are equal
        assert torch.equal(copied["weight"], state_dict["weight"])
        assert torch.equal(copied["bias"], state_dict["bias"])

    def test_copy_state_dict_non_tensors(self, lr_finder):
        """Test that non-tensor values are preserved."""
        state_dict = {
            "step": 100,
            "param_name": "test",
        }

        copied = lr_finder._copy_state_dict(state_dict)

        assert copied["step"] == 100
        assert copied["param_name"] == "test"

    def test_copy_state_dict_mixed(self, lr_finder):
        """Test copying state dict with mixed types."""
        state_dict = {
            "weight": torch.randn(5, 5),
            "step": 42,
            "name": "layer1",
        }

        copied = lr_finder._copy_state_dict(state_dict)

        assert copied["weight"] is not state_dict["weight"]
        assert torch.equal(copied["weight"], state_dict["weight"])
        assert copied["step"] == 42
        assert copied["name"] == "layer1"


class TestLRFinderSuggestLR:
    """Test _suggest_lr method."""

    def test_suggest_lr_insufficient_data(self, lr_finder):
        """Test suggestion with insufficient data points."""
        # Set minimal data
        lr_finder.lrs = [1e-5, 1e-4, 1e-3]
        lr_finder.losses = [2.0, 1.5, 1.2]

        suggested_lr = lr_finder._suggest_lr()

        # Should use min loss approach with < 10 points
        min_idx = np.argmin(lr_finder.losses)
        expected_lr = lr_finder.lrs[min_idx] / 10.0
        assert suggested_lr == pytest.approx(expected_lr)

    def test_suggest_lr_with_sufficient_data(self, lr_finder):
        """Test suggestion with sufficient data points."""
        # Create realistic loss curve (decreasing then increasing)
        lrs = np.logspace(-7, -1, 50)
        losses = [3.0 - 0.05 * i + 0.002 * i**2 for i in range(50)]

        lr_finder.lrs = lrs.tolist()
        lr_finder.losses = losses

        suggested_lr = lr_finder._suggest_lr()

        # Should return a valid LR in the range
        assert suggested_lr >= lrs[0]
        assert suggested_lr <= lrs[-1]

    def test_suggest_lr_minimum_bound(self, lr_finder):
        """Test that suggested LR respects minimum bound."""
        # Create data that would suggest very low LR
        lr_finder.lrs = [1e-8, 1e-7, 1e-6]
        lr_finder.losses = [2.0, 1.5, 1.2]

        suggested_lr = lr_finder._suggest_lr()

        # Should respect minimum LR of 1e-7
        assert suggested_lr >= 1e-7

    def test_suggest_lr_empty_data(self, lr_finder):
        """Test suggestion with empty data."""
        lr_finder.lrs = []
        lr_finder.losses = []

        # Should handle gracefully or raise error
        with pytest.raises(Exception):  # ValueError or IndexError
            lr_finder._suggest_lr()


class TestLRFinderGetResults:
    """Test get_results method."""

    def test_get_results_empty(self, lr_finder):
        """Test getting results before running range test."""
        results = lr_finder.get_results()

        assert "lrs" in results
        assert "losses" in results
        assert "suggested_lr" in results
        assert results["lrs"] == []
        assert results["losses"] == []
        assert results["suggested_lr"] is None

    def test_get_results_with_data(self, lr_finder):
        """Test getting results after setting data."""
        lr_finder.lrs = [1e-5, 1e-4, 1e-3]
        lr_finder.losses = [2.0, 1.5, 1.2]
        lr_finder.best_lr = 1e-4

        results = lr_finder.get_results()

        assert results["lrs"] == [1e-5, 1e-4, 1e-3]
        assert results["losses"] == [2.0, 1.5, 1.2]
        assert results["suggested_lr"] == 1e-4


class TestLRFinderPlot:
    """Test plot method."""

    def test_plot_no_data_warning(self, lr_finder):
        """Test that plotting with no data issues warning."""
        with patch("mammography.tuning.lr_finder.plt") as mock_plt:
            lr_finder.plot()
            # Should not create plot with no data
            mock_plt.subplots.assert_not_called()

    def test_plot_with_data(self, lr_finder):
        """Test plotting with valid data."""
        # Set some data
        lr_finder.lrs = list(np.logspace(-5, -2, 30))
        lr_finder.losses = [2.0 - 0.05 * i for i in range(30)]
        lr_finder.best_lr = 1e-3

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_lr_plot.png"
            lr_finder.plot(save_path=save_path)

            # Check that file was created
            assert save_path.exists()

    def test_plot_skip_parameters(self, lr_finder):
        """Test plot with skip_start and skip_end parameters."""
        # Set data with 30 points
        lr_finder.lrs = list(np.logspace(-5, -2, 30))
        lr_finder.losses = [2.0 - 0.05 * i for i in range(30)]
        lr_finder.best_lr = 1e-3

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_lr_plot.png"
            # Skip first 5 and last 5 points
            lr_finder.plot(save_path=save_path, skip_start=5, skip_end=5)

            assert save_path.exists()

    def test_plot_insufficient_data_after_skip(self, lr_finder):
        """Test plot when skip parameters remove all data."""
        lr_finder.lrs = [1e-5, 1e-4, 1e-3]
        lr_finder.losses = [2.0, 1.5, 1.2]

        # Skip too much data
        with patch("mammography.tuning.lr_finder.plt") as mock_plt:
            lr_finder.plot(skip_start=10, skip_end=10)
            # Should not create plot
            mock_plt.subplots.assert_not_called()


class TestLRFinderRangeTest:
    """Test range_test method."""

    def test_range_test_basic(self, lr_finder, dummy_dataloader):
        """Test basic range test execution."""
        suggested_lr = lr_finder.range_test(
            train_loader=dummy_dataloader,
            start_lr=1e-5,
            end_lr=1e-2,
            num_iter=10,
        )

        # Check results were recorded
        assert len(lr_finder.lrs) > 0
        assert len(lr_finder.losses) > 0
        assert len(lr_finder.lrs) == len(lr_finder.losses)
        assert suggested_lr is not None
        assert lr_finder.best_lr == suggested_lr

    def test_range_test_lr_progression(self, lr_finder, dummy_dataloader):
        """Test that learning rates increase exponentially."""
        lr_finder.range_test(
            train_loader=dummy_dataloader,
            start_lr=1e-5,
            end_lr=1e-2,
            num_iter=10,
        )

        # Check that LRs are increasing
        lrs = lr_finder.lrs
        for i in range(len(lrs) - 1):
            assert lrs[i + 1] > lrs[i]

        # Check first and last LR are close to expected
        assert lrs[0] == pytest.approx(1e-5, rel=0.1)
        # Last LR might not reach end_lr if stopped early

    def test_range_test_model_state_restored(
        self, dummy_model, dummy_optimizer, dummy_criterion, dummy_dataloader
    ):
        """Test that model and optimizer state are restored after range test."""
        device = torch.device("cpu")

        # Save initial state
        initial_model_state = {
            k: v.clone() for k, v in dummy_model.state_dict().items()
        }
        initial_optim_state = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in dummy_optimizer.state_dict().items()
        }

        lr_finder = LRFinder(
            model=dummy_model,
            optimizer=dummy_optimizer,
            criterion=dummy_criterion,
            device=device,
        )

        # Run range test
        lr_finder.range_test(
            train_loader=dummy_dataloader,
            start_lr=1e-5,
            end_lr=1e-2,
            num_iter=5,
        )

        # Check state is restored
        current_model_state = dummy_model.state_dict()
        for key in initial_model_state:
            assert torch.equal(initial_model_state[key], current_model_state[key])

    def test_range_test_divergence_stopping(self, lr_finder, dummy_dataloader):
        """Test that range test stops when loss diverges."""
        # Use very aggressive LR range to trigger divergence
        lr_finder.range_test(
            train_loader=dummy_dataloader,
            start_lr=1e-3,
            end_lr=10.0,
            num_iter=100,
            diverge_th=3.0,  # Low threshold to trigger early stopping
        )

        # Should stop before completing all iterations
        assert len(lr_finder.lrs) < 100

    def test_range_test_resets_results(self, lr_finder, dummy_dataloader):
        """Test that running range_test again resets previous results."""
        # First run
        lr_finder.range_test(
            train_loader=dummy_dataloader,
            start_lr=1e-5,
            end_lr=1e-2,
            num_iter=5,
        )
        first_lrs = lr_finder.lrs.copy()

        # Second run with different parameters
        lr_finder.range_test(
            train_loader=dummy_dataloader,
            start_lr=1e-6,
            end_lr=1e-1,
            num_iter=5,
        )
        second_lrs = lr_finder.lrs

        # Results should be different (reset)
        assert first_lrs != second_lrs

    def test_range_test_with_invalid_labels(self, lr_finder):
        """Test range test with invalid labels (negative)."""
        # Create dataloader with invalid labels
        x = torch.randn(16, 1, 10, 10)
        y = torch.full((16,), -1)  # All invalid labels
        dataset = TensorDataset(x, y, torch.zeros(16))
        dataloader = DataLoader(dataset, batch_size=8)

        # Should handle gracefully (skip invalid batches)
        suggested_lr = lr_finder.range_test(
            train_loader=dataloader,
            start_lr=1e-5,
            end_lr=1e-2,
            num_iter=5,
        )

        # May have limited or no results due to invalid labels
        assert suggested_lr is not None or len(lr_finder.lrs) == 0


class TestLRFinderEdgeCases:
    """Test edge cases and error handling."""

    def test_non_finite_loss_handling(self, lr_finder):
        """Test handling of NaN or Inf loss values."""
        # Create dataloader with data that might cause NaN
        x = torch.randn(16, 1, 10, 10)
        y = torch.randint(0, 4, (16,))
        dataset = TensorDataset(x, y, torch.zeros(16))
        dataloader = DataLoader(dataset, batch_size=8)

        # Mock the forward pass to return NaN loss
        with patch.object(lr_finder.model, "forward", return_value=torch.randn(8, 4)):
            with patch.object(
                lr_finder.criterion, "forward", return_value=torch.tensor(float("nan"))
            ):
                lr_finder.range_test(
                    train_loader=dataloader,
                    start_lr=1e-5,
                    end_lr=1e-2,
                    num_iter=10,
                )

                # Should stop early due to non-finite loss
                assert len(lr_finder.lrs) < 10

    def test_empty_dataloader(self, lr_finder):
        """Test behavior with empty dataloader."""
        # Create empty dataloader
        x = torch.randn(0, 1, 10, 10)
        y = torch.randint(0, 4, (0,))
        dataset = TensorDataset(x, y, torch.zeros(0))
        dataloader = DataLoader(dataset, batch_size=8)

        # Should handle empty dataloader gracefully
        with pytest.raises(Exception):  # StopIteration or similar
            lr_finder.range_test(
                train_loader=dataloader, start_lr=1e-5, end_lr=1e-2, num_iter=10
            )

    def test_single_batch_dataloader(self, lr_finder):
        """Test with dataloader containing single batch."""
        x = torch.randn(4, 1, 10, 10)
        y = torch.randint(0, 4, (4,))
        dataset = TensorDataset(x, y, torch.zeros(4))
        dataloader = DataLoader(dataset, batch_size=4)

        # Should handle by recycling the iterator
        suggested_lr = lr_finder.range_test(
            train_loader=dataloader, start_lr=1e-5, end_lr=1e-2, num_iter=5
        )

        assert suggested_lr is not None
        assert len(lr_finder.lrs) > 0
