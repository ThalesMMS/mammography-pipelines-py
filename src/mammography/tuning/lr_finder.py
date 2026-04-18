#
# lr_finder.py
# mammography-pipelines
#
# Learning rate range test utility for automatic learning rate discovery using exponential search.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""Learning rate finder implementation using range test method for optimal LR discovery."""
import logging
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class LRFinder:
    """
    Learning rate range test for automatic learning rate discovery.

    Implements the LR range test method: trains for a few iterations while
    exponentially increasing the learning rate from a very small value to a large value.
    Records the loss at each step to identify the optimal learning rate range.

    The suggested learning rate is typically:
    - The learning rate with steepest loss decrease (fastest learning)
    - OR 1/10th of the learning rate at minimum loss (conservative approach)

    Usage:
        lr_finder = LRFinder(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=device
        )
        suggested_lr = lr_finder.range_test(
            train_loader=train_loader,
            start_lr=1e-7,
            end_lr=1.0,
            num_iter=100
        )
        lr_finder.plot(save_path="outputs/lr_finder.png")

    Reference:
        Leslie N. Smith, "Cyclical Learning Rates for Training Neural Networks"
        https://arxiv.org/abs/1506.01186
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        amp_enabled: bool = False,
    ):
        """
        Initialize LR Finder with model and training components.

        Args:
            model: Neural network model to train
            optimizer: Optimizer instance (will be modified during range test)
            criterion: Loss function (e.g., nn.CrossEntropyLoss())
            device: Device for training (cuda/mps/cpu)
            amp_enabled: Whether to use automatic mixed precision
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.amp_enabled = amp_enabled
        self.logger = logging.getLogger("mammography.lr_finder")

        # Store initial state to restore after range test
        self.initial_state = {
            "model": self._copy_state_dict(model.state_dict()),
            "optimizer": self._copy_state_dict(optimizer.state_dict()),
        }

        # Results storage
        self.lrs: List[float] = []
        self.losses: List[float] = []
        self.best_lr: Optional[float] = None

    def _copy_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy state dict to avoid reference issues."""
        return {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in state_dict.items()}

    def range_test(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 1.0,
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_th: float = 5.0,
    ) -> float:
        """
        Perform learning rate range test.

        Trains the model for num_iter iterations, exponentially increasing
        the learning rate from start_lr to end_lr. Stops early if loss
        diverges beyond diverge_th times the minimum loss.

        Args:
            train_loader: DataLoader for training data
            start_lr: Starting learning rate (default: 1e-7)
            end_lr: Ending learning rate (default: 1.0)
            num_iter: Number of iterations for range test (default: 100)
            smooth_f: Loss smoothing factor for exponential moving average (default: 0.05)
            diverge_th: Stop if loss > diverge_th * min_loss (default: 5.0)

        Returns:
            Suggested learning rate (float)
        """
        self.logger.info(
            "Starting LR range test: lr=[%.2e, %.2e], num_iter=%d",
            start_lr, end_lr, num_iter
        )

        # Reset results
        self.lrs = []
        self.losses = []
        self.best_lr = None

        # Put model in training mode
        self.model.train()

        # Calculate learning rate multiplier per step
        lr_mult = (end_lr / start_lr) ** (1.0 / num_iter)
        current_lr = start_lr

        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        # Initialize scaler for AMP if enabled
        scaler = GradScaler() if self.amp_enabled and self.device.type == "cuda" else None

        # Track best loss and smoothed loss for divergence detection
        best_loss = float("inf")
        smoothed_loss = None

        # Iterate through data
        data_iter = iter(train_loader)
        pbar = tqdm(range(num_iter), desc="LR Range Test", leave=True)

        for iteration in pbar:
            # Get next batch
            try:
                batch = next(data_iter)
            except StopIteration:
                # Reset iterator if we run out of data
                data_iter = iter(train_loader)
                batch = next(data_iter)

            if batch is None:
                continue

            # Unpack batch (handle optional extra features)
            if len(batch) == 4:
                x, y, _, extra_features = batch
            else:
                x, y, _ = batch
                extra_features = None

            # Move to device
            x = x.to(device=self.device, non_blocking=True, memory_format=torch.channels_last)
            if isinstance(y, torch.Tensor):
                y = y.to(device=self.device, dtype=torch.long)
            else:
                y = torch.as_tensor(y, dtype=torch.long, device=self.device)

            # Filter out invalid labels
            mask = y >= 0
            if not mask.any():
                continue

            x = x[mask]
            y = y[mask]

            extra_tensor = None
            if extra_features is not None:
                extra_tensor = extra_features.to(device=self.device, non_blocking=True)
                extra_tensor = extra_tensor[mask]

            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)

            try:
                with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    logits = self.model(x, extra_tensor)
                    loss = self.criterion(logits, y)

                # Check for non-finite loss
                if not torch.isfinite(loss):
                    self.logger.warning("Non-finite loss detected at lr=%.2e, stopping", current_lr)
                    break

                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.warning("OOM at lr=%.2e, stopping", current_lr)
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    break
                raise

            # Record learning rate and loss
            loss_val = loss.item()
            self.lrs.append(current_lr)
            self.losses.append(loss_val)

            # Compute smoothed loss for divergence detection
            if smoothed_loss is None:
                smoothed_loss = loss_val
            else:
                smoothed_loss = smooth_f * loss_val + (1 - smooth_f) * smoothed_loss

            # Update best loss
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # If an aggressive sweep drives the loss close to zero, the
            # remaining high-LR points are not useful for choosing a stable LR.
            if (
                len(self.losses) >= 10
                and current_lr > start_lr * 1000
                and loss_val <= 0.01 * max(1.0, self.losses[0])
            ):
                self.logger.info("Loss saturated at lr=%.2e, stopping early", current_lr)
                break

            # Check for divergence
            if smoothed_loss > diverge_th * best_loss:
                self.logger.info("Loss diverging at lr=%.2e, stopping early", current_lr)
                break

            # Update progress bar
            pbar.set_postfix({"lr": f"{current_lr:.2e}", "loss": f"{loss_val:.4f}"})

            # Increase learning rate for next iteration
            current_lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

        # Restore initial model and optimizer state
        self.model.load_state_dict(self.initial_state["model"])
        self.optimizer.load_state_dict(self.initial_state["optimizer"])

        if not self.losses:
            self.best_lr = max(start_lr, 1e-7)
            self.logger.warning(
                "LR range test produced no valid losses; using fallback LR %.2e",
                self.best_lr,
            )
            return self.best_lr

        # Suggest learning rate
        self.best_lr = self._suggest_lr()

        self.logger.info("LR range test complete. Suggested LR: %.2e", self.best_lr)
        return self.best_lr

    def _suggest_lr(self) -> float:
        """
        Suggest optimal learning rate from range test results.

        Uses the steepest gradient method: finds the learning rate where
        the loss is decreasing most rapidly. Falls back to 1/10th of the
        LR at minimum loss if gradient method fails.

        Returns:
            Suggested learning rate (float)
        """
        if len(self.losses) < 10:
            self.logger.warning("Insufficient data points for LR suggestion, using min loss approach")
            min_idx = np.argmin(self.losses)
            # Use 1/10th of LR at minimum loss (conservative)
            suggested_lr = self.lrs[min_idx] / 10.0
            return max(suggested_lr, 1e-7)  # Ensure minimum LR

        # Compute loss gradient (derivative)
        losses_np = np.array(self.losses)
        lrs_np = np.array(self.lrs)

        # Use log scale for learning rates
        log_lrs = np.log10(lrs_np)

        # Smooth losses with moving average to reduce noise
        window = min(10, len(losses_np) // 5)
        smoothed_losses = np.convolve(
            losses_np,
            np.ones(window) / window,
            mode='valid'
        )

        # Compute gradient on smoothed losses
        gradients = np.gradient(smoothed_losses)

        # Find steepest negative gradient (fastest learning)
        # Ignore first 10% and last 10% to avoid edge effects
        start_idx = len(gradients) // 10
        end_idx = len(gradients) - len(gradients) // 10

        if start_idx >= end_idx:
            # Fallback if range too small
            min_idx = np.argmin(losses_np)
            suggested_lr = lrs_np[min_idx] / 10.0
            return max(suggested_lr, 1e-7)

        search_gradients = gradients[start_idx:end_idx]
        steepest_idx = start_idx + np.argmin(search_gradients)

        # Map back to original LR array accounting for smoothing offset
        offset = window // 2
        original_idx = min(steepest_idx + offset, len(lrs_np) - 1)
        suggested_lr = lrs_np[original_idx]

        self.logger.debug("Steepest gradient at idx=%d, lr=%.2e", original_idx, suggested_lr)
        return suggested_lr

    def plot(
        self,
        save_path: Optional[Path] = None,
        skip_start: int = 10,
        skip_end: int = 5,
        log_scale: bool = True,
    ) -> None:
        """
        Plot learning rate vs loss curve.

        Args:
            save_path: Path to save plot (if None, not saved)
            skip_start: Skip first N points (noisy initialization)
            skip_end: Skip last N points (diverging loss)
            log_scale: Use log scale for learning rate axis
        """
        if not self.lrs or not self.losses:
            self.logger.warning("No data to plot, run range_test() first")
            return

        # Prepare data
        lrs = self.lrs[skip_start : len(self.lrs) - skip_end]
        losses = self.losses[skip_start : len(self.losses) - skip_end]

        if not lrs:
            self.logger.warning("Not enough data points after skipping start/end")
            return

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lrs, losses, linewidth=2, label="Loss")

        # Mark suggested LR
        if self.best_lr is not None:
            ax.axvline(
                x=self.best_lr,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Suggested LR: {self.best_lr:.2e}"
            )

        # Formatting
        if log_scale:
            ax.set_xscale("log")
        ax.set_xlabel("Learning Rate", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Learning Rate Finder", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            self.logger.info("LR finder plot saved to %s", save_path)

        plt.close(fig)

    def get_results(self) -> Dict[str, Any]:
        """
        Get range test results as dictionary.

        Returns:
            Dictionary with keys: lrs, losses, suggested_lr
        """
        return {
            "lrs": self.lrs,
            "losses": self.losses,
            "suggested_lr": self.best_lr,
        }
