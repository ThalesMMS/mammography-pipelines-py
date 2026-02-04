#
# cancer_trainer.py
# mammography-pipelines
#
# Training and evaluation utilities for breast cancer detection models, including metrics computation and history tracking.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Training and evaluation utilities for breast cancer detection models.

WARNING: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

This module provides a complete training pipeline for binary cancer classification,
including training loops, evaluation, metrics computation, and prediction collection.
It supports automatic mixed precision (AMP) training for improved performance on
compatible hardware.

Components:
    - DensityHistoryEntry: Dataclass for tracking per-epoch metrics
    - train_one_epoch: Single epoch training with gradient clipping
    - evaluate: Model evaluation with metrics computation
    - collect_predictions: Gather predictions for analysis
    - fit_classifier: Complete multi-epoch training loop
    - get_sens_spec: Compute sensitivity and specificity metrics

Example usage:
    >>> from mammography.training.cancer_trainer import fit_classifier, evaluate
    >>> from mammography.models.cancer_models import MammographyModel, resolve_device
    >>> import torch
    >>> import torch.nn as nn
    >>>
    >>> # Setup model and training components
    >>> device = resolve_device()
    >>> model = MammographyModel().to(device)
    >>> criterion = nn.BCELoss()
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    >>>
    >>> # Train for multiple epochs
    >>> history = fit_classifier(
    ...     model=model,
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     criterion=criterion,
    ...     optimizer=optimizer,
    ...     device=device,
    ...     num_epochs=10,
    ...     amp_enabled=True
    ... )
    >>>
    >>> # Evaluate on test set
    >>> test_loss, test_acc = evaluate(
    ...     model=model,
    ...     loader=test_loader,
    ...     criterion=criterion,
    ...     device=device
    ... )
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.amp.grad_scaler import GradScaler
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class DensityHistoryEntry:
    """Per-epoch snapshot to ease export and later plotting."""

    epoch: int
    train_loss: float
    train_acc: float
    val_loss: Optional[float]
    val_acc: Optional[float]


def get_sens_spec(y_true, y_pred):
    """Compute sensitivity and specificity from the confusion matrix."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity


def _prepare_targets(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert labels 1–4 to range 0–3 and return a mask of valid examples."""

    if labels.ndim != 1:
        labels = labels.view(-1)
    mask = labels > 0
    return (labels[mask] - 1).long(), mask


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    amp_enabled: bool = False,
) -> Tuple[float, float]:
    """Run one training epoch for binary cancer classification.

    Args:
        model: The neural network model to train.
        loader: DataLoader providing (images, labels) batches.
        criterion: Loss function (typically BCELoss or BCEWithLogitsLoss).
        optimizer: Optimizer for updating model parameters.
        device: Device to run training on (cpu/cuda/mps).
        scaler: Optional GradScaler for mixed precision training.
        amp_enabled: Whether to use automatic mixed precision.

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    # Convert device string to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)

    model.train()
    total_loss = 0.0
    all_preds: List[float] = []
    all_labels: List[float] = []

    use_amp = amp_enabled and device.type in {"cuda", "mps"}
    use_scaler = scaler is not None and device.type == "cuda"

    for images, labels, *_ in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Ensure labels are float for BCE loss
        if labels.dtype != torch.float32:
            labels = labels.float()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            # Handle both single output and batched outputs
            outputs = outputs.view(-1)
            labels_flat = labels.view(-1)
            loss = criterion(outputs, labels_flat)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * labels_flat.size(0)

        # Store binary predictions (threshold 0.5) for accuracy computation
        preds = outputs.detach().cpu().view(-1).numpy()
        preds_binary = (preds > 0.5).astype(float)
        all_preds.extend(preds_binary)
        all_labels.extend(labels_flat.cpu().numpy())

    if len(all_labels) == 0:
        return 0.0, 0.0

    avg_loss = total_loss / len(all_labels)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return avg_loss, float(accuracy)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool = False,
) -> Tuple[float, float]:
    """Evaluate the model on validation/test data for binary cancer classification.

    Args:
        model: The neural network model to evaluate.
        loader: DataLoader providing (images, labels) batches.
        criterion: Loss function (typically BCELoss or BCEWithLogitsLoss).
        device: Device to run evaluation on (cpu/cuda/mps).
        amp_enabled: Whether to use automatic mixed precision.

    Returns:
        Tuple of (average_loss, accuracy) for the evaluation.
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[float] = []
    all_labels: List[float] = []

    use_amp = amp_enabled and device.type in {"cuda", "mps"}

    for images, labels, *_ in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Ensure labels are float for BCE loss
        if labels.dtype != torch.float32:
            labels = labels.float()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            # Handle both single output and batched outputs
            outputs = outputs.view(-1)
            labels_flat = labels.view(-1)
            loss = criterion(outputs, labels_flat)

        total_loss += loss.item() * labels_flat.size(0)

        # Store binary predictions (threshold 0.5) for accuracy computation
        preds = outputs.detach().cpu().view(-1).numpy()
        preds_binary = (preds > 0.5).astype(float)
        all_preds.extend(preds_binary)
        all_labels.extend(labels_flat.cpu().numpy())

    if len(all_labels) == 0:
        return 0.0, 0.0

    avg_loss = total_loss / len(all_labels)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return avg_loss, float(accuracy)


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
) -> Dict[str, Any]:
    """Collect predictions and labels from the model on a dataset.

    Args:
        model: The neural network model to evaluate.
        loader: DataLoader providing (images, labels) batches.
        device: Device to run inference on (cpu/cuda/mps).
        amp_enabled: Whether to use automatic mixed precision.

    Returns:
        Dictionary containing:
            - 'predictions': numpy array of predicted probabilities
            - 'labels': numpy array of ground truth labels
            - 'binary_predictions': numpy array of binary predictions (threshold 0.5)
    """
    model.eval()
    all_preds: List[float] = []
    all_labels: List[float] = []

    use_amp = amp_enabled and device.type in {"cuda", "mps"}

    for images, labels, *_ in tqdm(loader, desc="Collecting predictions", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Ensure labels are float for consistency
        if labels.dtype != torch.float32:
            labels = labels.float()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            outputs = outputs.view(-1)

        # Store raw predictions (probabilities)
        preds = outputs.detach().cpu().view(-1).numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.view(-1).cpu().numpy())

    predictions_array = np.array(all_preds)
    labels_array = np.array(all_labels)
    binary_preds = (predictions_array > 0.5).astype(float)

    return {
        "predictions": predictions_array,
        "labels": labels_array,
        "binary_predictions": binary_preds,
    }


def fit_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    num_epochs: Optional[int] = None,
    scaler: Optional[GradScaler] = None,
    amp_enabled: bool = False,
    # Backward compatibility parameters
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
) -> List[DensityHistoryEntry]:
    """Fit a binary classifier over multiple epochs with validation.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function (typically BCELoss or BCEWithLogitsLoss).
        optimizer: Optimizer for updating model parameters.
        device: Device to run training on (cpu/cuda/mps).
        num_epochs: Number of epochs to train (or use 'epochs' for backward compatibility).
        scaler: Optional GradScaler for mixed precision training.
        amp_enabled: Whether to use automatic mixed precision.
        epochs: Alias for num_epochs (backward compatibility).
        lr: Learning rate for creating default optimizer (backward compatibility).

    Returns:
        List of DensityHistoryEntry objects tracking metrics per epoch.
    """
    # Handle backward compatibility
    if epochs is not None and num_epochs is None:
        num_epochs = epochs

    if device is None:
        device = torch.device("cpu")

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    if optimizer is None:
        if lr is None:
            lr = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if num_epochs is None:
        num_epochs = 10

    history: List[DensityHistoryEntry] = []

    for epoch in range(1, num_epochs + 1):
        # Training phase
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )

        # Validation phase
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        # Record history
        entry = DensityHistoryEntry(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )
        history.append(entry)

        # Progress reporting
        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

    return history
