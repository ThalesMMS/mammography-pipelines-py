#
# normalization.py
# mammography-pipelines
#
# Utilities for computing and validating normalization statistics from datasets.
# Supports automatic detection of unnormalized data and computation of mean/std.
#
# Thales Matheus Mendonça Santos - January 2026
#
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class NormalizationStats:
    """
    Container for normalization statistics.

    Educational Note: Normalization statistics are crucial for consistent
    preprocessing across training and inference. These stats ensure that
    input data has the expected distribution for the model.

    Attributes:
        mean: Per-channel mean values (typically 3 values for RGB)
        std: Per-channel standard deviation values (typically 3 values for RGB)
        method: Method used to compute statistics ('auto', 'imagenet', 'custom')
        sample_size: Number of samples used to compute statistics (None for preset values)
    """
    mean: List[float]
    std: List[float]
    method: str = "auto"
    sample_size: Optional[int] = None

    def __post_init__(self):
        """Validate normalization statistics."""
        if len(self.mean) != len(self.std):
            raise ValueError(f"mean and std must have same length, got {len(self.mean)} vs {len(self.std)}")
        if len(self.mean) not in [1, 3]:
            raise ValueError(f"mean/std must have 1 or 3 values, got {len(self.mean)}")
        if any(s <= 0 for s in self.std):
            raise ValueError(f"std values must be positive, got {self.std}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationStats":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def imagenet_defaults(cls) -> "NormalizationStats":
        """Return ImageNet normalization statistics."""
        return cls(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            method="imagenet",
            sample_size=None
        )


def compute_normalization_stats(
    dataset: Union[Dataset, DataLoader],
    num_samples: Optional[int] = 1000,
    num_workers: int = 0,
    batch_size: int = 32,
) -> NormalizationStats:
    """
    Compute normalization statistics (mean and std) from dataset samples.

    Educational Note: Computing normalization statistics from your dataset
    ensures that the preprocessing matches your data distribution. This is
    especially important when working with medical images that may have
    different intensity ranges than natural images.

    Args:
        dataset: PyTorch Dataset or DataLoader to compute statistics from
        num_samples: Maximum number of samples to use (None = use all)
        num_workers: Number of data loading workers
        batch_size: Batch size for data loading

    Returns:
        NormalizationStats: Computed statistics with mean, std, and metadata

    Raises:
        ValueError: If dataset is empty or invalid
        RuntimeError: If computation fails

    Example:
        >>> from mammography.data.dataset import MammoDensityDataset
        >>> dataset = MammoDensityDataset(...)
        >>> stats = compute_normalization_stats(dataset, num_samples=500)
        >>> print(f"Mean: {stats.mean}, Std: {stats.std}")
    """
    try:
        # Handle DataLoader vs Dataset
        if isinstance(dataset, DataLoader):
            dataloader = dataset
            actual_sample_size = len(dataset.dataset) if hasattr(dataset, 'dataset') else None
        else:
            # Create DataLoader from Dataset
            if len(dataset) == 0:
                raise ValueError("Dataset is empty, cannot compute normalization statistics")

            # Limit samples if requested
            if num_samples is not None and num_samples < len(dataset):
                indices = torch.randperm(len(dataset))[:num_samples].tolist()
                subset = torch.utils.data.Subset(dataset, indices)
                actual_sample_size = num_samples
            else:
                subset = dataset
                actual_sample_size = len(dataset)

            # Create a robust collate function that handles None values
            # This avoids circular import and works with any dataset structure
            def simple_robust_collate(batch):
                """Filter out None values and handle tuples with optional None elements."""
                # Filter out top-level None items
                batch = [item for item in batch if item is not None]
                if not batch:
                    return None

                # Check if batch contains tuples (complex dataset) or simple tensors
                if isinstance(batch[0], tuple):
                    # Complex dataset - handle tuples manually to deal with None values inside them
                    # Support different tuple lengths: (image, label) or (image, label, metadata, embedding)
                    tuple_len = len(batch[0])

                    if tuple_len == 2:
                        # Simple tuple format: (image, label)
                        images = torch.stack([b[0] for b in batch], dim=0)
                        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
                        return images, labels

                    elif tuple_len >= 3:
                        # Full format: (image, label, metadata, optional_embedding)
                        images = torch.stack([b[0] for b in batch], dim=0)
                        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
                        metadata = [b[2] for b in batch]

                        # Handle optional embeddings
                        embeddings = None
                        if tuple_len > 3:
                            emb_list = [b[3] for b in batch if b[3] is not None]
                            if len(emb_list) == len(batch):
                                embeddings = torch.stack(emb_list, dim=0)

                        if embeddings is not None:
                            return images, labels, metadata, embeddings
                        else:
                            return images, labels, metadata
                else:
                    # Simple dataset (just tensors) - use default_collate
                    return torch.utils.data.dataloader.default_collate(batch)

            dataloader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                collate_fn=simple_robust_collate,  # Handle None values gracefully
            )

        logger.info(f"Computing normalization statistics from {actual_sample_size or 'all'} samples...")

        # Accumulate statistics
        channel_sum = None
        channel_sum_sq = None
        num_pixels = 0
        num_batches = 0

        for batch_data in dataloader:
            # Handle different batch formats (tuple, dict, or tensor)
            if isinstance(batch_data, (tuple, list)):
                images = batch_data[0]
            elif isinstance(batch_data, dict):
                images = batch_data.get('image', batch_data.get('data'))
            else:
                images = batch_data

            # Convert to float tensor if needed
            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images)
            images = images.float()

            # Ensure NCHW format (batch, channels, height, width)
            if images.ndim == 3:
                images = images.unsqueeze(1)  # Add channel dimension
            elif images.ndim != 4:
                logger.warning(f"Unexpected tensor shape: {images.shape}, skipping batch")
                continue

            batch_size_actual = images.shape[0]
            num_channels = images.shape[1]

            # Initialize accumulators on first batch
            if channel_sum is None:
                channel_sum = torch.zeros(num_channels)
                channel_sum_sq = torch.zeros(num_channels)

            # Compute per-channel statistics
            # Sum over batch, height, width dimensions
            for c in range(num_channels):
                channel_data = images[:, c, :, :]
                channel_sum[c] += channel_data.sum()
                channel_sum_sq[c] += (channel_data ** 2).sum()

            num_pixels += batch_size_actual * images.shape[2] * images.shape[3]
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("No valid batches processed, cannot compute statistics")

        # Compute final mean and std
        mean = channel_sum / num_pixels
        var = (channel_sum_sq / num_pixels) - (mean ** 2)
        std = torch.sqrt(torch.clamp(var, min=1e-8))  # Clamp to avoid sqrt of negative

        # Convert to lists
        mean_list = mean.tolist()
        std_list = std.tolist()

        logger.info(f"Computed normalization stats: mean={mean_list}, std={std_list}")

        return NormalizationStats(
            mean=mean_list,
            std=std_list,
            method="auto",
            sample_size=actual_sample_size
        )

    except Exception as e:
        logger.error(f"Failed to compute normalization statistics: {e}")
        raise


def validate_normalization(
    data: Union[torch.Tensor, np.ndarray],
    expected_mean: Optional[List[float]] = None,
    expected_std: Optional[List[float]] = None,
    tolerance: float = 0.1,
) -> Dict[str, Any]:
    """
    Validate that data appears to be properly normalized.

    Educational Note: This function helps detect when input data is not
    normalized as expected, which can cause training failures (NaN losses)
    or poor model performance. It compares actual statistics with expected
    values and warns if they differ significantly.

    Args:
        data: Tensor or array to validate (NCHW format expected)
        expected_mean: Expected per-channel mean (defaults to [0.485, 0.456, 0.406])
        expected_std: Expected per-channel std (defaults to [0.229, 0.224, 0.225])
        tolerance: Allowed relative deviation from expected values

    Returns:
        Dict with validation results:
            - is_normalized: bool indicating if data appears normalized
            - actual_mean: Computed mean values
            - actual_std: Computed std values
            - expected_mean: Expected mean values
            - expected_std: Expected std values
            - warnings: List of warning messages

    Example:
        >>> import torch
        >>> data = torch.randn(32, 3, 224, 224)
        >>> result = validate_normalization(data)
        >>> if not result['is_normalized']:
        ...     print(f"Warnings: {result['warnings']}")
    """
    # Convert to tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    data = data.float()

    # Ensure NCHW format
    if data.ndim == 3:
        data = data.unsqueeze(0)
    elif data.ndim != 4:
        raise ValueError(f"Expected 4D tensor (NCHW), got shape {data.shape}")

    num_channels = data.shape[1]

    # Set defaults if not provided
    if expected_mean is None:
        expected_mean = [0.485, 0.456, 0.406] if num_channels == 3 else [0.5]
    if expected_std is None:
        expected_std = [0.229, 0.224, 0.225] if num_channels == 3 else [0.5]

    # Validate input lengths
    if len(expected_mean) != num_channels:
        raise ValueError(f"expected_mean length {len(expected_mean)} != channels {num_channels}")
    if len(expected_std) != num_channels:
        raise ValueError(f"expected_std length {len(expected_std)} != channels {num_channels}")

    # Compute actual statistics
    actual_mean = []
    actual_std = []
    for c in range(num_channels):
        channel_data = data[:, c, :, :]
        actual_mean.append(float(channel_data.mean()))
        actual_std.append(float(channel_data.std()))

    # Check for warnings
    warnings = []
    is_normalized = True

    for c in range(num_channels):
        # Check mean deviation
        mean_deviation = abs(actual_mean[c] - expected_mean[c])
        mean_threshold = abs(expected_mean[c]) * tolerance if expected_mean[c] != 0 else tolerance
        if mean_deviation > mean_threshold:
            warnings.append(
                f"Channel {c}: mean={actual_mean[c]:.4f} deviates from expected={expected_mean[c]:.4f} "
                f"by {mean_deviation:.4f} (threshold={mean_threshold:.4f})"
            )
            is_normalized = False

        # Check std deviation
        std_deviation = abs(actual_std[c] - expected_std[c])
        std_threshold = abs(expected_std[c]) * tolerance if expected_std[c] != 0 else tolerance
        if std_deviation > std_threshold:
            warnings.append(
                f"Channel {c}: std={actual_std[c]:.4f} deviates from expected={expected_std[c]:.4f} "
                f"by {std_deviation:.4f} (threshold={std_threshold:.4f})"
            )
            is_normalized = False

    # Log warnings if data appears unnormalized
    if not is_normalized:
        logger.warning(f"Data appears to be unnormalized or incorrectly normalized:")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    return {
        "is_normalized": is_normalized,
        "actual_mean": actual_mean,
        "actual_std": actual_std,
        "expected_mean": expected_mean,
        "expected_std": expected_std,
        "warnings": warnings,
    }


def z_score_normalize(
    data: Union[torch.Tensor, np.ndarray],
    mean: Optional[Union[float, List[float]]] = None,
    std: Optional[Union[float, List[float]]] = None,
    eps: float = 1e-8,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply z-score normalization (standardization) to data.

    Z-score normalization transforms data to have zero mean and unit variance:
        normalized = (data - mean) / std

    Educational Note: Z-score normalization is a common preprocessing step
    that centers the data around zero and scales it to unit variance. This
    helps neural networks converge faster and more stably during training.

    Args:
        data: Input data (tensor or numpy array)
        mean: Mean value(s) for normalization (if None, computed from data)
        std: Standard deviation value(s) for normalization (if None, computed from data)
        eps: Small epsilon value to avoid division by zero

    Returns:
        Normalized data (same type as input)

    Example:
        >>> import torch
        >>> data = torch.randn(100, 3, 224, 224) * 50 + 100
        >>> normalized = z_score_normalize(data)
        >>> print(f"Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
        Mean: 0.0000, Std: 1.0000
    """
    is_numpy = isinstance(data, np.ndarray)

    # Convert to tensor if needed
    if is_numpy:
        data_tensor = torch.from_numpy(data).float()
    else:
        data_tensor = data.float()

    # Compute mean and std if not provided
    if mean is None:
        mean = data_tensor.mean()
    if std is None:
        std = data_tensor.std()

    # Convert scalar mean/std to tensor
    if isinstance(mean, (int, float)):
        mean = torch.tensor(mean, dtype=data_tensor.dtype, device=data_tensor.device)
    elif isinstance(mean, list):
        mean = torch.tensor(mean, dtype=data_tensor.dtype, device=data_tensor.device)
    elif isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).to(dtype=data_tensor.dtype, device=data_tensor.device)

    if isinstance(std, (int, float)):
        std = torch.tensor(std, dtype=data_tensor.dtype, device=data_tensor.device)
    elif isinstance(std, list):
        std = torch.tensor(std, dtype=data_tensor.dtype, device=data_tensor.device)
    elif isinstance(std, np.ndarray):
        std = torch.from_numpy(std).to(dtype=data_tensor.dtype, device=data_tensor.device)

    # Clamp std to avoid division by zero
    std = torch.clamp(std, min=eps)

    # Apply normalization
    normalized = (data_tensor - mean) / std

    # Convert back to numpy if input was numpy
    if is_numpy:
        return normalized.numpy()
    return normalized
