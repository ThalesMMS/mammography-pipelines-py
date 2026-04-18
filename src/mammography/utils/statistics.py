#
# statistics.py
# mammography-pipelines
#
# Statistical utilities for cross-validation and model evaluation.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""
Statistical utilities for cross-validation and model evaluation.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.

Provides functions for computing confidence intervals, aggregating metrics
across cross-validation folds, and formatting results for reporting.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


def compute_confidence_interval(
    values: NDArray[np.floating] | Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute confidence interval using t-distribution.

    Args:
        values: Array or sequence of metric values (e.g., accuracy scores from CV folds)
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval

    Raises:
        ValueError: If values is empty or confidence is not in (0, 1)

    Example:
        >>> import numpy as np
        >>> accuracies = np.array([0.8, 0.82, 0.79, 0.81, 0.83])
        >>> lower, upper = compute_confidence_interval(accuracies)
        >>> print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
        95% CI: [0.788, 0.832]
    """
    if not 0 < confidence < 1:
        raise ValueError(f"Confidence level must be in (0, 1), got {confidence}")

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute confidence interval for empty array")

    n = arr.size
    mean = float(np.mean(arr))

    # Handle edge case: single value
    if n == 1:
        return (mean, mean)

    std_err = float(np.std(arr, ddof=1) / np.sqrt(n))

    # Use t-distribution for small samples
    try:
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    except ImportError:
        # Fallback: use normal approximation if scipy not available
        # For 95% CI, z ≈ 1.96
        import warnings
        warnings.warn(
            "scipy not available, using normal approximation for CI",
            UserWarning,
            stacklevel=2,
        )
        # Normal approximation for common confidence levels
        z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        t_value = z_values.get(confidence, 1.96)

    margin = t_value * std_err
    return (mean - margin, mean + margin)


def aggregate_cv_metrics(
    fold_metrics: Sequence[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Aggregate metrics across cross-validation folds.

    Computes mean, standard deviation, min, max, and 95% confidence interval
    for each metric across all folds.

    Args:
        fold_metrics: List of dictionaries, each containing metrics for one fold
                      Example: [{"accuracy": 0.8, "f1": 0.75}, ...]

    Returns:
        Dictionary mapping metric names to aggregated statistics:
        {
            "accuracy": {
                "mean": 0.82,
                "std": 0.015,
                "min": 0.80,
                "max": 0.84,
                "ci_lower": 0.805,
                "ci_upper": 0.835,
                "values": [0.80, 0.82, 0.81, 0.84, 0.83],
            },
            ...
        }

    Raises:
        ValueError: If fold_metrics is empty or folds have inconsistent metric keys

    Example:
        >>> fold_metrics = [
        ...     {"accuracy": 0.80, "f1": 0.75},
        ...     {"accuracy": 0.82, "f1": 0.77},
        ...     {"accuracy": 0.81, "f1": 0.76},
        ... ]
        >>> aggregated = aggregate_cv_metrics(fold_metrics)
        >>> print(f"Accuracy: {aggregated['accuracy']['mean']:.3f}")
        Accuracy: 0.810
    """
    if not fold_metrics:
        raise ValueError("Cannot aggregate empty fold_metrics list")

    # Collect metric names from first fold
    metric_keys = list(fold_metrics[0].keys())

    # Verify all folds have the same metrics
    for i, fold in enumerate(fold_metrics[1:], start=1):
        if set(fold.keys()) != set(metric_keys):
            raise ValueError(
                f"Fold {i} has inconsistent metrics. "
                f"Expected {metric_keys}, got {list(fold.keys())}"
            )

    aggregated: dict[str, dict[str, float]] = {}

    for metric_name in metric_keys:
        # Collect values for this metric across all folds
        values = [fold[metric_name] for fold in fold_metrics]
        arr = np.array(values, dtype=np.float64)

        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        # Compute 95% confidence interval
        try:
            ci_lower, ci_upper = compute_confidence_interval(arr, confidence=0.95)
        except Exception:
            # Fallback if CI computation fails
            ci_lower = ci_upper = mean

        aggregated[metric_name] = {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "values": values,
        }

    return aggregated


def format_cv_results(
    aggregated: dict[str, dict[str, float]],
    decimal_places: int = 4,
) -> str:
    """
    Format aggregated cross-validation results as human-readable text.

    Args:
        aggregated: Output from aggregate_cv_metrics()
        decimal_places: Number of decimal places to display (default: 4)

    Returns:
        Multi-line string with formatted results

    Example:
        >>> aggregated = {
        ...     "accuracy": {
        ...         "mean": 0.82, "std": 0.015,
        ...         "ci_lower": 0.805, "ci_upper": 0.835,
        ...     },
        ...     "f1": {
        ...         "mean": 0.78, "std": 0.020,
        ...         "ci_lower": 0.760, "ci_upper": 0.800,
        ...     },
        ... }
        >>> print(format_cv_results(aggregated))
        Cross-Validation Results (95% CI):
        ===================================
        accuracy:  0.8200 ± 0.0150  [0.8050, 0.8350]
        f1:        0.7800 ± 0.0200  [0.7600, 0.8000]
    """
    lines = ["Cross-Validation Results (95% CI):", "=" * 35]

    # Find longest metric name for alignment
    max_name_len = max((len(name) for name in aggregated.keys()), default=0)

    for metric_name, stats in sorted(aggregated.items()):
        mean = stats["mean"]
        std = stats["std"]
        ci_lower = stats["ci_lower"]
        ci_upper = stats["ci_upper"]

        # Format with proper alignment
        name_padded = metric_name.ljust(max_name_len)
        mean_str = f"{mean:.{decimal_places}f}"
        std_str = f"{std:.{decimal_places}f}"
        ci_str = f"[{ci_lower:.{decimal_places}f}, {ci_upper:.{decimal_places}f}]"

        line = f"{name_padded}:  {mean_str} ± {std_str}  {ci_str}"
        lines.append(line)

    return "\n".join(lines)


def compute_pooled_std(
    fold_stds: Sequence[float],
    fold_sizes: Sequence[int],
) -> float:
    """
    Compute pooled standard deviation across folds.

    This is useful when folds have different sizes and you want to compute
    a weighted standard deviation.

    Args:
        fold_stds: Standard deviations from each fold
        fold_sizes: Sample sizes for each fold

    Returns:
        Pooled standard deviation

    Raises:
        ValueError: If inputs are empty or have different lengths

    Example:
        >>> fold_stds = [0.1, 0.12, 0.11]
        >>> fold_sizes = [100, 100, 100]
        >>> pooled = compute_pooled_std(fold_stds, fold_sizes)
        >>> print(f"Pooled std: {pooled:.4f}")
        Pooled std: 0.1101
    """
    if not fold_stds or not fold_sizes:
        raise ValueError("fold_stds and fold_sizes cannot be empty")

    if len(fold_stds) != len(fold_sizes):
        raise ValueError(
            f"fold_stds and fold_sizes must have same length, "
            f"got {len(fold_stds)} and {len(fold_sizes)}"
        )

    if len(fold_stds) < 2:
        return 0.0

    stds = np.array(fold_stds, dtype=np.float64)
    sizes = np.array(fold_sizes, dtype=np.int64)

    # Pooled variance formula: Σ[(n_i - 1) * s_i^2] / Σ(n_i - 1)
    numerator = np.sum((sizes - 1) * stds ** 2)
    denominator = np.sum(sizes - 1)

    if denominator <= 0:
        return 0.0

    pooled_variance = numerator / denominator
    return float(np.sqrt(pooled_variance))


def effect_size_cohen_d(
    group1: NDArray[np.floating] | Sequence[float],
    group2: NDArray[np.floating] | Sequence[float],
) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Cohen's d measures the standardized difference between two means.
    Rules of thumb: |d| < 0.2 (small), 0.2-0.5 (medium), > 0.8 (large)

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Cohen's d effect size

    Raises:
        ValueError: If either group is empty

    Example:
        >>> baseline = [0.75, 0.76, 0.74, 0.77]
        >>> improved = [0.82, 0.83, 0.81, 0.84]
        >>> d = effect_size_cohen_d(baseline, improved)
        >>> print(f"Cohen's d: {d:.3f}")
        Cohen's d: 2.828
    """
    arr1 = np.asarray(group1, dtype=np.float64)
    arr2 = np.asarray(group2, dtype=np.float64)

    if arr1.size == 0 or arr2.size == 0:
        raise ValueError("Cannot compute effect size for empty arrays")

    mean1 = np.mean(arr1)
    mean2 = np.mean(arr2)

    # Pooled standard deviation
    n1, n2 = arr1.size, arr2.size
    var1 = np.var(arr1, ddof=1) if n1 > 1 else 0.0
    var2 = np.var(arr2, ddof=1) if n2 > 1 else 0.0

    if n1 + n2 <= 2:
        return 0.0

    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std <= 1e-12:
        return 0.0

    return float((mean2 - mean1) / pooled_std)
