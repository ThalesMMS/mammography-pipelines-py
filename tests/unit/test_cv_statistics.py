from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

np = pytest.importorskip("numpy")

from mammography.utils.statistics import (
    aggregate_cv_metrics,
    compute_confidence_interval,
    compute_pooled_std,
    effect_size_cohen_d,
    format_cv_results,
)


# ============================================================================
# Tests for compute_confidence_interval
# ============================================================================


def test_compute_confidence_interval_basic() -> None:
    """Test basic confidence interval computation with known data."""
    values = np.array([0.8, 0.82, 0.79, 0.81, 0.83])
    lower, upper = compute_confidence_interval(values, confidence=0.95)

    # Check that CI is a tuple of two floats
    assert isinstance(lower, float)
    assert isinstance(upper, float)

    # Check that bounds are reasonable
    assert lower < upper
    assert lower < np.mean(values) < upper

    # Check that mean is at the center
    mean = np.mean(values)
    assert abs((lower + upper) / 2 - mean) < 1e-10


def test_compute_confidence_interval_single_value() -> None:
    """Test CI with single value - should return (value, value)."""
    values = np.array([0.85])
    lower, upper = compute_confidence_interval(values)

    assert lower == 0.85
    assert upper == 0.85


def test_compute_confidence_interval_two_values() -> None:
    """Test CI with two values - should work with t-distribution."""
    values = np.array([0.8, 0.9])
    lower, upper = compute_confidence_interval(values, confidence=0.95)

    assert lower < 0.85 < upper
    assert lower < upper


def test_compute_confidence_interval_list_input() -> None:
    """Test CI with list input (not numpy array)."""
    values = [0.75, 0.78, 0.76, 0.77, 0.79]
    lower, upper = compute_confidence_interval(values)

    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert lower < upper


def test_compute_confidence_interval_different_confidence_levels() -> None:
    """Test that different confidence levels produce different intervals."""
    values = np.array([0.8, 0.82, 0.79, 0.81, 0.83])

    lower_90, upper_90 = compute_confidence_interval(values, confidence=0.90)
    lower_95, upper_95 = compute_confidence_interval(values, confidence=0.95)
    lower_99, upper_99 = compute_confidence_interval(values, confidence=0.99)

    # 99% CI should be wider than 95% CI, which should be wider than 90% CI
    width_90 = upper_90 - lower_90
    width_95 = upper_95 - lower_95
    width_99 = upper_99 - lower_99

    assert width_90 < width_95 < width_99


def test_compute_confidence_interval_empty_array() -> None:
    """Test that empty array raises ValueError."""
    with pytest.raises(ValueError, match="empty array"):
        compute_confidence_interval(np.array([]))


def test_compute_confidence_interval_invalid_confidence() -> None:
    """Test that invalid confidence level raises ValueError."""
    values = np.array([0.8, 0.82, 0.79])

    with pytest.raises(ValueError, match="Confidence level must be in"):
        compute_confidence_interval(values, confidence=0.0)

    with pytest.raises(ValueError, match="Confidence level must be in"):
        compute_confidence_interval(values, confidence=1.0)

    with pytest.raises(ValueError, match="Confidence level must be in"):
        compute_confidence_interval(values, confidence=1.5)


def test_compute_confidence_interval_identical_values() -> None:
    """Test CI with identical values - should have zero width."""
    values = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
    lower, upper = compute_confidence_interval(values)

    # With identical values, std is 0, so CI should collapse to the mean
    assert lower == 0.8
    assert upper == 0.8


# ============================================================================
# Tests for aggregate_cv_metrics
# ============================================================================


def test_aggregate_cv_metrics_basic() -> None:
    """Test basic aggregation of CV metrics - main verification test."""
    fold_metrics = [
        {"accuracy": 0.80, "f1": 0.75},
        {"accuracy": 0.82, "f1": 0.77},
        {"accuracy": 0.81, "f1": 0.76},
    ]

    aggregated = aggregate_cv_metrics(fold_metrics)

    # Check that all metrics are present
    assert "accuracy" in aggregated
    assert "f1" in aggregated

    # Check structure of aggregated metrics
    for metric_name in ["accuracy", "f1"]:
        stats = aggregated[metric_name]
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "ci_lower" in stats
        assert "ci_upper" in stats
        assert "values" in stats

    # Check accuracy aggregation
    acc_stats = aggregated["accuracy"]
    assert acc_stats["mean"] == pytest.approx(0.81, abs=1e-6)
    assert acc_stats["min"] == 0.80
    assert acc_stats["max"] == 0.82
    assert acc_stats["values"] == [0.80, 0.82, 0.81]

    # Check that CI bounds are reasonable
    assert acc_stats["ci_lower"] < acc_stats["mean"] < acc_stats["ci_upper"]


def test_aggregate_cv_metrics_single_fold() -> None:
    """Test aggregation with single fold - std should be 0."""
    fold_metrics = [{"accuracy": 0.85, "kappa": 0.72}]

    aggregated = aggregate_cv_metrics(fold_metrics)

    acc_stats = aggregated["accuracy"]
    assert acc_stats["mean"] == 0.85
    assert acc_stats["std"] == 0.0
    assert acc_stats["min"] == 0.85
    assert acc_stats["max"] == 0.85
    # CI should collapse to the mean for single value
    assert acc_stats["ci_lower"] == 0.85
    assert acc_stats["ci_upper"] == 0.85


def test_aggregate_cv_metrics_many_folds() -> None:
    """Test aggregation with many folds (5-fold CV)."""
    fold_metrics = [
        {"accuracy": 0.80, "f1": 0.75, "auc": 0.88},
        {"accuracy": 0.82, "f1": 0.77, "auc": 0.89},
        {"accuracy": 0.81, "f1": 0.76, "auc": 0.87},
        {"accuracy": 0.84, "f1": 0.79, "auc": 0.90},
        {"accuracy": 0.83, "f1": 0.78, "auc": 0.88},
    ]

    aggregated = aggregate_cv_metrics(fold_metrics)

    # Check all metrics present
    assert set(aggregated.keys()) == {"accuracy", "f1", "auc"}

    # Check accuracy
    acc_stats = aggregated["accuracy"]
    expected_mean = (0.80 + 0.82 + 0.81 + 0.84 + 0.83) / 5
    assert acc_stats["mean"] == pytest.approx(expected_mean, abs=1e-6)
    assert acc_stats["std"] > 0  # Should have non-zero variance
    assert acc_stats["min"] == 0.80
    assert acc_stats["max"] == 0.84


def test_aggregate_cv_metrics_empty_list() -> None:
    """Test that empty fold_metrics raises ValueError."""
    with pytest.raises(ValueError, match="empty fold_metrics"):
        aggregate_cv_metrics([])


def test_aggregate_cv_metrics_inconsistent_keys() -> None:
    """Test that inconsistent metric keys across folds raises ValueError."""
    fold_metrics = [
        {"accuracy": 0.80, "f1": 0.75},
        {"accuracy": 0.82, "precision": 0.77},  # Different metric key
    ]

    with pytest.raises(ValueError, match="inconsistent metrics"):
        aggregate_cv_metrics(fold_metrics)


def test_aggregate_cv_metrics_values_preserved() -> None:
    """Test that original values are preserved in aggregated results."""
    fold_metrics = [
        {"accuracy": 0.80},
        {"accuracy": 0.82},
        {"accuracy": 0.81},
    ]

    aggregated = aggregate_cv_metrics(fold_metrics)

    assert aggregated["accuracy"]["values"] == [0.80, 0.82, 0.81]


# ============================================================================
# Tests for format_cv_results
# ============================================================================


def test_format_cv_results_basic() -> None:
    """Test basic formatting of CV results."""
    aggregated = {
        "accuracy": {
            "mean": 0.82,
            "std": 0.015,
            "ci_lower": 0.805,
            "ci_upper": 0.835,
        },
        "f1": {
            "mean": 0.78,
            "std": 0.020,
            "ci_lower": 0.760,
            "ci_upper": 0.800,
        },
    }

    formatted = format_cv_results(aggregated, decimal_places=4)

    # Check that output is a string
    assert isinstance(formatted, str)

    # Check that it contains the header
    assert "Cross-Validation Results" in formatted
    assert "95% CI" in formatted

    # Check that metric names are present
    assert "accuracy" in formatted
    assert "f1" in formatted

    # Check that values are formatted correctly
    assert "0.8200" in formatted
    assert "0.0150" in formatted

    # Check CI format
    assert "[0.8050, 0.8350]" in formatted


def test_format_cv_results_custom_decimal_places() -> None:
    """Test formatting with custom decimal places."""
    aggregated = {
        "accuracy": {
            "mean": 0.82345,
            "std": 0.01567,
            "ci_lower": 0.80778,
            "ci_upper": 0.83912,
        },
    }

    # Test with 2 decimal places
    formatted_2 = format_cv_results(aggregated, decimal_places=2)
    assert "0.82" in formatted_2
    assert "0.02" in formatted_2

    # Test with 6 decimal places
    formatted_6 = format_cv_results(aggregated, decimal_places=6)
    assert "0.823450" in formatted_6
    assert "0.015670" in formatted_6


def test_format_cv_results_sorted_output() -> None:
    """Test that metrics are sorted alphabetically in output."""
    aggregated = {
        "zeta": {"mean": 0.9, "std": 0.01, "ci_lower": 0.89, "ci_upper": 0.91},
        "alpha": {"mean": 0.8, "std": 0.02, "ci_lower": 0.78, "ci_upper": 0.82},
        "beta": {"mean": 0.85, "std": 0.015, "ci_lower": 0.835, "ci_upper": 0.865},
    }

    formatted = format_cv_results(aggregated)

    # Find positions of metric names in formatted string
    alpha_pos = formatted.find("alpha")
    beta_pos = formatted.find("beta")
    zeta_pos = formatted.find("zeta")

    # Check that they appear in alphabetical order
    assert alpha_pos < beta_pos < zeta_pos


def test_format_cv_results_alignment() -> None:
    """Test that metric names are properly aligned."""
    aggregated = {
        "accuracy": {"mean": 0.82, "std": 0.01, "ci_lower": 0.81, "ci_upper": 0.83},
        "f1": {"mean": 0.78, "std": 0.02, "ci_lower": 0.76, "ci_upper": 0.80},
    }

    formatted = format_cv_results(aggregated)

    # Both metric names should have colons at the same position
    lines = formatted.split("\n")
    metric_lines = [line for line in lines if ":" in line and ("accuracy" in line or "f1" in line)]

    # Check that colons align (both at position after longest metric name)
    colon_positions = [line.find(":") for line in metric_lines]
    assert len(set(colon_positions)) == 1  # All colons at same position


# ============================================================================
# Tests for compute_pooled_std
# ============================================================================


def test_compute_pooled_std_basic() -> None:
    """Test basic pooled standard deviation computation."""
    fold_stds = [0.1, 0.12, 0.11]
    fold_sizes = [100, 100, 100]

    pooled = compute_pooled_std(fold_stds, fold_sizes)

    # With equal sizes, pooled std should be close to mean of stds
    assert pooled > 0
    assert pooled == pytest.approx(0.1101, abs=1e-3)


def test_compute_pooled_std_equal_sizes() -> None:
    """Test pooled std with equal fold sizes."""
    fold_stds = [0.15, 0.15, 0.15]
    fold_sizes = [50, 50, 50]

    pooled = compute_pooled_std(fold_stds, fold_sizes)

    # With equal stds, pooled should equal the common std
    assert pooled == pytest.approx(0.15, abs=1e-6)


def test_compute_pooled_std_different_sizes() -> None:
    """Test pooled std with different fold sizes (weighted)."""
    fold_stds = [0.1, 0.2]
    fold_sizes = [100, 10]  # First fold has 10x more samples

    pooled = compute_pooled_std(fold_stds, fold_sizes)

    # Pooled std should be closer to 0.1 (larger fold)
    assert pooled < 0.15  # Closer to 0.1 than to 0.2


def test_compute_pooled_std_empty_inputs() -> None:
    """Test that empty inputs raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_pooled_std([], [])


def test_compute_pooled_std_mismatched_lengths() -> None:
    """Test that mismatched input lengths raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        compute_pooled_std([0.1, 0.2], [100, 100, 100])


def test_compute_pooled_std_single_fold() -> None:
    """Test pooled std with single fold."""
    fold_stds = [0.15]
    fold_sizes = [100]

    pooled = compute_pooled_std(fold_stds, fold_sizes)

    # With single fold, pooled std should equal the fold std
    # But denominator is (n-1) = 0, so we get 0.0
    assert pooled == 0.0


# ============================================================================
# Tests for effect_size_cohen_d
# ============================================================================


def test_effect_size_cohen_d_basic() -> None:
    """Test basic Cohen's d computation."""
    baseline = [0.75, 0.76, 0.74, 0.77]
    improved = [0.82, 0.83, 0.81, 0.84]

    d = effect_size_cohen_d(baseline, improved)

    # Improved should have positive effect size
    assert d > 0
    # This is a large effect (> 0.8)
    assert d > 0.8


def test_effect_size_cohen_d_no_difference() -> None:
    """Test Cohen's d when groups are identical."""
    group1 = [0.8, 0.82, 0.81, 0.83]
    group2 = [0.8, 0.82, 0.81, 0.83]

    d = effect_size_cohen_d(group1, group2)

    # Effect size should be zero
    assert d == pytest.approx(0.0, abs=1e-10)


def test_effect_size_cohen_d_negative() -> None:
    """Test Cohen's d with negative effect (group2 < group1)."""
    group1 = [0.9, 0.91, 0.89, 0.92]
    group2 = [0.7, 0.71, 0.69, 0.72]

    d = effect_size_cohen_d(group1, group2)

    # Effect size should be negative
    assert d < 0


def test_effect_size_cohen_d_numpy_arrays() -> None:
    """Test Cohen's d with numpy arrays as input."""
    group1 = np.array([0.75, 0.76, 0.74, 0.77])
    group2 = np.array([0.82, 0.83, 0.81, 0.84])

    d = effect_size_cohen_d(group1, group2)

    assert d > 0


def test_effect_size_cohen_d_empty_arrays() -> None:
    """Test that empty arrays raise ValueError."""
    with pytest.raises(ValueError, match="empty arrays"):
        effect_size_cohen_d([], [0.8, 0.9])

    with pytest.raises(ValueError, match="empty arrays"):
        effect_size_cohen_d([0.8, 0.9], [])


def test_effect_size_cohen_d_single_values() -> None:
    """Test Cohen's d with single values in each group."""
    group1 = [0.75]
    group2 = [0.85]

    d = effect_size_cohen_d(group1, group2)

    # Should compute, but variance is 0 so result is 0
    assert d == 0.0


def test_effect_size_cohen_d_zero_variance() -> None:
    """Test Cohen's d when both groups have zero variance."""
    group1 = [0.8, 0.8, 0.8]
    group2 = [0.9, 0.9, 0.9]

    d = effect_size_cohen_d(group1, group2)

    # Pooled std is 0, so effect size is 0
    assert d == 0.0


# ============================================================================
# Integration tests combining multiple functions
# ============================================================================


def test_aggregate_and_format_integration() -> None:
    """Test integration of aggregate_cv_metrics and format_cv_results."""
    fold_metrics = [
        {"accuracy": 0.80, "f1": 0.75},
        {"accuracy": 0.82, "f1": 0.77},
        {"accuracy": 0.81, "f1": 0.76},
    ]

    # Aggregate metrics
    aggregated = aggregate_cv_metrics(fold_metrics)

    # Format results
    formatted = format_cv_results(aggregated)

    # Check that formatted string contains expected values
    assert "accuracy" in formatted
    assert "f1" in formatted
    assert "0.8100" in formatted  # mean of accuracies


def test_full_cv_workflow() -> None:
    """Test complete workflow: fold metrics → aggregation → formatting."""
    # Simulate 5-fold cross-validation results
    fold_metrics = [
        {"accuracy": 0.85, "kappa": 0.78, "macro_f1": 0.82, "auc": 0.91},
        {"accuracy": 0.87, "kappa": 0.80, "macro_f1": 0.84, "auc": 0.92},
        {"accuracy": 0.84, "kappa": 0.77, "macro_f1": 0.81, "auc": 0.90},
        {"accuracy": 0.86, "kappa": 0.79, "macro_f1": 0.83, "auc": 0.91},
        {"accuracy": 0.88, "kappa": 0.81, "macro_f1": 0.85, "auc": 0.93},
    ]

    # Aggregate
    aggregated = aggregate_cv_metrics(fold_metrics)

    # Verify aggregation
    assert len(aggregated) == 4
    assert all(
        key in aggregated for key in ["accuracy", "kappa", "macro_f1", "auc"]
    )

    # Check accuracy stats
    acc = aggregated["accuracy"]
    assert acc["mean"] == pytest.approx(0.86, abs=1e-6)
    assert acc["min"] == 0.84
    assert acc["max"] == 0.88
    assert acc["ci_lower"] < acc["mean"] < acc["ci_upper"]

    # Format
    formatted = format_cv_results(aggregated, decimal_places=3)

    # Verify formatting
    assert "0.860" in formatted
    assert "accuracy" in formatted
    assert "kappa" in formatted
    assert "macro_f1" in formatted
    assert "auc" in formatted


def test_edge_case_all_zeros() -> None:
    """Test handling of all-zero metrics (degenerate case)."""
    fold_metrics = [
        {"metric": 0.0},
        {"metric": 0.0},
        {"metric": 0.0},
    ]

    aggregated = aggregate_cv_metrics(fold_metrics)

    stats = aggregated["metric"]
    assert stats["mean"] == 0.0
    assert stats["std"] == 0.0
    assert stats["min"] == 0.0
    assert stats["max"] == 0.0
    assert stats["ci_lower"] == 0.0
    assert stats["ci_upper"] == 0.0


def test_edge_case_high_variance() -> None:
    """Test handling of metrics with high variance across folds."""
    fold_metrics = [
        {"accuracy": 0.5},
        {"accuracy": 0.9},
        {"accuracy": 0.6},
        {"accuracy": 0.8},
    ]

    aggregated = aggregate_cv_metrics(fold_metrics)

    stats = aggregated["accuracy"]
    # High variance should result in wide confidence interval
    ci_width = stats["ci_upper"] - stats["ci_lower"]
    assert ci_width > 0.1  # Expect substantial uncertainty
