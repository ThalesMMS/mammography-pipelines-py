"""
Helpers for taming noisy NumPy warnings in numerical routines.

These warnings are environment-specific and can be triggered by BLAS-backed
matmul even when the results are finite. We filter only the matmul warnings
so that other numeric warnings remain visible.
"""

from contextlib import contextmanager
import warnings


def resolve_pca_svd_solver(
    n_samples: int,
    n_features: int,
    n_components: int,
    requested: str | None,
) -> str:
    """Pick a PCA solver, favoring randomized SVD for large matrices."""
    if not requested:
        requested = "auto"
    if requested != "auto":
        return requested
    min_dim = min(n_samples, n_features)
    if n_components >= min_dim:
        return "full"
    if n_samples * n_features >= 1_000_000 and n_components <= min_dim // 2:
        return "randomized"
    return "full"


@contextmanager
def suppress_numpy_matmul_warnings():
    """Suppress spurious NumPy RuntimeWarnings emitted by matmul."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        yield
