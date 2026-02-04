#
# reproducibility.py
# mammography-pipelines
#
# Utilities for ensuring reproducible experiments across runs.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""
Reproducibility utilities for deterministic training and evaluation.

Provides functions to fix random seeds across Python, NumPy, PyTorch, and CUDA
to ensure reproducible results in deep learning experiments.
"""

import random
import numpy as np
import torch
from typing import Optional


def fix_seeds(seed: int = 42, deterministic: bool = False) -> None:
    """
    Fix random seeds for reproducibility across all libraries.

    This is a wrapper around seed_everything from mammography.utils.common
    for backward compatibility with tests and legacy code.

    Args:
        seed: Random seed value (default: 42)
        deterministic: Enable deterministic CUDA operations (slower but reproducible)

    Note:
        Deterministic mode may reduce performance but ensures bit-exact reproducibility
        on CUDA operations. Use only when exact reproducibility is critical.

    Example:
        >>> fix_seeds(42, deterministic=True)
        >>> # All random operations are now reproducible
    """
    from mammography.utils.common import seed_everything

    seed_everything(seed, deterministic=deterministic)


def get_random_state() -> dict:
    """
    Capture current random state from all RNG sources.

    Returns:
        Dictionary containing random states from Python, NumPy, and PyTorch

    Example:
        >>> state = get_random_state()
        >>> # Do some random operations
        >>> restore_random_state(state)  # Restore to previous state
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_state(state: dict) -> None:
    """
    Restore random state from a previously saved state dictionary.

    Args:
        state: Dictionary containing random states (from get_random_state())

    Example:
        >>> state = get_random_state()
        >>> # Do some random operations
        >>> restore_random_state(state)  # Restore to previous state
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
