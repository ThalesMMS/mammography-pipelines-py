"""
Mammography training utilities and loops.

This module provides training loops, evaluation functions, and utilities for
training deep learning models on mammography data.
"""

# Legacy training engine
from mammography.training.engine import *  # noqa: F401, F403

# Cancer classification training
from mammography.training.cancer_trainer import (  # noqa: F401
    DensityHistoryEntry,
    collect_predictions,
    evaluate,
    fit_classifier,
    get_sens_spec,
    train_one_epoch,
)

__all__ = [
    # Cancer training exports
    "DensityHistoryEntry",
    "get_sens_spec",
    "train_one_epoch",
    "evaluate",
    "collect_predictions",
    "fit_classifier",
]
