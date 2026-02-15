#
# __init__.py
# mammography-pipelines
#
# Re-exports hyperparameter tuning utilities for model optimization.
#
# Thales Matheus Mendonça Santos - January 2026
#
"""Hyperparameter tuning utilities for model optimization."""

from mammography.tuning.search_space import (
    SearchSpace,
    CategoricalParam,
    IntParam,
    FloatParam,
)
try:
    from mammography.tuning.optuna_tuner import OptunaTuner
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OptunaTuner = None  # type: ignore[assignment]

__all__ = [
    "SearchSpace",
    "CategoricalParam",
    "IntParam",
    "FloatParam",
    "OptunaTuner",
]
