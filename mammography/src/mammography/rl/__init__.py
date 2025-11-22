#
# __init__.py
# mammography-pipelines-py
#
# Re-exports RL refinement stubs, configs, policies, and simple evaluation helpers.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""RL refinement stubs exposed as a tiny public surface for consumers."""

from .env import EnvConfig, MammoGymEnv, SimpleMammoEnv
from .policy import BasePolicy, RandomPolicy, ThresholdPolicy
from .train import TrainConfig, Trainer
from .eval import EvalConfig, evaluate_policy

__all__ = [
    "EnvConfig",
    "MammoGymEnv",
    "SimpleMammoEnv",
    "BasePolicy",
    "RandomPolicy",
    "ThresholdPolicy",
    "TrainConfig",
    "Trainer",
    "EvalConfig",
    "evaluate_policy",
]
