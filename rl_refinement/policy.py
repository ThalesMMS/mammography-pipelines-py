"""Placeholder policies used by the rl_refinement stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence
import random


class BasePolicy(Protocol):
    def act(self, obs: Sequence[float]) -> int:  # pragma: no cover - protocol only
        ...

    def reset(self, *, seed: int | None = None) -> None:  # pragma: no cover
        ...


@dataclass
class RandomPolicy:
    action_count: int
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def act(self, obs: Sequence[float]) -> int:  # noqa: ARG002
        return self.rng.randrange(self.action_count)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)


@dataclass
class ThresholdPolicy:
    threshold: float = 0.8

    def act(self, obs: Sequence[float]) -> int:
        val_kappa = obs[1]
        if val_kappa >= self.threshold:
            return 1  # mantém estratégia
        return 2  # intensifica ajustes

    def reset(self, *, seed: int | None = None) -> None:  # noqa: ARG002
        return
