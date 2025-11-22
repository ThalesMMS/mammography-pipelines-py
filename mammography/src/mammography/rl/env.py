#
# env.py
# mammography-pipelines-py
#
# Lightweight environment stubs for RL refinement experiments, including a gym-compatible wrapper.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Lightweight environment stubs for RL refinement experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Sequence
import random
import numpy as np

try:  # pragma: no cover - optional dependency
    import gymnasium as gym
    _GYM_USES_GYMNASIUM = True
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore
        _GYM_USES_GYMNASIUM = False
    except Exception:  # pragma: no cover
        gym = None  # type: ignore
        _GYM_USES_GYMNASIUM = False


Observation = Tuple[float, float, float, float]


@dataclass
class EnvConfig:
    """Configuration toggles for the toy environment."""

    seed: int | None = None
    max_steps: int = 10
    reward_scale: float = 1.0
    baseline_kappa: float = 0.6
    target_kappa: float = 0.8


@dataclass
class SimpleMammoEnv:
    """Simple environment that emulates validation metrics adjustments."""

    config: EnvConfig = field(default_factory=EnvConfig)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.config.seed)
        self.action_space = 3  # 0 = relax, 1 = keep, 2 = intensify
        self._step = 0
        self.state: Observation = self._sample_state()

    def seed(self, value: int | None = None) -> None:
        self.rng.seed(value)

    def _sample_state(self) -> Observation:
        return (
            self.rng.uniform(0.6, 0.9),  # val_acc
            self.rng.uniform(0.4, 0.8),  # val_kappa
            self.rng.uniform(0.1, 1.0),  # train_loss
            self.rng.uniform(0.1, 1.0),  # val_loss
        )

    def reset(self) -> Observation:
        self._step = 0
        self.state = self._sample_state()
        return self.state

    def step(self, action: int) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if not 0 <= action < self.action_space:
            raise ValueError(f"ação inválida: {action}")
        val_acc, val_kappa, train_loss, val_loss = self.state
        delta = (action - 1) * 0.01
        # Tiny reward shaping: action 0 decreases effort, 2 intensifies, 1 keeps state.
        val_acc = max(0.0, min(1.0, val_acc + delta))
        val_kappa = max(0.0, min(1.0, val_kappa + delta * 1.5))
        train_loss = max(0.0, train_loss - delta)
        val_loss = max(0.0, val_loss - delta * 0.5)
        self.state = (val_acc, val_kappa, train_loss, val_loss)
        self._step += 1

        reward = (
            (val_kappa - self.config.baseline_kappa) * self.config.reward_scale
            - abs(train_loss - val_loss) * 0.1
        )
        done = self._step >= self.config.max_steps or val_kappa >= self.config.target_kappa
        info = {"step": self._step, "val_acc": val_acc, "val_kappa": val_kappa}
        return self.state, reward, done, info


def rollout(env: SimpleMammoEnv, policy: Sequence[int]) -> Dict[str, Any]:
    """Runs a short rollout using a static sequence of actions."""

    obs = env.reset()
    total_reward = 0.0
    for idx, action in enumerate(policy):
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            return {"steps": idx + 1, "reward": total_reward, "info": info}
    return {"steps": len(policy), "reward": total_reward, "info": {"val_kappa": obs[1]}}


if gym is not None:  # pragma: no cover - requires gym install
    class MammoGymEnv(gym.Env):  # type: ignore[misc]
        """Gym/Gymnasium wrapper over SimpleMammoEnv for SB3 algorithms."""

        metadata = {"render_modes": []}

        def __init__(self, config: EnvConfig | None = None):
            super().__init__()
            self.config = config or EnvConfig()
            self._env = SimpleMammoEnv(self.config)
            low = np.zeros(4, dtype=np.float32)
            high = np.ones(4, dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            self.action_space = gym.spaces.Discrete(self._env.action_space)

        def reset(self, *, seed: int | None = None, options: dict | None = None):  # noqa: D401
            if seed is not None:
                self._env.seed(seed)
            obs = np.array(self._env.reset(), dtype=np.float32)
            if _GYM_USES_GYMNASIUM:
                return obs, {}
            return obs

        def step(self, action: int):  # noqa: D401
            obs, reward, done, info = self._env.step(int(action))
            obs_arr = np.array(obs, dtype=np.float32)
            if _GYM_USES_GYMNASIUM:
                return obs_arr, float(reward), done, False, info
            return obs_arr, float(reward), done, info
else:  # pragma: no cover - fallback
    class MammoGymEnv:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("gym/gymnasium não está instalado; instale para usar PPO/A2C.")
