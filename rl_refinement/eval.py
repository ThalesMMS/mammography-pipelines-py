"""Minimal evaluation helpers for the RL refinement stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .env import EnvConfig, SimpleMammoEnv
from .policy import BasePolicy, RandomPolicy


@dataclass
class EvalConfig:
    episodes: int = 3
    max_steps: int = 10
    seed: int | None = None


def evaluate_policy(policy: BasePolicy | None = None, config: EvalConfig | None = None) -> dict[str, float]:
    cfg = config or EvalConfig()
    env_cfg = EnvConfig(seed=cfg.seed, max_steps=cfg.max_steps)
    env = SimpleMammoEnv(env_cfg)
    agent = policy or RandomPolicy(action_count=env.action_space, seed=cfg.seed)

    rewards: list[float] = []
    for episode in range(cfg.episodes):
        agent.reset(seed=(cfg.seed or 0) + episode)
        obs = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done and steps < cfg.max_steps:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            total += reward
            steps += 1
        rewards.append(total)
    return {"episodes": float(cfg.episodes), "mean_reward": sum(rewards) / len(rewards)}
