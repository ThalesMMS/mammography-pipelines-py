"""Stub training loop for the RL refinement stage."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Any

from .env import EnvConfig, SimpleMammoEnv, MammoGymEnv
from .policy import BasePolicy, RandomPolicy, ThresholdPolicy

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import PPO, A2C  # type: ignore
    from stable_baselines3.common.evaluation import evaluate_policy as sb3_evaluate  # type: ignore
except Exception:  # pragma: no cover
    PPO = None  # type: ignore
    A2C = None  # type: ignore
    sb3_evaluate = None

try:  # Optional dependency for YAML configs.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None


@dataclass
class TrainConfig:
    algorithm: str = "random"
    episodes: int = 5
    max_steps: int = 10
    reward_scale: float = 1.0
    seed: int | None = None
    total_timesteps: int = 5000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    outdir: Path = Path("outputs/rl_refinement/run_stub")
    dry_run: bool = False


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        env_cfg = EnvConfig(
            seed=config.seed,
            max_steps=config.max_steps,
            reward_scale=config.reward_scale,
        )
        self.env = SimpleMammoEnv(env_cfg)
        self.config = config
        self.policy: BasePolicy | None = None
        if config.algorithm == "threshold":
            self.policy = ThresholdPolicy()
        elif config.algorithm == "random":
            self.policy = RandomPolicy(action_count=self.env.action_space, seed=config.seed)

    def run(self) -> dict[str, float]:
        if self.config.algorithm in {"ppo", "a2c"}:
            return self._run_sb3()

        summary = {
            "algorithm": self.config.algorithm,
            "episodes": float(self.config.episodes),
            "mean_reward": 0.0,
        }
        if self.config.dry_run:
            return summary

        if self.policy is None:
            raise RuntimeError(f"Algoritmo {self.config.algorithm} não possui policy definida.")

        episode_rewards: list[float] = []
        for episode in range(1, self.config.episodes + 1):
            self.policy.reset(seed=(self.config.seed or 0) + episode)
            obs = self.env.reset()
            done = False
            total = 0.0
            steps = 0
            while not done and steps < self.config.max_steps:
                action = self.policy.act(obs)
                obs, reward, done, _ = self.env.step(action)
                total += reward
                steps += 1
            episode_rewards.append(total)
        summary["mean_reward"] = sum(episode_rewards) / len(episode_rewards)
        return summary

    def _run_sb3(self) -> dict[str, float]:
        if self.config.algorithm not in {"ppo", "a2c"}:
            raise RuntimeError("SB3 training solicitado para algoritmo incompatível.")
        if (self.config.algorithm == "ppo" and PPO is None) or (self.config.algorithm == "a2c" and A2C is None):
            raise RuntimeError(
                "stable-baselines3 não está instalado. Use `pip install stable-baselines3 gymnasium` para habilitar PPO/A2C."
            )
        summary: dict[str, float] = {
            "algorithm": self.config.algorithm,
            "episodes": float(self.config.episodes),
            "total_timesteps": float(self.config.total_timesteps),
        }
        if self.config.dry_run:
            return summary
        algo_cls = PPO if self.config.algorithm == "ppo" else A2C
        env = MammoGymEnv(self.env.config)
        model = algo_cls(
            "MlpPolicy",
            env,
            verbose=0,
            seed=self.config.seed,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
        )
        model.learn(total_timesteps=self.config.total_timesteps)
        if sb3_evaluate is not None:
            mean_reward, std_reward = sb3_evaluate(
                model,
                env,
                n_eval_episodes=max(1, self.config.episodes),
            )
            summary["mean_reward"] = float(mean_reward)
            summary["reward_std"] = float(std_reward)
        else:
            summary["mean_reward"] = 0.0
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        model.save(str(self.config.outdir / f"{self.config.algorithm}_policy"))
        return summary


def _write_summary(outdir: Path, data: dict[str, float], config: TrainConfig) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.json"
    payload = {
        **data,
        "algorithm": config.algorithm,
        "episodes": float(config.episodes),
        "total_timesteps": float(config.total_timesteps),
        "learning_rate": float(config.learning_rate),
        "gamma": float(config.gamma),
        "seed": config.seed,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rl_refinement.train",
        description="Stub de treinamento para refinamento por RL (Stage 3).",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument(
        "--algo",
        choices=["random", "threshold", "ppo", "a2c"],
        default="random",
        help="Algoritmo usado para rl-refine (default: random).",
    )
    parser.add_argument("--total-timesteps", type=int, default=5000, help="Timesteps SB3 quando --algo for PPO/A2C.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate usado pelo agente PPO/A2C.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Fator de desconto para PPO/A2C.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/rl_refinement/run_stub"))
    parser.add_argument("--threshold-policy", action="store_true", help="Atalho legado equivalente a --algo threshold.")
    parser.add_argument("--policy-config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_policy_payload(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {}
    text = resolved.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    if yaml is not None:
        return yaml.safe_load(text) or {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    payload = _load_policy_payload(args.policy_config)
    algo = str(payload.get("algorithm", args.algo)).lower()
    if args.threshold_policy:
        algo = "threshold"
    config = TrainConfig(
        algorithm=algo,
        episodes=int(payload.get("episodes", args.episodes)),
        max_steps=int(payload.get("window", args.max_steps)),
        reward_scale=float(payload.get("reward_scale", args.reward_scale)),
        seed=args.seed,
        total_timesteps=int(payload.get("total_timesteps", args.total_timesteps)),
        learning_rate=float(payload.get("learning_rate", args.learning_rate)),
        gamma=float(payload.get("gamma", args.gamma)),
        outdir=args.outdir,
        dry_run=args.dry_run,
    )
    trainer = Trainer(config)
    summary = trainer.run()
    summary["timestamp"] = time.time()
    if not config.dry_run:
        _write_summary(config.outdir, summary, config)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(main())
