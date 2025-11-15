from __future__ import annotations

from rl_refinement.env import EnvConfig, SimpleMammoEnv


def test_env_reset_and_step_shapes() -> None:
    env = SimpleMammoEnv(EnvConfig(seed=123, max_steps=5))
    obs = env.reset()
    assert isinstance(obs, tuple)
    assert len(obs) == 4

    next_obs, reward, done, info = env.step(1)
    assert isinstance(next_obs, tuple)
    assert len(next_obs) == 4
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_env_allows_multiple_steps_until_done() -> None:
    env = SimpleMammoEnv(EnvConfig(seed=42, max_steps=3))
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(2)
        steps += 1
        if steps > 5:
            break
    assert steps <= 5
