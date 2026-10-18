"""VMAS wrapper smoke tests.

Skipped when the optional ``vmas`` package is not installed.
"""
import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("vmas") is None,
    reason="vmas is not installed (pip install -r requirements-vmas.txt)",
)


def test_vmas_env_info_and_rollout():
    from src.envs.vmas_env import VMASEnv

    env = VMASEnv(scenario="navigation", time_limit=5, seed=0)
    info = env.get_env_info()

    assert info["n_agents"] == env.n_agents >= 1
    assert info["n_actions"] == env.n_actions >= 2
    assert info["episode_limit"] == 5
    assert info["obs_shape"] == env.get_obs_size() > 0
    assert info["state_shape"] == info["obs_shape"] * info["n_agents"]

    obs, state = env.reset()
    assert len(obs) == env.n_agents
    assert all(o.shape == (info["obs_shape"],) and o.dtype == np.float32 for o in obs)
    assert state.shape == (info["state_shape"],)
    assert env.get_avail_actions() == [[1] * info["n_actions"]] * info["n_agents"]

    steps = 0
    terminated = False
    while not terminated:
        reward, terminated, step_info = env.step(np.zeros(env.n_agents, dtype=int))
        steps += 1
        assert np.isfinite(reward)
    # A fixed action cannot end navigation early, so this episode is truncated
    # at the horizon and reported as such (so the learner bootstraps through it).
    assert steps == 5
    assert step_info["episode_limit"] is True
    env.close()


def test_vmas_env_rejects_unsupported_modes():
    from src.envs.vmas_env import VMASEnv

    with pytest.raises(ValueError, match="num_envs must be 1"):
        VMASEnv(scenario="navigation", num_envs=2)
    with pytest.raises(ValueError, match="continuous"):
        VMASEnv(scenario="navigation", continuous_actions=True)
    with pytest.raises(ValueError, match="reward_aggregation"):
        VMASEnv(scenario="navigation", reward_aggregation="max")


def test_vmas_env_rejects_out_of_range_actions():
    from src.envs.vmas_env import VMASEnv

    env = VMASEnv(scenario="navigation", time_limit=3)
    env.reset()
    with pytest.raises(ValueError, match="outside"):
        env.step(np.full(env.n_agents, env.n_actions))


def test_vmas_simulator_runs_on_cpu_by_default():
    """The env stays on CPU even for GPU training runs (num_envs=1)."""
    from src.envs.vmas_env import VMASEnv
    from src.main import _load_config

    env = VMASEnv(scenario="navigation", time_limit=3)
    assert env.device == "cpu"

    config = _load_config("vmas", "envs")
    assert config["env"] == "vmas"
    assert config["env_args"]["device"] == "cpu"
