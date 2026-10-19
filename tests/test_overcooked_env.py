"""Overcooked wrapper smoke tests.

Skipped when the optional ``overcooked-ai`` package is not installed.
"""
import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("overcooked_ai_py") is None,
    reason="overcooked-ai is not installed (pip install -r requirements-overcooked.txt)",
)


def test_overcooked_env_info_and_rollout():
    from src.envs.overcooked_env import OvercookedEnv

    env = OvercookedEnv(layout="cramped_room", horizon=6, seed=0)
    info = env.get_env_info()

    assert info["n_agents"] == 2
    assert info["n_actions"] == 6
    assert info["episode_limit"] == 6
    assert info["obs_shape"] == env.get_obs_size() > 0
    assert info["state_shape"] == info["obs_shape"] * 2
    assert env.get_avail_actions() == [[1] * 6, [1] * 6]

    obs, state = env.reset()
    assert len(obs) == 2
    assert all(o.shape == (info["obs_shape"],) and o.dtype == np.float32 for o in obs)
    assert state.shape == (info["state_shape"],)

    steps = 0
    terminated = False
    soups = 0
    while not terminated:
        reward, terminated, step_info = env.step(np.zeros(2, dtype=int))
        steps += 1
        soups += step_info["soups"]
        assert np.isfinite(reward)
    # Overcooked has no early terminal: the episode always ends on the horizon,
    # and it must be reported as a truncation so the learner bootstraps through.
    assert steps == 6
    assert step_info["episode_limit"] is True


def test_overcooked_lossless_obs_mode_is_larger():
    from src.envs.overcooked_env import OvercookedEnv

    featurized = OvercookedEnv(layout="cramped_room", horizon=3, obs="featurized")
    lossless = OvercookedEnv(layout="cramped_room", horizon=3, obs="lossless")
    assert featurized.get_obs_size() == 96
    assert lossless.get_obs_size() == 5 * 4 * 26


def test_overcooked_composes_the_shaped_reward():
    """OvercookedEnv.step returns the sparse reward only; shaping lives in env_info."""
    from src.envs.overcooked_env import OvercookedEnv

    info = {"sparse_r_by_agent": [20.0, 0.0], "shaped_r_by_agent": [3.0, 5.0]}

    sparse = OvercookedEnv(layout="cramped_room", horizon=3, reward_shaping="sparse")
    assert sparse._compose_reward(info) == 20.0          # shaping ignored

    shaped = OvercookedEnv(layout="cramped_room", horizon=3, reward_shaping="shaped")
    assert shaped._compose_reward(info) == 28.0          # 20 delivery + 3 + 5 shaping
    # and with no events at all both are zero (nothing invented)
    empty = {"sparse_r_by_agent": [0.0, 0.0], "shaped_r_by_agent": [0.0, 0.0]}
    assert sparse._compose_reward(empty) == 0.0
    assert shaped._compose_reward(empty) == 0.0


def test_overcooked_rejects_bad_config():
    from src.envs.overcooked_env import OvercookedEnv

    with pytest.raises(ValueError, match="obs must be"):
        OvercookedEnv(layout="cramped_room", obs="rgb")
    with pytest.raises(ValueError, match="reward_shaping"):
        OvercookedEnv(layout="cramped_room", reward_shaping="dense")
    with pytest.raises(ValueError, match="horizon"):
        OvercookedEnv(layout="cramped_room", horizon=0)


def test_overcooked_rejects_out_of_range_actions():
    from src.envs.overcooked_env import OvercookedEnv

    env = OvercookedEnv(layout="cramped_room", horizon=3)
    env.reset()
    with pytest.raises(ValueError, match="outside"):
        env.step(np.array([6, 0]))
    with pytest.raises(ValueError, match="expected 2 actions"):
        env.step(np.array([0]))


def test_overcooked_presets_cover_five_difficulties():
    from src.main import _load_config

    base = _load_config("overcooked", "envs")
    assert base["env_args"]["obs"] == "lossless"     # cheaper *and* more informative
    assert base["env_args"]["reward_shaping"] == "shaped"  # sparse is undiscoverable here

    ladder = {
        "overcooked_cramped_room": ("cramped_room", 250.0),
        "overcooked_coordination_ring": ("coordination_ring", 250.0),
        "overcooked_asymmetric_advantages": ("asymmetric_advantages", 280.0),
        "overcooked_counter_circuit": ("counter_circuit_o_1order", 140.0),
        "overcooked_forced_coordination": ("forced_coordination", 120.0),
    }
    for name, (layout, return_max) in ladder.items():
        config = _load_config(name, "envs")
        assert config["env"] == "overcooked"
        assert config["env_args"]["layout"] == layout
        assert config["env_args"]["reward_shaping"] == "shaped"  # inherited
        assert config["overcooked_return_max"] == return_max
        assert config["overcooked_return_min"] == 0.0
