"""Overcooked-AI (https://github.com/HumanCompatibleAI/overcooked_ai) as a pymarl MultiAgentEnv.

Overcooked-AI is a two-player cooperative cooking gridworld: 6 discrete actions
per agent (stay, N/S/E/W, interact) and a shared reward of 20 per delivered
soup. That fits the Q-learners here directly, and unlike LBF's binary return the
episode return takes several distinct values (0, 20, 40, ...), so RAC's one-hot
return bins actually get support in every slot.

Two choices are exposed because they change the numbers a lot:

* ``reward_shaping``: "sparse" (only 20 per soup) or "shaped" (the package's
  BASE_REW_SHAPING_PARAMS: 3 per placement in pot, 3 per dish pickup, 5 per soup
  pickup). Sparse is the default: the bins are then multiples of 20 and the
  task is the one usually reported.
* ``obs``: "featurized" (96 floats per agent, ``mdp.featurize_state``) or
  "lossless" (520 per agent, the full grid encoding). Featurized is cheaper for
  the RNN; use "lossless" to compare against published baselines.

The episode always ends on the horizon (Overcooked has no early terminal), so
``info["episode_limit"]`` is always true and the learner bootstraps through the
last step.
"""
import numpy as np

from .multiagentenv import MultiAgentEnv


class OvercookedEnv(MultiAgentEnv):
    def __init__(
        self,
        layout="cramped_room",
        horizon=200,
        obs="featurized",
        reward_shaping="sparse",
        random_start_pos=True,
        rnd_obj_prob_thresh=0.0,
        seed=None,
        **kwargs
    ):
        try:
            from overcooked_ai_py.mdp.actions import Action
            from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as _OCEnv
            from overcooked_ai_py.mdp.overcooked_mdp import (
                BASE_REW_SHAPING_PARAMS,
                OvercookedGridworld,
            )
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "overcooked_ai is not installed. Install it with "
                "`pip install -r requirements-overcooked.txt`."
            ) from exc

        if obs not in ("featurized", "lossless"):
            raise ValueError(f"overcooked: obs must be 'featurized' or 'lossless', got {obs!r}.")
        if reward_shaping not in ("sparse", "shaped"):
            raise ValueError(
                f"overcooked: reward_shaping must be 'sparse' or 'shaped', got {reward_shaping!r}."
            )
        if int(horizon) <= 0:
            raise ValueError(f"overcooked: horizon must be positive, got {horizon}.")

        self._Action = Action
        self.layout = str(layout)
        self._obs_mode = obs
        self._reward_shaping = reward_shaping
        self.episode_limit = int(horizon)
        self._seed = None if seed is None else int(seed)

        # `rew_shaping_params=None` means the package default (shaped), so sparse
        # has to be requested explicitly by zeroing every term.
        shaping = None
        if reward_shaping == "sparse":
            shaping = {key: 0 for key in BASE_REW_SHAPING_PARAMS}
        self._mdp = OvercookedGridworld.from_layout_name(
            self.layout, rew_shaping_params=shaping
        )
        start_state_fn = None
        if bool(random_start_pos):
            start_state_fn = self._mdp.get_random_start_state_fn(
                random_start_pos=True, rnd_obj_prob_thresh=float(rnd_obj_prob_thresh)
            )
        self._env = _OCEnv.from_mdp(
            self._mdp, start_state_fn=start_state_fn, horizon=self.episode_limit
        )
        self.n_agents = 2
        self.n_actions = len(Action.INDEX_TO_ACTION)

        self._soups = 0
        self._episode_count = 0
        self._obs = self._encode(self._reset_env())
        obs_sizes = {int(o.shape[0]) for o in self._obs}
        if len(obs_sizes) != 1:
            raise ValueError(f"overcooked: agents have different observation sizes {sorted(obs_sizes)}.")
        self._obs_size = obs_sizes.pop()
        # `sparse_r_by_agent` is reported by the env even when shaping is on.
        self._sparse_per_soup = 20

    # ------------------------------------------------------------------ #
    # pymarl MultiAgentEnv interface
    # ------------------------------------------------------------------ #
    def reset(self):
        if self._seed is not None:
            np.random.seed(self._seed + self._episode_count)
        self._episode_count += 1
        self._obs = self._encode(self._reset_env())
        self._soups = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        indices = [int(a) for a in np.asarray(actions).reshape(-1)]
        if len(indices) != self.n_agents:
            raise ValueError(f"overcooked: expected {self.n_agents} actions, got {len(indices)}.")
        for value in indices:
            if not 0 <= value < self.n_actions:
                raise ValueError(f"overcooked: action {value} is outside [0, {self.n_actions}).")
        joint_action = tuple(self._Action.INDEX_TO_ACTION[i] for i in indices)

        state, reward, done, info = self._env.step(joint_action)
        self._obs = self._encode(state)
        delivered = int(np.sum(np.asarray(info.get("sparse_r_by_agent", 0)))) // self._sparse_per_soup
        self._soups += delivered
        return (
            float(reward),
            bool(done),
            {"episode_limit": bool(done), "soups": delivered},
        )

    def get_obs(self):
        return list(self._obs)

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return self._obs_size

    def get_state(self):
        return np.concatenate(self._obs)

    def get_state_size(self):
        return self._obs_size * self.n_agents

    def get_avail_actions(self):
        return [[1] * self.n_actions for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return [1] * self.n_actions

    def get_total_actions(self):
        return self.n_actions

    def get_stats(self):
        return {}

    def render(self):
        return None

    def close(self):
        pass

    def seed(self):
        return self._seed

    def save_replay(self):
        pass

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _reset_env(self):
        self._env.reset()
        return self._env.state

    def _encode(self, state):
        if self._obs_mode == "lossless":
            encoded = np.asarray(self._mdp.lossless_state_encoding(state))
            return [row.astype(np.float32).reshape(-1) for row in encoded]
        encoded = np.asarray(self._mdp.featurize_state(state, self._env.mlam))
        return [row.astype(np.float32).reshape(-1) for row in encoded]
