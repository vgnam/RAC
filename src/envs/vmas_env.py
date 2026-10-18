"""VMAS (Vectorized Multi-Agent Simulator) as a pymarl MultiAgentEnv.

VMAS is a GPU-vectorised 2D physics simulator (https://github.com/proroklab/VectorizedMultiAgentSimulator)
whose scenarios are continuous-control by default but expose a discrete action
space as well (``continuous_actions=False``); this wrapper only wires the
discrete path, because the learners here are Q-learners over a categorical
action selector.

Scope, and why:

* ``num_envs = 1``. The episode runner drives a single environment
  (``batch_size_run = 1``); VMAS's vectorisation would need the parallel runner.
  A bigger ``num_envs`` is rejected rather than silently ignored.
* the team reward is the mean (default) or the sum of the per-agent rewards.
  VMAS scenarios usually broadcast one shared reward to every agent, so the mean
  keeps the return scale independent of ``n_agents``.
* VMAS rewards are often negative (navigation pays the distance to the goal), so
  the RAC/HC-RAC context bins need a return range that matches the scenario:
  run ~20k steps, read ``episode_returns_min`` / ``episode_returns_max`` from the
  metrics CSV, then set ``vmas_return_min`` / ``vmas_return_max`` (and
  ``corac_return_range`` for HC-RAC) to that range.
"""
import numpy as np
import torch as th

from .multiagentenv import MultiAgentEnv


class VMASEnv(MultiAgentEnv):
    def __init__(
        self,
        scenario="navigation",
        num_envs=1,
        continuous_actions=False,
        time_limit=100,
        seed=None,
        device="cpu",
        reward_aggregation="mean",
        **scenario_kwargs
    ):
        if bool(continuous_actions):
            raise ValueError(
                "vmas: only discrete actions are supported (continuous_actions=false); "
                "the action selector here is categorical."
            )
        if int(num_envs) != 1:
            raise ValueError(
                "vmas: num_envs must be 1 because the episode runner drives one "
                "environment; vectorised rollouts need the parallel runner."
            )
        if reward_aggregation not in ("mean", "sum"):
            raise ValueError(
                f"vmas: reward_aggregation must be 'mean' or 'sum', got {reward_aggregation!r}."
            )
        try:
            from vmas import make_env
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "vmas is not installed. Install it with `pip install vmas` "
                "(see requirements-vmas.txt)."
            ) from exc

        self.scenario = str(scenario)
        self._seed = None if seed is None else int(seed)
        self._episode_count = 0
        self._reward_aggregation = reward_aggregation
        self.episode_limit = int(time_limit)
        self._device = th.device(device)
        self.device = str(self._device)
        # The simulator is meant to run on CPU even when the policy is on a GPU:
        # with num_envs=1 every step would otherwise pay a host<->device transfer
        # plus a kernel launch for a handful of bodies. Warn instead of silently
        # accepting a much slower configuration.
        if self._device.type != "cpu":
            print(
                "vmas: simulator device is {} but num_envs=1; CPU is usually "
                "faster here. Pass env_args.device=cpu to silence this.".format(
                    self._device
                )
            )
        self._env = make_env(
            scenario=self.scenario,
            num_envs=1,
            device=self._device,
            continuous_actions=False,
            max_steps=self.episode_limit,
            seed=self._seed,
            # Gives us the explicit truncation flag, which is what the runner's
            # `info["episode_limit"]` convention expects.
            terminated_truncated=True,
            **scenario_kwargs
        )
        self.n_agents = int(self._env.n_agents)

        action_sizes = {int(self._env.get_agent_action_space(a).n) for a in self._env.agents}
        if len(action_sizes) != 1:
            raise ValueError(
                "vmas: agents have different discrete action spaces "
                f"{sorted(action_sizes)}; this wrapper assumes a shared one."
            )
        self.n_actions = action_sizes.pop()

        # get_env_info() is read before the first runner reset, so the observation
        # shape has to be known from construction.
        initial = self._to_obs(self._env.reset(seed=self._seed))
        obs_sizes = {int(o.shape[0]) for o in initial}
        if len(obs_sizes) != 1:
            raise ValueError(
                "vmas: agents have different observation sizes "
                f"{sorted(obs_sizes)}; this wrapper assumes a shared one."
            )
        self._obs_size = obs_sizes.pop()
        self._obs = initial

    # ------------------------------------------------------------------ #
    # pymarl MultiAgentEnv interface
    # ------------------------------------------------------------------ #
    def reset(self):
        seed = None if self._seed is None else self._seed + self._episode_count
        self._episode_count += 1
        self._obs = self._to_obs(self._env.reset(seed=seed))
        return self.get_obs(), self.get_state()

    def step(self, actions):
        acts = []
        for action in np.asarray(actions).reshape(-1):
            value = int(action)
            if not 0 <= value < self.n_actions:
                raise ValueError(
                    f"vmas: action {value} is outside [0, {self.n_actions})."
                )
            acts.append(th.tensor([[value]], dtype=th.long, device=self._device))
        obs, rewards, terminated, truncated, _ = self._env.step(acts)
        self._obs = self._to_obs(obs)

        per_agent = [float(r[0]) for r in rewards]
        reward = float(np.mean(per_agent)) if self._reward_aggregation == "mean" else float(np.sum(per_agent))
        done = bool(terminated[0]) or bool(truncated[0])
        info = {"episode_limit": bool(truncated[0]) and not bool(terminated[0])}
        return reward, done, info

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
        try:
            return self._env.render(mode="rgb_array")
        except Exception:
            return None

    def close(self):
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

    def seed(self):
        return self._seed

    def save_replay(self):
        pass

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_obs(obs):
        """vmas returns a list of (num_envs, obs_size) tensors -> list of 1-D float32 arrays."""
        if isinstance(obs, dict):
            raise ValueError(
                "vmas: dict observations (dict_spaces=true) are not supported; "
                "disable dict_spaces."
            )
        out = []
        for agent_obs in obs:
            array = agent_obs.detach().cpu().reshape(-1).numpy().astype(np.float32)
            out.append(array)
        return out
