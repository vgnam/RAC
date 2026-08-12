import numpy as np

# MATE 0.1.0 uses ``np.bool8`` throughout its environment implementation.
# NumPy 2 removed that alias in favor of ``np.bool_``.  Define the old name
# before importing MATE so the environment works with either NumPy major
# version without requiring a process-wide package downgrade.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import mate

from src.envs.multiagentenv import MultiAgentEnv
from src.utils.dict2namedtuple import convert

_TARGET_AGENTS = {
    "greedy": mate.GreedyTargetAgent,
    "heuristic": mate.HeuristicTargetAgent,
    "naive": mate.NaiveTargetAgent,
    "random": mate.RandomTargetAgent,
}


class MateEnv(MultiAgentEnv):

    def __init__(self, batch_size=None, **kwargs):
        if "env_args" in kwargs:
            args = kwargs["env_args"]
            if isinstance(args, dict):
                args = convert(args)
        else:
            args = convert(kwargs)
        self.args = args

        self.mate_config = getattr(args, "mate_config", "MATE-4v8-9.yaml")
        self.levels = int(getattr(args, "levels", 5))
        self.episode_limit = int(getattr(args, "episode_limit", 1000))
        self.frame_skip = int(getattr(args, "frame_skip", 1))
        if self.frame_skip < 1:
            raise ValueError(f"frame_skip must be positive, got {self.frame_skip}.")
        self.relative_coordinates = bool(
            getattr(args, "relative_coordinates", False)
        )
        self.rescale_observation = bool(
            getattr(args, "rescale_observation", False)
        )
        self.repeat_reward_individual_done = bool(
            getattr(args, "repeat_reward_individual_done", False)
        )
        self.coverage_reward = bool(getattr(args, "coverage_reward", False))
        self.coverage_reward_coefficient = float(
            getattr(args, "coverage_reward_coefficient", 1.0)
        )
        self.reward_reduction = str(
            getattr(args, "reward_reduction", "mean")
        ).lower()
        seed = getattr(args, "seed", None)
        self.seed_value = seed
        target_agent_seed = getattr(args, "target_agent_seed", seed)
        if target_agent_seed is None:
            target_agent_seed = 0

        target_agent_name = getattr(args, "target_agent", "greedy").lower()
        assert target_agent_name in _TARGET_AGENTS, (
            f"Unknown target_agent {target_agent_name!r}. "
            f"Choose from {list(_TARGET_AGENTS.keys())}."
        )
        target_agent_cls = _TARGET_AGENTS[target_agent_name]

        # Match the preprocessing order used by the official MATE camera
        # baselines: discrete actions -> single camera team -> relative
        # coordinates -> rescaled observations -> repeated team reward/done ->
        # shared coverage reward. Frame skipping is handled in step() because it
        # is implemented by the benchmark's RLlib adapter rather than MATE.
        base_env = mate.make(
            "MultiAgentTracking-v0",
            config=self.mate_config,
        )
        base_env = mate.DiscreteCamera(base_env, levels=self.levels)
        self.env = mate.MultiCamera(
            base_env,
            target_agent=target_agent_cls(seed=int(target_agent_seed)),
        )
        if self.relative_coordinates:
            self.env = mate.RelativeCoordinates(self.env)
        if self.rescale_observation:
            self.env = mate.RescaledObservation(self.env)
        if self.repeat_reward_individual_done or self.coverage_reward:
            self.env = mate.RepeatedRewardIndividualDone(self.env)
        if self.coverage_reward:
            self.env = mate.AuxiliaryCameraRewards(
                self.env,
                coefficients={"coverage_rate": self.coverage_reward_coefficient},
                reduction=self.reward_reduction,
            )
        if seed is not None:
            self.env.seed(int(seed))

        self.n_agents = self.env.num_teammates
        self.n_actions = self.levels * self.levels
        self.obs_size = int(np.prod(self.env.observation_space.spaces[0].shape))
        self.state_size = self.n_agents * self.obs_size

        self._obs = None
        self.steps = 0
        self.coverage_rate_sum = 0.0
        self.coverage_rate_samples = 0

    def reset(self):
        obs = self.env.reset()
        self._obs = np.asarray(obs, dtype=np.float32)
        self.steps = 0
        self.coverage_rate_sum = 0.0
        self.coverage_rate_samples = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64).ravel()
        reward = 0.0
        terminated = False

        # The official MATE baselines repeat one selected action for five raw
        # simulator frames and sum the fragment rewards.
        for _ in range(self.frame_skip):
            obs, fragment_reward, done, _ = self.env.step(actions)
            fragment_reward = np.asarray(fragment_reward, dtype=np.float64)
            reward += float(fragment_reward.mean())

            self.coverage_rate_sum += float(self.env.coverage_rate)
            self.coverage_rate_samples += 1

            done = np.asarray(done, dtype=np.bool_)
            terminated = bool(done.all())
            if terminated:
                break

        self._obs = np.asarray(obs, dtype=np.float32)
        self.steps += 1

        info = {
            "episode_limit": False,
            "coverage_rate": self.coverage_rate_sum / self.coverage_rate_samples,
        }
        if self.steps >= self.episode_limit:
            terminated = True
            info["episode_limit"] = True
        return reward, terminated, info

    def get_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return self.obs_size

    def get_state(self):
        return np.concatenate(self._obs)

    def get_state_size(self):
        return self.state_size

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return [1 for _ in range(self.n_actions)]

    def get_total_actions(self):
        return self.n_actions

    def get_stats(self):
        return {}

    def get_env_info(self):
        return super().get_env_info()

    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.seed_value = seed
        self.env.seed(seed)

    def save_replay(self):
        pass
