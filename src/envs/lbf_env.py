"""Level-Based Foraging (lbforaging, gymnasium API) as a pymarl MultiAgentEnv.

The team reward is the sum of the agents' rewards, so an episode return lies in
[0, 1] (1 = every food item collected).
"""
import gymnasium as gym
import lbforaging  # noqa: F401  (registers the Foraging-* environments)
import numpy as np

from .multiagentenv import MultiAgentEnv


class LBFEnv(MultiAgentEnv):
    def __init__(self, key="Foraging-8x8-2p-2f-coop-v3", time_limit=50, seed=None, **kwargs):
        self._env = gym.make(key, max_episode_steps=time_limit)
        self._seed = seed
        self.n_agents = self._env.unwrapped.n_agents
        self.n_actions = int(self._env.action_space[0].n)
        self.episode_limit = time_limit
        self._obs = None
        self._episode_count = 0

    def reset(self):
        seed = None
        if self._seed is not None:
            seed = self._seed + self._episode_count
        self._episode_count += 1
        obs, _ = self._env.reset(seed=seed)
        self._obs = [np.asarray(o, dtype=np.float32) for o in obs]
        return self.get_obs(), self.get_state()

    def step(self, actions):
        actions = [int(a) for a in np.asarray(actions).reshape(-1)]
        obs, rewards, terminated, truncated, _ = self._env.step(actions)
        self._obs = [np.asarray(o, dtype=np.float32) for o in obs]
        info = {"episode_limit": bool(truncated and not terminated)}
        return float(np.sum(rewards)), bool(terminated or truncated), info

    def get_obs(self):
        return list(self._obs)

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return int(self._env.observation_space[0].shape[0])

    def get_state(self):
        return np.concatenate(self._obs)

    def get_state_size(self):
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return [1] * self.n_actions

    def get_total_actions(self):
        return self.n_actions

    def get_stats(self):
        return {}

    def render(self):
        pass

    def close(self):
        self._env.close()

    def seed(self):
        return self._seed

    def save_replay(self):
        pass

    def get_env_info(self):
        return {"state_shape": self.get_state_size(),
                "obs_shape": self.get_obs_size(),
                "n_actions": self.get_total_actions(),
                "n_agents": self.n_agents,
                "episode_limit": self.episode_limit}
