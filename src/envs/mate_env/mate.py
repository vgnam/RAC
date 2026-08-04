import numpy as np

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
        seed = getattr(args, "seed", None)
        self.seed_value = seed

        target_agent_name = getattr(args, "target_agent", "greedy").lower()
        assert target_agent_name in _TARGET_AGENTS, (
            f"Unknown target_agent {target_agent_name!r}. "
            f"Choose from {list(_TARGET_AGENTS.keys())}."
        )
        target_agent_cls = _TARGET_AGENTS[target_agent_name]

        self.env = mate.make(
            "MultiAgentTracking-v0",
            config=self.mate_config,
            wrappers=[
                mate.WrapperSpec(mate.DiscreteCamera, levels=self.levels),
                mate.WrapperSpec(
                    mate.MultiCamera,
                    target_agent=target_agent_cls(seed=0 if seed is None else int(seed)),
                ),
            ],
        )
        if seed is not None:
            self.env.seed(int(seed))

        self.n_agents = self.env.num_teammates
        self.n_actions = self.levels * self.levels
        self.obs_size = int(np.prod(self.env.observation_space.spaces[0].shape))
        self.state_size = self.n_agents * self.obs_size

        self._obs = None
        self.steps = 0

    def reset(self):
        obs = self.env.reset()
        self._obs = np.asarray(obs, dtype=np.float32)
        self.steps = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64).ravel()
        obs, reward, done, _ = self.env.step(actions)
        self._obs = np.asarray(obs, dtype=np.float32)
        self.steps += 1

        terminated = bool(done)
        info = {"episode_limit": False}
        if self.steps >= self.episode_limit:
            terminated = True
            info["episode_limit"] = True
        return float(reward), terminated, info

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
