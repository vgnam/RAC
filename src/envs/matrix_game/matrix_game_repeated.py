from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.envs.multiagentenv import MultiAgentEnv

import numpy as np


# Each cell lists the equally likely payoffs of a joint action.
PAYOFF_TABLES = {
    "climbing": [[[11], [-30], [0]], [[-30], [7], [6]], [[0], [0], [5]]],
    # Kapetanakis & Kudenko (2002): (b, b) pays 14 or 0, expected 7.
    "partially_stochastic_climbing": [[[11], [-30], [0]], [[-30], [14, 0], [6]], [[0], [0], [5]]],
    "penalty": [[[10], [0], [-15]], [[0], [2], [0]], [[-15], [0], [10]]],
}


class Matrix_Game_Repeated_Env(MultiAgentEnv):
    """A one-shot matrix game repeated for ``episode_limit`` steps.

    Teammates keep their (greedy) policy for the whole episode, so the
    teammate behaviour is persistent within an episode while the payoff noise
    of stochastic cells is drawn independently at every step.
    """

    def __init__(
            self,
            game="climbing",
            n_agents=2,
            n_actions=3,
            episode_limit=2,
            obs_last_action=False,
            state_last_action=False,
            is_print=False,
            print_rew=False,
            print_steps=1000,
            seed=None,
            **kwargs
    ):
        if game not in PAYOFF_TABLES:
            raise ValueError("Unknown repeated matrix game: {}".format(game))
        self.game = game
        self.matrix_table = PAYOFF_TABLES[game]
        self.n_agents = 2
        self.n_actions = 3
        self.episode_limit = int(episode_limit)
        self._rng = np.random.RandomState(seed)

        self._episode_steps = 0
        self._episode_count = 0
        self._all_optimal = True
        self.battles_won = 0
        self.battles_game = 0

        self.unit_dim = 1

    def step(self, actions):
        """Returns reward, terminated, info."""
        a0, a1 = int(actions[0]), int(actions[1])
        cell = self.matrix_table[a0][a1]
        reward = float(cell[self._rng.randint(len(cell))])

        self._episode_steps += 1
        self._all_optimal = self._all_optimal and a0 == 0 and a1 == 0

        terminated = self._episode_steps >= self.episode_limit
        info = {}
        if terminated:
            info["battle_won"] = self._all_optimal
            self._episode_count += 1
            self.battles_game += 1
            self.battles_won += int(self._all_optimal)
        return reward, terminated, info

    def get_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        return np.array([self._episode_steps])

    def get_obs_size(self):
        return 1

    def get_state(self):
        return np.array([0 for _ in range(self.n_agents)])

    def get_state_size(self):
        return self.n_agents

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return [1] * self.n_actions

    def get_total_actions(self):
        return self.n_actions

    def reset(self):
        self._episode_steps = 0
        self._all_optimal = True
        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        pass

    def get_env_info(self):
        return {"state_shape": self.get_state_size(),
                "obs_shape": self.get_obs_size(),
                "n_actions": self.get_total_actions(),
                "n_agents": self.n_agents,
                "episode_limit": self.episode_limit,
                "unit_dim": self.unit_dim}

    def get_stats(self):
        return {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / max(self.battles_game, 1),
        }
