"""Execution tremble: agents choose intended actions, the env executes K_eps(. | intended).

The episode buffer keeps the intended actions returned by the controller; only the
executed ones reach the wrapped env. For deterministic matrix games the wrapper also
reports the exact expected reward of the intended joint action under tremble
(``intended_value``) and whether it is optimal in G_eps (``intended_optimal``), so
evaluation is free of the sampling noise of ``return_mean``.
"""
import numpy as np

from src.tremble.games import MatrixGame, noisy_payoff, tremble


class TrembleWrapper:
    def __init__(self, env, tremble_eps, seed=None):
        self.env = env
        self.tremble_eps = float(tremble_eps)
        self.rng = np.random.default_rng(seed)
        self.n_actions = env.get_total_actions()
        self.noisy_table = None
        table = getattr(env, "matrix_table", None)
        if table is not None and np.asarray(table).dtype != object and np.asarray(table).ndim == env.n_agents:
            game = MatrixGame(type(env).__name__, np.asarray(table, dtype=float)[..., None])
            self.noisy_table = noisy_payoff(game, self.tremble_eps)

    def step(self, actions):
        intended = np.asarray(actions, dtype=np.int64).reshape(-1)
        executed = tremble(intended, self.tremble_eps, self.n_actions, self.rng)
        reward, terminated, info = self.env.step(executed)
        if self.noisy_table is not None:
            value = self.noisy_table[tuple(intended)]
            info["intended_value"] = float(value)
            info["intended_optimal"] = float(value >= self.noisy_table.max() - 1e-9)
        return reward, terminated, info

    def __getattr__(self, name):
        return getattr(self.env, name)
