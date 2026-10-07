"""Common-payoff matrix games and the exact tremble-perturbed game G_eps.

Payoff tables mirror ``src/envs/matrix_game``. A game stores, for every joint
action, K equally likely reward outcomes (K = 1 for deterministic games).
"""
from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class MatrixGame:
    name: str
    outcomes: np.ndarray  # shape (A,) * n + (K,)

    @property
    def n_agents(self):
        return self.outcomes.ndim - 1

    @property
    def n_actions(self):
        return self.outcomes.shape[0]

    @property
    def mean_payoff(self):
        return self.outcomes.mean(axis=-1)

    def sample_rewards(self, joint_actions, rng):
        """joint_actions: int array (L, n) -> rewards (L,)."""
        cells = self.outcomes[tuple(joint_actions.T)]  # (L, K)
        pick = rng.integers(cells.shape[1], size=len(cells))
        return cells[np.arange(len(cells)), pick].astype(float)


def _deterministic(name, table):
    return MatrixGame(name, np.asarray(table, dtype=float)[..., None])


def climbing():
    return _deterministic("climbing", [[11, -30, 0], [-30, 7, 6], [0, 0, 5]])


def penalty(k=-15):
    return _deterministic("penalty", [[10, 0, k], [0, 2, 0], [k, 0, 10]])


def matrix_game_3():
    return _deterministic("matrix_game_3", [[8, -12, -12], [-12, 6, 0], [-12, 0, 6]])


def asymmetric_2x2():
    """Counterexample 3 of the proposal: u(A,A)=10, u(A,B)=-10, u(B,A)=0, u(B,B)=8."""
    return _deterministic("asymmetric_2x2", [[10, -10], [0, 8]])


def climbing_partially_stochastic():
    """(b,b) pays 14 or 0 with equal probability (mean 7)."""
    out = np.array([[[11, 11], [-30, -30], [0, 0]],
                    [[-30, -30], [14, 0], [6, 6]],
                    [[0, 0], [0, 0], [5, 5]]], dtype=float)
    return MatrixGame("climbing_partial_stoch", out)


def climbing_fully_stochastic():
    out = np.array([[[10, 12], [-65, 5], [8, -8]],
                    [[-65, 5], [14, 0], [12, 0]],
                    [[8, -8], [12, 0], [10, 0]]], dtype=float)
    return MatrixGame("climbing_full_stoch", out)


def n_agent_coordination(n):
    """All 0 -> 10, all 1 -> 5, mixed with fewer than half choosing 1 -> -10, else 3."""
    table = np.empty((2,) * n)
    for joint in product(range(2), repeat=n):
        ones = sum(joint)
        if ones == 0:
            table[joint] = 10
        elif ones == n:
            table[joint] = 5
        elif ones < n / 2:
            table[joint] = -10
        else:
            table[joint] = 3
    return _deterministic("n_agent_{}".format(n), table)


def tremble_kernel(n_actions, eps):
    """K[intended, executed]: keep with prob 1-eps, slip uniformly otherwise."""
    if n_actions == 1:
        return np.ones((1, 1))
    k = np.full((n_actions, n_actions), eps / (n_actions - 1))
    np.fill_diagonal(k, 1.0 - eps)
    return k


def noisy_payoff(game, eps):
    """r_eps over intended joint actions: contract the kernel along every agent axis."""
    k = tremble_kernel(game.n_actions, eps)
    table = game.mean_payoff
    for axis in range(game.n_agents):
        table = np.moveaxis(np.tensordot(k, table, axes=([1], [axis])), 0, axis)
    return table


def individual_optimum(table):
    """Q_i(a_i) = max over teammates' intentions, for every agent: (n, A)."""
    n = table.ndim
    return np.stack([
        table.max(axis=tuple(j for j in range(n) if j != i)) for i in range(n)
    ])


def tremble(intended, eps, n_actions, rng):
    """Sample executed actions from K_eps(. | intended); intended: int array (..., n)."""
    if eps == 0 or n_actions == 1:
        return intended.copy()
    slip = rng.random(intended.shape) < eps
    offset = rng.integers(1, n_actions, size=intended.shape)
    return np.where(slip, (intended + offset) % n_actions, intended)
