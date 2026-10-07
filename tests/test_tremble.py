import numpy as np
import pytest

from src.tremble import games as G
from src.tremble import learners as L


@pytest.mark.parametrize("eps,diag", [
    (0.0, [11.0, 7.0, 5.0]),
    (0.05, [8.51, 5.05, 4.63]),
    (0.10, [6.26, 3.28, 4.22]),
    (0.15, [4.22, 1.71, 3.76]),
    (0.20, [2.42, 0.32, 3.26]),
])
def test_climbing_noisy_diagonal_matches_proposal(eps, diag):
    table = G.noisy_payoff(G.climbing(), eps)
    assert np.allclose(np.diag(table), diag, atol=0.006)


def test_climbing_switch_at_one_sixth():
    table = G.noisy_payoff(G.climbing(), 1 / 6)
    assert table[0, 0] == pytest.approx(table[2, 2])


@pytest.mark.parametrize("n,all0,all1", [(2, 9.32, 4.82), (4, 6.47, 4.62), (8, 3.27, 4.33)])
def test_n_agent_game_matches_proposal(n, all0, all1):
    table = G.noisy_payoff(G.n_agent_coordination(n), 0.05)
    assert table[(0,) * n] == pytest.approx(all0, abs=0.006)
    assert table[(1,) * n] == pytest.approx(all1, abs=0.006)


def test_counterexample_3_values():
    fixed = L.exact_fixed_points(G.asymmetric_2x2(), 0.25)
    assert np.allclose(fixed["TR-IQL (ours)"], [[4.25, 3.25], [4.25, 3.25]])
    assert np.allclose(fixed["Hurwicz per agent (eta=(1-eps)^(n-1))"], [[5, 6], [7.5, 3.5]])


def test_tremble_sampling_matches_kernel():
    rng = np.random.default_rng(0)
    executed = G.tremble(np.zeros((200000, 1), dtype=int), 0.3, 3, rng)
    freq = np.bincount(executed[:, 0], minlength=3) / len(executed)
    assert np.allclose(freq, G.tremble_kernel(3, 0.3)[0], atol=0.005)


def test_tr_iql_recovers_fixed_point_on_deterministic_game():
    rng = np.random.default_rng(0)
    game = G.climbing()
    q = L.tremble_robust_iql(game, 0.0, rng, n_blocks=200, block_len=20)
    assert np.allclose(q, G.individual_optimum(game.mean_payoff))
