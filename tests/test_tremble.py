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


def test_tremble_learner_fused_unroll_matches_stepwise():
    from types import SimpleNamespace

    import torch as th

    from src.controllers.basic_controller import BasicMAC
    from src.learners.tremble_iql_learner import TrembleIQLLearner
    from src.modules.agents.rnn_agent import RNNAgent

    th.manual_seed(0)
    bs, T, n, obs_dim, n_actions = 3, 7, 4, 5, 6
    args = SimpleNamespace(rnn_hidden_dim=16, n_actions=n_actions,
                           obs_last_action=True, obs_agent_id=True)
    actions = th.randint(n_actions, (bs, T, n))
    data = {"obs": th.randn(bs, T, n, obs_dim),
            "actions_onehot": th.nn.functional.one_hot(actions, n_actions).float()}
    batch = type("Batch", (), {"__getitem__": lambda self, k: data[k],
                               "batch_size": bs, "max_seq_length": T, "device": "cpu"})()
    mac = SimpleNamespace(args=args, n_agents=n)
    mac._build_inputs = lambda b, t: BasicMAC._build_inputs(mac, b, t)
    mac._build_sequence_inputs = lambda b: BasicMAC._build_sequence_inputs(mac, b)
    agent = RNNAgent(obs_dim + n_actions + n, args)
    learner = SimpleNamespace(mac=mac, n_agents=n)
    learner._unroll_stepwise = lambda a, b: TrembleIQLLearner._unroll_stepwise(learner, a, b)

    fused = TrembleIQLLearner._unroll(learner, agent, batch)
    stepwise = TrembleIQLLearner._unroll_stepwise(learner, agent, batch)
    assert fused.shape == (bs, T, n, n_actions)
    assert th.allclose(fused, stepwise, atol=1e-6)


def _block_learner(block_len=4, pool_z=3.0, beta=0.1, var_floor=1e-4, buffer_size=64):
    from types import SimpleNamespace

    import torch as th

    from src.learners.tremble_block_learner import TrembleBlockLearner

    args = SimpleNamespace(n_agents=2, lr=1e-3, optim_alpha=0.99, optim_eps=1e-5, block_len=block_len,
                           pool_z=pool_z, block_beta=beta, block_var_floor=var_floor, learner_log_interval=1)
    agent = th.nn.Linear(3, 2)
    mac = SimpleNamespace(agent=agent, parameters=agent.parameters, blocks=0)
    mac.new_block = lambda t_env: setattr(mac, "blocks", mac.blocks + 1)
    learner = TrembleBlockLearner(mac, None, None, args)
    buffer = SimpleNamespace(buffer_size=buffer_size, buffer_index=0)
    learner.set_buffer(buffer)
    return learner, buffer


def _feed_block(learner, buffer, returns):
    import torch as th

    for r in returns:
        buffer.buffer_index = (buffer.buffer_index + 1) % buffer.buffer_size  # run.py inserts first
        ep = type("Ep", (), {"batch_size": 1,
                             "__getitem__": lambda self, k, r=r: th.tensor([[[float(r)], [0.0]]])})()
        learner.observe_episode(ep, t_env=0)


def test_block_learner_pools_blocks_by_mean_return():
    learner, buffer = _block_learner()
    _feed_block(learner, buffer, [0.0, 1.0, 0.0, 1.0])      # first block is adopted
    assert learner.decisions["replace"] == 1 and learner.mac.blocks == 1
    _feed_block(learner, buffer, [0.0, 1.0, 1.0, 0.0])      # same mean: merged
    assert learner.decisions["merge"] == 1
    assert learner.best[1] < learner.block_stats[0][1]      # pooled variance shrinks
    _feed_block(learner, buffer, [-9.0, -10.0, -9.0, -10.0])  # clearly worse: ignored
    assert learner.decisions["ignore"] == 1
    w = learner.slot_weight
    assert (w[learner.slot_block == 0] == 1.0).all() and (w[learner.slot_block == 2] == 0.1).all()
    _feed_block(learner, buffer, [9.0, 10.0, 9.0, 10.0])    # clearly better: replaces, old blocks drop to beta
    assert learner.decisions["replace"] == 2
    assert (learner.slot_weight[learner.slot_block == 0] == 0.1).all()
    assert (learner.slot_weight[learner.slot_block == 3] == 1.0).all()


def test_block_learner_unfinished_block_has_zero_weight():
    learner, buffer = _block_learner()
    _feed_block(learner, buffer, [1.0] * 4)
    _feed_block(learner, buffer, [1.0, 1.0])                # half a block
    pending = learner.slot_block[:buffer.buffer_index] == -1
    assert pending.sum() == 2 and (learner.slot_weight[:buffer.buffer_index][pending] == 0).all()


def test_block_learner_calibration_moves_beta_towards_ceiling():
    learner, buffer = _block_learner(var_floor=1.0)
    learner.args.calib_kappa, learner.args.calib_lr, learner.args.hyst_beta_min = 1.0, 0.5, 0.01
    learner.args.calib_bonus, learner.args.calib_bonus_anneal = 0.0, 1
    learner.beta_neg, learner.best = 0.1, (5.0, 1.0)            # ceiling = 5 + 1 = 6
    learner._update_beta(v_opt=9.0, t_env=0)                     # over-optimistic: less optimism
    assert learner.beta_neg > 0.1
    raised = learner.beta_neg
    learner._update_beta(v_opt=3.0, t_env=0)                     # under the ceiling: more optimism
    assert learner.beta_neg < raised
    for _ in range(50):
        learner._update_beta(v_opt=100.0, t_env=0)
    assert learner.beta_neg == 1.0                               # capped at plain (nominal) TD
    learner.args.calib_bonus, learner.args.calib_bonus_anneal = 10.0, 100
    learner._update_beta(v_opt=10.0, t_env=0)                    # the bonus lifts the ceiling to 16
    assert learner.ceiling == 16.0 and learner.beta_neg < 1.0
