import unittest
from types import SimpleNamespace

import torch as th
import torch.nn.functional as F

from src.envs.matrix_game.matrix_game_repeated import Matrix_Game_Repeated_Env
from src.modules.agents.new_ree_agent import HyperREEAgent
from src.modules.contrastive_context import ContrastiveContextLabeler


class _Batch:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]


def _batch(actions, rewards):
    """actions: (bs, T, n_agents) ints, rewards: (bs, T) floats."""
    bs, steps, n_agents = actions.shape
    obs = th.arange(steps + 1, dtype=th.float32).view(1, -1, 1, 1).expand(bs, -1, n_agents, 1)
    padded_actions = th.cat([actions, th.zeros(bs, 1, n_agents, dtype=th.long)], dim=1)
    reward = th.cat([rewards, th.zeros(bs, 1)], dim=1).unsqueeze(-1)
    return _Batch({
        "obs": obs.contiguous(),
        "actions_onehot": F.one_hot(padded_actions, 3).float(),
        "reward": reward,
    })


def _args(**overrides):
    args = dict(slot_number=4, seed=0, corac_segment_len=1, corac_n_hard=2,
                corac_n_freq=8, corac_embed_dim=8, context_dim=16, corac_bank_size=8)
    args.update(overrides)
    return SimpleNamespace(**args)


def _agent_args(**overrides):
    args = dict(slot_number=4, rnn_hidden_dim=8, n_actions=3, agent_recurrent=True)
    args.update(overrides)
    return SimpleNamespace(**args)


class RepeatedMatrixGameTest(unittest.TestCase):
    def test_climbing_pays_per_step_and_terminates(self):
        env = Matrix_Game_Repeated_Env(game="climbing", episode_limit=3, seed=0)
        env.reset()
        total = 0.0
        for step in range(3):
            reward, terminated, info = env.step([0, 0])
            total += reward
            self.assertEqual(terminated, step == 2)
        self.assertEqual(total, 33.0)
        self.assertTrue(info["battle_won"])

    def test_partially_stochastic_cell_is_14_or_0(self):
        env = Matrix_Game_Repeated_Env(game="partially_stochastic_climbing", episode_limit=200, seed=0)
        env.reset()
        rewards = {env.step([1, 1])[0] for _ in range(200)}
        self.assertEqual(rewards, {0.0, 14.0})


class ContrastiveContextLabelerTest(unittest.TestCase):
    def setUp(self):
        th.manual_seed(0)
        bs, steps = 16, 4
        self.actions = th.randint(0, 3, (bs, steps, 2))
        self.rewards = th.randn(bs, steps)
        self.batch = _batch(self.actions, self.rewards)
        self.mask = th.ones(bs, steps, 1)

    def test_contexts_shape_and_return_features(self):
        labeler = ContrastiveContextLabeler(1, 3, 2, _args())
        contexts, loss, stats = labeler.label(self.batch, self.mask, t_env=0)
        self.assertEqual(contexts.shape, (16, 5, 2, 16))
        self.assertTrue(th.isfinite(loss))
        # phi(R) is shared by the agents of an episode and identical over time.
        phi = contexts[..., :8]
        self.assertTrue(th.equal(phi[:, :, 0], phi[:, :, 1]))
        self.assertTrue(th.equal(phi[:, 0], phi[:, -1]))
        # The first cosine frequency is the constant 1; the second is monotone in R.
        returns = self.rewards.sum(1)
        order = returns.argsort()
        second = phi[order, 0, 0, 1]
        self.assertTrue(bool((second[1:] <= second[:-1] + 1e-6).all()))

    def test_bank_is_fifo_and_bounded(self):
        labeler = ContrastiveContextLabeler(1, 3, 2, _args(corac_bank_size=8))
        for _ in range(3):
            labeler.label(self.batch, self.mask, t_env=0)
        support = labeler.support()
        self.assertEqual(support.shape, (2, 8, 16))
        contexts, _, _ = labeler.label(self.batch, self.mask, t_env=0)
        # The newest contexts are in the bank.
        newest = contexts[-8:, 0].transpose(0, 1)  # (n_agents, 8, C)
        self.assertTrue(th.allclose(labeler.support().sort(dim=1)[0], newest.sort(dim=1)[0]))

    def test_short_episodes_get_zero_trajectory_embedding(self):
        batch = _batch(self.actions[:, :1], self.rewards[:, :1])
        labeler = ContrastiveContextLabeler(1, 3, 2, _args())
        contexts, loss, _ = labeler.label(batch, self.mask[:, :1], t_env=0)
        self.assertIsNone(loss)  # no two disjoint windows in a 1-step episode
        self.assertTrue(th.equal(contexts[..., 8:], th.zeros_like(contexts[..., 8:])))

    def test_outcome_swap_keeps_state_and_never_reproduces_anchor(self):
        rewards = th.randint(0, 2, (16, 4)).float()
        batch = _batch(self.actions, rewards)
        labeler = ContrastiveContextLabeler(1, 3, 2, _args(corac_n_hard=4))
        features = labeler.step_features(batch)
        anchor = features[:, :1].transpose(1, 2).reshape(-1, 1, features.shape[-1])
        episode_of_row = th.arange(16).repeat_interleave(2)
        agent_of_row = th.arange(2).repeat(16)
        hard, valid = labeler._outcome_swapped_negatives(
            features, self.mask[:, :, 0], anchor, episode_of_row, agent_of_row
        )
        start = labeler.n_state_features
        stop = start + labeler.n_outcome_features
        self.assertTrue(bool(valid.any()))
        state_part = hard[..., :start]
        self.assertTrue(th.equal(state_part, anchor[:, None, :, :start].expand_as(state_part)))
        same = (hard[..., start:stop] == anchor[:, None, :, start:stop]).all(-1).all(-1)
        self.assertFalse(bool((same & valid).any()))


class HyperTwinTest(unittest.TestCase):
    def setUp(self):
        th.manual_seed(0)
        self.inputs = th.randn(2, 3, 2, 5)  # (bs, T, n_agents, F)

    def _per_step(self, agent, contexts):
        """Reference: run forward() step by step for each context."""
        bs, steps, n_agents, _ = self.inputs.shape
        out = []
        for c in contexts:
            h = agent.init_hidden().expand(bs * n_agents, -1)
            qs = []
            for t in range(steps):
                q, h = agent(self.inputs[:, t].reshape(bs * n_agents, -1),
                             c.expand(bs * n_agents, -1), h)
                qs.append(q.view(bs, n_agents, -1))
            out.append(th.stack(qs, dim=1))
        return th.stack(out, dim=3)  # (bs, T, n_agents, M, A)

    def test_rac_slots_unchanged(self):
        agent = HyperREEAgent(5, _agent_args())
        hidden = agent.init_hidden().expand(2, 2, -1)
        q_all, _ = agent.counterfactual_sequence(self.inputs, hidden)
        reference = self._per_step(agent, th.eye(4))
        self.assertTrue(th.allclose(q_all, reference, atol=1e-5))

    def test_context_bank_matches_forward_and_starts_without_optimism(self):
        agent = HyperREEAgent(5, _agent_args(context_dim=6, twin_hyper_hidden=16))
        contexts = th.randn(5, 6)
        hidden = agent.init_hidden().expand(2, 2, -1)
        q_all, _ = agent.counterfactual_sequence(self.inputs, hidden, contexts=contexts)
        self.assertEqual(q_all.shape, (2, 3, 2, 5, 3))
        self.assertTrue(th.allclose(q_all, self._per_step(agent, contexts), atol=1e-5))
        # Zero-delta init: every context shares the base head at the start.
        self.assertTrue(th.allclose(q_all, q_all[..., :1, :].expand_as(q_all)))

    def test_per_agent_bank(self):
        agent = HyperREEAgent(5, _agent_args(context_dim=6, twin_hyper_hidden=16))
        with th.no_grad():
            agent.hyper_head.delta[-1].weight.normal_()
        bank = th.randn(2, 4, 6)  # (n_agents, M, C)
        hidden = agent.init_hidden().expand(2, 2, -1)
        q_all, _ = agent.counterfactual_sequence(self.inputs, hidden, contexts=bank)
        self.assertEqual(q_all.shape, (2, 3, 2, 4, 3))
        for agent_id in range(2):
            reference = self._per_step(agent, bank[agent_id])[:, :, agent_id]
            self.assertTrue(th.allclose(q_all[:, :, agent_id], reference, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
