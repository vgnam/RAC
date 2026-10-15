import unittest
from types import SimpleNamespace

import torch as th
import torch.nn.functional as F

from src.envs.matrix_game.matrix_game_repeated import Matrix_Game_Repeated_Env
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
    args = dict(slot_number=4, seed=0, corac_segment_len=1, corac_gap=0,
                corac_n_hard=2, corac_lambda_final=0.0, corac_warmup_steps=0,
                corac_blend_steps=1)
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

    def test_lambda_one_ranks_by_return(self):
        labeler = ContrastiveContextLabeler(1, 3, 2, _args(corac_lambda_final=1.0))
        for _ in range(3):  # fill the running CDF
            _, context, loss, _ = labeler.label(self.batch, self.mask, t_env=0)
        self.assertIsNone(loss)
        returns = self.rewards.sum(1)
        order = returns.argsort()
        ranked = context[order, 0]
        self.assertTrue(bool((ranked[1:] >= ranked[:-1]).all()))

    def test_outcome_swap_keeps_state_and_action(self):
        labeler = ContrastiveContextLabeler(1, 3, 2, _args())
        features = labeler.step_features(self.batch)
        anchor = features[:2, :1].transpose(1, 2).reshape(-1, 1, features.shape[-1])
        episode_of_row = th.tensor([0, 0, 1, 1])
        agent_of_row = th.tensor([0, 1, 0, 1])
        hard, valid = labeler._outcome_swapped_negatives(
            features, self.mask[:, :, 0], anchor, episode_of_row, agent_of_row
        )
        n_state = labeler.n_state_features
        self.assertTrue(valid.all())
        state_part = hard[..., :n_state]
        self.assertTrue(th.equal(state_part, anchor[:, None, :, :n_state].expand_as(state_part)))
        # Agent-id block is untouched as well.
        id_start = n_state + labeler.n_outcome_features
        self.assertTrue(th.equal(hard[..., id_start:], anchor[:, None, :, id_start:].expand_as(hard[..., id_start:])))

    def test_outcome_swap_never_reproduces_anchor(self):
        # Discrete payoffs: most same-action neighbours share the anchor's reward.
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
        same = (hard[..., start:stop] == anchor[:, None, :, start:stop]).all(-1).all(-1)
        self.assertFalse(bool((same & valid).any()))
        self.assertTrue(bool(valid.any()))

    def test_corac_trains_and_returns_valid_contexts(self):
        labeler = ContrastiveContextLabeler(1, 3, 2, _args())
        optimiser = th.optim.Adam(labeler.parameters(), lr=1e-3)
        onehot, context, loss, stats = labeler.label(self.batch, self.mask, t_env=10)
        self.assertEqual(onehot.shape, (16, 5, 2, 4))
        self.assertTrue(th.isfinite(loss))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        self.assertTrue(((context >= 0) & (context < 4)).all())
        self.assertIn("corac_nce_accuracy", stats)


if __name__ == "__main__":
    unittest.main()
