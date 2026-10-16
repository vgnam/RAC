import os
import unittest
from types import SimpleNamespace

import torch as th
import torch.nn.functional as F

from src.controllers.dual_ree_controller import DualREEMAC
from src.envs.matrix_game.matrix_game_repeated import Matrix_Game_Repeated_Env
from src.learners.dual_episode_ree_q_learner import DualEpisodeREEQLearner
from src.modules.agents.new_ree_agent import HyperREEAgent
from src.modules.contrastive_context import ContrastiveContextLabeler


class _Batch:
    def __init__(self, data):
        self.data = data
        self.batch_size = data["obs"].shape[0]
        self.max_seq_length = data["obs"].shape[1]

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

    def test_return_std_floor_keeps_phi_informative_when_returns_concentrate(self):
        """MATE late training: return level ~500, batch spread ~10.

        Without a floor relative to the level, z explodes and phi collapses into
        a constant binary code, so every context looks the same.
        """
        saturations = {}
        for min_std_frac in (0.0, 0.05):
            labeler = ContrastiveContextLabeler(
                1, 3, 2, _args(corac_min_std_frac=min_std_frac)
            )
            with th.no_grad():
                labeler.return_mean.fill_(500.0)
                labeler.return_var.fill_(1e-8)  # collapsed EMA spread
                labeler.stats_initialised.fill_(True)
            returns = th.tensor([490.0, 500.0, 510.0])
            phi = labeler.return_features(returns)
            saturations[min_std_frac] = float((phi.abs() > 0.999).float().mean())
            # Return ordering must survive the floor.
            self.assertLessEqual(phi[2, 1].item(), phi[0, 1].item() + 1e-6)

        # No floor: z is ~1e4, sigmoid saturates to {0, 0.5, 1} and phi becomes a
        # binary code. With the floor phi stays a smooth, order preserving code.
        self.assertGreater(saturations[0.0], 0.8)
        self.assertLess(saturations[0.05], 0.5)

    def test_pair_sampler_handles_episodes_holding_barely_two_windows(self):
        """length == 2 * segment_len: interior starts have no disjoint partner.

        The sampler used to draw one and then index an empty candidate list.
        """
        for seg, length in ((2, 4), (3, 6), (3, 7), (3, 8), (16, 32), (16, 33)):
            labeler = ContrastiveContextLabeler(1, 3, 2, _args(corac_segment_len=seg))
            lengths = th.full((4,), length)
            for _ in range(60):
                rows, t1s, t2s = labeler._sample_pairs(lengths)
                self.assertEqual(len(rows), 4)
                for t1, t2 in zip(t1s, t2s):
                    self.assertGreaterEqual(abs(t1 - t2), seg, (seg, length, t1, t2))
                    self.assertGreaterEqual(min(t1, t2), 0)
                    self.assertLessEqual(max(t1, t2) + seg, length)

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


class _FakeMac:
    """Minimal MAC stand-in so the learner can be built without an env."""

    def __init__(self, context_dim):
        self.twin_agent = SimpleNamespace(context_dim=context_dim)
        self._parameter = th.nn.Parameter(th.zeros(1))

    def parameters(self):
        return [self._parameter]

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass


def _learner_args(**overrides):
    args = dict(
        env="matrix_game_3", slot_number=4, n_actions=3, n_agents=2,
        mixer=None, lr=5e-4, optim_alpha=0.99, optim_eps=1e-5,
        grad_norm_clip=10, learner_log_interval=1000, double_q=True,
        gamma=0.99, target_update_interval=200, kl_weight=1.0,
        obs_last_action=True, obs_agent_id=False,
        context_labeler="return_bins",
        seed=0, device="cpu",
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class LearnerContextChecksTest(unittest.TestCase):
    """The CoRAC wiring must fail early and loudly, and it must persist."""

    def _scheme(self):
        return {
            "obs": {"vshape": 1, "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (3,), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }

    def test_context_dim_without_corac_labeler_fails_early(self):
        with self.assertRaisesRegex(ValueError, "only meaningful with"):
            DualEpisodeREEQLearner(
                _FakeMac(4), self._scheme(), _Logger(),
                _learner_args(context_dim=48),
            )

    def test_corac_context_dim_must_match_twin(self):
        with self.assertRaisesRegex(ValueError, "twin agent reads contexts"):
            DualEpisodeREEQLearner(
                _FakeMac(4), self._scheme(), _Logger(),
                _learner_args(context_labeler="corac", context_dim=16,
                              corac_n_freq=8, corac_embed_dim=8),
            )

    def test_twin_target_clamp_bounds_the_bootstrap(self):
        """The twin bootstraps with max_a' Q, which ratchets without bound when a
        context carries little reward signal (hc_rac_audit.md, 3.1-3.2). With the
        clamp on, no target may leave the batch's realisable return range."""
        scheme = {
            "obs": {"vshape": 1, "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "actions_onehot": {"vshape": (3,), "group": "agents"},
            "avail_actions": {"vshape": (3,), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        actions = th.randint(0, 3, (6, 4, 2))
        rewards = th.full((6, 4), 0.01)
        bt, steps, n_agents = actions.shape
        padded = th.cat([actions, th.zeros(bt, 1, n_agents, dtype=th.long)], dim=1)
        batch = _Batch({
            "obs": th.zeros(bt, steps + 1, n_agents, 1),
            "actions": padded.unsqueeze(-1),
            "actions_onehot": F.one_hot(padded, 3).float(),
            "avail_actions": th.ones(bt, steps + 1, n_agents, 3, dtype=th.int),
            "reward": th.cat([rewards, th.zeros(bt, 1)], dim=1).unsqueeze(-1),
            "terminated": th.cat(
                [th.zeros(bt, steps - 1, 1), th.ones(bt, 1, 1), th.zeros(bt, 1, 1)],
                dim=1,
            ),
            "filled": th.ones(bt, steps + 1, 1, dtype=th.long),
        })
        ordinary = []
        for clamp in (False, True):
            args = _learner_args(
                context_labeler="corac", context_dim=6, corac_n_freq=3,
                corac_embed_dim=3, corac_n_hard=0, corac_bank_size=4,
                twin_hyper_hidden=8, rnn_hidden_dim=8, n_actions=3, n_agents=2,
                slot_number=4, twin_target_clamp=clamp,
                agent="rnn", twin_agent="hyper_ree_agent",
                mac="dual_ree_mac", action_selector="epsilon_greedy",
                agent_output_type="q", epsilon_start=1.0, epsilon_finish=0.05,
                epsilon_anneal_time=100,
            )
            mac = DualREEMAC(scheme, {"agents": 2}, args)
            logger = _Logger()
            learner = DualEpisodeREEQLearner(mac, scheme, logger, args)
            # A wildly optimistic twin head: without the clamp every bootstrap
            # target is ~gamma * 1000, far outside the episode returns (~0.04).
            with th.no_grad():
                mac.twin_agent.hyper_head.base_bias.fill_(1000.0)
                learner.target_mac.twin_agent.hyper_head.base_bias.fill_(1000.0)
            learner.train(batch, t_env=1, episode_num=0)
            ordinary.append(logger.stats["twin_target_mean"][-1][1])
        unclamped, clamped = ordinary
        self.assertGreater(unclamped, 100.0)
        self.assertLessEqual(clamped, 0.05)

    def test_labeler_state_round_trips_through_checkpoints(self):
        import tempfile
        scheme = self._scheme()
        args = _learner_args(context_labeler="corac", context_dim=16,
                             corac_n_freq=8, corac_embed_dim=8,
                             corac_segment_len=1, corac_n_hard=2,
                             corac_bank_size=4)
        learner = DualEpisodeREEQLearner(_FakeMac(16), scheme, _Logger(), args)
        batch = _batch(th.randint(0, 3, (6, 4, 2)), th.randn(6, 4))
        mask = th.ones(6, 4, 1)
        learner.context_labeler.label(batch, mask, t_env=0)

        with tempfile.TemporaryDirectory() as path:
            learner.save_models(path)
            self.assertTrue(os.path.exists(os.path.join(path, "context_labeler.th")))
            restored = DualEpisodeREEQLearner(_FakeMac(16), scheme, _Logger(), args)
            before = restored.context_labeler.support().clone()
            restored.load_models(path)

        self.assertFalse(th.equal(before, restored.context_labeler.support()))
        for name in ("return_mean", "return_var", "bank_count"):
            self.assertTrue(
                th.equal(
                    getattr(learner.context_labeler, name),
                    getattr(restored.context_labeler, name),
                ),
                name,
            )
        self.assertTrue(
            th.allclose(
                learner.context_labeler.bank, restored.context_labeler.bank)
        )


class _Logger:
    """Logger stub recording the scalars the learner logs."""

    def __init__(self):
        self.stats = {}

    def log_stat(self, key, value, t_env):
        self.stats.setdefault(key, []).append((t_env, value))

    @property
    def console_logger(self):
        from types import SimpleNamespace
        return SimpleNamespace(info=lambda *a, **k: None,
                               warning=lambda *a, **k: None)


class ContextAblationTest(unittest.TestCase):
    """hc_rac_method.md 5: (a) phi(g) only, (b) u only, (c) both."""

    def setUp(self):
        th.manual_seed(0)
        actions = th.randint(0, 3, (16, 4, 2))
        rewards = th.randn(16, 4)
        self.batch = _batch(actions, rewards)
        self.mask = th.ones(16, 4, 1)

    def test_return_only_needs_no_encoder_and_no_tcl_loss(self):
        labeler = ContrastiveContextLabeler(
            1, 3, 2, _args(context_dim=None, corac_use_trajectory=False)
        )
        self.assertEqual(labeler.context_dim, 8)  # corac_n_freq only
        self.assertIsNone(labeler.output)
        contexts, loss, stats = labeler.label(self.batch, self.mask, t_env=0)
        self.assertEqual(contexts.shape, (16, 5, 2, 8))
        self.assertIsNone(loss)  # no trajectory half, nothing contrastive to train
        self.assertEqual(stats["corac_embed_batch_std"], 0.0)
        # phi is the whole context: shared by agents, constant over time.
        self.assertTrue(th.equal(contexts[:, 0], contexts[:, -1]))
        self.assertTrue(th.equal(contexts[:, 0, 0], contexts[:, 0, 1]))

    def test_trajectory_only_context_matches_the_embedding(self):
        labeler = ContrastiveContextLabeler(
            1, 3, 2, _args(context_dim=None, corac_use_return=False)
        )
        self.assertEqual(labeler.context_dim, 8)  # corac_embed_dim only
        contexts, loss, stats = labeler.label(self.batch, self.mask, t_env=0)
        self.assertEqual(contexts.shape, (16, 5, 2, 8))
        self.assertIsNotNone(loss)
        with th.no_grad():
            features = labeler.step_features(self.batch)
            embedding = labeler.episode_embedding(features, th.full((16,), 4))
        self.assertTrue(th.allclose(contexts[:, 0], embedding, atol=1e-6))
        self.assertEqual(stats["corac_phi_saturated"], 0.0)
        self.assertGreater(stats["corac_embed_batch_std"], 0.0)

    def test_no_context_half_enabled_fails_early(self):
        with self.assertRaisesRegex(ValueError, "at least one context half"):
            ContrastiveContextLabeler(
                1, 3, 2,
                _args(context_dim=None, corac_use_return=False,
                      corac_use_trajectory=False),
            )

    def test_derived_context_dim_reaches_both_labeler_and_twin(self):
        """An ablation must be a one-flag change: no context_dim to keep in sync."""
        scheme = {
            "obs": {"vshape": 1, "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "actions_onehot": {"vshape": (3,), "group": "agents"},
            "avail_actions": {"vshape": (3,), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        bt, steps, n_agents = 6, 4, 2
        actions = th.randint(0, 3, (bt, steps, n_agents))
        padded = th.cat([actions, th.zeros(bt, 1, n_agents, dtype=th.long)], dim=1)
        batch = _Batch({
            "obs": th.zeros(bt, steps + 1, n_agents, 1),
            "actions": padded.unsqueeze(-1),
            "actions_onehot": F.one_hot(padded, 3).float(),
            "avail_actions": th.ones(bt, steps + 1, n_agents, 3, dtype=th.int),
            "reward": th.cat([th.rand(bt, steps), th.zeros(bt, 1)], dim=1).unsqueeze(-1),
            "terminated": th.cat(
                [th.zeros(bt, steps - 1, 1), th.ones(bt, 1, 1), th.zeros(bt, 1, 1)],
                dim=1,
            ),
            "filled": th.ones(bt, steps + 1, 1, dtype=th.long),
        })
        args = _learner_args(
            context_labeler="corac", corac_use_return=True,
            corac_use_trajectory=False, corac_n_freq=3, corac_embed_dim=3,
            twin_hyper_hidden=8, rnn_hidden_dim=8, agent="rnn",
            twin_agent="hyper_ree_agent", mac="dual_ree_mac",
            action_selector="epsilon_greedy", agent_output_type="q",
            epsilon_start=1.0, epsilon_finish=0.05, epsilon_anneal_time=100,
        )
        self.assertFalse(hasattr(args, "context_dim"))
        mac = DualREEMAC(scheme, {"agents": 2}, args)
        self.assertEqual(mac.twin_agent.context_dim, 3)
        learner = DualEpisodeREEQLearner(mac, scheme, _Logger(), args)
        self.assertEqual(learner.context_labeler.context_dim, 3)
        learner.train(batch, t_env=1, episode_num=0)


if __name__ == "__main__":
    unittest.main()
