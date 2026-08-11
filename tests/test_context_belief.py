import unittest
from types import SimpleNamespace

import torch as th
import torch.nn.functional as F

from src.modules.context_belief import (
    ContextBeliefModel,
    posterior_optimistic_q,
)


class _Batch:
    def __init__(self, data):
        self.data = data
        self.batch_size = data["obs"].shape[0]
        self.max_seq_length = data["obs"].shape[1]
        self.device = data["obs"].device

    def __getitem__(self, key):
        return self.data[key]


class PosteriorOptimisticQTest(unittest.TestCase):
    def test_uniform_belief_recovers_rac_max(self):
        context_q = th.tensor([[[[1.0, 3.0, 2.0], [4.0, 0.0, 1.0]]]])
        belief = th.full((1, 1, 3), 1.0 / 3.0)

        decision_q, diagnostics = posterior_optimistic_q(
            context_q,
            belief,
            optimism_min=0.25,
            optimism_max=1.0,
        )

        self.assertTrue(
            th.allclose(decision_q, context_q.max(dim=-1)[0])
        )
        self.assertTrue(
            th.allclose(
                diagnostics["optimism_weight"],
                th.ones((1, 1)),
            )
        )

    def test_confident_belief_keeps_nonzero_optimism(self):
        context_q = th.tensor([[[[1.0, 5.0, 0.0]]]])
        belief = th.tensor([[[1.0, 0.0, 0.0]]])

        decision_q, diagnostics = posterior_optimistic_q(
            context_q,
            belief,
            optimism_min=0.25,
            optimism_max=1.0,
        )

        posterior_mean = diagnostics["posterior_mean"]
        self.assertGreater(decision_q.item(), posterior_mean.item())
        self.assertGreaterEqual(
            diagnostics["optimism_weight"].item(),
            0.25,
        )

    def test_context_shift_temporarily_increases_optimism(self):
        context_q = th.tensor([[[[1.0, 5.0, 0.0]]]])
        belief = th.tensor([[[0.98, 0.01, 0.01]]])
        low_shift_q, low = posterior_optimistic_q(
            context_q,
            belief,
            context_shift=th.zeros((1, 1)),
        )
        high_shift_q, high = posterior_optimistic_q(
            context_q,
            belief,
            context_shift=th.ones((1, 1)),
        )

        self.assertGreater(
            high["optimism_weight"].item(),
            low["optimism_weight"].item(),
        )
        self.assertGreater(high_shift_q.item(), low_shift_q.item())


class ContextBeliefModelTest(unittest.TestCase):
    def test_causal_filter_and_dynamics_loss_are_finite(self):
        th.manual_seed(7)
        batch_size, time_steps = 2, 4
        n_agents, obs_dim, n_actions, n_contexts = 2, 5, 3, 3
        args = SimpleNamespace(
            slot_number=n_contexts,
            belief_hidden_dim=8,
            belief_decoder_hidden_dim=16,
            belief_projection_dim=4,
            belief_include_agent_id=True,
            belief_temperature=1.0,
            belief_transition_stay=0.9,
            belief_prior_strength=1.0,
            belief_reward_loss_weight=1.0,
            belief_kl_weight=1e-3,
            belief_balance_weight=1e-2,
            seed=7,
        )
        model = ContextBeliefModel(obs_dim, n_actions, n_agents, args)

        action_indices = th.randint(
            n_actions,
            (batch_size, time_steps, n_agents),
        )
        batch = _Batch({
            "obs": th.randn(batch_size, time_steps, n_agents, obs_dim),
            "actions_onehot": F.one_hot(
                action_indices,
                num_classes=n_actions,
            ).float(),
            "reward": th.randn(batch_size, time_steps, 1),
        })

        outputs = model.infer_sequence(batch)
        self.assertEqual(
            outputs["belief"].shape,
            (batch_size, time_steps, n_agents, n_contexts),
        )
        self.assertTrue(
            th.allclose(
                outputs["belief"].sum(dim=-1),
                th.ones(batch_size, time_steps, n_agents),
            )
        )
        # The zero-initialized evidence head gives a uniform, zero-shift first
        # belief before any self-supervised update.
        self.assertTrue(
            th.allclose(
                outputs["belief"][:, 0],
                th.full(
                    (batch_size, n_agents, n_contexts),
                    1.0 / n_contexts,
                ),
            )
        )
        self.assertTrue(
            th.allclose(
                outputs["shift"][:, 0],
                th.zeros(batch_size, n_agents),
            )
        )

        assignments = F.gumbel_softmax(
            outputs["logits"],
            tau=1.0,
            hard=True,
            dim=-1,
        )
        transition_mask = th.ones(batch_size, time_steps - 1, 1)
        losses = model.dynamics_loss(
            batch,
            outputs["belief"],
            transition_mask,
            args,
            context_assignments=assignments,
        )
        for value in losses.values():
            self.assertTrue(th.isfinite(value).item())
        losses["loss"].backward()
        self.assertIsNotNone(model.evidence_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
