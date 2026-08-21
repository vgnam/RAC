import unittest
from types import SimpleNamespace

import torch as th
import torch.nn.functional as F

from src.modules.agents.mate_entity_gru_agent import (
    MateEntityGRUAgent,
    MateEntityGRUTwinAgent,
)


def make_args():
    return SimpleNamespace(
        n_actions=25,
        n_agents=4,
        obs_shape=126,
        obs_last_action=True,
        rnn_hidden_dim=256,
        mate_num_cameras=4,
        mate_num_targets=8,
        mate_num_obstacles=9,
        mate_entity_dim=32,
        mate_entity_mlp_hidden_dim=32,
        mate_attention_heads=4,
        mate_fusion_dim=64,
        mate_previous_action_embedding_dim=8,
        agent_orthogonal_init=False,
        slot_number=10,
    )


class MateEntityGRUAgentTest(unittest.TestCase):
    def setUp(self):
        th.manual_seed(7)
        self.args = make_args()
        self.agent = MateEntityGRUAgent(126 + 25, self.args)
        self.agent.eval()

    def make_observation(self, batch_size=2):
        observation = th.randn(batch_size, 126)
        observation[:, 0:3] = th.tensor([4.0, 8.0, 9.0])
        observation[:, 3] = th.arange(batch_size) % 4

        # Visibility flags occupy the final feature of every entity record.
        observation[:, 22 + 4 : 62 : 5] = 0.0
        observation[:, 62 + 3 : 98 : 4] = 0.0
        observation[:, 98 + 6 : 126 : 7] = 0.0
        return observation

    def test_forward_shapes_and_all_hidden_sets_are_finite(self):
        observation = self.make_observation()
        previous_action = th.zeros(2, 25)
        inputs = th.cat([observation, previous_action], dim=-1)
        hidden = th.zeros(2, 256)

        q_values, next_hidden = self.agent(inputs, hidden)

        self.assertEqual(q_values.shape, (2, 25))
        self.assertEqual(next_hidden.shape, (2, 256))
        self.assertTrue(th.isfinite(q_values).all())
        self.assertTrue(th.isfinite(next_hidden).all())

    def test_hidden_entity_features_do_not_change_spatial_encoding(self):
        observation = self.make_observation(batch_size=1)
        changed = observation.clone()

        # Change features of hidden target 0, obstacle 0, and camera 1 while
        # leaving their zero visibility masks untouched.
        changed[:, 22:26] += 1000.0
        changed[:, 62:65] -= 1000.0
        changed[:, 105:111] += 1000.0

        first = self.agent.encode_spatial_observation(observation)
        second = self.agent.encode_spatial_observation(changed)
        self.assertTrue(th.allclose(first, second, atol=1.0e-6, rtol=1.0e-6))

    def test_self_camera_is_removed_from_teammate_attention_mask(self):
        observation = self.make_observation(batch_size=1)
        observation[:, 3] = 2.0
        observation[:, 98 + 6 : 126 : 7] = 1.0

        parsed = self.agent._parse_observation(observation)
        teammate_visible = parsed[-1]

        self.assertEqual(teammate_visible.tolist(), [[True, True, False, True]])

    def test_previous_action_has_learned_start_token(self):
        actions = th.zeros(3, 25)
        actions[1, 0] = 1.0
        actions[2, 17] = 1.0

        indices = self.agent.previous_action_indices(actions)

        self.assertEqual(indices.tolist(), [25, 0, 17])

    def test_gru_uses_previous_hidden_state(self):
        observation = self.make_observation(batch_size=1)
        previous_action = th.zeros(1, 25)
        previous_action[:, 4] = 1.0
        inputs = th.cat([observation, previous_action], dim=-1)

        q_zero, _ = self.agent(inputs, th.zeros(1, 256))
        q_memory, _ = self.agent(inputs, th.ones(1, 256))

        self.assertFalse(th.allclose(q_zero, q_memory))

    def test_fused_sequence_matches_step_by_step_recurrence(self):
        batch_size, time_steps, n_agents = 2, 5, 4
        observation = self.make_observation(batch_size * time_steps * n_agents)
        observation = observation.reshape(batch_size, time_steps, n_agents, 126)
        observation[..., 3] = th.arange(n_agents).view(1, 1, n_agents)

        previous_action = th.zeros(batch_size, time_steps, n_agents, 25)
        action_indices = th.randint(0, 25, (batch_size, time_steps - 1, n_agents))
        previous_action[:, 1:].scatter_(
            -1, action_indices.unsqueeze(-1), 1.0
        )
        inputs = th.cat([observation, previous_action], dim=-1)
        initial_hidden = th.randn(batch_size, n_agents, 256)

        step_hidden = initial_hidden.clone()
        step_outputs = []
        for timestep in range(time_steps):
            q_values, flat_hidden = self.agent(
                inputs[:, timestep].reshape(batch_size * n_agents, -1),
                step_hidden,
            )
            step_outputs.append(q_values.reshape(batch_size, n_agents, 25))
            step_hidden = flat_hidden.reshape(batch_size, n_agents, 256)
        step_outputs = th.stack(step_outputs, dim=1)

        fused_outputs, fused_hidden = self.agent.forward_sequence(
            inputs, initial_hidden
        )

        self.assertTrue(
            th.allclose(step_outputs, fused_outputs, atol=2.0e-5, rtol=1.0e-5)
        )
        self.assertTrue(
            th.allclose(step_hidden, fused_hidden, atol=2.0e-5, rtol=1.0e-5)
        )

    def test_invalid_layout_fails_early(self):
        args = make_args()
        args.mate_num_targets = 7
        with self.assertRaisesRegex(ValueError, "Unexpected MATE camera observation"):
            MateEntityGRUAgent(126 + 25, args)

    def test_twin_sequence_matches_step_by_step_recurrence(self):
        twin = MateEntityGRUTwinAgent(151, self.args).eval()
        batch_size, time_steps, n_agents = 2, 4, 4
        observation = self.make_observation(batch_size * time_steps * n_agents)
        observation = observation.reshape(batch_size, time_steps, n_agents, 126)
        observation[..., 3] = th.arange(n_agents).view(1, 1, n_agents)
        previous_action = th.zeros(batch_size, time_steps, n_agents, 25)
        previous_action[:, 1:, :, 3] = 1.0
        inputs = th.cat([observation, previous_action], dim=-1)
        contexts = F.one_hot(
            th.randint(0, 10, (batch_size, time_steps, n_agents)),
            num_classes=10,
        ).float()
        initial_hidden = th.randn(batch_size, n_agents, 256)

        step_hidden = initial_hidden.clone()
        step_outputs = []
        for timestep in range(time_steps):
            q_values, flat_hidden = twin(
                inputs[:, timestep].reshape(batch_size * n_agents, -1),
                contexts[:, timestep],
                step_hidden,
            )
            step_outputs.append(q_values.reshape(batch_size, n_agents, 25))
            step_hidden = flat_hidden.reshape(batch_size, n_agents, 256)

        fused_outputs, fused_hidden = twin.forward_sequence(
            inputs, contexts, initial_hidden
        )
        self.assertTrue(
            th.allclose(
                th.stack(step_outputs, dim=1),
                fused_outputs,
                atol=2.0e-5,
                rtol=1.0e-5,
            )
        )
        self.assertTrue(
            th.allclose(step_hidden, fused_hidden, atol=2.0e-5, rtol=1.0e-5)
        )

    def test_vectorized_counterfactual_matches_each_context_head(self):
        twin = MateEntityGRUTwinAgent(151, self.args).eval()
        batch_size, time_steps, n_agents = 1, 3, 4
        observation = self.make_observation(batch_size * time_steps * n_agents)
        observation = observation.reshape(batch_size, time_steps, n_agents, 126)
        observation[..., 3] = th.arange(n_agents).view(1, 1, n_agents)
        inputs = th.cat(
            [observation, th.zeros(batch_size, time_steps, n_agents, 25)],
            dim=-1,
        )
        initial_hidden = th.zeros(batch_size, n_agents, 256)
        all_context_q, _ = twin.counterfactual_sequence(inputs, initial_hidden)

        for context_index in range(10):
            contexts = th.zeros(batch_size, time_steps, n_agents, 10)
            contexts[..., context_index] = 1.0
            selected_q, _ = twin.forward_sequence(
                inputs, contexts, initial_hidden
            )
            self.assertTrue(
                th.allclose(
                    selected_q,
                    all_context_q[..., context_index, :],
                    atol=2.0e-5,
                    rtol=1.0e-5,
                )
            )


if __name__ == "__main__":
    unittest.main()
