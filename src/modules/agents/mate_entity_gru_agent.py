import math

import torch as th
import torch.nn as nn


class MaskAwareEntityAttention(nn.Module):
    """Attend from the controlled camera to one typed set of visible entities."""

    def __init__(self, model_dim, n_heads):
        super().__init__()
        if model_dim % n_heads != 0:
            raise ValueError(
                f"entity_dim ({model_dim}) must be divisible by "
                f"attention_heads ({n_heads})."
            )

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.output = nn.Linear(model_dim, model_dim)
        self.output_norm = nn.LayerNorm(model_dim)
        self.empty_summary = nn.Parameter(th.zeros(model_dim))

    def forward(self, camera, entities, visible):
        """
        Args:
            camera: ``(batch, model_dim)`` controlled-camera representation.
            entities: ``(batch, count, model_dim)`` typed entity tokens.
            visible: ``(batch, count)`` boolean visibility mask.
        """
        batch_size, entity_count, _ = entities.shape
        visible = visible.bool()

        query = self.query(camera).view(batch_size, self.n_heads, self.head_dim)
        keys = self.key(entities).view(
            batch_size, entity_count, self.n_heads, self.head_dim
        )
        values = self.value(entities).view(
            batch_size, entity_count, self.n_heads, self.head_dim
        )

        scores = th.einsum("bhd,bnhd->bhn", query, keys) * self.scale
        scores = scores.masked_fill(
            ~visible.unsqueeze(1), th.finfo(scores.dtype).min
        )

        # Multiplying by the mask and renormalizing makes an all-hidden set safe:
        # its summary is replaced by the learned empty-set representation below.
        weights = th.softmax(scores, dim=-1) * visible.unsqueeze(1).to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        summary = th.einsum("bhn,bnhd->bhd", weights, values).reshape(
            batch_size, self.model_dim
        )
        summary = self.output_norm(self.output(summary))

        empty = self.empty_summary.unsqueeze(0).expand(batch_size, -1)
        has_visible_entity = visible.any(dim=-1, keepdim=True)
        return th.where(has_visible_entity, summary, empty)


def _make_entity_encoder(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.LayerNorm(output_dim),
    )


class MateEntityGRUAgent(nn.Module):
    """MATE camera Q-network with typed entity attention and recurrent memory.

    The controller still supplies the standard flat PyMARL input. This module
    parses only the raw MATE observation, embeds the previous one-hot action,
    and leaves the IQL learner and its 25-action Q head unchanged.
    """

    PRESERVED_DIM = 13
    SELF_CAMERA_DIM = 9
    TARGET_DIM = 4
    OBSTACLE_DIM = 3
    TEAMMATE_CAMERA_DIM = 6

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.n_actions = int(args.n_actions)
        self.n_cameras = int(
            getattr(args, "mate_num_cameras", getattr(args, "n_agents", 4))
        )
        self.n_targets = int(getattr(args, "mate_num_targets", 8))
        self.n_obstacles = int(getattr(args, "mate_num_obstacles", 9))
        self.obs_dim = int(args.obs_shape)

        if not bool(getattr(args, "obs_last_action", False)):
            raise ValueError(
                "MateEntityGRUAgent requires obs_last_action=true so it can "
                "construct the learned previous-action embedding."
            )

        expected_obs_dim = (
            self.PRESERVED_DIM
            + self.SELF_CAMERA_DIM
            + self.n_targets * (self.TARGET_DIM + 1)
            + self.n_obstacles * (self.OBSTACLE_DIM + 1)
            + self.n_cameras * (self.TEAMMATE_CAMERA_DIM + 1)
        )
        if self.obs_dim != expected_obs_dim:
            raise ValueError(
                "Unexpected MATE camera observation size. The configured "
                f"entity layout requires {expected_obs_dim}, got {self.obs_dim}. "
                "Check mate_num_cameras/targets/obstacles and the MATE config."
            )
        if input_shape < self.obs_dim + self.n_actions:
            raise ValueError(
                "Controller input does not contain the previous one-hot action: "
                f"expected at least {self.obs_dim + self.n_actions}, got {input_shape}."
            )

        entity_dim = int(getattr(args, "mate_entity_dim", 128))
        entity_hidden_dim = int(getattr(args, "mate_entity_mlp_hidden_dim", 128))
        attention_heads = int(getattr(args, "mate_attention_heads", 4))
        action_embedding_dim = int(
            getattr(args, "mate_previous_action_embedding_dim", 32)
        )
        fusion_dim = int(getattr(args, "mate_fusion_dim", 256))
        hidden_dim = int(args.rnn_hidden_dim)

        self.self_encoder = _make_entity_encoder(
            self.PRESERVED_DIM + self.SELF_CAMERA_DIM,
            entity_hidden_dim,
            entity_dim,
        )
        self.target_encoder = _make_entity_encoder(
            self.TARGET_DIM, entity_hidden_dim, entity_dim
        )
        self.obstacle_encoder = _make_entity_encoder(
            self.OBSTACLE_DIM, entity_hidden_dim, entity_dim
        )
        self.teammate_encoder = _make_entity_encoder(
            self.TEAMMATE_CAMERA_DIM, entity_hidden_dim, entity_dim
        )

        self.target_attention = MaskAwareEntityAttention(entity_dim, attention_heads)
        self.obstacle_attention = MaskAwareEntityAttention(entity_dim, attention_heads)
        self.teammate_attention = MaskAwareEntityAttention(entity_dim, attention_heads)

        self.spatial_fusion = nn.Sequential(
            nn.Linear(4 * entity_dim, fusion_dim),
            nn.SiLU(),
            nn.LayerNorm(fusion_dim),
        )
        # Index n_actions is reserved for the all-zero action at episode start.
        self.previous_action_embedding = nn.Embedding(
            self.n_actions + 1, action_embedding_dim
        )
        # A full GRU (rather than GRUCell) supports both one-step rollout and a
        # fused cuDNN sequence path during replay training.
        self.rnn = nn.GRU(
            fusion_dim + action_embedding_dim,
            hidden_dim,
            batch_first=True,
        )
        self.q_head = nn.Linear(hidden_dim, self.n_actions)

        if bool(getattr(args, "agent_orthogonal_init", False)):
            self._orthogonal_init()

    def _orthogonal_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.q_head.weight, gain=1.0)

        for name, parameter in self.rnn.named_parameters():
            if "weight" in name:
                for gate in parameter.chunk(3, dim=0):
                    nn.init.orthogonal_(gate)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def init_hidden(self):
        return self.q_head.weight.new_zeros(1, self.args.rnn_hidden_dim)

    def _split_inputs(self, inputs):
        observation = inputs[:, : self.obs_dim]
        previous_action = inputs[
            :, self.obs_dim : self.obs_dim + self.n_actions
        ]
        return observation, previous_action

    def previous_action_indices(self, previous_action):
        has_previous_action = previous_action.abs().sum(dim=-1) > 0.0
        indices = previous_action.argmax(dim=-1)
        no_previous_action = th.full_like(indices, self.n_actions)
        return th.where(has_previous_action, indices, no_previous_action)

    def _parse_observation(self, observation):
        offset = 0
        preserved = observation[:, offset : offset + self.PRESERVED_DIM]
        offset += self.PRESERVED_DIM
        self_camera = observation[:, offset : offset + self.SELF_CAMERA_DIM]
        offset += self.SELF_CAMERA_DIM

        targets = observation[
            :, offset : offset + self.n_targets * (self.TARGET_DIM + 1)
        ].reshape(-1, self.n_targets, self.TARGET_DIM + 1)
        offset += self.n_targets * (self.TARGET_DIM + 1)

        obstacles = observation[
            :, offset : offset + self.n_obstacles * (self.OBSTACLE_DIM + 1)
        ].reshape(-1, self.n_obstacles, self.OBSTACLE_DIM + 1)
        offset += self.n_obstacles * (self.OBSTACLE_DIM + 1)

        teammates = observation[
            :, offset : offset + self.n_cameras * (self.TEAMMATE_CAMERA_DIM + 1)
        ].reshape(-1, self.n_cameras, self.TEAMMATE_CAMERA_DIM + 1)

        target_visible = targets[..., -1] > 0.5
        obstacle_visible = obstacles[..., -1] > 0.5
        teammate_visible = teammates[..., -1] > 0.5

        # MATE includes the controlled camera in the teammate block. It is
        # already represented by the private self-camera token, so remove it.
        agent_indices = preserved[:, 3].long().clamp(0, self.n_cameras - 1)
        own_camera = th.nn.functional.one_hot(
            agent_indices, num_classes=self.n_cameras
        ).bool()
        teammate_visible = teammate_visible & ~own_camera

        return (
            th.cat([preserved, self_camera], dim=-1),
            targets[..., : self.TARGET_DIM],
            target_visible,
            obstacles[..., : self.OBSTACLE_DIM],
            obstacle_visible,
            teammates[..., : self.TEAMMATE_CAMERA_DIM],
            teammate_visible,
        )

    def encode_spatial_observation(self, observation):
        (
            self_camera,
            targets,
            target_visible,
            obstacles,
            obstacle_visible,
            teammates,
            teammate_visible,
        ) = self._parse_observation(observation)

        self_token = self.self_encoder(self_camera)
        target_tokens = self.target_encoder(targets)
        obstacle_tokens = self.obstacle_encoder(obstacles)
        teammate_tokens = self.teammate_encoder(teammates)

        target_summary = self.target_attention(
            self_token, target_tokens, target_visible
        )
        obstacle_summary = self.obstacle_attention(
            self_token, obstacle_tokens, obstacle_visible
        )
        teammate_summary = self.teammate_attention(
            self_token, teammate_tokens, teammate_visible
        )
        return self.spatial_fusion(
            th.cat(
                [
                    self_token,
                    target_summary,
                    obstacle_summary,
                    teammate_summary,
                ],
                dim=-1,
            )
        )

    def forward(self, inputs, hidden_state):
        recurrent_output, hidden = self.recurrent_step(inputs, hidden_state)
        q_values = self.q_head(recurrent_output)
        return q_values, hidden

    def recurrent_step(self, inputs, hidden_state):
        """Encode one rollout step and update the shared recurrent core."""
        observation, previous_action = self._split_inputs(inputs)
        spatial = self.encode_spatial_observation(observation)
        action_indices = self.previous_action_indices(previous_action)
        action_embedding = self.previous_action_embedding(action_indices)

        recurrent_input = th.cat([spatial, action_embedding], dim=-1).unsqueeze(1)
        hidden_in = hidden_state.reshape(
            1, -1, self.args.rnn_hidden_dim
        ).contiguous()
        recurrent_output, hidden = self.rnn(recurrent_input, hidden_in)
        hidden = hidden.squeeze(0)
        recurrent_output = recurrent_output.squeeze(1)
        return recurrent_output, hidden

    def forward_sequence(self, inputs, hidden_state):
        """Fused replay-training path.

        Args:
            inputs: controller inputs shaped ``(batch, time, agents, features)``.
            hidden_state: initial states shaped ``(batch, agents, hidden_dim)``.

        Entity encoding and attention are evaluated in one large batch across
        every episode timestep. The temporal recurrence is then executed by a
        single fused GRU call instead of hundreds of Python GRUCell calls.
        """
        recurrent_output, final_hidden = self.recurrent_sequence(
            inputs, hidden_state
        )
        return self.q_head(recurrent_output), final_hidden

    def recurrent_sequence(self, inputs, hidden_state):
        """Encode a complete episode and return its recurrent features."""
        batch_size, time_steps, n_agents, input_dim = inputs.shape
        flat_inputs = inputs.reshape(batch_size * time_steps * n_agents, input_dim)
        observation, previous_action = self._split_inputs(flat_inputs)

        spatial = self.encode_spatial_observation(observation).reshape(
            batch_size, time_steps, n_agents, -1
        )
        action_indices = self.previous_action_indices(previous_action)
        action_embedding = self.previous_action_embedding(action_indices).reshape(
            batch_size, time_steps, n_agents, -1
        )
        recurrent_input = th.cat([spatial, action_embedding], dim=-1)

        # cuDNN expects each agent trajectory to be a separate batch item.
        recurrent_input = recurrent_input.permute(0, 2, 1, 3).reshape(
            batch_size * n_agents, time_steps, -1
        )
        initial_hidden = hidden_state.reshape(
            1, batch_size * n_agents, self.args.rnn_hidden_dim
        ).contiguous()
        recurrent_output, final_hidden = self.rnn(recurrent_input, initial_hidden)
        recurrent_output = recurrent_output.reshape(
            batch_size, n_agents, time_steps, self.args.rnn_hidden_dim
        ).permute(0, 2, 1, 3).contiguous()
        final_hidden = final_hidden.squeeze(0).reshape(
            batch_size, n_agents, self.args.rnn_hidden_dim
        )
        return recurrent_output, final_hidden


class MateEntityGRUTwinAgent(MateEntityGRUAgent):
    """Entity-GRU RAC twin with a vectorized context-conditioned Q head."""

    context_independent_recurrent_core = True

    def __init__(self, input_shape, args):
        super().__init__(input_shape, args)
        del self.q_head
        self.slot_number = int(args.slot_number)
        hidden_dim = int(args.rnn_hidden_dim)
        self.context_weight = nn.Linear(
            self.slot_number, hidden_dim * self.n_actions
        )
        self.context_bias = nn.Linear(self.slot_number, self.n_actions)

        if bool(getattr(args, "agent_orthogonal_init", False)):
            nn.init.orthogonal_(self.context_weight.weight, gain=1.0)
            nn.init.zeros_(self.context_weight.bias)
            nn.init.orthogonal_(self.context_bias.weight, gain=1.0)
            nn.init.zeros_(self.context_bias.bias)

    def init_hidden(self):
        return self.context_bias.weight.new_zeros(
            1, self.args.rnn_hidden_dim
        )

    def _context_q(self, recurrent_output, context):
        context = context.reshape(-1, self.slot_number)
        recurrent_output = recurrent_output.reshape(
            -1, self.args.rnn_hidden_dim
        )
        weights = self.context_weight(context).reshape(
            -1, self.args.rnn_hidden_dim, self.n_actions
        )
        bias = self.context_bias(context)
        return th.bmm(recurrent_output.unsqueeze(1), weights).squeeze(1) + bias

    def forward(self, inputs, context, hidden_state):
        recurrent_output, hidden = self.recurrent_step(inputs, hidden_state)
        return self._context_q(recurrent_output, context), hidden

    def forward_sequence(self, inputs, context, hidden_state):
        recurrent_output, final_hidden = self.recurrent_sequence(
            inputs, hidden_state
        )
        batch_size, time_steps, n_agents, _ = recurrent_output.shape
        q_values = self._context_q(recurrent_output, context).reshape(
            batch_size, time_steps, n_agents, self.n_actions
        )
        return q_values, final_hidden

    def _all_context_q(self, recurrent_output):
        """Compute Q for every categorical context without re-encoding state."""
        identity = th.eye(
            self.slot_number,
            device=recurrent_output.device,
            dtype=recurrent_output.dtype,
        )
        weights = self.context_weight(identity).reshape(
            self.slot_number,
            self.args.rnn_hidden_dim,
            self.n_actions,
        )
        bias = self.context_bias(identity)
        return th.einsum("...h,kha->...ka", recurrent_output, weights) + bias

    def counterfactual_step(self, inputs, hidden_state):
        recurrent_output, hidden = self.recurrent_step(inputs, hidden_state)
        return self._all_context_q(recurrent_output), hidden

    def counterfactual_sequence(self, inputs, hidden_state):
        recurrent_output, final_hidden = self.recurrent_sequence(
            inputs, hidden_state
        )
        return self._all_context_q(recurrent_output), final_hidden
