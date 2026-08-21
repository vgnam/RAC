import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


def normalized_categorical_entropy(probabilities, eps=1e-8):
    """Return categorical entropy in [0, 1] along the last dimension."""
    n_categories = probabilities.shape[-1]
    if n_categories <= 1:
        return th.zeros_like(probabilities[..., 0])
    probabilities = probabilities.clamp_min(eps)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    return entropy / math.log(n_categories)


def normalized_js_divergence(posterior, prior, eps=1e-8):
    """Return Jensen-Shannon divergence in [0, 1] along the last dimension."""
    posterior = posterior.clamp_min(eps)
    prior = prior.clamp_min(eps)
    midpoint = 0.5 * (posterior + prior)
    js = 0.5 * (
        (posterior * (posterior.log() - midpoint.log())).sum(dim=-1)
        + (prior * (prior.log() - midpoint.log())).sum(dim=-1)
    )
    return (js / math.log(2.0)).clamp(0.0, 1.0)


def posterior_optimistic_q(
    context_q,
    belief,
    context_shift=None,
    optimism_min=0.25,
    optimism_max=1.0,
):
    """Combine posterior plausibility with RAC's cooperative optimism.

    Args:
        context_q: Q values with shape (..., n_actions, n_contexts).
        belief: categorical context belief with shape (..., n_contexts).
        context_shift: optional normalized shift score with shape (...).

    The lower bound on ``alpha`` deliberately keeps an optimistic-max term even
    when the context posterior is confident. With a uniform posterior and
    ``optimism_max == 1``, the rule exactly recovers RAC's max over contexts.
    """
    if not 0.0 <= optimism_min <= optimism_max <= 1.0:
        raise ValueError(
            "Expected 0 <= optimism_min <= optimism_max <= 1, got "
            f"{optimism_min} and {optimism_max}."
        )
    if context_q.shape[-1] != belief.shape[-1]:
        raise ValueError(
            "Context-Q and belief dimensions differ: "
            f"{context_q.shape[-1]} != {belief.shape[-1]}."
        )

    belief = belief / belief.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    posterior_mean = (context_q * belief.unsqueeze(-2)).sum(dim=-1)
    optimistic_max = context_q.max(dim=-1)[0]

    uncertainty = normalized_categorical_entropy(belief)
    if context_shift is not None:
        # Union of two uncertainty signals: either ambiguity or a surprising
        # regime shift is sufficient to temporarily restore stronger optimism.
        shift = context_shift.clamp(0.0, 1.0)
        uncertainty = 1.0 - (1.0 - uncertainty) * (1.0 - shift)

    alpha = optimism_min + (optimism_max - optimism_min) * uncertainty
    decision_q = (
        (1.0 - alpha.unsqueeze(-1)) * posterior_mean
        + alpha.unsqueeze(-1) * optimistic_max
    )
    return decision_q, {
        "posterior_mean": posterior_mean,
        "optimistic_max": optimistic_max,
        "optimism_weight": alpha,
        "uncertainty": uncertainty,
    }


class ContextBeliefModel(nn.Module):
    """Causal categorical belief filter over locally induced dynamics.

    The filter consumes only information available before the current action:
    the current local observation and the previous local action/shared reward.
    A lightweight mixture dynamics decoder supplies a self-supervised learning
    signal by predicting projected observation deltas and rewards.
    """

    def __init__(self, obs_dim, n_actions, n_agents, args):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.n_agents = int(n_agents)
        self.n_contexts = int(args.slot_number)
        self.hidden_dim = int(getattr(args, "belief_hidden_dim", 64))
        self.include_agent_id = bool(
            getattr(args, "belief_include_agent_id", True)
        )
        self.temperature = float(getattr(args, "belief_temperature", 1.0))
        self.transition_stay = float(
            getattr(args, "belief_transition_stay", 0.95)
        )
        self.prior_strength = float(
            getattr(args, "belief_prior_strength", 1.0)
        )
        if self.temperature <= 0.0:
            raise ValueError("belief_temperature must be positive.")
        if not 0.0 <= self.transition_stay <= 1.0:
            raise ValueError("belief_transition_stay must lie in [0, 1].")
        if self.prior_strength < 0.0:
            raise ValueError("belief_prior_strength must be non-negative.")

        id_dim = self.n_agents if self.include_agent_id else 0
        filter_input_dim = self.obs_dim + self.n_actions + 1 + id_dim
        self.filter_encoder = nn.Sequential(
            nn.Linear(filter_input_dim, self.hidden_dim),
            nn.ReLU(),
        )
        # A full GRU supports both causal one-step filtering at execution and
        # a fused cuDNN sequence path during replay training.
        self.filter_rnn = nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            batch_first=True,
        )
        self.evidence_head = nn.Linear(self.hidden_dim, self.n_contexts)
        # Start from an exactly uniform belief rather than arbitrary confidence.
        nn.init.zeros_(self.evidence_head.weight)
        nn.init.zeros_(self.evidence_head.bias)

        projection_dim = min(
            self.obs_dim,
            int(getattr(args, "belief_projection_dim", 16)),
        )
        if projection_dim <= 0:
            raise ValueError("belief_projection_dim must be positive.")
        self.projection_dim = projection_dim
        projection_generator = th.Generator(device="cpu")
        projection_generator.manual_seed(
            int(getattr(args, "seed", 0) or 0) + 7919
        )
        projection = th.randn(
            projection_dim,
            self.obs_dim,
            generator=projection_generator,
        )
        projection = F.normalize(projection, dim=-1)
        self.register_buffer("observation_projection", projection)

        decoder_hidden_dim = int(
            getattr(args, "belief_decoder_hidden_dim", 128)
        )
        decoder_input_dim = self.obs_dim + self.n_actions + id_dim
        decoder_output_dim = self.n_contexts * (projection_dim + 1)
        self.dynamics_decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, decoder_output_dim),
        )

    def initial_state(self, batch_size, device):
        hidden = th.zeros(
            batch_size,
            self.n_agents,
            self.hidden_dim,
            device=device,
        )
        belief = th.full(
            (batch_size, self.n_agents, self.n_contexts),
            1.0 / self.n_contexts,
            device=device,
        )
        return hidden, belief

    def _agent_ids(self, batch_size, device):
        return th.eye(self.n_agents, device=device).unsqueeze(0).expand(
            batch_size, -1, -1
        )

    def filter_step(
        self,
        observation,
        previous_action,
        previous_reward,
        hidden,
        previous_belief,
    ):
        batch_size = observation.shape[0]
        observation = observation.reshape(batch_size, self.n_agents, self.obs_dim)
        if previous_reward.dim() == 2:
            previous_reward = previous_reward.unsqueeze(1)
        if previous_reward.shape[1] == 1:
            previous_reward = previous_reward.expand(-1, self.n_agents, -1)

        inputs = [observation, previous_action, previous_reward]
        if self.include_agent_id:
            inputs.append(self._agent_ids(batch_size, observation.device))
        inputs = th.cat(inputs, dim=-1)

        encoded = self.filter_encoder(inputs.reshape(batch_size * self.n_agents, -1))
        _, next_hidden = self.filter_rnn(
            encoded.unsqueeze(1),
            hidden.reshape(
                1, batch_size * self.n_agents, self.hidden_dim
            ).contiguous(),
        )
        next_hidden = next_hidden.squeeze(0).reshape(
            batch_size, self.n_agents, self.hidden_dim
        )

        uniform = th.full_like(previous_belief, 1.0 / self.n_contexts)
        predictive_prior = (
            self.transition_stay * previous_belief
            + (1.0 - self.transition_stay) * uniform
        )
        evidence_logits = self.evidence_head(next_hidden) / self.temperature
        posterior_logits = (
            evidence_logits
            + self.prior_strength * predictive_prior.clamp_min(1e-8).log()
        )
        posterior = F.softmax(posterior_logits, dim=-1)
        shift = normalized_js_divergence(posterior, predictive_prior)
        return posterior, posterior_logits, next_hidden, predictive_prior, shift

    def infer_sequence(self, batch):
        batch_size = batch.batch_size
        hidden, previous_belief = self.initial_state(batch_size, batch.device)
        time_steps = batch.max_seq_length

        observations = batch["obs"].reshape(
            batch_size, time_steps, self.n_agents, self.obs_dim
        )
        previous_actions = th.cat(
            [
                th.zeros_like(batch["actions_onehot"][:, :1]),
                batch["actions_onehot"][:, :-1],
            ],
            dim=1,
        )
        previous_rewards = th.cat(
            [
                th.zeros_like(batch["reward"][:, :1]),
                batch["reward"][:, :-1],
            ],
            dim=1,
        ).unsqueeze(2).expand(-1, -1, self.n_agents, -1)

        inputs = [observations, previous_actions, previous_rewards]
        if self.include_agent_id:
            agent_ids = self._agent_ids(batch_size, batch.device)
            inputs.append(
                agent_ids.unsqueeze(1).expand(-1, time_steps, -1, -1)
            )
        inputs = th.cat(inputs, dim=-1)
        encoded = self.filter_encoder(
            inputs.reshape(batch_size * time_steps * self.n_agents, -1)
        ).reshape(batch_size, time_steps, self.n_agents, self.hidden_dim)

        encoded = encoded.permute(0, 2, 1, 3).reshape(
            batch_size * self.n_agents, time_steps, self.hidden_dim
        )
        recurrent_output, final_hidden = self.filter_rnn(
            encoded,
            hidden.reshape(
                1, batch_size * self.n_agents, self.hidden_dim
            ).contiguous(),
        )
        recurrent_output = recurrent_output.reshape(
            batch_size, self.n_agents, time_steps, self.hidden_dim
        ).permute(0, 2, 1, 3)
        evidence_logits = self.evidence_head(recurrent_output) / self.temperature

        beliefs = []
        logits = []
        priors = []
        shifts = []

        uniform = th.full_like(previous_belief, 1.0 / self.n_contexts)
        for timestep in range(time_steps):
            predictive_prior = (
                self.transition_stay * previous_belief
                + (1.0 - self.transition_stay) * uniform
            )
            posterior_logits = (
                evidence_logits[:, timestep]
                + self.prior_strength
                * predictive_prior.clamp_min(1e-8).log()
            )
            belief = F.softmax(posterior_logits, dim=-1)
            shift = normalized_js_divergence(belief, predictive_prior)
            beliefs.append(belief)
            logits.append(posterior_logits)
            priors.append(predictive_prior)
            shifts.append(shift)
            previous_belief = belief

        return {
            "belief": th.stack(beliefs, dim=1),
            "logits": th.stack(logits, dim=1),
            "predictive_prior": th.stack(priors, dim=1),
            "shift": th.stack(shifts, dim=1),
        }

    def dynamics_loss(
        self,
        batch,
        beliefs,
        transition_mask,
        args,
        context_assignments=None,
    ):
        observations = batch["obs"].reshape(
            batch.batch_size,
            batch.max_seq_length,
            self.n_agents,
            self.obs_dim,
        )
        actions = batch["actions_onehot"][:, :-1]
        current_observations = observations[:, :-1]
        next_observations = observations[:, 1:]
        batch_size, time_steps, n_agents, _ = current_observations.shape

        decoder_inputs = [current_observations, actions]
        if self.include_agent_id:
            agent_ids = self._agent_ids(batch_size, observations.device)
            decoder_inputs.append(
                agent_ids.unsqueeze(1).expand(-1, time_steps, -1, -1)
            )
        decoder_inputs = th.cat(decoder_inputs, dim=-1)
        predictions = self.dynamics_decoder(
            decoder_inputs.reshape(batch_size * time_steps * n_agents, -1)
        )
        predictions = predictions.reshape(
            batch_size,
            time_steps,
            n_agents,
            self.n_contexts,
            self.projection_dim + 1,
        )
        predicted_delta = predictions[..., : self.projection_dim]
        predicted_reward = predictions[..., self.projection_dim]

        observation_delta = next_observations - current_observations
        projected_delta = F.linear(
            observation_delta,
            self.observation_projection,
        )
        delta_error = F.smooth_l1_loss(
            predicted_delta,
            projected_delta.unsqueeze(-2).expand_as(predicted_delta),
            reduction="none",
        ).mean(dim=-1)

        rewards = batch["reward"][:, :-1]
        rewards = rewards.unsqueeze(2).expand(-1, -1, n_agents, -1).squeeze(-1)
        reward_error = F.smooth_l1_loss(
            predicted_reward,
            rewards.unsqueeze(-1).expand_as(predicted_reward),
            reduction="none",
        )
        reward_weight = float(
            getattr(args, "belief_reward_loss_weight", 1.0)
        )
        per_context_error = delta_error + reward_weight * reward_error

        belief = beliefs[:, :-1]
        reconstruction_weights = belief
        if context_assignments is not None:
            reconstruction_weights = context_assignments[:, :-1]
        agent_mask = transition_mask.expand(-1, -1, n_agents)
        denominator = agent_mask.sum().clamp_min(1.0)
        reconstruction = (
            (reconstruction_weights * per_context_error).sum(dim=-1)
            * agent_mask
        ).sum() / denominator

        uniform_log_prob = -math.log(self.n_contexts)
        per_item_kl = (
            belief.clamp_min(1e-8)
            * (belief.clamp_min(1e-8).log() - uniform_log_prob)
        ).sum(dim=-1)
        prior_kl = (per_item_kl * agent_mask).sum() / denominator

        marginal = (
            belief * agent_mask.unsqueeze(-1)
        ).sum(dim=(0, 1, 2)) / denominator
        balance_kl = (
            marginal.clamp_min(1e-8)
            * (marginal.clamp_min(1e-8).log() - uniform_log_prob)
        ).sum()

        kl_weight = float(getattr(args, "belief_kl_weight", 1e-3))
        balance_weight = float(
            getattr(args, "belief_balance_weight", 1e-2)
        )
        total = reconstruction + kl_weight * prior_kl + balance_weight * balance_kl
        entropy = normalized_categorical_entropy(belief)
        entropy = (entropy * agent_mask).sum() / denominator
        return {
            "loss": total,
            "reconstruction": reconstruction,
            "prior_kl": prior_kl,
            "balance_kl": balance_kl,
            "entropy": entropy,
        }
