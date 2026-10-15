"""CoRAC: contrastive, return-anchored context labels for RAC.

RAC labels every replay episode with ``bin(R)``, so environment luck that
inflates the episode return leaks into the optimistic context. CoRAC instead

1. learns a segment encoder with InfoNCE whose positives are two disjoint
   segments of the *same* episode (teammate behaviour is persistent within an
   episode, per-step payoff noise is not) and whose hard negatives keep the
   anchor's (observation, action) sequence but swap in the outcomes
   (reward, observation change) of nearest neighbours from other episodes;
2. regresses the normalised episode return from the stop-gradient episode
   embedding, giving an estimate of E[R | persistent behaviour];
3. turns that estimate into an ordinal context with a running empirical CDF
   and equal-mass bins.

The luck knob ``lambda`` mixes the encoder estimate with the realised return
before ranking: lambda = 1 is RAC with quantile bins, lambda = 0 is pure CoRAC.
Labels are only used for training; execution is unchanged from RAC.
"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveContextLabeler(nn.Module):
    def __init__(self, obs_dim, n_actions, n_agents, args):
        super().__init__()
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.slot_number = int(args.slot_number)

        self.segment_len = int(getattr(args, "corac_segment_len", 1))
        self.gap = int(getattr(args, "corac_gap", 0))
        self.temperature = float(getattr(args, "corac_temperature", 0.1))
        self.n_hard = int(getattr(args, "corac_n_hard", 4))
        self.tau_plus = float(getattr(args, "corac_tau_plus", 1.0 / self.slot_number))
        self.debias = bool(getattr(args, "corac_debias", True))
        self.lambda_final = float(getattr(args, "corac_lambda_final", 0.0))
        self.warmup_steps = int(getattr(args, "corac_warmup_steps", 10000))
        self.blend_steps = int(getattr(args, "corac_blend_steps", 10000))
        self.stat_momentum = float(getattr(args, "corac_stat_momentum", 0.01))
        self.pool_size = int(getattr(args, "corac_pool_size", 4096))
        self.train_encoder = self.lambda_final < 1.0

        embed_dim = int(getattr(args, "corac_embed_dim", 32))
        hidden_dim = int(getattr(args, "corac_hidden_dim", 64))
        proj_dim = min(obs_dim, int(getattr(args, "corac_proj_dim", 32)))

        # Fixed random projections: obs-change features and the kNN key.
        generator = th.Generator().manual_seed(int(getattr(args, "seed", 0) or 0))
        self.register_buffer(
            "delta_proj",
            th.randn(obs_dim, proj_dim, generator=generator) / math.sqrt(proj_dim),
        )
        self.register_buffer(
            "key_proj",
            th.randn(obs_dim, proj_dim, generator=generator) / math.sqrt(proj_dim),
        )

        # x_t = [o_t, onehot(a_t), r_t, P(o_{t+1} - o_t), agent_id]
        self.n_state_features = obs_dim + n_actions
        self.n_outcome_features = 1 + proj_dim
        step_dim = self.n_state_features + self.n_outcome_features + n_agents
        self.step_net = nn.Sequential(
            nn.Linear(step_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.output = nn.Linear(hidden_dim, embed_dim)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        quantile_size = int(getattr(args, "corac_quantile_buffer", 4096))
        self.register_buffer("quantile_values", th.zeros(quantile_size))
        self.register_buffer("quantile_count", th.zeros((), dtype=th.long))
        self.register_buffer("return_mean", th.zeros(()))
        self.register_buffer("return_var", th.ones(()))
        self.register_buffer("stats_initialised", th.zeros((), dtype=th.bool))

    # ------------------------------------------------------------------ #
    # Features and encoder
    # ------------------------------------------------------------------ #
    def step_features(self, batch):
        """Per-step features, shape (bs, T, n_agents, F), for t = 0..T-1."""
        obs = batch["obs"]
        bs, t_plus_one = obs.shape[:2]
        steps = t_plus_one - 1
        delta = (obs[:, 1:] - obs[:, :-1]) @ self.delta_proj
        reward = batch["reward"][:, :-1].unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        agent_ids = th.eye(self.n_agents, device=obs.device).view(
            1, 1, self.n_agents, self.n_agents
        ).expand(bs, steps, -1, -1)
        return th.cat(
            [
                obs[:, :-1],
                batch["actions_onehot"][:, :-1],
                reward,
                delta,
                agent_ids,
            ],
            dim=-1,
        )

    def encode(self, segments):
        """Encode segments of shape (..., L, F) into unit vectors (..., d)."""
        hidden = self.step_net(segments)
        weights = th.softmax(self.attention(hidden), dim=-2)
        pooled = (weights * hidden).sum(dim=-2)
        return F.normalize(self.output(pooled), dim=-1)

    def _swap_outcomes(self, segments, outcomes):
        """Replace the (reward, obs-change) block of every step."""
        start = self.n_state_features
        stop = start + self.n_outcome_features
        return th.cat([segments[..., :start], outcomes, segments[..., stop:]], dim=-1)

    def episode_embedding(self, features, lengths):
        """Mean embedding of non-overlapping segments, shape (bs, n_agents, d)."""
        bs, steps = features.shape[:2]
        seg = self.segment_len
        n_segments = steps // seg
        if n_segments == 0:
            return None, th.zeros(bs, dtype=th.bool, device=features.device)
        windows = features[:, : n_segments * seg].reshape(
            bs, n_segments, seg, self.n_agents, -1
        ).transpose(2, 3)  # (bs, n_seg, n_agents, L, F)
        embeddings = self.encode(windows)  # (bs, n_seg, n_agents, d)
        seg_ends = (th.arange(n_segments, device=features.device) + 1) * seg
        valid = (seg_ends.unsqueeze(0) <= lengths.unsqueeze(1)).float()  # (bs, n_seg)
        counts = valid.sum(dim=1).clamp(min=1.0)
        mean = (embeddings * valid.view(bs, n_segments, 1, 1)).sum(dim=1) / counts.view(bs, 1, 1)
        return F.normalize(mean, dim=-1), valid.sum(dim=1) > 0

    # ------------------------------------------------------------------ #
    # Contrastive objective
    # ------------------------------------------------------------------ #
    def _sample_pairs(self, lengths):
        """Sample (episode, t1, t2) with two disjoint segments of length L."""
        seg, min_sep = self.segment_len, self.segment_len + self.gap
        rows, t1s, t2s = [], [], []
        for b, length in enumerate(lengths.tolist()):
            last_start = int(length) - seg
            if last_start < min_sep:
                continue
            t1 = int(th.randint(0, last_start + 1, (1,)).item())
            candidates = [t for t in range(last_start + 1) if abs(t - t1) >= min_sep]
            t2 = candidates[int(th.randint(0, len(candidates), (1,)).item())]
            rows.append(b)
            t1s.append(t1)
            t2s.append(t2)
        return rows, t1s, t2s

    def contrastive_loss(self, features, lengths, step_mask):
        rows, t1s, t2s = self._sample_pairs(lengths)
        if len(rows) < 2:
            return None, {}
        device = features.device
        seg = self.segment_len
        offsets = th.arange(seg, device=device)
        rows_t = th.tensor(rows, device=device)
        idx1 = th.tensor(t1s, device=device).unsqueeze(1) + offsets  # (E, L)
        idx2 = th.tensor(t2s, device=device).unsqueeze(1) + offsets

        # (E, L, n_agents, F) -> (E * n_agents, L, F); row = episode-major
        anchor = features[rows_t.unsqueeze(1), idx1].transpose(1, 2).reshape(-1, seg, features.shape[-1])
        positive = features[rows_t.unsqueeze(1), idx2].transpose(1, 2).reshape(-1, seg, features.shape[-1])
        n_rows = anchor.shape[0]
        episode_of_row = rows_t.repeat_interleave(self.n_agents)
        agent_of_row = th.arange(self.n_agents, device=device).repeat(len(rows))

        hard, hard_valid = self._outcome_swapped_negatives(
            features, step_mask, anchor, episode_of_row, agent_of_row
        )

        u = self.encode(anchor)
        u_pos = self.encode(positive)
        pos_logit = (u * u_pos).sum(-1, keepdim=True) / self.temperature  # (N, 1)

        easy_logits = u @ u_pos.t() / self.temperature  # (N, N)
        # Same-episode rows (other agents of the same episode) are not negatives.
        same_episode = episode_of_row.unsqueeze(0) == episode_of_row.unsqueeze(1)
        easy_logits = easy_logits.masked_fill(same_episode, float("-inf"))

        negatives = [easy_logits]
        if hard is not None:
            u_hard = self.encode(hard)  # (N, n_hard, d)
            hard_logits = (u.unsqueeze(1) * u_hard).sum(-1) / self.temperature
            negatives.append(hard_logits.masked_fill(~hard_valid, float("-inf")))
        neg_logits = th.cat(negatives, dim=1)

        if self.debias:
            # Chuang et al. (2020): correct the negative term for false negatives.
            neg_exp = th.exp(neg_logits)
            n_neg = th.isfinite(neg_logits).sum(dim=1).clamp(min=1).float()
            pos_exp = th.exp(pos_logit.squeeze(1))
            neg_mean = neg_exp.sum(dim=1) / n_neg
            corrected = (neg_mean - self.tau_plus * pos_exp) / (1.0 - self.tau_plus)
            corrected = th.clamp(corrected, min=math.exp(-1.0 / self.temperature))
            loss = -(pos_logit.squeeze(1) - th.log(pos_exp + n_neg * corrected))
        else:
            logits = th.cat([pos_logit, neg_logits], dim=1)
            loss = -F.log_softmax(logits, dim=1)[:, 0]

        with th.no_grad():
            all_logits = th.cat([pos_logit, neg_logits], dim=1)
            accuracy = (all_logits.argmax(dim=1) == 0).float().mean()
            stats = {"corac_nce_accuracy": accuracy.item()}
            if hard is not None:
                hard_masked = hard_logits.masked_fill(~hard_valid, float("-inf"))
                beats_hard = (pos_logit.squeeze(1) > hard_masked.max(dim=1)[0]).float()
                stats["corac_pos_beats_hard"] = beats_hard.mean().item()
                stats["corac_hard_valid"] = hard_valid.float().mean().item()
        return loss.mean(), stats

    def _outcome_swapped_negatives(self, features, step_mask, anchor, episode_of_row, agent_of_row):
        """Anchor (o, a) sequence with outcomes from same-action neighbours.

        Returns (n_rows, n_hard, L, F) segments and a (n_rows, n_hard) mask.
        """
        if self.n_hard <= 0:
            return None, None
        device = features.device
        bs, steps, n_agents, n_feat = features.shape
        obs_dim = self.delta_proj.shape[0]
        n_rows, seg = anchor.shape[:2]

        # Candidate pool: every valid (episode, step, agent).
        pool = features.reshape(-1, n_feat)
        pool_episode = th.arange(bs, device=device).view(bs, 1, 1).expand(bs, steps, n_agents).reshape(-1)
        pool_agent = th.arange(n_agents, device=device).view(1, 1, n_agents).expand(bs, steps, n_agents).reshape(-1)
        pool_valid = step_mask.reshape(bs, steps, 1).expand(bs, steps, n_agents).reshape(-1) > 0
        # Long episodes (e.g. 200 steps x 8 agents): search a random valid subset.
        valid_idx = pool_valid.nonzero(as_tuple=True)[0]
        if valid_idx.numel() > self.pool_size:
            valid_idx = valid_idx[th.randperm(valid_idx.numel(), device=device)[: self.pool_size]]
        pool, pool_episode, pool_agent = pool[valid_idx], pool_episode[valid_idx], pool_agent[valid_idx]
        pool_valid = pool_valid[valid_idx]
        pool_action = pool[:, obs_dim: obs_dim + self.n_actions].argmax(-1)
        pool_key = pool[:, :obs_dim] @ self.key_proj

        # Anchor steps flattened: (n_rows * L, F).
        flat = anchor.reshape(-1, n_feat)
        flat_episode = episode_of_row.repeat_interleave(seg)
        flat_agent = agent_of_row.repeat_interleave(seg)
        flat_action = flat[:, obs_dim: obs_dim + self.n_actions].argmax(-1)
        flat_key = flat[:, :obs_dim] @ self.key_proj

        start = self.n_state_features
        stop = start + self.n_outcome_features
        dist = th.cdist(flat_key, pool_key)  # (n_rows * L, P)
        # Random tie-breaking: identical observations are common.
        dist = dist + 1e-3 * th.rand_like(dist)
        # With discrete outcomes a neighbour often has exactly the anchor's
        # outcome; swapping it in would reproduce the anchor, not a negative.
        new_outcome = th.cdist(flat[:, start:stop], pool[:, start:stop]) > 1e-6
        allowed = (
            (flat_action.unsqueeze(1) == pool_action.unsqueeze(0))
            & (flat_agent.unsqueeze(1) == pool_agent.unsqueeze(0))
            & (flat_episode.unsqueeze(1) != pool_episode.unsqueeze(0))
            & pool_valid.unsqueeze(0)
            & new_outcome
        )
        dist = dist.masked_fill(~allowed, float("inf"))
        k = min(self.n_hard, dist.shape[1])
        nearest_dist, nearest = th.topk(dist, k, dim=1, largest=False)  # (n_rows * L, k)
        found = th.isfinite(nearest_dist)

        outcomes = pool[:, start: start + self.n_outcome_features][nearest]  # (n_rows*L, k, O)
        outcomes = outcomes.view(n_rows, seg, k, -1).transpose(1, 2)  # (n_rows, k, L, O)
        hard = self._swap_outcomes(anchor.unsqueeze(1).expand(-1, k, -1, -1), outcomes)
        valid = found.view(n_rows, seg, k).all(dim=1)
        return hard, valid

    # ------------------------------------------------------------------ #
    # Return anchoring and labels
    # ------------------------------------------------------------------ #
    def luck_weight(self, t_env):
        if self.lambda_final >= 1.0 or t_env < self.warmup_steps:
            return 1.0
        progress = min(1.0, (t_env - self.warmup_steps) / max(self.blend_steps, 1))
        return 1.0 + progress * (self.lambda_final - 1.0)

    def _update_return_stats(self, returns):
        mean, var = returns.mean(), returns.var(unbiased=False)
        if not bool(self.stats_initialised):
            self.return_mean.copy_(mean)
            self.return_var.copy_(var.clamp(min=1e-6))
            self.stats_initialised.fill_(True)
        else:
            m = self.stat_momentum
            self.return_mean.mul_(1 - m).add_(m * mean)
            self.return_var.mul_(1 - m).add_(m * var)

    def _push_quantile_values(self, values):
        size = self.quantile_values.shape[0]
        values = values.detach().reshape(-1)[-size:]
        start = int(self.quantile_count) % size
        idx = (th.arange(values.shape[0], device=values.device) + start) % size
        self.quantile_values[idx] = values
        self.quantile_count.add_(values.shape[0])

    def _empirical_cdf(self, values):
        filled = int(min(int(self.quantile_count), self.quantile_values.shape[0]))
        reference = self.quantile_values[:filled]
        below = (reference.unsqueeze(0) < values.reshape(-1, 1)).float().sum(1)
        equal = (reference.unsqueeze(0) == values.reshape(-1, 1)).float().sum(1)
        # Mid-rank so that tied returns share a bin.
        return ((below + 0.5 * equal) / max(filled, 1)).view_as(values)

    def label(self, batch, mask, t_env):
        """Return one-hot contexts (bs, T+1, n_agents, K) and the auxiliary loss."""
        features = self.step_features(batch)
        step_mask = mask[:, :, 0] if mask.dim() == 3 else mask
        lengths = step_mask.sum(dim=1).long()
        returns = (batch["reward"][:, :-1, 0] * step_mask).sum(dim=1)  # (bs,)
        self._update_return_stats(returns)
        z = (returns - self.return_mean) / self.return_var.clamp(min=1e-6).sqrt()

        lam = self.luck_weight(t_env)
        aux_loss = None
        stats = {"corac_lambda": lam}
        values = z.unsqueeze(1).expand(-1, self.n_agents)

        if self.train_encoder:
            nce, nce_stats = self.contrastive_loss(features, lengths, step_mask)
            stats.update(nce_stats)
            with th.no_grad():
                embedding, has_segments = self.episode_embedding(features, lengths)
            if embedding is not None and has_segments.any():
                prediction = self.value_head(embedding).squeeze(-1)  # (bs, n_agents)
                target = z.unsqueeze(1).expand_as(prediction)
                head_loss = ((prediction - target) ** 2)[has_segments].mean()
                aux_loss = head_loss if nce is None else nce + head_loss
                stats["corac_head_loss"] = head_loss.item()
                if nce is not None:
                    stats["corac_nce_loss"] = nce.item()
                mixed = (1.0 - lam) * prediction.detach() + lam * values
                # Episodes too short for a segment keep the realised return.
                values = th.where(has_segments.unsqueeze(1), mixed, values)

        kappa = self._empirical_cdf(values) if int(self.quantile_count) > 0 else th.full_like(values, 0.5)
        self._push_quantile_values(values)
        context = th.clamp((kappa * self.slot_number).long(), 0, self.slot_number - 1)

        steps = batch["obs"].shape[1]
        onehot = F.one_hot(context, self.slot_number).float()  # (bs, n_agents, K)
        onehot = onehot.unsqueeze(1).expand(-1, steps, -1, -1).contiguous()
        stats["corac_context_entropy"] = _entropy(context, self.slot_number)
        return onehot, context, aux_loss, stats


def _entropy(context, slot_number):
    counts = th.bincount(context.reshape(-1), minlength=slot_number).float()
    probs = counts / counts.sum().clamp(min=1.0)
    return float(-(probs * th.log(probs.clamp(min=1e-12))).sum().item())
