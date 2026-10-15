"""CoRAC (HC-RAC): continuous contexts for RAC through a hypernetwork.

RAC labels every replay episode with a one-hot ``bin(R)`` slot. Coarse slots merge
"did nothing" with "captured prey" episodes, and the slot thresholds have to be
tuned per environment. CoRAC instead gives every (episode, agent) a continuous
context

    c_e = [phi(g_e) ; sg(u_e)]

* ``phi(g_e)``: cosine features of the episode return, squashed to [0, 1] with a
  running mean/std (shared by all agents). It keeps RAC's signal, including the
  teammate luck that lets RAC escape relative overgeneralization, at full
  resolution and without thresholds.
* ``u_e``: a trajectory embedding learned with Trajectory Contrastive Learning
  (Wang et al., 2021): two windows of the same trajectory are positives; windows
  of other trajectories and "outcome-swapped" windows (the anchor's observations
  and actions with the outcomes of same-action neighbours from other episodes,
  as in CORRO) are negatives. It is stop-gradient, so TD never turns it into a
  return-to-go code.

The twin Q reads c_e through a hypernetwork (``twin_hyper_hidden`` > 0); the
optimistic teacher is the max over a bank of recent contexts. Labels are only used
for training; execution is unchanged from RAC.
See paper_revision/hc_rac_method.md.
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

        self.n_freq = int(getattr(args, "corac_n_freq", 16))
        self.embed_dim = int(getattr(args, "corac_embed_dim", 32))
        self.context_dim = self.n_freq + self.embed_dim
        configured = getattr(args, "context_dim", None)
        if configured is not None and int(configured) != self.context_dim:
            raise ValueError(
                "context_dim must equal corac_n_freq + corac_embed_dim "
                f"({self.context_dim}), got {configured}"
            )
        self.segment_len = int(getattr(args, "corac_segment_len", 1))
        self.temperature = float(getattr(args, "corac_temperature", 0.1))
        self.n_hard = int(getattr(args, "corac_n_hard", 4))
        self.pool_size = int(getattr(args, "corac_pool_size", 4096))
        self.stat_momentum = float(getattr(args, "corac_stat_momentum", 0.01))
        bank_size = int(getattr(args, "corac_bank_size", 64))

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
        self.register_buffer("frequencies", math.pi * th.arange(self.n_freq, dtype=th.float32))

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
        self.output = nn.Linear(hidden_dim, self.embed_dim)

        self.register_buffer("return_mean", th.zeros(()))
        self.register_buffer("return_var", th.ones(()))
        self.register_buffer("stats_initialised", th.zeros((), dtype=th.bool))
        # FIFO bank of recent contexts per agent: the support of the teacher's max.
        self.register_buffer("bank", th.zeros(n_agents, bank_size, self.context_dim))
        self.register_buffer("bank_count", th.zeros((), dtype=th.long))

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
        """Mean embedding of non-overlapping windows, shape (bs, n_agents, d).

        Episodes too short for two windows (no contrastive pair) get a zero vector.
        """
        bs, steps = features.shape[:2]
        seg = self.segment_len
        n_segments = steps // seg
        has_pair = lengths >= 2 * seg
        if n_segments == 0:
            return features.new_zeros(bs, self.n_agents, self.embed_dim)
        windows = features[:, : n_segments * seg].reshape(
            bs, n_segments, seg, self.n_agents, -1
        ).transpose(2, 3)  # (bs, n_seg, n_agents, L, F)
        embeddings = self.encode(windows)  # (bs, n_seg, n_agents, d)
        seg_ends = (th.arange(n_segments, device=features.device) + 1) * seg
        valid = (seg_ends.unsqueeze(0) <= lengths.unsqueeze(1)).float()  # (bs, n_seg)
        counts = valid.sum(dim=1).clamp(min=1.0)
        mean = (embeddings * valid.view(bs, n_segments, 1, 1)).sum(dim=1) / counts.view(bs, 1, 1)
        return F.normalize(mean, dim=-1) * has_pair.view(bs, 1, 1).float()

    # ------------------------------------------------------------------ #
    # Trajectory contrastive objective
    # ------------------------------------------------------------------ #
    def _sample_pairs(self, lengths):
        """Sample (episode, t1, t2): two disjoint windows of the same trajectory."""
        seg = self.segment_len
        rows, t1s, t2s = [], [], []
        for b, length in enumerate(lengths.tolist()):
            last_start = int(length) - seg
            if last_start < seg:
                continue
            t1 = int(th.randint(0, last_start + 1, (1,)).item())
            candidates = [t for t in range(last_start + 1) if abs(t - t1) >= seg]
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
        episode_of_row = rows_t.repeat_interleave(self.n_agents)
        agent_of_row = th.arange(self.n_agents, device=device).repeat(len(rows))

        hard, hard_valid = self._outcome_swapped_negatives(
            features, step_mask, anchor, episode_of_row, agent_of_row
        )

        u = self.encode(anchor)
        u_pos = self.encode(positive)
        pos_logit = (u * u_pos).sum(-1, keepdim=True) / self.temperature  # (N, 1)

        easy_logits = u @ u_pos.t() / self.temperature  # (N, N)
        # Same-episode rows (the positive, other agents of the episode) are not negatives.
        same_episode = episode_of_row.unsqueeze(0) == episode_of_row.unsqueeze(1)
        easy_logits = easy_logits.masked_fill(same_episode, float("-inf"))

        negatives = [easy_logits]
        if hard is not None:
            u_hard = self.encode(hard)  # (N, n_hard, d)
            hard_logits = (u.unsqueeze(1) * u_hard).sum(-1) / self.temperature
            hard_logits = hard_logits.masked_fill(~hard_valid, float("-inf"))
            negatives.append(hard_logits)
        logits = th.cat([pos_logit] + negatives, dim=1)
        loss = -F.log_softmax(logits, dim=1)[:, 0]

        with th.no_grad():
            stats = {"corac_nce_accuracy": (logits.argmax(dim=1) == 0).float().mean().item()}
            if hard is not None:
                beats_hard = (pos_logit.squeeze(1) > hard_logits.max(dim=1)[0]).float()
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

        outcomes = pool[:, start:stop][nearest]  # (n_rows*L, k, O)
        outcomes = outcomes.view(n_rows, seg, k, -1).transpose(1, 2)  # (n_rows, k, L, O)
        hard = self._swap_outcomes(anchor.unsqueeze(1).expand(-1, k, -1, -1), outcomes)
        valid = found.view(n_rows, seg, k).all(dim=1)
        return hard, valid

    # ------------------------------------------------------------------ #
    # Contexts
    # ------------------------------------------------------------------ #
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

    def return_features(self, returns):
        """phi(g): cosine features of the return squashed to [0, 1], shape (..., n_freq)."""
        z = (returns - self.return_mean) / self.return_var.clamp(min=1e-6).sqrt()
        squashed = th.sigmoid(1.702 * z)  # ~ standard normal CDF, order preserving
        return th.cos(squashed.unsqueeze(-1) * self.frequencies)

    def _push_bank(self, contexts):
        """contexts (bs, n_agents, C) -> FIFO bank (n_agents, M, C)."""
        size = self.bank.shape[1]
        contexts = contexts.detach().transpose(0, 1)[:, -size:]  # (n_agents, B, C)
        start = int(self.bank_count) % size
        idx = (th.arange(contexts.shape[1], device=contexts.device) + start) % size
        self.bank[:, idx] = contexts
        self.bank_count.add_(contexts.shape[1])

    def support(self):
        """Bank of recent contexts, (n_agents, M_filled, C): the support of the teacher's max."""
        filled = int(min(int(self.bank_count), self.bank.shape[1]))
        return self.bank[:, :filled]

    def label(self, batch, mask, t_env):
        """Return contexts (bs, T+1, n_agents, C), the TCL loss (or None) and stats."""
        features = self.step_features(batch)
        step_mask = mask[:, :, 0] if mask.dim() == 3 else mask
        lengths = step_mask.sum(dim=1).long()
        returns = (batch["reward"][:, :-1, 0] * step_mask).sum(dim=1)  # (bs,)
        self._update_return_stats(returns)

        loss, stats = self.contrastive_loss(features, lengths, step_mask)
        with th.no_grad():
            embedding = self.episode_embedding(features, lengths)  # (bs, n_agents, d), sg
            phi = self.return_features(returns).unsqueeze(1).expand(-1, self.n_agents, -1)
            contexts = th.cat([phi, embedding], dim=-1)  # (bs, n_agents, C)
        self._push_bank(contexts)

        if loss is not None:
            stats["corac_nce_loss"] = loss.item()
        stats["corac_bank_size"] = float(min(int(self.bank_count), self.bank.shape[1]))
        steps = batch["obs"].shape[1]
        return contexts.unsqueeze(1).expand(-1, steps, -1, -1).contiguous(), loss, stats
