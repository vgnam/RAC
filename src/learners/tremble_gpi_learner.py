import copy

import numpy as np
import torch as th
from torch.optim import RMSprop

from src.learners.tremble_iql_learner import agent_features


class TrembleGPILearner:
    """TR-IQL v4: per-state optimism over teammates' intentions via block heads (GPI).

    Blocks are the same as in TR-IQL v2: every agent acts with a frozen policy
    snapshot for ``block_len`` episodes (``TrembleBlockMAC`` with ``freeze_acting``),
    so teammates' intentions are fixed within a block and only tremble varies.

    * Block heads. Each of the last ``gpi_blocks`` blocks owns two linear heads on
      the shared RNN torso (one per half of its episodes, alternating). A head is
      trained by double-Q TD with a target network only on its own block's episodes,
      bootstrapping on itself: Q_b(s, a) estimates the value of a when teammates
      follow block b's intentions, averaged over execution tremble.
    * Optimism. Q_opt(s, a) = Q_{b*, B}(s, a) with b* = argmax_b Q_{b, A}(s, a) over
      the live blocks: a per-state max over teammates' intentions (the double
      estimator of TR-IQL: select on one half of a block, evaluate on the other).
      In a single-state game Q_b(a_i) = r_eps(a_i, a_-i^b), so Q_opt is the TR-IQL
      target max_{a_-i} r_eps(a_i, a_-i); in multi-step games it is generalised
      policy improvement over the blocks' joint policies.
    * Coverage. A head enters the max for (agent, action) only after both halves
      have taken it ``gpi_min_count`` times; if no live head covers it, the target
      is the current decision value (no update), so untried actions do not drift.
    * Decision. The acting output layer ``agent.fc2`` is regressed onto Q_opt
      (weight ``distill_weight``); greedy actions follow it.

    A head is re-initialised from ``agent.fc2`` when its block slot is reused.
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        agent = mac.agent
        assert hasattr(agent, "fc2") and hasattr(agent, "encoder") and hasattr(agent, "recurrent"), \
            "TrembleGPILearner needs the default RNN agent (agent: rnn; agent_recurrent may be False)"

        self.H = int(args.gpi_blocks)
        self.block_len = int(args.block_len)
        self.min_count = int(args.gpi_min_count)
        hidden = agent.fc2.in_features
        self.heads = th.nn.Linear(hidden, 2 * self.H * self.n_actions)
        for h in range(self.H):
            self._reset_head(h, also_target=False)
        self.target_agent = copy.deepcopy(agent)
        self.target_heads = copy.deepcopy(self.heads)

        self.params = list(mac.parameters()) + list(self.heads.parameters())
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.buffer = None
        self.slot_owner = None   # block id of each replay slot (-1: empty)
        self.slot_half = None    # 0 / 1: which head of its block the episode trains
        self.counts = np.zeros((2 * self.H, self.n_agents, self.n_actions), dtype=np.int64)
        self.cur_block = 0
        self.ep_in_block = 0
        self.block_returns = []
        self.last_block_mean = 0.0
        self.best_block_mean = -np.inf

        self.last_target_update_episode = 0
        steps = getattr(args, "target_update_interval_steps", None)
        self.target_update_interval_steps = int(steps) if steps is not None else None
        self.last_target_update_t = 0
        self.log_stats_t = -args.learner_log_interval - 1

    # ----- block bookkeeping -------------------------------------------------------

    def set_buffer(self, buffer):
        """Share run.py's replay buffer; episodes are tagged with their block."""
        self.buffer = buffer
        self.slot_owner = np.full(buffer.buffer_size, -1, dtype=np.int64)
        self.slot_half = np.zeros(buffer.buffer_size, dtype=np.int64)

    def _head_index(self, block, half):
        return (block % self.H) * 2 + half

    def _reset_head(self, h, also_target=True):
        """Both heads of block slot h <- the decision layer; coverage counts cleared."""
        fc2 = self.mac.agent.fc2
        modules = [self.heads] + ([self.target_heads] if also_target else [])
        with th.no_grad():
            for module in modules:
                w = module.weight.view(2 * self.H, self.n_actions, -1)
                b = module.bias.view(2 * self.H, self.n_actions)
                w[2 * h:2 * h + 2] = fc2.weight.to(w.device)
                b[2 * h:2 * h + 2] = fc2.bias.to(b.device)
        if hasattr(self, "counts"):
            self.counts[2 * h:2 * h + 2] = 0

    def observe_episode(self, episode_batch, t_env):
        assert self.buffer is not None, "TrembleGPILearner needs set_buffer() (see run.py)"
        assert episode_batch.batch_size == 1, "TrembleGPILearner requires batch_size_run=1"
        # run.py inserts the episode into the buffer just before calling this.
        slot = (self.buffer.buffer_index - 1) % self.buffer.buffer_size
        half = self.ep_in_block % 2
        self.slot_owner[slot] = self.cur_block
        self.slot_half[slot] = half

        filled = episode_batch["filled"][0, :-1, 0].bool()
        if self.min_count > 0:
            terminated = episode_batch["terminated"][0, :-1, 0].bool()
            valid = filled.clone()
            valid[1:] &= ~terminated[:-1]
            actions = episode_batch["actions"][0, :-1, :, 0][valid].cpu().numpy()  # (T', n_agents)
            h = self._head_index(self.cur_block, half)
            for i in range(self.n_agents):
                self.counts[h, i] += np.bincount(actions[:, i], minlength=self.n_actions)
        self.block_returns.append(episode_batch["reward"].sum().item())

        self.ep_in_block += 1
        if self.ep_in_block >= self.block_len:
            self.last_block_mean = float(np.mean(self.block_returns))
            self.best_block_mean = max(self.best_block_mean, self.last_block_mean)
            self.block_returns = []
            self.cur_block += 1
            self.ep_in_block = 0
            self._reset_head(self.cur_block % self.H)  # its previous block falls out of the window
            self.mac.new_block(t_env)

    def _live_blocks(self):
        first = max(0, self.cur_block - self.H + 1)
        return np.arange(first, self.cur_block + 1)

    # ----- training ----------------------------------------------------------------

    def train(self, batch, t_env, episode_num):
        """One step on episodes of the live blocks (``batch`` from run.py is unused)."""
        n_stored = self.buffer.episodes_in_buffer
        first = max(0, self.cur_block - self.H + 1)
        live_slots = np.nonzero(self.slot_owner[:n_stored] >= first)[0]
        if len(live_slots) == 0:
            return
        ids = np.random.choice(live_slots, self.args.batch_size, replace=len(live_slots) < self.args.batch_size)
        owners, halves = self.slot_owner[ids], self.slot_half[ids]
        batch = self.buffer[ids]
        batch = batch[:, :batch.max_t_filled()]
        if batch.device != self.args.device:
            batch.to(self.args.device)
        device = batch.device
        bs, T = batch.batch_size, batch.max_seq_length
        n, A, H2 = self.n_agents, self.n_actions, 2 * self.H

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail = batch["avail_actions"]

        head_idx = th.as_tensor(owners % self.H * 2 + halves, device=device)
        rows = th.arange(bs, device=device)
        covered = th.as_tensor(self.counts >= max(self.min_count, 1), device=device)  # (2H, n, A)

        # Block heads: TD on their own block's episodes, bootstrapping on themselves.
        feats = agent_features(self.mac, self.mac.agent, batch, n)
        q_heads = self.heads(feats).view(bs, T, n, H2, A)
        q_b = q_heads.permute(0, 3, 1, 2, 4)[rows, head_idx]  # (bs, T, n, A)
        chosen = th.gather(q_b[:, :-1], dim=3, index=actions).squeeze(3)
        with th.no_grad():
            t_feats = agent_features(self.mac, self.target_agent, batch, n)
            t_heads = self.target_heads(t_feats).view(bs, T, n, H2, A)
            t_b = t_heads.permute(0, 3, 1, 2, 4)[rows, head_idx][:, 1:]
            usable = (avail[:, 1:] > 0)
            if self.min_count > 0:
                cov_b = covered[head_idx].unsqueeze(1)  # (bs, 1, n, A)
                usable_cov = usable & cov_b
                usable = th.where(usable_cov.any(dim=3, keepdim=True), usable_cov, usable)
            q_next = q_b[:, 1:].detach().masked_fill(~usable, -9999999)
            best_next = q_next.max(dim=3, keepdim=True)[1]
            target_max = th.gather(t_b, 3, best_next).squeeze(3)
            targets = rewards + self.args.gamma * (1 - terminated) * target_max
        td_mask = mask.expand_as(chosen)
        td_loss = (((chosen - targets) * td_mask) ** 2).sum() / td_mask.sum()

        # Optimistic target: per-state max over live blocks (select on half A, value on half B).
        with th.no_grad():
            live = self._live_blocks()
            h_a = th.as_tensor(live % self.H * 2, device=device)
            h_b = h_a + 1
            q_sel, q_val = t_heads[:, :, :, h_a], t_heads[:, :, :, h_b]  # (bs, T, n, L, A)
            ok = (covered[h_a] & covered[h_b]).permute(1, 0, 2)[None, None]  # (1, 1, n, L, A)
            any_ok = ok.any(dim=3)  # (1, 1, n, A)
            best = q_sel.masked_fill(~ok, -float("inf")).argmax(dim=3, keepdim=True)
            q_gpi = th.gather(q_val, 3, best).squeeze(3)  # (bs, T, n, A)
            q_dec_target = self.target_agent.fc2(t_feats)
            q_opt = th.where(any_ok.expand_as(q_gpi), q_gpi, q_dec_target)

        q_dec = self.mac.agent.fc2(feats)
        d_mask = mask.unsqueeze(3) * (avail[:, :-1] > 0).float()
        distill_loss = (((q_dec[:, :-1] - q_opt[:, :-1]) ** 2) * d_mask).sum() / d_mask.sum().clamp(min=1)

        loss = td_loss + float(self.args.distill_weight) * distill_loss
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if self.target_update_interval_steps is not None:
            update = t_env - self.last_target_update_t >= self.target_update_interval_steps
        else:
            update = (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0
        if update:
            self.target_agent.load_state_dict(self.mac.agent.state_dict())
            self.target_heads.load_state_dict(self.heads.state_dict())
            self.last_target_update_episode = episode_num
            self.last_target_update_t = t_env

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            n_td = td_mask.sum().item()
            n_d = d_mask.sum().item()
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("distill_loss", distill_loss.item(), t_env)
            self.logger.log_stat("grad_norm", float(grad_norm), t_env)
            self.logger.log_stat("q_taken_mean", (chosen * td_mask).sum().item() / n_td, t_env)
            self.logger.log_stat("target_mean", (targets * td_mask).sum().item() / n_td, t_env)
            self.logger.log_stat("q_opt_mean", (q_opt[:, :-1] * d_mask).sum().item() / max(n_d, 1), t_env)
            self.logger.log_stat("q_dec_mean", (q_dec[:, :-1].detach() * d_mask).sum().item() / max(n_d, 1), t_env)
            self.logger.log_stat("gpi_gain_mean",
                                 ((q_opt - q_dec_target)[:, :-1] * d_mask).sum().item() / max(n_d, 1), t_env)
            self.logger.log_stat("gpi_covered_frac", any_ok.float().mean().item(), t_env)
            self.logger.log_stat("n_blocks", float(self.cur_block), t_env)
            self.logger.log_stat("block_last_mean", self.last_block_mean, t_env)
            if np.isfinite(self.best_block_mean):
                self.logger.log_stat("block_best_mean", self.best_block_mean, t_env)
            self.log_stats_t = t_env

    # ----- misc --------------------------------------------------------------------

    def show_matrix_info(self, batch, t_env):
        with th.no_grad():
            q = self.mac.agent.fc2(agent_features(self.mac, self.mac.agent, batch, self.n_agents))[0, 0]
        rows = ["agent {}: {}".format(i, " ".join("{:7.3f}".format(v) for v in q[i].tolist()))
                for i in range(self.n_agents)]
        print("t_env {} TR-IQL v4 Q(intended action):\n  {}".format(t_env, "\n  ".join(rows)))

    def cuda(self):
        self.mac.cuda()
        self.heads.to(self.args.device)
        self.target_agent.to(self.args.device)
        self.target_heads.to(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.heads.state_dict(), "{}/heads.th".format(path))
        th.save(self.target_agent.state_dict(), "{}/target_agent.th".format(path))
        th.save(self.target_heads.state_dict(), "{}/target_heads.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        load = lambda f: th.load("{}/{}".format(path, f), map_location=lambda storage, loc: storage)
        self.heads.load_state_dict(load("heads.th"))
        self.target_agent.load_state_dict(load("target_agent.th"))
        self.target_heads.load_state_dict(load("target_heads.th"))
        self.optimiser.load_state_dict(load("opt.th"))
