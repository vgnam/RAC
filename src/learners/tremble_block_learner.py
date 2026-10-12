import copy

import numpy as np
import torch as th
from torch.optim import RMSprop

from src.learners.tremble_iql_learner import unroll_agent


class TrembleBlockLearner:
    """TR-IQL v2: block-level tremble-robust optimism for multi-step games.

    Same principle as ``TrembleIQLLearner`` (optimistic over teammates' intentions,
    nominal over execution tremble, commitment blocks, variance-aware pooling of
    blocks), but the pooling test runs once per block on the block's mean episode
    return instead of once per (state, action):

    * Within a block every agent acts with a frozen snapshot of its policy
      (``TrembleBlockMAC`` with ``freeze_acting``), so the block's episodes are
      i.i.d. given the joint intention: their mean return estimates the intention's
      value under tremble and its standard error is ``var(returns) / block_len``.
    * The best group of blocks is kept as (mean, variance). A finished block replaces
      it when better by more than ``pool_z`` standard errors, is merged into it
      (inverse-variance weighting) when not significantly different, and is ignored
      when worse.
    * Q is trained every episode by double-Q TD with a target network on the shared
      replay buffer. Episodes are sampled with weight 1 when their block is not
      significantly worse than the current best group and ``block_beta`` otherwise,
      so Q tracks the best joint intention found so far; old episodes are re-weighted
      whenever the best group changes. Episodes of the unfinished block have weight 0.

    In single-state matrix games the block test is the per-action test of
    ``TrembleIQLLearner``; with ``block_beta`` > 0 Q is a weighted rather than an
    exact best-group average.
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.params = list(mac.parameters())
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_agent = copy.deepcopy(mac.agent)

        self.block_len = int(args.block_len)
        self.pool_z = float(args.pool_z)
        self.beta = float(args.block_beta)
        self.var_floor = float(args.block_var_floor)

        self.buffer = None
        self.slot_block = None   # block id of each replay slot, -1 while its block is unfinished
        self.slot_weight = None  # sampling weight of each replay slot
        self.block_stats = {}    # block id -> (mean return, variance of the mean)
        self.best = None         # (mean, variance) of the best group of blocks
        self.n_blocks = 0
        self.block_returns = []
        self.block_slots = []
        self.decisions = {"replace": 0, "merge": 0, "ignore": 0}
        self.last_block_mean = 0.0

        self.last_target_update_episode = 0
        self.log_stats_t = -args.learner_log_interval - 1

    def set_buffer(self, buffer):
        """Share run.py's replay buffer; the learner samples it with block weights."""
        self.buffer = buffer
        self.slot_block = np.full(buffer.buffer_size, -1, dtype=np.int64)
        self.slot_weight = np.zeros(buffer.buffer_size)

    def observe_episode(self, episode_batch, t_env):
        assert self.buffer is not None, "TrembleBlockLearner needs set_buffer() (see run.py)"
        assert episode_batch.batch_size == 1, "TrembleBlockLearner requires batch_size_run=1"
        # run.py inserts the episode into the buffer just before calling this.
        slot = (self.buffer.buffer_index - 1) % self.buffer.buffer_size
        self.slot_block[slot] = -1
        self.slot_weight[slot] = 0.0
        self.block_slots.append(slot)
        self.block_returns.append(episode_batch["reward"].sum().item())
        if len(self.block_returns) >= self.block_len:
            self._end_block(t_env)

    def _end_block(self, t_env):
        returns = np.asarray(self.block_returns, dtype=np.float64)
        m = float(returns.mean())
        v = max(float(returns.var(ddof=1)) / len(returns) if len(returns) > 1 else 0.0, self.var_floor)
        if self.best is None:
            self.best, decision = (m, v), "replace"
        else:
            mu, var = self.best
            score = (m - mu) / np.sqrt(v + var)
            if score > self.pool_z:
                self.best, decision = (m, v), "replace"
            elif score >= -self.pool_z:
                precision = 1.0 / v + 1.0 / var
                self.best, decision = ((m / v + mu / var) / precision, 1.0 / precision), "merge"
            else:
                decision = "ignore"
        self.decisions[decision] += 1
        self.last_block_mean = m

        block_id = self.n_blocks
        self.block_stats[block_id] = (m, v)
        self.slot_block[self.block_slots] = block_id
        self.n_blocks += 1
        self.block_returns, self.block_slots = [], []
        self._refresh_weights()
        self.mac.new_block(t_env)

    def _refresh_weights(self):
        mu, var = self.best
        live = np.unique(self.slot_block[self.slot_block >= 0])
        # Forget statistics of blocks no longer in the buffer.
        self.block_stats = {b: self.block_stats[b] for b in live}
        self.slot_weight[:] = 0.0
        for b in live:
            m, v = self.block_stats[b]
            comparable = (m - mu) / np.sqrt(v + var) >= -self.pool_z
            self.slot_weight[self.slot_block == b] = 1.0 if comparable else self.beta

    def train(self, batch, t_env, episode_num):
        """One double-Q TD step on a block-weighted sample (``batch`` from run.py is unused)."""
        n_stored = self.buffer.episodes_in_buffer
        weights = self.slot_weight[:n_stored]
        if weights.sum() <= 0:
            # Nothing comparable to the best group left (block_beta=0): sample finished blocks uniformly.
            weights = (self.slot_block[:n_stored] >= 0).astype(np.float64)
        total = weights.sum()
        if total <= 0:
            return
        ids = np.random.choice(n_stored, self.args.batch_size, p=weights / total)
        batch = self.buffer[ids]
        batch = batch[:, :batch.max_t_filled()]
        if batch.device != self.args.device:
            batch.to(self.args.device)

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        q = unroll_agent(self.mac, self.mac.agent, batch, self.n_agents)
        chosen = th.gather(q[:, :-1], dim=3, index=actions).squeeze(3)
        with th.no_grad():
            target_q = unroll_agent(self.mac, self.target_agent, batch, self.n_agents)[:, 1:]
            target_q[avail_actions[:, 1:] == 0] = -9999999
            q_live = q.detach().clone()
            q_live[avail_actions == 0] = -9999999
            best_next = q_live[:, 1:].max(dim=3, keepdim=True)[1]
            target_max = th.gather(target_q, 3, best_next).squeeze(3)
            targets = rewards + self.args.gamma * (1 - terminated) * target_max

        mask = mask.expand_as(chosen)
        td_error = (chosen - targets) * mask
        loss = (td_error ** 2).sum() / mask.sum()
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self.target_agent.load_state_dict(self.mac.agent.state_dict())
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            n = mask.sum().item()
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", float(grad_norm), t_env)
            self.logger.log_stat("q_taken_mean", (chosen * mask).sum().item() / n, t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / n, t_env)
            self.logger.log_stat("n_blocks", float(self.n_blocks), t_env)
            self.logger.log_stat("block_best_mean", self.best[0], t_env)
            self.logger.log_stat("block_best_se", float(np.sqrt(self.best[1])), t_env)
            self.logger.log_stat("block_last_mean", self.last_block_mean, t_env)
            n_decisions = max(sum(self.decisions.values()), 1)
            for key, count in self.decisions.items():
                self.logger.log_stat("block_{}_frac".format(key), count / n_decisions, t_env)
            self.decisions = {key: 0 for key in self.decisions}
            self.logger.log_stat("buffer_full_weight_frac",
                                 float((weights == 1.0).sum()) / max(float((weights > 0).sum()), 1.0), t_env)
            self.log_stats_t = t_env

    def show_matrix_info(self, batch, t_env):
        with th.no_grad():
            q = unroll_agent(self.mac, self.mac.agent, batch, self.n_agents)[0, 0]
        rows = ["agent {}: {}".format(i, " ".join("{:7.3f}".format(v) for v in q[i].tolist()))
                for i in range(self.n_agents)]
        print("t_env {} TR-IQL v2 Q(intended action):\n  {}".format(t_env, "\n  ".join(rows)))

    def cuda(self):
        self.mac.cuda()
        self.target_agent.to(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.target_agent.state_dict(), "{}/target_agent.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        load = lambda f: th.load("{}/{}".format(path, f), map_location=lambda storage, loc: storage)
        self.target_agent.load_state_dict(load("target_agent.th"))
        self.optimiser.load_state_dict(load("opt.th"))
