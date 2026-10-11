import copy

import torch as th
from torch.optim import Adam

from src.components.episode_buffer import ReplayBuffer
from src.tremble.games import chi2_kappa


class TrembleIQLLearner:
    """Tremble-Robust IQL: optimistic over teammates' intentions, nominal or robust
    over execution tremble.

    Q (the acting network, ``mac.agent``) is indexed by intended actions. Within a
    block (see ``TrembleBlockMAC``) teammates' intentions are fixed, so regressing
    Q^e on the block's TD targets estimates the expectation over execution tremble.
    At the end of the block Q(s, a_i) <- max(Q(s, a_i), Q^e(s, a_i)), which over
    blocks approaches max over teammates' intentions.

    ``dro_eps_max`` (pooled only) scores each block by mean - kappa * std with
    kappa = chi2_kappa(tremble_eps, dro_eps_max, n_agents), a lower bound on its value
    under any tremble rate in [tremble_eps, dro_eps_max] (exact for matrix games; for
    multi-step games it is applied per step). Maximising noisy block means is
    biased upwards, most for high-variance (risky) intentions, so two estimators are
    available (``estimator``):

    ``pooled`` (default): Q stores the mean of the best group of blocks and the
    auxiliary network S its log variance. A new block replaces the group when it is
    higher by more than ``pool_z`` standard errors, is merged into it (inverse-variance
    weighting) when not significantly different, and is ignored when lower. Recurring
    teammate intentions are pooled without being observed, the estimate sharpens over
    time and Q can move down. The block variance is pooled over states per (agent,
    action), which is exact for single-state matrix games.

    ``double``: Q^e is fitted separately on the two halves of the block; the first
    half decides whether the block beats the best block so far (S tracks its score),
    the second half supplies the value written into Q.
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.scheme = {k: v for k, v in scheme.items() if k != "filled"}  # re-added by the buffer
        self.n_agents = args.n_agents

        self.qe_agent = copy.deepcopy(mac.agent)
        self.sel_agent = copy.deepcopy(mac.agent)
        self.q_opt = Adam(mac.parameters(), lr=args.fit_lr)
        self.qe_opt = Adam(self.qe_agent.parameters(), lr=args.fit_lr)
        self.sel_opt = Adam(self.sel_agent.parameters(), lr=args.fit_lr)
        self.estimator = getattr(args, "estimator", "pooled")
        eps_max = getattr(args, "dro_eps_max", None)
        self.kappa = 0.0 if eps_max is None else chi2_kappa(
            args.env_args.get("tremble_eps", 0.0), eps_max, self.n_agents)
        if self.kappa and self.estimator != "pooled":
            raise ValueError("dro_eps_max requires estimator: pooled")
        if self.estimator == "pooled":
            # S outputs log variance; start very uncertain so a first block is adopted as is.
            with th.no_grad():
                self.sel_agent.fc2.weight.zero_()
                self.sel_agent.fc2.bias.fill_(args.init_log_var)

        self.block_buffer = None
        self.n_blocks = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

    def observe_episode(self, episode_batch, t_env):
        if self.block_buffer is None:
            self.block_buffer = ReplayBuffer(self.scheme, {"agents": self.n_agents}, self.args.block_len,
                                             episode_batch.max_seq_length, device=self.args.device)
        self.block_buffer.insert_episode_batch(episode_batch)
        if self.block_buffer.episodes_in_buffer >= self.args.block_len:
            self._end_block(t_env)
            self.block_buffer.buffer_index = 0
            self.block_buffer.episodes_in_buffer = 0
            self.mac.new_block(t_env)

    def train(self, batch, t_env, episode_num):
        """All updates happen at block ends (``observe_episode``)."""

    def _unroll(self, agent, batch):
        hidden = agent.init_hidden().unsqueeze(0).expand(batch.batch_size, self.n_agents, -1)
        outs = []
        for t in range(batch.max_seq_length):
            q, hidden = agent(self.mac._build_inputs(batch, t), hidden)
            outs.append(q.view(batch.batch_size, self.n_agents, -1))
        return th.stack(outs, dim=1)  # (bs, T, n_agents, n_actions)

    def _fit(self, agent, opt, batch, target, mask, actions=None):
        """Regress agent outputs on target; on the taken actions only if ``actions`` is given."""
        for _ in range(self.args.fit_iters):
            out = self._unroll(agent, batch)[:, :-1]
            if actions is not None:
                out = th.gather(out, dim=3, index=actions).squeeze(3)
            err = (out - target) ** 2 * mask
            loss = err.sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(agent.parameters(), self.args.grad_norm_clip)
            opt.step()
        return loss.item()

    def _end_block(self, t_env):
        block = self.block_buffer[:self.block_buffer.episodes_in_buffer]
        rewards = block["reward"][:, :-1]
        actions = block["actions"][:, :-1]
        terminated = block["terminated"][:, :-1].float()
        mask = block["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.expand(-1, -1, self.n_agents)

        with th.no_grad():
            q_all = self._unroll(self.mac.agent, block)
            q_next = q_all[:, 1:].clone()
            q_next[block["avail_actions"][:, 1:] == 0] = -9999999
            targets = rewards + self.args.gamma * (1 - terminated) * q_next.max(dim=3)[0]

        q_old = q_all[:, :-1]
        if self.estimator == "pooled":
            q_target, sel_target, stats = self._pooled_targets(block, targets, actions, mask, q_old)
        else:
            q_target, sel_target, stats = self._double_targets(block, targets, actions, mask, q_old)

        full_mask = mask.unsqueeze(3)
        self._fit(self.sel_agent, self.sel_opt, block, sel_target, full_mask)
        q_loss = self._fit(self.mac.agent, self.q_opt, block, q_target, full_mask)
        self.n_blocks += 1

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("q_fit_loss", q_loss, t_env)
            for key, value in stats.items():
                self.logger.log_stat(key, value, t_env)
            self.logger.log_stat("n_blocks", float(self.n_blocks), t_env)
            self.log_stats_t = t_env

    def _pooled_targets(self, block, targets, actions, mask, q_old):
        self._fit(self.qe_agent, self.qe_opt, block, targets, mask, actions)
        with th.no_grad():
            m = th.gather(self._unroll(self.qe_agent, block)[:, :-1], 3, actions).squeeze(3)
            chosen = th.zeros_like(q_old).scatter_(3, actions, 1.0) * mask.unsqueeze(3)
            count = chosen.sum(dim=(0, 1))  # (n_agents, n_actions)
            sq_res = (((targets - m) ** 2) * mask).unsqueeze(3) * chosen
            sigma2 = sq_res.sum(dim=(0, 1)) / (count - 1).clamp(min=1)
            v_group = (sigma2 / count.clamp(min=1)).clamp(min=1e-4)
            v = (chosen * v_group).sum(dim=3).clamp(min=1e-4)  # per-sample block variance of m
            if self.kappa:
                m = m - self.kappa * (chosen * sigma2.sqrt()).sum(dim=3)  # chi^2-DRO block score

            log_var_all = self._unroll(self.sel_agent, block)[:, :-1]
            var_old = th.gather(log_var_all, 3, actions).squeeze(3).exp()
            mu_old = th.gather(q_old, 3, actions).squeeze(3)
            score = (m - mu_old) / th.sqrt(v + var_old)
            replace = score > self.args.pool_z
            merge = score.abs() <= self.args.pool_z
            precision = 1.0 / v + 1.0 / var_old
            mu_new = th.where(replace, m, th.where(merge, (m / v + mu_old / var_old) / precision, mu_old))
            var_new = th.where(replace, v, th.where(merge, 1.0 / precision, var_old))

            onehot = chosen.bool()
            q_target, sel_target = q_old.clone(), log_var_all.clone()
            q_target[onehot] = mu_new.unsqueeze(3).expand_as(onehot)[onehot]
            sel_target[onehot] = var_new.log().unsqueeze(3).expand_as(onehot)[onehot]
        n = mask.sum().item()
        stats = {"block_replace_frac": ((replace & (mask > 0)).float().sum().item() / n),
                 "block_merge_frac": ((merge & (mask > 0)).float().sum().item() / n),
                 "dro_kappa": self.kappa}
        return q_target, sel_target, stats

    def _double_targets(self, block, targets, actions, mask, q_old):
        half = block.batch_size // 2
        values = []
        for part in (slice(0, half), slice(half, block.batch_size)):
            self._fit(self.qe_agent, self.qe_opt, block[part], targets[part], mask[part], actions[part])
            with th.no_grad():
                out = self._unroll(self.qe_agent, block)[:, :-1]
                values.append(th.gather(out, dim=3, index=actions).squeeze(3))
        v_select, v_value = values

        with th.no_grad():
            sel_all = self._unroll(self.sel_agent, block)[:, :-1]
            improve = (v_select > th.gather(sel_all, 3, actions).squeeze(3)) & (mask > 0)
            sel_target = sel_all.clone()
            q_target = q_old.clone()
            onehot = th.zeros_like(q_old, dtype=th.bool).scatter_(3, actions, True) & improve.unsqueeze(3)
            sel_target[onehot] = v_select.unsqueeze(3).expand_as(onehot)[onehot]
            q_target[onehot] = v_value.unsqueeze(3).expand_as(onehot)[onehot]
        return q_target, sel_target, {"block_improve_frac": improve.float().sum().item() / mask.sum().item()}

    def show_matrix_info(self, batch, t_env):
        with th.no_grad():
            q = self._unroll(self.mac.agent, batch)[0, 0]
        rows = ["agent {}: {}".format(i, " ".join("{:7.3f}".format(v) for v in q[i].tolist()))
                for i in range(self.n_agents)]
        print("t_env {} TR-IQL Q(intended action):\n  {}".format(t_env, "\n  ".join(rows)))

    def cuda(self):
        self.mac.cuda()
        self.qe_agent.to(self.args.device)
        self.sel_agent.to(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.qe_agent.state_dict(), "{}/qe_agent.th".format(path))
        th.save(self.sel_agent.state_dict(), "{}/sel_agent.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        load = lambda f: th.load("{}/{}".format(path, f), map_location=lambda storage, loc: storage)
        self.qe_agent.load_state_dict(load("qe_agent.th"))
        self.sel_agent.load_state_dict(load("sel_agent.th"))
