import copy

import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop

from src.components.episode_buffer import EpisodeBatch
from src.learners.q_learner import QLearner


class UCVaRQLearner(QLearner):
    """Upper-CVaR IQL: IQL whose regression target is the mean of the top ``ucvar_alpha``
    tail of the TD target instead of its mean.

    If teammates play every joint action with probability >= alpha, every
    teammate-induced target distribution has density ratio <= 1/alpha w.r.t. the observed
    one, so max over teammates <= CVaR+_alpha (equality for deterministic environments).
    Rockafellar-Uryasev: CVaR+_alpha(Y) = min_eta eta + E[(Y - eta)_+] / alpha.

    * ``eta_mac`` (a copy of the agent network) tracks the (1 - alpha)-quantile of the
      double-Q target Y by the pinball loss.
    * Q regresses on eta + (Y - eta)_+ / alpha, with eta detached.

    Acting, exploration, target network and double Q are those of IQL; ``ucvar_alpha=1``
    is IQL. Only 1-step targets and no mixer.

    Lenient schedule (``ucvar_lenient: True``): alpha is per (observation, action),
    alpha(n) = end - (end - start) * exp(-n / ucvar_count_scale), where n counts the
    collected visits of the SimHash bucket of the observation with that action (shared by
    all agents). Rarely tried pairs stay optimistic while teammates still explore; often
    tried pairs fall back to IQL instead of staying optimistic about environment noise.
    """

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        assert self.mixer is None, "UCVaR-IQL is an independent learner (mixer: None)"
        assert getattr(args, "n_step", 1) == 1, "UCVaR-IQL uses 1-step targets"
        self.alpha = float(args.ucvar_alpha)
        assert 0.0 < self.alpha <= 1.0, "ucvar_alpha must be in (0, 1]"
        self.lenient = bool(getattr(args, "ucvar_lenient", False))
        if self.lenient:
            self.alpha_start = float(args.ucvar_alpha_start)
            self.alpha_end = float(args.ucvar_alpha_end)
            self.count_scale = float(args.ucvar_count_scale)
            assert 0.0 < self.alpha_start <= self.alpha_end <= 1.0, "need 0 < alpha_start <= alpha_end <= 1"
            assert self.count_scale > 0, "ucvar_count_scale must be positive"
            self.n_actions = args.n_actions
            bits = int(args.ucvar_hash_bits)
            gen = th.Generator().manual_seed(int(args.seed))
            self.hash_proj = th.randn(scheme["obs"]["vshape"], bits, generator=gen)
            self.hash_weights = 2 ** th.arange(bits, dtype=th.long)
            self.counts = th.zeros((2 ** bits) * self.n_actions, dtype=th.long)
        self.eta_mac = copy.deepcopy(mac)
        self.q_params = self.params
        self.eta_params = list(self.eta_mac.parameters())
        self.params = self.q_params + self.eta_params
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def _cells(self, obs, actions):
        """Flat (SimHash bucket of obs, action) index; obs (..., obs_dim), actions (..., 1)."""
        bits = (obs.float() @ self.hash_proj > 0).long()
        return (bits * self.hash_weights).sum(-1) * self.n_actions + actions.squeeze(-1).long()

    def _alpha(self, obs, actions):
        n = self.counts[self._cells(obs, actions)].float()
        return self.alpha_end - (self.alpha_end - self.alpha_start) * th.exp(-n / self.count_scale)

    def observe_episode(self, episode_batch, t_env):
        if not self.lenient:
            return
        terminated = episode_batch["terminated"][:, :-1].float()
        filled = episode_batch["filled"][:, :-1].float()
        filled[:, 1:] = filled[:, 1:] * (1 - terminated[:, :-1])
        cells = self._cells(episode_batch["obs"][:, :-1], episode_batch["actions"][:, :-1])
        cells = cells[filled.squeeze(-1).bool()]  # (bs, T, n_agents) -> visited cells
        self.counts.index_add_(0, cells.reshape(-1), th.ones_like(cells.reshape(-1)))

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        self.mac.init_hidden(batch.batch_size)
        mac_out = self.mac.forward_sequence(batch)
        chosen_q = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        self.eta_mac.init_hidden(batch.batch_size)
        eta_out = self.eta_mac.forward_sequence(batch)
        chosen_eta = th.gather(eta_out[:, :-1], dim=3, index=actions).squeeze(3)

        if self.lenient:
            alpha = self._alpha(batch["obs"][:, :-1], actions)
        else:
            alpha = th.full_like(chosen_q, self.alpha)

        with th.no_grad():
            self.target_mac.init_hidden(batch.batch_size)
            target_mac_out = self.target_mac.forward_sequence(batch)[:, 1:]
            target_mac_out[avail_actions[:, 1:] == 0] = -9999999
            if self.args.double_q:
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            else:
                target_max_qvals = target_mac_out.max(dim=3)[0]
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
            eta_d = chosen_eta.detach()
            cvar_targets = eta_d + F.relu(targets - eta_d) / alpha

        mask = mask.expand_as(chosen_q)
        n = mask.sum().clamp(min=1)
        q_loss = (((chosen_q - cvar_targets) * mask) ** 2).sum() / n
        u = targets - chosen_eta
        tau = 1.0 - alpha
        eta_loss = ((u * (tau - (u < 0).float())) * mask).sum() / n
        loss = q_loss + eta_loss

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.q_params, self.args.grad_norm_clip)
        eta_grad_norm = th.nn.utils.clip_grad_norm_(self.eta_params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            with th.no_grad():
                # Discounted return-to-go of the sampled episodes (truncated at the episode end).
                ret = th.zeros_like(rewards[:, 0])
                returns = th.zeros_like(rewards)
                for t in reversed(range(rewards.shape[1])):
                    ret = rewards[:, t] + self.args.gamma * ret * mask[:, t, :1]
                    returns[:, t] = ret
                returns = returns.expand_as(chosen_q)
            stat = lambda x: ((x * mask).sum() / n).item()
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("eta_loss", eta_loss.item(), t_env)
            self.logger.log_stat("grad_norm", float(grad_norm), t_env)
            self.logger.log_stat("eta_grad_norm", float(eta_grad_norm), t_env)
            self.logger.log_stat("q_taken_mean", stat(chosen_q), t_env)
            self.logger.log_stat("target_mean", stat(targets), t_env)
            self.logger.log_stat("cvar_target_mean", stat(cvar_targets), t_env)
            self.logger.log_stat("eta_mean", stat(chosen_eta), t_env)
            self.logger.log_stat("tail_frac", stat((targets > eta_d).float()), t_env)
            self.logger.log_stat("return_to_go_mean", stat(returns), t_env)
            self.logger.log_stat("q_minus_return", stat(chosen_q - returns), t_env)
            if self.lenient:
                self.logger.log_stat("alpha_mean", stat(alpha), t_env)
                self.logger.log_stat("alpha_low_frac", stat((alpha < 0.5).float()), t_env)
                catch = getattr(self.args, "ucvar_log_action", None)
                if catch is not None:
                    m = mask * (actions.squeeze(3) == catch).float()
                    if m.sum() > 0:
                        self.logger.log_stat("alpha_action{}_mean".format(catch),
                                             ((alpha * m).sum() / m.sum()).item(), t_env)
                self.logger.log_stat("hash_cells_visited", float((self.counts > 0).sum()), t_env)
            self.log_stats_t = t_env

    def cuda(self):
        super().cuda()
        self.eta_mac.cuda()
        if self.lenient:
            device = self.args.device
            self.hash_proj = self.hash_proj.to(device)
            self.hash_weights = self.hash_weights.to(device)
            self.counts = self.counts.to(device)

    def save_models(self, path):
        super().save_models(path)
        th.save(self.eta_mac.agent.state_dict(), "{}/eta_agent.th".format(path))
        if self.lenient:
            th.save({"hash_proj": self.hash_proj, "counts": self.counts}, "{}/ucvar_counts.th".format(path))

    def load_models(self, path):
        super().load_models(path)
        self.eta_mac.agent.load_state_dict(
            th.load("{}/eta_agent.th".format(path), map_location=lambda storage, loc: storage))
        if self.lenient:
            state = th.load("{}/ucvar_counts.th".format(path), map_location=self.counts.device)
            self.hash_proj = state["hash_proj"].to(self.counts.device)
            self.counts = state["counts"].to(self.counts.device)
