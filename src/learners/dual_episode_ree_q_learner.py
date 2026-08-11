import sys
import os

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
from src.components.episode_buffer import EpisodeBatch
from src.modules.mixers.vdn import VDNMixer
from src.modules.mixers.qmix import QMixer
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.distributions import Categorical


class DualEpisodeREEQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0
        self.last_target_update_t = 0
        self.target_update_interval_steps = getattr(
            args, "target_update_interval_steps", None
        )
        if self.target_update_interval_steps is not None:
            self.target_update_interval_steps = int(
                self.target_update_interval_steps
            )
            if self.target_update_interval_steps <= 0:
                raise ValueError(
                    "target_update_interval_steps must be positive, got "
                    f"{self.target_update_interval_steps}."
                )

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.possible_returns = None
        if 'matrix_game_3' in args.env:
            assert self.args.slot_number == 4, "Should set slot_number to 4 for matrix_game_3."
            # For matrix game, possible return contains -12, 0, 6, 8
            self.possible_returns = [-12, 0, 6, 8]

        if 'stag_hunt' in args.env:
            # For pp, we consider continuous return within [-100, 50], divide into [slot_number] slots?
            # slot_number intervals, slot_number+1 points
            self.possible_returns = np.linspace(-200, 50, num=args.slot_number + 1)

        if 'sc2' in args.env:
            # For SMAC and SMACv2, we consider continuous returns within [?, ?] ?
            self.possible_returns = np.linspace(0, 25, num=args.slot_number + 1)

        if 'mate' in args.env:
            return_min = float(getattr(args, 'mate_return_min', 0.0))
            return_max = float(getattr(args, 'mate_return_max', 1000.0))
            if return_min >= return_max:
                raise ValueError(
                    'mate_return_min must be smaller than mate_return_max, '
                    f'got {return_min} >= {return_max}.'
                )
            self.possible_returns = np.linspace(
                return_min, return_max, num=args.slot_number + 1
            )

        print("possible returns", self.possible_returns)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Shape cumulative return
        batch_rewards = batch["reward"]  # (bs, max_seq_length, 1)
        episode_returns = th.sum(batch_rewards, dim=1)      # (bs, 1)
        returns_expand = episode_returns.unsqueeze(dim=1).expand(-1, batch.max_seq_length * self.args.n_agents, -1)
        returns_expand = returns_expand.reshape(batch.batch_size, batch.max_seq_length, self.args.n_agents, 1).cpu().numpy()

        # ======================For matrix game, we should define index according to the return=========================
        if "matrix_game_3" in self.args.env:
            return_index = np.searchsorted(self.possible_returns, returns_expand)  # shape=(bs, max_seq_length, n_agents, 1)
            return_indices = return_index.squeeze(axis=-1)      # (bs, max_seq_length, n_agents)

            onehot_return_indices = np.eye(self.args.slot_number, dtype=np.int32)[return_indices]  # (bs, max_seq_length, n_agents, slot_number)
            onehot_return_indices = th.from_numpy(onehot_return_indices).to(device=self.args.device, dtype=th.float32)

        elif "stag_hunt" in self.args.env or "sc2" in self.args.env or "mate" in self.args.env:
            returns_expand_flat = returns_expand.reshape(-1)    # (bs*max_seq_length*n_agents)
            return_index = np.digitize(returns_expand_flat, bins=self.possible_returns, right=False) - 1
            return_index = np.clip(return_index, 0, self.args.slot_number - 1)
            return_indices = return_index.reshape(returns_expand.shape[:-1])  # (bs, max_seq_length, n_agents)
            # (bs, max_seq_length, n_agents, slot_number)
            onehot_return_indices = np.eye(self.args.slot_number, dtype=np.int32)[return_indices]
            onehot_return_indices = th.from_numpy(onehot_return_indices).to(device=self.args.device, dtype=th.float32)

        else:
            raise Exception("Not implemented.")

        belief_outputs = None
        belief_losses = None
        teacher_diagnostics = None
        if bool(getattr(self.args, "use_context_belief", False)):
            belief_outputs = self.mac.infer_context_sequence(batch)
            belief_tau = float(getattr(self.args, "belief_gumbel_tau", 1.0))
            if belief_tau <= 0.0:
                raise ValueError("belief_gumbel_tau must be positive.")
            context_indices = F.gumbel_softmax(
                belief_outputs["logits"],
                tau=belief_tau,
                hard=True,
                dim=-1,
            )
            belief_losses = self.mac.context_dynamics_loss(
                batch,
                belief_outputs["belief"],
                mask,
                context_assignments=context_indices,
            )
        else:
            context_indices = onehot_return_indices

        # Calculate estimated Q-Values
        mac_out = []
        twin_mac_out = []
        self.mac.init_hidden(batch.batch_size)
        self.mac.init_twin_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, twin_agent_outs = self.mac.train_forward(batch, context_indices[:, t], t=t)
            mac_out.append(agent_outs)
            twin_mac_out.append(twin_agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        twin_mac_out = th.stack(twin_mac_out, dim=1)    # (bs, max_seq_length, n_agents, n_actions)

        # Calculate argmax_{z}q^{i}(\tau^{i},z,a^{i}), and make mac_out to approximate it
        # (bs*max_seq_length*n_agents, slot_number, slot_number)
        indices = th.eye(self.args.slot_number, device=self.args.device).unsqueeze(dim=0).expand(batch.batch_size * batch.max_seq_length * self.args.n_agents, -1, -1)
        indices = indices.reshape(batch.batch_size, batch.max_seq_length, self.args.n_agents, self.args.slot_number, self.args.slot_number)
        counterfactual_twin_mac_out = []
        self.mac.init_twin_counter_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            counter_twin_agent_outs = self.mac.counter_forward(batch, indices[:, t], t=t)
            counterfactual_twin_mac_out.append(counter_twin_agent_outs)
        counterfactual_twin_mac_out = th.stack(counterfactual_twin_mac_out, dim=1)      # (bs, max_seq_length, n_agents, slot_number, n_actions)
        counterfactual_twin_mac_out = counterfactual_twin_mac_out[:, :-1].permute(0, 1, 2, 4, 3)    # (bs, max_seq_length-1, n_agents, n_actions, slot_number)
        if belief_outputs is not None:
            teacher_context_q, teacher_diagnostics = self.mac.aggregate_context_q(
                counterfactual_twin_mac_out,
                belief_outputs["belief"][:, :-1],
                belief_outputs["shift"][:, :-1],
            )
        else:
            teacher_context_q = counterfactual_twin_mac_out.max(dim=-1)[0]

        # Use counterfactual_twin_mac_out to guide the update of mac_out.
        # Calculate the corresponding softmax action distribution, shape=(bs, max_seq_length-1, n_agents, n_actions)
        mac_q_dist = Categorical(logits=mac_out[:, :-1])
        teacher_mac_q_dist = Categorical(logits=teacher_context_q.clone().detach())

        import torch.distributions as D
        kl_constraint = D.kl_divergence(mac_q_dist, teacher_mac_q_dist)     # Maybe (bs, max_seq_length-1, n_agents)
        kl_mask = mask.clone().expand_as(kl_constraint)
        masked_kl_constraint = kl_constraint * kl_mask
        kl_loss = masked_kl_constraint.sum() / kl_mask.sum()

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_twin_action_qvals = th.gather(twin_mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_twin_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        self.target_mac.init_twin_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, target_twin_agent_outs = self.target_mac.train_forward(batch, context_indices[:, t].detach(), t=t)
            target_mac_out.append(target_agent_outs)
            target_twin_mac_out.append(target_twin_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_twin_mac_out = th.stack(target_twin_mac_out[1:], dim=1)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # From OG deepmarl
        target_twin_mac_out[avail_actions[:, 1:] == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            twin_mac_out_detach = twin_mac_out.clone().detach()
            twin_mac_out_detach[avail_actions == 0] = -9999999
            twin_cur_max_actions = twin_mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_twin_max_qvals = th.gather(target_twin_mac_out, 3, twin_cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_twin_max_qvals = target_twin_mac_out.max(dim=3)[0]

        N = getattr(self.args, "n_step", 1)
        assert N == 1, "Should use N=1 for fair comparison"

        # Calculate 1-step Q-Learning targets
        rewards = rewards.expand(-1, -1, self.args.n_agents)       # (bs, max_seq_length-1, n_agents)
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        twin_targets = rewards + self.args.gamma * (1 - terminated) * target_twin_max_qvals
        # print((targets == rewards).all())

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        twin_td_error = (chosen_twin_action_qvals - twin_targets.detach())

        mask = mask.expand_as(twin_td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        masked_twin_td_error = twin_td_error * mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask.sum()
        twin_td_loss = (masked_twin_td_error ** 2).sum() / mask.sum()

        loss = twin_td_loss + td_loss + self.args.kl_weight * kl_loss
        if belief_losses is not None:
            loss = loss + float(
                getattr(self.args, "belief_dynamics_weight", 1.0)
            ) * belief_losses["loss"]

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if self.target_update_interval_steps is not None:
            if t_env - self.last_target_update_t >= self.target_update_interval_steps:
                self._update_targets()
                self.last_target_update_t = t_env
        elif (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("kl_loss", (self.args.kl_weight * kl_loss).item(), t_env)
            self.logger.log_stat("twin_td_loss", twin_td_loss.item(), t_env)

            mask_elems = mask.sum()
            valid_twin_td_errors = twin_td_error.detach().abs()[mask.bool()]
            self.logger.log_stat(
                "twin_td_error_abs",
                valid_twin_td_errors.mean().item(),
                t_env,
            )
            self.logger.log_stat(
                "twin_td_error_max",
                valid_twin_td_errors.max().item(),
                t_env,
            )
            self.logger.log_stat(
                "twin_td_error_p95",
                th.quantile(valid_twin_td_errors.float(), 0.95).item(),
                t_env,
            )
            self.logger.log_stat(
                "twin_q_taken_mean",
                ((chosen_twin_action_qvals.detach() * mask).sum() / mask_elems).item(),
                t_env,
            )
            self.logger.log_stat(
                "twin_target_mean",
                ((twin_targets.detach() * mask).sum() / mask_elems).item(),
                t_env,
            )

            terminal_flags = terminated.expand_as(twin_td_error)
            terminal_mask = mask * terminal_flags
            nonterminal_mask = mask * (1.0 - terminal_flags)
            squared_twin_td_error = twin_td_error.detach().pow(2)
            terminal_elems = terminal_mask.sum()
            if terminal_elems.item() > 0:
                self.logger.log_stat(
                    "twin_loss_terminal",
                    (
                        (squared_twin_td_error * terminal_mask).sum()
                        / terminal_elems
                    ).item(),
                    t_env,
                )
            nonterminal_elems = nonterminal_mask.sum()
            if nonterminal_elems.item() > 0:
                self.logger.log_stat(
                    "twin_loss_nonterminal",
                    (
                        (squared_twin_td_error * nonterminal_mask).sum()
                        / nonterminal_elems
                    ).item(),
                    t_env,
                )

            teacher_entropy = teacher_mac_q_dist.entropy().detach()
            self.logger.log_stat(
                "kl_teacher_entropy",
                ((teacher_entropy * kl_mask).sum() / kl_mask.sum()).item(),
                t_env,
            )

            if belief_losses is not None:
                belief_mask = mask
                belief_mask_elems = belief_mask.sum().clamp_min(1.0)
                shift = belief_outputs["shift"][:, :-1]
                self.logger.log_stat(
                    "context_belief_loss",
                    belief_losses["loss"].item(),
                    t_env,
                )
                self.logger.log_stat(
                    "context_reconstruction_loss",
                    belief_losses["reconstruction"].item(),
                    t_env,
                )
                self.logger.log_stat(
                    "context_prior_kl",
                    belief_losses["prior_kl"].item(),
                    t_env,
                )
                self.logger.log_stat(
                    "context_balance_kl",
                    belief_losses["balance_kl"].item(),
                    t_env,
                )
                self.logger.log_stat(
                    "context_uncertainty_mean",
                    belief_losses["entropy"].item(),
                    t_env,
                )
                self.logger.log_stat(
                    "context_shift_mean",
                    ((shift * belief_mask).sum() / belief_mask_elems).item(),
                    t_env,
                )
                valid_shift = shift[belief_mask.bool()]
                self.logger.log_stat(
                    "context_shift_max",
                    valid_shift.max().item(),
                    t_env,
                )

                optimism_weight = teacher_diagnostics["optimism_weight"]
                self.logger.log_stat(
                    "context_optimism_weight",
                    (
                        (optimism_weight * belief_mask).sum()
                        / belief_mask_elems
                    ).item(),
                    t_env,
                )
                action_mask = belief_mask.unsqueeze(-1)
                action_mask_elems = (
                    action_mask.sum() * self.args.n_actions
                ).clamp_min(1.0)
                posterior_q = teacher_diagnostics["posterior_mean"]
                optimistic_q = teacher_diagnostics["optimistic_max"]
                self.logger.log_stat(
                    "context_posterior_q_mean",
                    ((posterior_q * action_mask).sum() / action_mask_elems).item(),
                    t_env,
                )
                self.logger.log_stat(
                    "context_optimistic_q_mean",
                    ((optimistic_q * action_mask).sum() / action_mask_elems).item(),
                    t_env,
                )
                self.logger.log_stat(
                    "context_optimism_gap",
                    (
                        ((optimistic_q - posterior_q) * action_mask).sum()
                        / action_mask_elems
                    ).item(),
                    t_env,
                )

                belief = belief_outputs["belief"][:, :-1]
                agent_belief_mask = belief_mask.unsqueeze(-1)
                belief_slot_mass = (
                    belief * agent_belief_mask
                ).sum(dim=(0, 1, 2)) / belief_mask_elems
                for slot, mass in enumerate(belief_slot_mass):
                    self.logger.log_stat(
                        f"belief_slot_{slot}_mass",
                        mass.item(),
                        t_env,
                    )

            self.logger.log_stat("episode_returns_min", episode_returns.min().item(), t_env)
            self.logger.log_stat("episode_returns_max", episode_returns.max().item(), t_env)
            episode_return_slots = return_indices[:, 0, 0].reshape(-1)
            slot_counts = np.bincount(
                episode_return_slots, minlength=self.args.slot_number
            )
            slot_fractions = slot_counts / max(1, episode_return_slots.size)
            for slot, fraction in enumerate(slot_fractions):
                self.logger.log_stat(
                    f"return_slot_{slot}_fraction", float(fraction), t_env
                )

            # self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.to(self.args.device)
            self.target_mixer.to(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right, but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def show_matrix_info(self, batch, t_env):
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            # agent_outs = self.mac.forward(batch, t=t, show_h=bool(1-t))
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time, threads, steps, agents, actions
        actions_dim = mac_out.shape[3]
        print("Episode %i, The learned matrix payoff is:" % t_env)
        payoff = ""

        indices = th.eye(self.args.slot_number, device=self.args.device).unsqueeze(dim=0).expand(batch.batch_size * batch.max_seq_length * self.args.n_agents, -1, -1)
        indices = indices.reshape(batch.batch_size, batch.max_seq_length, self.args.n_agents, self.args.slot_number, self.args.slot_number)
        counterfactual_twin_mac_out = []
        self.mac.init_twin_counter_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            counter_twin_agent_outs = self.mac.counter_forward(batch, indices[:, t], t=t)
            counterfactual_twin_mac_out.append(counter_twin_agent_outs)
        counterfactual_twin_mac_out = th.stack(counterfactual_twin_mac_out, dim=1)  # (bs, max_seq_length, n_agents, slot_number, n_actions)
        counterfactual_twin_mac_out = counterfactual_twin_mac_out[:, :-1].permute(0, 1, 2, 4, 3)  # (bs, max_seq_length-1, n_agents, n_actions, slot_number)
        argmax_counter_twin_mac_out = counterfactual_twin_mac_out.max(dim=-1)[0]  # (bs, max_seq_length-1, n_agents, n_actions)

        for ai in range(actions_dim):
            for aj in range(actions_dim):
                actions = th.tensor([[ai, aj]]).to(**dict(dtype=th.int64, device=self.args.device))
                actions = actions.unsqueeze(0).unsqueeze(-1).repeat(batch.batch_size, batch.max_seq_length - 1, 1, 1)
                # print(actions.shape, actions)
                chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
                counter_chosen_action_qvals = th.gather(argmax_counter_twin_mac_out, dim=3, index=actions).squeeze(3)   # (bs, max_seq_length-1, n_agents)

                # Slot number denotes [-12, 0, 6, 8]
                context_action_qvals = th.gather(counterfactual_twin_mac_out, dim=3, index=actions.unsqueeze(-1).expand(-1, -1, -1, -1, self.args.slot_number)).squeeze(3)    # (bs, max_seq_length-1, n_agents, slot_number)

                if self.mixer is not None:
                    raise Exception("Only for fully decentralized training.")
                else:
                    mixer_qvals = th.zeros((1, 1, 1))
                sp = "{0:.4}".format(str(chosen_action_qvals[0, 0, 0].item())) + "||" \
                     + "{0:.4}".format(str(chosen_action_qvals[0, 0, 1].item())) \
                     + "||" + "{0:.4}".format(str(counter_chosen_action_qvals[0, 0, 0].item())) \
                     + "||" + "{0:.4}".format(str(counter_chosen_action_qvals[0, 0, 1].item())) \
                     + "||" + "{0:.4}".format(str(context_action_qvals[0, 0, 0, 0].item())) \
                     + "||" + "{0:.4}".format(str(context_action_qvals[0, 0, 0, 1].item())) \
                     + "||" + "{0:.4}".format(str(context_action_qvals[0, 0, 0, 2].item())) \
                     + "||" + "{0:.4}".format(str(context_action_qvals[0, 0, 0, 3].item())) \
                     + "||" + "{0:.4}".format(str(context_action_qvals[0, 0, 1, 0].item())) \
                     + "||" + "{0:.4}".format(str(context_action_qvals[0, 0, 1, 1].item())) \
                     + "||" + "{0:.4}".format(str(context_action_qvals[0, 0, 1, 2].item())) \
                     + "||" + "{0:.4}".format(str(context_action_qvals[0, 0, 1, 3].item())) \
                     + "||" + "{0:.4}".format(str(mixer_qvals[0, 0, 0].item())) + "     "
                payoff += sp
                # print(ai, aj, chosen_action_qvals[0, 0, 0].item(),
                # chosen_action_qvals[0, 0, 1].item(), mixer_qvals[0, 0, 0].item())
            payoff += "\n"
        print(payoff)
        # max_actions = mac_out.max(dim=3)[1]
        max_actions = batch["actions"][:, :-1, :, 0]
        print("Max actions is:", max_actions[0, 0, 0].item(), max_actions[0, 0, 1].item(),
              "  ||   Reward is", batch["reward"][0, 0, 0].item())
        # chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        # chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        # print(mac_out.shape)
        # print(mac_out)
        # print(batch["actions"][:, :-1])
        # print(batch["actions"][:, :-1].shape)
        # exit()
        # print(self.mixer.state_dict())

    def show_mmdp_info(self, batch, t_env):
        pass
