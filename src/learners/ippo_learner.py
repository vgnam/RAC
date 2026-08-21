import os

import torch as th
from torch.optim import Adam

from src.components.episode_buffer import EpisodeBatch
from src.modules.critics.independent_value import IndependentValueCritic


class IPPOLearner:
    """Independent PPO with a shared decentralized actor and local critic."""

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.critic = IndependentValueCritic(scheme, args)

        self.actor_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.actor_params + self.critic_params
        self.optimiser = Adam(
            [
                {"params": self.actor_params, "lr": args.lr},
                {"params": self.critic_params, "lr": args.critic_lr},
            ],
            eps=float(getattr(args, "ppo_adam_eps", 1.0e-5)),
        )
        self.log_stats_t = -self.args.learner_log_interval - 1

    @staticmethod
    def _scheduled_value(schedule, t_env, default):
        if not schedule:
            return float(default)
        points = sorted((int(t), float(value)) for t, value in schedule)
        if t_env <= points[0][0]:
            return points[0][1]
        for (left_t, left_value), (right_t, right_value) in zip(
            points[:-1], points[1:]
        ):
            if t_env < right_t:
                if right_t == left_t:
                    return right_value
                fraction = (t_env - left_t) / float(right_t - left_t)
                return left_value + fraction * (right_value - left_value)
        return points[-1][1]

    def _actor_probabilities(self, batch):
        self.mac.init_hidden(batch.batch_size)
        probabilities = [
            self.mac.forward(batch, t=timestep, test_mode=False)
            for timestep in range(batch.max_seq_length)
        ]
        return th.stack(probabilities, dim=1)

    def _advantages_and_returns(self, batch, old_values, mask):
        rewards = batch["reward"][:, :-1].float()
        terminated = batch["terminated"][:, :-1].float()
        n_agents = int(self.args.n_agents)
        transition_count = rewards.size(1)

        advantages = old_values[:, :-1].new_zeros(
            batch.batch_size, transition_count, n_agents
        )
        gae = old_values.new_zeros(batch.batch_size, n_agents)
        gamma = float(self.args.gamma)
        gae_lambda = float(getattr(self.args, "gae_lambda", 0.95))

        for timestep in reversed(range(transition_count)):
            nonterminal = 1.0 - terminated[:, timestep]
            reward = rewards[:, timestep].expand(-1, n_agents)
            delta = (
                reward
                + gamma * nonterminal * old_values[:, timestep + 1]
                - old_values[:, timestep]
            )
            gae = delta + gamma * gae_lambda * nonterminal * gae
            gae = gae * mask[:, timestep]
            advantages[:, timestep] = gae

        returns = advantages + old_values[:, :-1]
        return advantages, returns

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        del episode_num
        actions = batch["actions"][:, :-1]
        mask = batch["filled"][:, :-1].float()
        terminated = batch["terminated"][:, :-1].float()
        if mask.size(1) > 1:
            mask[:, 1:] *= 1.0 - terminated[:, :-1]
        agent_mask = mask.expand(-1, -1, int(self.args.n_agents))
        valid = agent_mask.bool()
        valid_count = agent_mask.sum().clamp_min(1.0)

        with th.no_grad():
            old_probabilities = self._actor_probabilities(batch)[:, :-1]
            old_action_probabilities = th.gather(
                old_probabilities, dim=3, index=actions
            ).squeeze(3).clamp_min(1.0e-8)
            old_log_probabilities = old_action_probabilities.log()
            old_values = self.critic(batch)
            advantages, returns = self._advantages_and_returns(
                batch, old_values, agent_mask
            )

            valid_advantages = advantages[valid]
            advantage_mean = valid_advantages.mean()
            advantage_std = valid_advantages.std(unbiased=False).clamp_min(
                1.0e-8
            )
            advantages = (advantages - advantage_mean) / advantage_std
            advantages = advantages * agent_mask

        clip_param = float(getattr(self.args, "ppo_clip_param", 0.2))
        value_clip = float(getattr(self.args, "ppo_value_clip", 0.2))
        value_coefficient = float(
            getattr(self.args, "ppo_value_loss_coefficient", 1.0)
        )
        entropy_coefficient = self._scheduled_value(
            getattr(self.args, "ppo_entropy_schedule", None),
            t_env,
            getattr(self.args, "ppo_entropy_coefficient", 0.0),
        )
        ppo_epochs = int(getattr(self.args, "ppo_epochs", 4))
        target_kl = float(getattr(self.args, "ppo_target_kl", 0.0))
        grad_clip = float(
            getattr(self.args, "ppo_grad_norm_clip", self.args.grad_norm_clip)
        )

        epoch_stats = []
        for _ in range(ppo_epochs):
            probabilities = self._actor_probabilities(batch)[:, :-1]
            action_probabilities = th.gather(
                probabilities, dim=3, index=actions
            ).squeeze(3).clamp_min(1.0e-8)
            log_probabilities = action_probabilities.log()
            log_ratio = log_probabilities - old_log_probabilities
            ratio = log_ratio.exp()

            surrogate = ratio * advantages
            clipped_surrogate = ratio.clamp(
                1.0 - clip_param, 1.0 + clip_param
            ) * advantages
            policy_loss = -(
                th.minimum(surrogate, clipped_surrogate) * agent_mask
            ).sum() / valid_count

            values = self.critic(batch)[:, :-1]
            value_error = (values - returns).pow(2)
            if value_clip > 0.0:
                clipped_values = old_values[:, :-1] + (
                    values - old_values[:, :-1]
                ).clamp(-value_clip, value_clip)
                clipped_value_error = (clipped_values - returns).pow(2)
                value_error = th.maximum(value_error, clipped_value_error)
            value_loss = 0.5 * (value_error * agent_mask).sum() / valid_count

            entropy = -(
                probabilities.clamp_min(1.0e-8).log() * probabilities
            ).sum(dim=-1)
            entropy = (entropy * agent_mask).sum() / valid_count

            loss = (
                policy_loss
                + value_coefficient * value_loss
                - entropy_coefficient * entropy
            )
            self.optimiser.zero_grad()
            loss.backward()
            if grad_clip > 0.0:
                grad_norm = th.nn.utils.clip_grad_norm_(
                    self.params, grad_clip
                )
            else:
                grad_norm = th.sqrt(
                    sum(
                        parameter.grad.detach().pow(2).sum()
                        for parameter in self.params
                        if parameter.grad is not None
                    )
                )
            self.optimiser.step()

            with th.no_grad():
                approximate_kl = (
                    ((ratio - 1.0) - log_ratio) * agent_mask
                ).sum() / valid_count
                clip_fraction = (
                    ((ratio - 1.0).abs() > clip_param).float() * agent_mask
                ).sum() / valid_count
            epoch_stats.append(
                (
                    loss.item(),
                    policy_loss.item(),
                    value_loss.item(),
                    entropy.item(),
                    approximate_kl.item(),
                    clip_fraction.item(),
                    float(grad_norm),
                )
            )
            if target_kl > 0.0 and approximate_kl.item() > target_kl:
                break

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            names = (
                "loss",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "grad_norm",
            )
            averaged = [
                sum(values) / len(values) for values in zip(*epoch_stats)
            ]
            for name, value in zip(names, averaged):
                self.logger.log_stat("ppo_" + name, value, t_env)
            self.logger.log_stat(
                "ppo_entropy_coefficient", entropy_coefficient, t_env
            )

            with th.no_grad():
                predicted_values = self.critic(batch)[:, :-1][valid]
                target_returns = returns[valid]
                return_variance = target_returns.var(unbiased=False)
                explained_variance = 1.0 - (
                    (target_returns - predicted_values).var(unbiased=False)
                    / return_variance.clamp_min(1.0e-8)
                )
            self.logger.log_stat(
                "ppo_explained_variance", explained_variance.item(), t_env
            )
            self.log_stats_t = t_env

    def cuda(self):
        self.mac.cuda()
        self.critic.to(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), os.path.join(path, "critic.th"))
        th.save(self.optimiser.state_dict(), os.path.join(path, "opt.th"))

    def load_models(self, path):
        self.mac.load_models(path)
        map_location = lambda storage, location: storage
        self.critic.load_state_dict(
            th.load(
                os.path.join(path, "critic.th"),
                map_location=map_location,
            )
        )
        self.optimiser.load_state_dict(
            th.load(os.path.join(path, "opt.th"), map_location=map_location)
        )
