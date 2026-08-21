from copy import copy

import torch as th
import torch.nn as nn

from src.modules.agents.rnn_agent import RNNAgent


class IndependentValueCritic(nn.Module):
    """Shared local value function used by IPPO.

    Each value estimate only consumes the corresponding agent's decentralized
    controller input. Parameters are shared across homogeneous agents, while
    the optional one-hot agent id lets the value function specialize.
    """

    def __init__(self, scheme, args):
        super().__init__()
        self.args = args
        self.n_agents = int(args.n_agents)

        input_shape = int(scheme["obs"]["vshape"])
        if args.obs_last_action:
            input_shape += int(scheme["actions_onehot"]["vshape"][0])
        if args.obs_agent_id:
            input_shape += self.n_agents

        critic_args = copy(args)
        critic_args.n_actions = 1
        critic_args.rnn_hidden_dim = int(
            getattr(args, "critic_hidden_dim", args.rnn_hidden_dim)
        )
        critic_args.agent_mlp_dims = list(
            getattr(
                args,
                "critic_mlp_dims",
                getattr(args, "agent_mlp_dims", []),
            )
            or []
        )
        critic_args.agent_activation = getattr(
            args,
            "critic_activation",
            getattr(args, "agent_activation", "relu"),
        )
        critic_args.agent_recurrent = bool(
            getattr(args, "critic_recurrent", True)
        )
        critic_args.agent_orthogonal_init = bool(
            getattr(args, "critic_orthogonal_init", True)
        )
        self.value_net = RNNAgent(input_shape, critic_args)

    def _build_sequence_inputs(self, batch):
        batch_size = batch.batch_size
        time_steps = batch.max_seq_length
        inputs = [batch["obs"]]

        if self.args.obs_last_action:
            previous_actions = th.cat(
                [
                    th.zeros_like(batch["actions_onehot"][:, :1]),
                    batch["actions_onehot"][:, :-1],
                ],
                dim=1,
            )
            inputs.append(previous_actions)

        if self.args.obs_agent_id:
            agent_ids = th.eye(self.n_agents, device=batch.device)
            agent_ids = agent_ids.view(1, 1, self.n_agents, self.n_agents)
            inputs.append(
                agent_ids.expand(batch_size, time_steps, -1, -1)
            )

        return th.cat(inputs, dim=-1)

    def forward(self, batch):
        inputs = self._build_sequence_inputs(batch)
        batch_size, time_steps, n_agents, input_dim = inputs.shape
        hidden = self.value_net.init_hidden().unsqueeze(0).expand(
            batch_size, n_agents, -1
        )

        values = []
        for timestep in range(time_steps):
            value, hidden = self.value_net(
                inputs[:, timestep].reshape(
                    batch_size * n_agents, input_dim
                ),
                hidden,
            )
            hidden = hidden.view(batch_size, n_agents, -1)
            values.append(value.view(batch_size, n_agents))
        return th.stack(values, dim=1)
