import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..contrastive_context import corac_context_dim


def _twin_context_dim(args):
    """Context width: CoRAC derives it from the enabled halves, RAC uses slots."""
    if str(getattr(args, "context_labeler", "return_bins")) == "corac":
        default = corac_context_dim(args)
    else:
        default = args.slot_number
    return int(getattr(args, "context_dim", None) or default)


# We consider two possible implementations of this agent, hyper-net, or input return_slot_index into fc_2
# These two implementations are proposed to deal with obs with high dimensions.


def _build_encoder(input_shape, args):
    encoder_dims = list(getattr(args, "agent_mlp_dims", []) or [])
    activation_name = str(getattr(args, "agent_activation", "relu")).lower()
    activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
    if activation_name not in activations:
        raise ValueError(
            f"Unsupported agent_activation {activation_name!r}; "
            f"choose from {list(activations)}."
        )

    if not encoder_dims:
        return None, nn.Linear(input_shape, args.rnn_hidden_dim), args.rnn_hidden_dim

    layers = []
    in_features = input_shape
    for out_features in encoder_dims:
        layers.extend([
            nn.Linear(in_features, int(out_features)),
            activations[activation_name](),
        ])
        in_features = int(out_features)
    return nn.Sequential(*layers), None, in_features


def _initialize_encoder(encoder, fc1, output_layers, args):
    if not bool(getattr(args, "agent_orthogonal_init", False)):
        return

    activation_name = str(getattr(args, "agent_activation", "relu")).lower()
    gain = nn.init.calculate_gain(activation_name)
    modules = list(encoder.modules()) if encoder is not None else [fc1]
    for module in modules:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=gain)
            nn.init.zeros_(module.bias)
    for module in output_layers:
        nn.init.orthogonal_(module.weight, gain=1.0)
        nn.init.zeros_(module.bias)


def _encode(inputs, encoder, fc1):
    if encoder is not None:
        return encoder(inputs)
    return F.relu(fc1(inputs))


class NormalREEAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NormalREEAgent, self).__init__()
        self.args = args
        self.slot_number = args.slot_number
        self.recurrent = bool(getattr(args, "agent_recurrent", True))

        self.encoder, self.fc1, rnn_input_dim = _build_encoder(input_shape, args)
        if self.recurrent:
            self.rnn = nn.GRUCell(rnn_input_dim, args.rnn_hidden_dim)
        else:
            if rnn_input_dim != args.rnn_hidden_dim:
                raise ValueError(
                    "A feed-forward REE agent requires the final "
                    f"agent_mlp_dims value ({rnn_input_dim}) to equal "
                    f"rnn_hidden_dim ({args.rnn_hidden_dim})."
                )
            self.rnn = None
        self.fc2 = nn.Linear((args.rnn_hidden_dim + self.slot_number), args.n_actions)
        _initialize_encoder(self.encoder, self.fc1, [self.fc2], args)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc2.weight.new_zeros(1, self.args.rnn_hidden_dim)

    def forward(self, inputs, return_indices, hidden_state):
        x = _encode(inputs, self.encoder, self.fc1)
        if self.recurrent:
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(x, h_in)
        else:
            h = x
        return_indices = return_indices.reshape(-1, self.args.slot_number)
        concat_inps = th.cat([h, return_indices], dim=-1)
        q = self.fc2(concat_inps)
        return q, h


class ZeroDeltaHyperHead(nn.Module):
    """Hypernetwork for a context-conditioned linear Q head (HC-RAC).

    W(c), b(c) = base head + MLP(c). The MLP's last layer starts at zero, so every
    context initially shares the base head and a max over contexts starts without
    optimism; contexts only diverge once the data separates them.
    """

    def __init__(self, context_dim, hidden_dim, n_actions, hyper_hidden):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        bound = 1.0 / hidden_dim ** 0.5
        self.base_weight = nn.Parameter(th.empty(hidden_dim, n_actions).uniform_(-bound, bound))
        self.base_bias = nn.Parameter(th.zeros(n_actions))
        self.delta = nn.Sequential(
            nn.Linear(context_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, hidden_dim * n_actions + n_actions),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, context):
        """context (..., C) -> weight (..., H, A), bias (..., A)."""
        delta = self.delta(context)
        split = self.hidden_dim * self.n_actions
        weight = self.base_weight + delta[..., :split].reshape(
            *context.shape[:-1], self.hidden_dim, self.n_actions
        )
        return weight, self.base_bias + delta[..., split:]


def context_q_all(h, weight, bias):
    """Q for every context of a bank.

    h (..., H) with shared contexts weight (M, H, A), or h (..., n_agents, H) with
    per-agent contexts weight (n_agents, M, H, A); returns (..., [n_agents,] M, A).
    """
    if weight.dim() == 3:
        return th.einsum("...h,mha->...ma", h, weight) + bias
    return th.einsum("...nh,nmha->...nma", h, weight) + bias


class HyperREEAgent(nn.Module):
    # The recurrent core never sees the context, so Q for many contexts shares one pass.
    context_independent_recurrent_core = True

    def __init__(self, input_shape, args):
        super(HyperREEAgent, self).__init__()
        self.args = args
        self.slot_number = args.slot_number
        # RAC: one-hot slots (context_dim = slot_number); HC-RAC: continuous context
        # vector whose width follows from corac_use_return / corac_use_trajectory.
        self.context_dim = _twin_context_dim(args)
        self.recurrent = bool(getattr(args, "agent_recurrent", True))

        self.encoder, self.fc1, rnn_input_dim = _build_encoder(input_shape, args)
        if self.recurrent:
            self.rnn = nn.GRUCell(rnn_input_dim, args.rnn_hidden_dim)
        else:
            if rnn_input_dim != args.rnn_hidden_dim:
                raise ValueError(
                    "A feed-forward REE agent requires the final "
                    f"agent_mlp_dims value ({rnn_input_dim}) to equal "
                    f"rnn_hidden_dim ({args.rnn_hidden_dim})."
                )
            self.rnn = None

        hyper_hidden = int(getattr(args, "twin_hyper_hidden", 0) or 0)
        output_layers = []
        if hyper_hidden > 0:
            self.hyper_head = ZeroDeltaHyperHead(
                self.context_dim, args.rnn_hidden_dim, args.n_actions, hyper_hidden
            )
        else:
            self.hyper_head = None
            self.fc2_w_net = nn.Linear(self.context_dim, args.rnn_hidden_dim * args.n_actions)
            self.fc2_b_net = nn.Linear(self.context_dim, args.n_actions)
            output_layers = [self.fc2_w_net, self.fc2_b_net]
        _initialize_encoder(self.encoder, self.fc1, output_layers, args)

    def init_hidden(self):
        # make hidden states on same device as model
        return next(self.parameters()).new_zeros(1, self.args.rnn_hidden_dim)

    def _context_params(self, context):
        """context (..., C) -> weight (..., H, A), bias (..., A)."""
        if self.hyper_head is not None:
            return self.hyper_head(context)
        weight = self.fc2_w_net(context).reshape(
            *context.shape[:-1], self.args.rnn_hidden_dim, self.args.n_actions
        )
        return weight, self.fc2_b_net(context)

    def _core_step(self, inputs, hidden_state):
        x = _encode(inputs, self.encoder, self.fc1)
        if not self.recurrent:
            return x
        return self.rnn(x, hidden_state.reshape(-1, self.args.rnn_hidden_dim))

    def forward(self, inputs, return_indices, hidden_state):
        h = self._core_step(inputs, hidden_state)
        weight, bias = self._context_params(return_indices.reshape(-1, self.context_dim))
        q = th.bmm(h.unsqueeze(1), weight) + bias.unsqueeze(1)
        return q, h.reshape(-1, 1, self.args.rnn_hidden_dim)

    def _contexts_or_slots(self, contexts, reference):
        if contexts is None:  # RAC: every one-hot slot
            contexts = th.eye(self.context_dim, device=reference.device, dtype=reference.dtype)
        return self._context_params(contexts)

    def counterfactual_step(self, inputs, hidden_state, contexts=None):
        """inputs (bs * n_agents, F) -> Q (bs * n_agents, M, A) for shared contexts (M, C)."""
        h = self._core_step(inputs, hidden_state)
        weight, bias = self._contexts_or_slots(contexts, h)
        return context_q_all(h, weight, bias), h

    def counterfactual_sequence(self, inputs, hidden_state, contexts=None):
        """inputs (bs, T, n_agents, F) -> Q (bs, T, n_agents, M, A).

        ``contexts`` is (M, C) shared by all agents or (n_agents, M, C) per agent.
        """
        batch_size, time_steps, n_agents, _ = inputs.shape
        h = hidden_state.reshape(batch_size * n_agents, -1)
        outputs = []
        for t in range(time_steps):
            h = self._core_step(inputs[:, t].reshape(batch_size * n_agents, -1), h)
            outputs.append(h.view(batch_size, n_agents, -1))
        h_seq = th.stack(outputs, dim=1)
        weight, bias = self._contexts_or_slots(contexts, h_seq)
        return context_q_all(h_seq, weight, bias), h.view(batch_size, n_agents, -1)
