import torch as th
import torch.nn as nn
import torch.nn.functional as F


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


class HyperREEAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(HyperREEAgent, self).__init__()
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

        self.fc2_w_net = nn.Linear(args.slot_number, args.rnn_hidden_dim * args.n_actions)
        self.fc2_b_net = nn.Linear(args.slot_number, args.n_actions)
        _initialize_encoder(
            self.encoder,
            self.fc1,
            [self.fc2_w_net, self.fc2_b_net],
            args,
        )

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc2_b_net.weight.new_zeros(1, self.args.rnn_hidden_dim)

    def forward(self, inputs, return_indices, hidden_state):
        x = _encode(inputs, self.encoder, self.fc1)
        if self.recurrent:
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(x, h_in)
        else:
            h = x
        return_indices = return_indices.reshape(-1, self.args.slot_number)

        fc2_w = self.fc2_w_net(return_indices)  # (-1, rnn_hidden_dim*n_actions)
        fc2_b = self.fc2_b_net(return_indices)  # (-1, n_actions)
        fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        fc2_b = fc2_b.reshape(-1, 1, self.args.n_actions)

        h = h.reshape(-1, 1, self.args.rnn_hidden_dim)
        q = th.bmm(h, fc2_w) + fc2_b
        return q, h
