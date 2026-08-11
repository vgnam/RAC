import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.recurrent = bool(getattr(args, "agent_recurrent", True))

        encoder_dims = list(getattr(args, "agent_mlp_dims", []) or [])
        activation_name = str(getattr(args, "agent_activation", "relu")).lower()
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
        if activation_name not in activations:
            raise ValueError(
                f"Unsupported agent_activation {activation_name!r}; "
                f"choose from {list(activations)}."
            )

        if encoder_dims:
            layers = []
            in_features = input_shape
            for out_features in encoder_dims:
                layers.extend([
                    nn.Linear(in_features, int(out_features)),
                    activations[activation_name](),
                ])
                in_features = int(out_features)
            self.encoder = nn.Sequential(*layers)
            self.fc1 = None
            rnn_input_dim = in_features
        else:
            # Preserve the original network unless an environment opts in.
            self.encoder = None
            self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
            rnn_input_dim = args.rnn_hidden_dim

        if self.recurrent:
            self.rnn = nn.GRUCell(rnn_input_dim, args.rnn_hidden_dim)
        else:
            if rnn_input_dim != args.rnn_hidden_dim:
                raise ValueError(
                    "A feed-forward agent requires the final agent_mlp_dims "
                    f"value ({rnn_input_dim}) to equal rnn_hidden_dim "
                    f"({args.rnn_hidden_dim})."
                )
            self.rnn = None
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if bool(getattr(args, "agent_orthogonal_init", False)):
            gain = nn.init.calculate_gain(activation_name)
            modules = list(self.encoder.modules()) if self.encoder is not None else [self.fc1]
            for module in modules:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=gain)
                    nn.init.zeros_(module.bias)
            nn.init.orthogonal_(self.fc2.weight, gain=1.0)
            nn.init.zeros_(self.fc2.bias)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc2.weight.new_zeros(1, self.args.rnn_hidden_dim)

    def forward(self, inputs, hidden_state):
        if self.encoder is not None:
            x = self.encoder(inputs)
        else:
            x = F.relu(self.fc1(inputs))
        if self.recurrent:
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(x, h_in)
        else:
            h = x
        q = self.fc2(h)
        return q, h
