import torch as th
import torch.nn as nn
import torch.nn.functional as F

class FFAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(FFAgent, self).__init__()
        self.args = args

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
            if in_features != args.rnn_hidden_dim:
                raise ValueError(
                    "The final agent_mlp_dims value must equal rnn_hidden_dim; "
                    f"got {in_features} and {args.rnn_hidden_dim}."
                )
            self.encoder = nn.Sequential(*layers)
            self.fc1 = None
            self.fc2 = None
            self.fc3 = nn.Linear(in_features, args.n_actions)
        else:
            # Preserve the original feed-forward agent for other environments.
            self.encoder = None
            self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
            self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if bool(getattr(args, "agent_orthogonal_init", False)):
            gain = nn.init.calculate_gain(activation_name)
            modules = (
                list(self.encoder.modules())
                if self.encoder is not None
                else [self.fc1, self.fc2]
            )
            for module in modules:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=gain)
                    nn.init.zeros_(module.bias)
            nn.init.orthogonal_(self.fc3.weight, gain=1.0)
            nn.init.zeros_(self.fc3.bias)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc3.weight.new_zeros(1, self.args.rnn_hidden_dim)

    def forward(self, inputs, hidden_state):
        if self.encoder is not None:
            h = self.encoder(inputs)
        else:
            x = F.relu(self.fc1(inputs))
            h = F.relu(self.fc2(x))
        q = self.fc3(h)
        return q, h
