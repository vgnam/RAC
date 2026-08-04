import torch as th
import torch.nn as nn
import torch.nn.functional as F


# We consider two possible implementations of this agent, hyper-net, or input return_slot_index into fc_2
# These two implementations are proposed to deal with obs with high dimensions.
class NormalREEAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NormalREEAgent, self).__init__()
        self.args = args
        self.slot_number = args.slot_number

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear((args.rnn_hidden_dim + self.slot_number), args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, return_indices, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        return_indices = return_indices.reshape(-1, self.args.slot_number)
        concat_inps = th.cat([h, return_indices], dim=-1)
        q = self.fc2(concat_inps)
        return q, h


class HyperREEAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(HyperREEAgent, self).__init__()
        self.args = args
        self.slot_number = args.slot_number

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.fc2_w_net = nn.Linear(args.slot_number, args.rnn_hidden_dim * args.n_actions)
        self.fc2_b_net = nn.Linear(args.slot_number, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, return_indices, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        return_indices = return_indices.reshape(-1, self.args.slot_number)

        fc2_w = self.fc2_w_net(return_indices)  # (-1, rnn_hidden_dim*n_actions)
        fc2_b = self.fc2_b_net(return_indices)  # (-1, n_actions)
        fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        fc2_b = fc2_b.reshape(-1, 1, self.args.n_actions)

        h = h.reshape(-1, 1, self.args.rnn_hidden_dim)
        q = th.bmm(h, fc2_w) + fc2_b
        return q, h