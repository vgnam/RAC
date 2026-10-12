import copy

import torch as th

from .basic_controller import BasicMAC


class TrembleBlockMAC(BasicMAC):
    """Commitment exploration for Tremble-Robust IQL.

    Training is split into blocks whose boundaries every agent knows. At the start
    of a block each agent independently decides, with the annealed epsilon, whether
    to explore; an exploring agent commits to one intention policy for the whole
    block, the others act greedily (Q only changes at block ends). Teammates'
    intentions are thus fixed within a block, so the block's TD targets estimate
    the value of a_i against one fixed teammate policy, averaged over tremble.

    ``explore_policy``: "action" commits to one uniformly drawn action (single-state
    matrix games); "random_linear" commits to argmax(W x + b) over the agent's
    inputs with W, b ~ N(0, 1) redrawn per block, a deterministic state-dependent
    policy for multi-step games; "noisy_greedy" commits to the greedy policy
    perturbed by that random linear score, argmax(z(Q) + explore_noise * z(W x + b))
    over available actions (z: per-state standardisation), which explores around
    the current policy instead of replacing it.

    ``freeze_acting`` (TR-IQL v2, where Q is trained every episode): act with a
    snapshot of the agent taken at the start of each block, so intentions stay
    fixed within a block while learning continues. Test episodes use the live agent.
    """

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.explore_policy = getattr(args, "explore_policy", "action")
        self.explore_noise = float(getattr(args, "explore_noise", 1.0))
        self.freeze_acting = bool(getattr(args, "freeze_acting", False))
        self.acting_agent = copy.deepcopy(self.agent) if self.freeze_acting else None
        self.explore = None
        self.committed = None
        self.weights = None
        self.bias = None
        self.new_block(t_env=0)

    def new_block(self, t_env):
        epsilon = self.action_selector.schedule.eval(t_env)
        self.action_selector.epsilon = epsilon
        self.explore = th.rand(self.n_agents) < epsilon
        self.committed = th.randint(self.args.n_actions, (self.n_agents,))
        self.weights = th.randn(self.n_agents, self.input_shape, self.args.n_actions)
        self.bias = th.randn(self.n_agents, self.args.n_actions)
        if self.acting_agent is not None:
            self.acting_agent.load_state_dict(self.agent.state_dict())

    def _random_linear_logits(self, ep_batch, t_ep):
        inputs = self._build_inputs(ep_batch, t_ep).view(ep_batch.batch_size, self.n_agents, -1)
        return th.einsum("bni,nia->bna", inputs.float().cpu(), self.weights) + self.bias

    def _committed_actions(self, ep_batch, t_ep, like, bs=slice(None), q=None, avail=None):
        if self.explore_policy == "action":
            return self.committed.to(like.device).unsqueeze(0).expand_as(like)
        logits = self._random_linear_logits(ep_batch, t_ep)[bs].to(like.device)
        if self.explore_policy == "noisy_greedy":
            def z(x):
                x = x.masked_fill(avail == 0, 0.0)
                n = avail.sum(dim=2, keepdim=True).clamp(min=1)
                mean = x.sum(dim=2, keepdim=True) / n
                std = (((x - mean) ** 2).masked_fill(avail == 0, 0.0).sum(dim=2, keepdim=True) / n).sqrt()
                return (x - mean) / (std + 1e-6)
            score = z(q) + self.explore_noise * z(logits)
            return score.masked_fill(avail == 0, -float("inf")).max(dim=2)[1]
        return logits.max(dim=2)[1]

    def _acting_forward(self, ep_batch, t):
        inputs = self._build_inputs(ep_batch, t)
        q, self.hidden_states = self.acting_agent(inputs, self.hidden_states)
        return q.view(ep_batch.batch_size, self.n_agents, -1)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if self.acting_agent is not None and not test_mode:
            q = self._acting_forward(ep_batch, t_ep)[bs].clone()
        else:
            q = self.forward(ep_batch, t_ep, test_mode=test_mode)[bs].clone()
        avail = avail_actions[bs]
        q[avail == 0] = -float("inf")
        greedy = q.max(dim=2)[1]
        if test_mode:
            return greedy
        explore = self.explore.to(greedy.device).unsqueeze(0).expand_as(greedy)
        q_finite = q.masked_fill(avail == 0, 0.0)
        committed = self._committed_actions(ep_batch, t_ep, greedy, bs=bs, q=q_finite, avail=avail)
        return th.where(explore, committed, greedy)

    def cuda(self):
        super().cuda()
        if self.acting_agent is not None:
            self.acting_agent.to(self.args.device)

    def load_models(self, path):
        super().load_models(path)
        if self.acting_agent is not None:
            self.acting_agent.load_state_dict(self.agent.state_dict())
