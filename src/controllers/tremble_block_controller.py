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
    policy for multi-step games.
    """

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.explore_policy = getattr(args, "explore_policy", "action")
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

    def _committed_actions(self, ep_batch, t_ep, like):
        if self.explore_policy == "action":
            return self.committed.to(like.device).unsqueeze(0).expand_as(like)
        inputs = self._build_inputs(ep_batch, t_ep).view(ep_batch.batch_size, self.n_agents, -1)
        logits = th.einsum("bni,nia->bna", inputs.float().cpu(), self.weights) + self.bias
        return logits.max(dim=2)[1].to(like.device)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        q = self.forward(ep_batch, t_ep, test_mode=test_mode)[bs].clone()
        q[avail_actions[bs] == 0] = -float("inf")
        greedy = q.max(dim=2)[1]
        if test_mode:
            return greedy
        explore = self.explore.to(greedy.device).unsqueeze(0).expand_as(greedy)
        committed = self._committed_actions(ep_batch, t_ep, greedy)[bs]
        return th.where(explore, committed, greedy)
