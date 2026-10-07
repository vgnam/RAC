import torch as th

from .basic_controller import BasicMAC


class TrembleBlockMAC(BasicMAC):
    """Commitment exploration for Tremble-Robust IQL.

    Training is split into blocks whose boundaries every agent knows. At the start
    of a block each agent independently decides, with the annealed epsilon, whether
    to explore; an exploring agent commits to one uniformly drawn intended action
    for the whole block, the others act greedily. Teammates' intentions are thus
    fixed within a block, so the block's mean reward estimates r_eps(a_i, a_-i).
    """

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.explore = None
        self.committed = None
        self.new_block(t_env=0)

    def new_block(self, t_env):
        epsilon = self.action_selector.schedule.eval(t_env)
        self.action_selector.epsilon = epsilon
        self.explore = th.rand(self.n_agents) < epsilon
        self.committed = th.randint(self.args.n_actions, (self.n_agents,))

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        q = self.forward(ep_batch, t_ep, test_mode=test_mode)[bs].clone()
        q[avail_actions[bs] == 0] = -float("inf")
        greedy = q.max(dim=2)[1]
        if test_mode:
            return greedy
        explore = self.explore.to(greedy.device).unsqueeze(0).expand_as(greedy)
        committed = self.committed.to(greedy.device).unsqueeze(0).expand_as(greedy)
        return th.where(explore, committed, greedy)
