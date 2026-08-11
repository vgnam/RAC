import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.agents import REGISTRY as agent_REGISTRY
from src.modules.context_belief import (
    ContextBeliefModel,
    posterior_optimistic_q,
)
from src.components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class DualREEMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.use_context_belief = bool(
            getattr(args, "use_context_belief", False)
        )
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.input_shape = input_shape
        self.agent_output_type = args.agent_output_type

        self.context_model = None
        if self.use_context_belief:
            obs_shape = scheme["obs"]["vshape"]
            if isinstance(obs_shape, int):
                obs_dim = obs_shape
            else:
                obs_dim = 1
                for dimension in obs_shape:
                    obs_dim *= dimension
            self.context_model = ContextBeliefModel(
                obs_dim,
                args.n_actions,
                args.n_agents,
                args,
            )
            optimism_min = float(
                getattr(args, "belief_optimism_min", 0.25)
            )
            optimism_max = float(
                getattr(args, "belief_optimism_max", 1.0)
            )
            if not 0.0 < optimism_min <= optimism_max <= 1.0:
                raise ValueError(
                    "Belief-aware RAC requires 0 < belief_optimism_min <= "
                    "belief_optimism_max <= 1 so cooperative optimism is "
                    "never removed."
                )

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self.twin_hidden_states = None
        self.twin_counter_hidden_states = None
        self.belief_hidden_states = None
        self.context_beliefs = None
        self.last_belief_diagnostics = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        if self.use_context_belief and bool(
            getattr(self.args, "belief_direct_action", True)
        ):
            belief, shift = self._online_belief_step(ep_batch, t)
            indices = th.eye(
                self.args.slot_number,
                device=ep_batch.device,
            ).view(1, 1, self.args.slot_number, self.args.slot_number)
            indices = indices.expand(
                ep_batch.batch_size,
                self.n_agents,
                -1,
                -1,
            )
            context_q = self.counter_forward(ep_batch, indices, t=t)
            # (batch, agents, actions, contexts)
            context_q = context_q.permute(0, 1, 3, 2)
            decision_q, diagnostics = self.aggregate_context_q(
                context_q,
                belief,
                shift,
            )
            self.last_belief_diagnostics = {
                key: value.detach() for key, value in diagnostics.items()
            }
            self.last_belief_diagnostics["shift"] = shift.detach()
            return decision_q

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    # def forward(self, ep_batch, t, test_mode=False):
    #     indices = th.eye(self.args.slot_number, device=self.args.device).unsqueeze(dim=0).expand(ep_batch.batch_size * self.args.n_agents, -1, -1)
    #     indices = indices.reshape(ep_batch.batch_size, self.args.n_agents, self.args.slot_number, self.args.slot_number)
    #     counter_twin_agent_outs = self.counter_forward(ep_batch, indices, t=t) # (bs, n_agents, slot_number, n_actions)
    #     counter_twin_agent_outs = counter_twin_agent_outs.permute(0, 1, 3, 2)  # (bs, n_agents, n_actions, slot_number)
    #
    #     argmax_counter_twin_agent_outs = counter_twin_agent_outs.max(dim=-1)[0]  # (bs, n_agents, n_actions)
    #
    #     return argmax_counter_twin_agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def train_forward(self, ep_batch, return_indices, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        twin_agent_outs, self.twin_hidden_states = self.twin_agent(agent_inputs, return_indices, self.twin_hidden_states)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), \
            twin_agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def counter_forward(self, ep_batch, return_indices, t):
        # return_indices.shape=(bs, n_agents, slot_number, slot_number)
        agent_inputs = self._build_inputs(ep_batch, t)      # (bs*n_agents, input_shape)
        agent_inputs_rep = agent_inputs.unsqueeze(dim=1).expand(-1, self.args.slot_number, -1)
        agent_inputs_rep = agent_inputs_rep.reshape(-1, self.input_shape)   # (bs*n_agents*slot_number, input_shape)
        return_indices = return_indices.reshape(-1, self.args.slot_number)  # (bs*n_agents*slot_number, slot_number)
        twin_counter_agent_outs, self.twin_counter_hidden_states = self.twin_agent(agent_inputs_rep, return_indices, self.twin_counter_hidden_states)

        return twin_counter_agent_outs.view(ep_batch.batch_size, self.n_agents, self.args.slot_number, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        if self.use_context_belief:
            self.init_twin_counter_hidden(batch_size)
            (
                self.belief_hidden_states,
                self.context_beliefs,
            ) = self.context_model.initial_state(batch_size, self.args.device)
            self.last_belief_diagnostics = None

    def init_twin_hidden(self, batch_size):
        self.twin_hidden_states = self.twin_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def init_twin_counter_hidden(self, batch_size):
        self.twin_counter_hidden_states = self.twin_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents * self.args.slot_number, -1)

    def parameters(self):
        parameters = list(self.agent.parameters()) + list(self.twin_agent.parameters())
        if self.context_model is not None:
            parameters += list(self.context_model.parameters())
        return parameters

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.twin_agent.load_state_dict(other_mac.twin_agent.state_dict())
        if self.context_model is not None:
            self.context_model.load_state_dict(other_mac.context_model.state_dict())

    def cuda(self):
        self.agent.to(self.args.device)
        self.twin_agent.to(self.args.device)
        if self.context_model is not None:
            self.context_model.to(self.args.device)

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.twin_agent.state_dict(), "{}/twin_agent.th".format(path))
        if self.context_model is not None:
            th.save(
                self.context_model.state_dict(),
                "{}/context_belief.th".format(path),
            )

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.twin_agent.load_state_dict(th.load("{}/twin_agent.th".format(path), map_location=lambda storage, loc: storage))
        if self.context_model is not None:
            self.context_model.load_state_dict(
                th.load(
                    "{}/context_belief.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )

    def infer_context_sequence(self, batch):
        if self.context_model is None:
            raise RuntimeError("Context belief is not enabled for this controller.")
        return self.context_model.infer_sequence(batch)

    def context_dynamics_loss(
        self,
        batch,
        beliefs,
        transition_mask,
        context_assignments=None,
    ):
        if self.context_model is None:
            raise RuntimeError("Context belief is not enabled for this controller.")
        return self.context_model.dynamics_loss(
            batch,
            beliefs,
            transition_mask,
            self.args,
            context_assignments=context_assignments,
        )

    def aggregate_context_q(self, context_q, belief, shift=None):
        return posterior_optimistic_q(
            context_q,
            belief,
            context_shift=shift,
            optimism_min=float(
                getattr(self.args, "belief_optimism_min", 0.25)
            ),
            optimism_max=float(
                getattr(self.args, "belief_optimism_max", 1.0)
            ),
        )

    def _online_belief_step(self, ep_batch, t):
        if self.belief_hidden_states is None or self.context_beliefs is None:
            (
                self.belief_hidden_states,
                self.context_beliefs,
            ) = self.context_model.initial_state(
                ep_batch.batch_size,
                ep_batch.device,
            )

        if t == 0:
            previous_action = th.zeros_like(ep_batch["actions_onehot"][:, t])
            previous_reward = th.zeros_like(ep_batch["reward"][:, t])
        else:
            previous_action = ep_batch["actions_onehot"][:, t - 1]
            previous_reward = ep_batch["reward"][:, t - 1]

        observation = ep_batch["obs"][:, t].reshape(
            ep_batch.batch_size,
            self.n_agents,
            -1,
        )
        (
            belief,
            _,
            self.belief_hidden_states,
            _,
            shift,
        ) = self.context_model.filter_step(
            observation,
            previous_action,
            previous_reward,
            self.belief_hidden_states,
            self.context_beliefs,
        )
        self.context_beliefs = belief
        return belief, shift

    def _build_agents(self, input_shape):
        # REE builds two agents, q^{i}(\tau^{i},a^{i}), q_{twin}^{i}(\tau^{i},r^{i},a^{i})
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.twin_agent = agent_REGISTRY[self.args.twin_agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
