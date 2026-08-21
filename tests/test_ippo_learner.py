import math
from types import SimpleNamespace

import torch as th

from src.components.episode_buffer import EpisodeBatch
from src.controllers.basic_controller import BasicMAC
from src.learners.ippo_learner import IPPOLearner


class RecordingLogger:
    def __init__(self):
        self.stats = {}

    def log_stat(self, name, value, timestep):
        self.stats[name] = (float(value), timestep)


def make_args():
    return SimpleNamespace(
        n_agents=3,
        n_actions=4,
        obs_last_action=True,
        obs_agent_id=True,
        agent="rnn",
        agent_output_type="pi_logits",
        action_selector="multinomial",
        epsilon_start=0.0,
        epsilon_finish=0.0,
        epsilon_anneal_time=1,
        test_greedy=True,
        mask_before_softmax=True,
        rnn_hidden_dim=8,
        agent_mlp_dims=[16, 8],
        agent_activation="tanh",
        agent_recurrent=True,
        agent_orthogonal_init=True,
        critic_hidden_dim=8,
        critic_mlp_dims=[16, 8],
        critic_activation="tanh",
        critic_recurrent=True,
        critic_orthogonal_init=True,
        lr=5.0e-4,
        critic_lr=5.0e-4,
        ppo_adam_eps=1.0e-5,
        ppo_epochs=2,
        ppo_clip_param=0.2,
        ppo_value_clip=0.2,
        ppo_value_loss_coefficient=1.0,
        ppo_entropy_coefficient=0.01,
        ppo_entropy_schedule=None,
        ppo_target_kl=0.0,
        ppo_grad_norm_clip=10.0,
        gamma=0.99,
        gae_lambda=0.95,
        grad_norm_clip=10.0,
        learner_log_interval=1,
        device="cpu",
    )


def make_batch(args):
    batch_size = 2
    sequence_length = 5
    observation_size = 6
    scheme = {
        "obs": {"vshape": observation_size, "group": "agents"},
        "actions": {
            "vshape": (1,),
            "group": "agents",
            "dtype": th.long,
        },
        "actions_onehot": {
            "vshape": (args.n_actions,),
            "group": "agents",
        },
        "avail_actions": {
            "vshape": (args.n_actions,),
            "group": "agents",
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,)},
    }
    batch = EpisodeBatch(
        scheme,
        {"agents": args.n_agents},
        batch_size,
        sequence_length,
    )
    transition_data = batch.data.transition_data
    transition_data["obs"].normal_()
    transition_data["avail_actions"].fill_(1.0)
    transition_data["reward"][:, :-1].normal_()
    transition_data["filled"].fill_(1)

    actions = th.randint(
        args.n_actions,
        (batch_size, sequence_length - 1, args.n_agents, 1),
    )
    transition_data["actions"][:, :-1] = actions
    transition_data["actions_onehot"][:, :-1].scatter_(
        dim=-1, index=actions, value=1.0
    )
    transition_data["terminated"][:, sequence_length - 2] = 1.0
    return batch, scheme


def test_ippo_updates_actor_and_logs_finite_statistics():
    th.manual_seed(7)
    args = make_args()
    batch, scheme = make_batch(args)
    mac = BasicMAC(scheme, {"agents": args.n_agents}, args)
    logger = RecordingLogger()
    learner = IPPOLearner(mac, scheme, logger, args)
    actor_before = [parameter.detach().clone() for parameter in mac.parameters()]

    learner.train(batch, t_env=100, episode_num=2)

    assert any(
        not th.allclose(before, after)
        for before, after in zip(actor_before, mac.parameters())
    )
    expected_stats = {
        "ppo_loss",
        "ppo_policy_loss",
        "ppo_value_loss",
        "ppo_entropy",
        "ppo_approx_kl",
        "ppo_clip_fraction",
        "ppo_grad_norm",
        "ppo_explained_variance",
    }
    assert expected_stats.issubset(logger.stats)
    assert all(math.isfinite(logger.stats[name][0]) for name in expected_stats)
