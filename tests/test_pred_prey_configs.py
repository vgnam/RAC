from src.main import _load_config


VALUE_BASED_SHARED = {
    "lr": 0.0005,
    "batch_size": 32,
    "buffer_size": 5000,
    "training_iters": 1,
    "target_update_interval": 200,
    "gamma": 0.99,
    "n_step": 1,
    "epsilon_start": 1.0,
    "epsilon_finish": 0.05,
    "epsilon_anneal_time": 50000,
}


def test_pred_prey_value_based_presets_share_training_schedule():
    configs = [
        _load_config("pred_prey_iql", "algs"),
        _load_config("pred_prey_rac", "algs"),
        _load_config("pred_prey_belief_rac", "algs"),
        _load_config("pred_prey_ucvar_iql", "algs"),
    ]

    for config in configs:
        for key, expected in VALUE_BASED_SHARED.items():
            assert config[key] == expected


def test_pred_prey_presets_select_expected_algorithms():
    iql = _load_config("pred_prey_iql", "algs")
    rac = _load_config("pred_prey_rac", "algs")
    belief_rac = _load_config("pred_prey_belief_rac", "algs")
    ippo = _load_config("pred_prey_ippo", "algs")

    assert (iql["name"], iql["learner"]) == ("iql", "q_learner")
    assert (rac["name"], rac["learner"]) == (
        "dual_iql_ree",
        "dual_episode_ree_q_learner",
    )
    assert not rac.get("use_context_belief", False)
    assert belief_rac["name"] == "belief_dual_iql_ree"
    assert belief_rac["use_context_belief"] is True

    assert ippo["name"] == "ippo"
    assert ippo["learner"] == "ippo_learner"
    assert ippo["runner"] == "parallel"
    assert ippo["on_policy"] is True
    assert ippo["batch_size_run"] == ippo["batch_size"] == 4
    assert ippo["buffer_size"] == 4


def test_pred_prey_task_budget_is_shared_by_all_presets():
    environment = _load_config("pred_prey_punish", "envs")

    assert environment["env"] == "stag_hunt"
    assert environment["t_max"] == 1_000_000
    assert environment["test_interval"] == 5_000
    assert environment["test_nepisode"] == 32


def test_pred_prey_ghost_only_adds_stochastic_capture_reward():
    base = _load_config("pred_prey_punish", "envs")
    ghost = _load_config("pred_prey_ghost", "envs")

    changed = {k for k in ghost["env_args"] if ghost["env_args"][k] != base["env_args"].get(k)}
    assert changed <= {"random_ghosts", "random_ghosts_prob", "random_ghosts_mul", "random_ghosts_indicator"}
    assert ghost["env_args"]["random_ghosts"] is True
    assert ghost["env_args"]["random_ghosts_mul"] < 0
    assert ghost["t_max"] == base["t_max"]
