from src.main import _load_config


def test_belief_entity_gru_obstacle_free_preset_is_consistent():
    config = _load_config("mate_belief_entity_gru_4v8_0", "envs")

    assert config["env_args"]["mate_config"] == "MATE-4v8-0.yaml"
    assert config["mate_num_cameras"] == 4
    assert config["mate_num_targets"] == 8
    assert config["mate_num_obstacles"] == 0
    assert config["agent"] == "mate_entity_gru"
    assert config["twin_agent"] == "mate_entity_gru_twin"


def test_belief_entity_gru_target_sparse_preset_is_consistent():
    config = _load_config("mate_belief_entity_gru_4v2_9", "envs")

    assert config["env_args"]["mate_config"] == "MATE-4v2-9.yaml"
    assert config["mate_num_cameras"] == 4
    assert config["mate_num_targets"] == 2
    assert config["mate_num_obstacles"] == 9
    assert config["agent"] == "mate_entity_gru"
    assert config["twin_agent"] == "mate_entity_gru_twin"


def test_corac_mate_preset_is_consistent():
    alg = _load_config("corac_mate", "algs")
    env = _load_config("mate_belief_entity_gru_4v8_0", "envs")

    assert alg["context_labeler"] == "corac"
    assert alg["corac_segment_len"] == 16  # 500-step episodes need real windows
    assert alg["twin_target_clamp"] is True
    # Stationary contexts: fixed phi range (RAC's slot range), EMA context
    # encoder, slow bank turnover.
    assert alg["corac_return_range"] == [0.0, 1000.0]
    assert alg["corac_target_tau"] > 0
    assert 0 < alg["corac_bank_push"] < alg["corac_bank_size"]
    assert alg["corac_use_return"] is True and alg["corac_use_trajectory"] is True
    # context_dim is derived from the halve flags, so it must not be pinned here.
    assert "context_dim" not in alg
    # The env preset is still the one that picks the camera-entity twin.
    assert env["agent"] == "mate_entity_gru"
    assert env["twin_agent"] == "mate_entity_gru_twin"
    assert env["mate_num_obstacles"] == 0


def test_corac_mate_ablations_flip_one_flag_each():
    return_only = _load_config("corac_mate_return_only", "algs")
    assert return_only["corac_use_return"] is True
    assert return_only["corac_use_trajectory"] is False
    assert return_only["corac_segment_len"] == 16  # inherited from corac_mate

    trajectory_only = _load_config("corac_mate_trajectory_only", "algs")
    assert trajectory_only["corac_use_return"] is False
    assert trajectory_only["corac_use_trajectory"] is True

    from src.modules.contrastive_context import corac_context_dim
    from types import SimpleNamespace

    assert corac_context_dim(SimpleNamespace(**return_only)) == return_only["corac_n_freq"]
    assert corac_context_dim(SimpleNamespace(**trajectory_only)) == trajectory_only["corac_embed_dim"]


def test_corac_mate_stability_ladder_changes_one_thing_per_step():
    """A -> D must be a nested ladder, one stabiliser per step (run_mate_stability.py)."""
    from src.main import recursive_dict_update

    env = _load_config("mate_belief_entity_gru_4v8_0", "envs")

    def effective(name):
        # Same order as src/main.py: algorithm defaults, then env overrides.
        merged = {}
        recursive_dict_update(merged, _load_config(name, "algs"))
        recursive_dict_update(merged, env)
        return merged

    a, b, c, d = (effective(n) for n in
                  ("corac_mate", "corac_mate_anneal", "corac_mate_argmax", "corac_mate_stable"))

    # A: constant optimism, per-action max over contexts.
    assert a["kl_weight_final"] is None
    assert a["teacher_context_select"] == "action_max"
    assert a["kl_logit_temperature_frac"] == 0.0

    # B: anneal the optimism away instead of keeping it for the whole run.
    # (kl_weight itself must not change between steps: the env preset owns it.)
    assert b["kl_weight_final"] == 0.0
    assert 0 < b["kl_anneal_start"] < b["kl_anneal_end"]
    assert b["kl_weight"] == a["kl_weight"]
    assert b["teacher_context_select"] == "action_max"

    # C: one context per step instead of a per-action max; B is inherited.
    assert c["teacher_context_select"] == "context_argmax"
    assert c["kl_weight_final"] == 0.0
    assert c["kl_anneal_end"] == b["kl_anneal_end"]

    # D: soft top-k (replaces C's single-context argmax) + scale-free KL.
    assert d["teacher_context_select"] == "topk_mean"
    assert d["teacher_topk"] >= 2
    assert d["kl_logit_temperature_frac"] > 0
    assert d["kl_weight_final"] == 0.0
    # The stationary-context settings from corac_mate must survive the ladder.
    for key in ("corac_return_range", "corac_target_tau", "corac_bank_push", "twin_target_clamp"):
        assert d[key] == a[key], key

    assert len({a["name"], b["name"], c["name"], d["name"]}) == 4  # distinct runs


def test_corac_lbf_preset_matches_the_binary_return():
    from src.main import recursive_dict_update

    alg = _load_config("corac_lbf", "algs")
    env = _load_config("lbf", "envs")
    merged = {}
    recursive_dict_update(merged, alg)
    recursive_dict_update(merged, env)
    assert merged["env"] == "lbf"
    # The LBF env preset must not carry a MATE-scale kl_weight over this one.
    assert merged["kl_weight"] == alg["kl_weight"]

    # phi must not drift or saturate once every episode returns 1.
    assert alg["corac_return_range"] == [0.0, 1.0]
    assert alg["corac_target_tau"] > 0
    assert 0 < alg["corac_bank_push"] < alg["corac_bank_size"]
    # Optimism loop: annealed, top-k, scale-free.
    assert alg["kl_weight_final"] == 0.0
    assert 0 < alg["kl_anneal_start"] < alg["kl_anneal_end"]
    assert alg["teacher_context_select"] == "topk_mean"
    assert alg["kl_logit_temperature_frac"] > 0
    # The MATE-scale weight (10 from dual_iql_ree) must not be reused on [0, 1].
    assert alg["kl_weight"] < 10.0
    assert alg["twin_target_clamp"] is True
    assert alg["context_labeler"] == "corac"
