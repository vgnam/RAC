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
    assert alg["corac_min_std_frac"] == 0.05
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
