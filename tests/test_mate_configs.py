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
