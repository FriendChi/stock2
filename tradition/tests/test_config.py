from tradition.config import build_tradition_config


def test_build_tradition_config_supports_override():
    config = build_tradition_config({"default_fund_code": "012832"})
    assert config["default_fund_code"] == "012832"
    assert "buy_and_hold" in config["strategy_param_dict"]


def test_build_tradition_config_contains_split_and_optimization_defaults():
    config = build_tradition_config()
    assert config["data_split_dict"]["train_ratio"] == 0.6
    assert config["data_split_dict"]["valid_ratio"] == 0.2
    assert config["data_split_dict"]["test_ratio"] == 0.2
    assert config["optimization_config"]["default_target_strategy_name"] == "multi_factor_score"
    assert config["optimization_config"]["target_metric"] == "sharpe"
    assert config["optimization_config"]["top_k"] == 5
    assert config["walk_forward_config"]["window_size"] == 700
    assert config["walk_forward_config"]["step_size"] == 60
