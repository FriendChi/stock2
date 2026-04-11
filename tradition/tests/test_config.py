from tradition.config import build_tradition_config, resolve_effective_code_dict


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


def test_resolve_effective_code_dict_adds_linked_codes_for_primary_code():
    config = build_tradition_config({"default_fund_code": "007301"})
    effective_code_dict = resolve_effective_code_dict(config)
    assert "007301" in effective_code_dict
    assert "512480" in effective_code_dict
    assert "000510" in effective_code_dict


def test_resolve_effective_code_dict_keeps_original_set_when_no_linked_codes():
    config = build_tradition_config(
        {
            "default_fund_code": "012832",
            "code_dict": {"012832": "新能源"},
            "linked_code_dict": {"007301": ["512480", "000510"]},
        }
    )
    effective_code_dict = resolve_effective_code_dict(config)
    assert effective_code_dict == {"012832": "新能源"}
