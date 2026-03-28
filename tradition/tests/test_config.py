from tradition.config import build_tradition_config


def test_build_tradition_config_supports_override():
    config = build_tradition_config({"default_fund_code": "012832"})
    assert config["default_fund_code"] == "012832"
    assert "buy_and_hold" in config["strategy_param_dict"]
