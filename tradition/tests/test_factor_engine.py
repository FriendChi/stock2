import pandas as pd

from tradition.factor_engine import (
    FACTOR_POOL_DICT,
    build_factor_search_param_spec_dict,
    resolve_factor_param_dict,
    build_factor_table,
    build_multi_factor_score,
    calculate_factor_momentum,
    calculate_ma_trend_state,
    rolling_zscore,
)


def test_rolling_zscore_returns_zero_for_constant_series():
    series = pd.Series([1.0] * 10)
    zscore = rolling_zscore(series, window=3)
    assert (zscore == 0.0).all()


def test_calculate_factor_momentum_keeps_pct_change_semantics():
    series = pd.Series([1.0, 1.2, 1.5], dtype=float)
    momentum = calculate_factor_momentum(series, window=1)
    assert round(float(momentum.iloc[-1]), 6) == 0.25


def test_calculate_ma_trend_state_returns_series():
    series = pd.Series([1.0, 1.1, 1.2, 1.3], dtype=float)
    trend_state = calculate_ma_trend_state(series, window=2)
    assert len(trend_state) == len(series)
    assert pd.notna(trend_state.iloc[-1])


def test_build_factor_table_contains_expected_columns():
    price_series = pd.Series(
        [1.0, 1.01, 1.02, 1.04, 1.03, 1.05, 1.08, 1.09, 1.10, 1.11],
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )
    factor_df = build_factor_table(
        price_series=price_series,
        strategy_params={
            "enabled_factor_list": [
                "momentum_short",
                "momentum_mid",
                "momentum_long",
                "ma_trend_state",
                "ma_slope",
                "price_position",
                "breakout_strength",
                "volatility",
                "drawdown",
            ],
            "factor_param_dict": {
                "momentum_short": {"window": 2},
                "momentum_mid": {"window": 3},
                "momentum_long": {"window": 4},
                "ma_trend_state": {"window": 3},
                "ma_slope": {"window": 2, "lookback": 1},
                "price_position": {"window": 3},
                "breakout_strength": {"window": 3},
                "volatility": {"window": 2},
                "drawdown": {"window": 4},
            },
            "score_window": 3,
            "factor_weight_dict": {
                "momentum_short": 0.12,
                "momentum_mid": 0.12,
                "momentum_long": 0.12,
                "ma_trend_state": 0.15,
                "ma_slope": 0.12,
                "price_position": 0.10,
                "breakout_strength": 0.10,
                "volatility": 0.10,
                "drawdown": 0.07,
            },
            "entry_threshold": 0.2,
            "exit_threshold": 0.0,
        },
    )
    expected_cols = {
        "momentum_short",
        "momentum_mid",
        "momentum_long",
        "ma_trend_state",
        "ma_slope",
        "price_position",
        "breakout_strength",
        "volatility",
        "drawdown",
    }
    assert expected_cols.issubset(set(factor_df.columns))


def test_factor_pool_contains_expected_grouping():
    assert FACTOR_POOL_DICT["momentum_short"]["group"] == "趋势/动量"
    assert FACTOR_POOL_DICT["volatility"]["group"] == "波动"
    assert FACTOR_POOL_DICT["ma_trend_state"]["param_spec"]["window"]["search_space"] == (40, 120, 5)
    assert FACTOR_POOL_DICT["price_position"]["param_spec"]["window"]["search_space"] == (30, 120, 5)
    assert FACTOR_POOL_DICT["breakout_strength"]["param_spec"]["window"]["search_space"] == (30, 120, 5)
    assert FACTOR_POOL_DICT["volatility"]["param_spec"]["window"]["search_space"] == (10, 40, 5)
    assert FACTOR_POOL_DICT["drawdown"]["param_spec"]["window"]["search_space"] == (30, 120, 5)


def test_build_factor_search_param_spec_dict_collects_enabled_factor_ranges():
    search_param_spec_dict = build_factor_search_param_spec_dict(
        strategy_params={
            "enabled_factor_list": [
                "momentum_short",
                "momentum_mid",
                "momentum_long",
                "ma_trend_state",
                "price_position",
                "breakout_strength",
                "volatility",
                "drawdown",
            ],
            "factor_weight_dict": {
                "momentum_short": 0.2,
                "momentum_mid": 0.2,
                "momentum_long": 0.2,
                "ma_trend_state": 0.1,
                "price_position": 0.1,
                "breakout_strength": 0.1,
                "volatility": 0.2,
                "drawdown": 0.1,
            },
        }
    )
    assert search_param_spec_dict["momentum_short"]["window"] == (10, 30, 5)
    assert search_param_spec_dict["momentum_mid"]["window"] == (30, 60, 5)
    assert search_param_spec_dict["momentum_long"]["window"] == (60, 120, 5)
    assert search_param_spec_dict["ma_trend_state"]["window"] == (40, 120, 5)
    assert search_param_spec_dict["price_position"]["window"] == (30, 120, 5)
    assert search_param_spec_dict["breakout_strength"]["window"] == (30, 120, 5)
    assert search_param_spec_dict["volatility"]["window"] == (10, 40, 5)
    assert search_param_spec_dict["drawdown"]["window"] == (30, 120, 5)


def test_resolve_factor_param_dict_merges_defaults_and_overrides():
    factor_param_dict = resolve_factor_param_dict(
        strategy_params={
            "enabled_factor_list": ["momentum_short", "ma_slope"],
            "factor_weight_dict": {
                "momentum_short": 0.5,
                "ma_slope": 0.5,
            },
            "factor_param_dict": {
                "momentum_short": {"window": 25},
                "ma_slope": {"lookback": 2},
            },
        }
    )
    assert factor_param_dict["momentum_short"]["window"] == 25
    assert factor_param_dict["ma_slope"]["window"] == 20
    assert factor_param_dict["ma_slope"]["lookback"] == 2


def test_resolve_factor_param_dict_raises_for_unknown_factor_name():
    try:
        resolve_factor_param_dict(
            strategy_params={
                "enabled_factor_list": ["momentum_short"],
                "factor_weight_dict": {"momentum_short": 1.0},
                "factor_param_dict": {
                    "unknown_factor": {"window": 20},
                },
            }
        )
        raise AssertionError("预期应抛出 ValueError")
    except ValueError as exc:
        assert "factor_param_dict 中存在未定义因子" in str(exc)


def test_resolve_factor_param_dict_raises_for_unknown_param_name():
    try:
        resolve_factor_param_dict(
            strategy_params={
                "enabled_factor_list": ["momentum_short"],
                "factor_weight_dict": {"momentum_short": 1.0},
                "factor_param_dict": {
                    "momentum_short": {"unknown_param": 20},
                },
            }
        )
        raise AssertionError("预期应抛出 ValueError")
    except ValueError as exc:
        assert "momentum_short 存在未定义参数" in str(exc)


def test_build_factor_search_param_spec_dict_raises_for_non_searchable_param():
    try:
        build_factor_search_param_spec_dict(
            strategy_params={
                "enabled_factor_list": ["ma_slope"],
                "factor_weight_dict": {"ma_slope": 1.0},
                "search_factor_param_name_dict": {
                    "ma_slope": ["lookback"],
                },
            }
        )
        raise AssertionError("预期应抛出 ValueError")
    except ValueError as exc:
        assert "ma_slope.lookback 未定义搜索范围" in str(exc)


def test_build_factor_search_param_spec_dict_raises_for_disabled_factor():
    try:
        build_factor_search_param_spec_dict(
            strategy_params={
                "enabled_factor_list": ["momentum_short"],
                "factor_weight_dict": {"momentum_short": 1.0},
                "search_factor_param_name_dict": {
                    "volatility": ["window"],
                },
            }
        )
        raise AssertionError("预期应抛出 ValueError")
    except ValueError as exc:
        assert "search_factor_param_name_dict 中存在未启用因子" in str(exc)


def test_build_multi_factor_score_returns_score_series():
    price_series = pd.Series(
        [1.0, 1.01, 1.02, 1.04, 1.03, 1.05, 1.08, 1.09, 1.10, 1.11],
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )
    factor_df, score_series = build_multi_factor_score(
        price_series=price_series,
        strategy_params={
            "enabled_factor_list": [
                "momentum_short",
                "momentum_mid",
                "momentum_long",
                "ma_trend_state",
                "ma_slope",
                "price_position",
                "breakout_strength",
                "volatility",
                "drawdown",
            ],
            "factor_param_dict": {
                "momentum_short": {"window": 2},
                "momentum_mid": {"window": 3},
                "momentum_long": {"window": 4},
                "ma_trend_state": {"window": 3},
                "ma_slope": {"window": 2, "lookback": 1},
                "price_position": {"window": 3},
                "breakout_strength": {"window": 3},
                "volatility": {"window": 2},
                "drawdown": {"window": 4},
            },
            "score_window": 3,
            "factor_weight_dict": {
                "momentum_short": 0.12,
                "momentum_mid": 0.12,
                "momentum_long": 0.12,
                "ma_trend_state": 0.15,
                "ma_slope": 0.12,
                "price_position": 0.10,
                "breakout_strength": 0.10,
                "volatility": 0.10,
                "drawdown": 0.07,
            },
            "entry_threshold": 0.2,
            "exit_threshold": 0.0,
        },
    )
    assert len(score_series) == len(price_series)
    assert score_series.name == "multi_factor_score"
    assert len(factor_df) == len(price_series)
