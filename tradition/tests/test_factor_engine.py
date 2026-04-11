import pandas as pd

from tradition.factor_engine import (
    FACTOR_POOL_DICT,
    build_factor_search_param_spec_dict,
    build_factor_table,
    build_multi_factor_score,
    build_single_factor_series,
    calculate_factor_momentum,
    calculate_ma_trend_state,
    calculate_trend_r2,
    calculate_trend_residual,
    calculate_trend_tvalue,
    resolve_factor_name_list_by_group,
    resolve_factor_param_dict,
    rolling_zscore,
)


def build_sample_strategy_params():
    return {
        "enabled_factor_list": [
            "momentum",
            "ma_trend_state",
            "ma_slope",
            "trend_r2",
            "trend_tvalue",
            "price_position",
            "breakout_strength",
            "donchian_breakout",
            "trend_residual",
            "volatility",
            "drawdown",
            "risk_adjusted_momentum",
            "sharpe_like_trend",
        ],
        "factor_param_dict": {
            "momentum": {"window": 2},
            "ma_trend_state": {"window": 3},
            "ma_slope": {"window": 3, "lookback": 1},
            "trend_r2": {"window": 4},
            "trend_tvalue": {"window": 4},
            "price_position": {"window": 3},
            "breakout_strength": {"window": 3},
            "donchian_breakout": {"window": 3},
            "trend_residual": {"window": 4},
            "volatility": {"window": 2},
            "drawdown": {"window": 4},
            "risk_adjusted_momentum": {"window": 3},
            "sharpe_like_trend": {"window": 3},
        },
        "score_window": 3,
        "factor_weight_dict": {
            "momentum": 0.10,
            "ma_trend_state": 0.08,
            "ma_slope": 0.08,
            "trend_r2": 0.08,
            "trend_tvalue": 0.08,
            "price_position": 0.08,
            "breakout_strength": 0.08,
            "donchian_breakout": 0.07,
            "trend_residual": 0.07,
            "volatility": 0.08,
            "drawdown": 0.08,
            "risk_adjusted_momentum": 0.10,
            "sharpe_like_trend": 0.10,
        },
        "entry_threshold": 0.2,
        "exit_threshold": 0.0,
    }


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


def test_calculate_trend_r2_matches_manual_window_result():
    series = pd.Series(
        [1.0001, 1.0001, 1.0001, 1.0002, 1.0002, 1.0002, 1.0002, 1.0005, 1.0004, 1.0005,
         1.0003, 1.0032, 0.9927, 0.9901, 1.0168, 1.0207, 1.0346, 1.0212, 1.0263, 1.0061],
        dtype=float,
    )
    trend_r2 = calculate_trend_r2(series, window=20)
    assert abs(float(trend_r2.iloc[-1]) - 0.36950525132966916) < 1e-12


def test_calculate_trend_tvalue_matches_manual_window_result():
    series = pd.Series(
        [1.0001, 1.0001, 1.0001, 1.0002, 1.0002, 1.0002, 1.0002, 1.0005, 1.0004, 1.0005,
         1.0003, 1.0032, 0.9927, 0.9901, 1.0168, 1.0207, 1.0346, 1.0212, 1.0263, 1.0061],
        dtype=float,
    )
    trend_tvalue = calculate_trend_tvalue(series, window=20)
    assert abs(float(trend_tvalue.iloc[-1]) - 3.2479237362902875) < 1e-12


def test_calculate_trend_residual_does_not_collapse_to_constant_zero():
    series = pd.Series([1.0, 1.01, 1.03, 1.02, 1.05, 1.08, 1.04, 1.09], dtype=float)
    trend_residual = calculate_trend_residual(series, window=4)
    non_na = trend_residual.dropna()
    assert len(non_na) > 0
    assert non_na.nunique() > 1
    assert not (non_na == 0.0).all()


def test_build_factor_table_contains_expected_columns():
    price_series = pd.Series(
        [1.0, 1.01, 1.02, 1.04, 1.03, 1.05, 1.08, 1.09, 1.10, 1.11, 1.13, 1.15],
        index=pd.date_range("2024-01-01", periods=12, freq="D"),
    )
    factor_df = build_factor_table(
        price_series=price_series,
        strategy_params=build_sample_strategy_params(),
    )
    expected_cols = set(build_sample_strategy_params()["enabled_factor_list"])
    assert expected_cols.issubset(set(factor_df.columns))


def test_factor_pool_contains_expected_grouping():
    assert FACTOR_POOL_DICT["momentum"]["group"] == "趋势强度"
    assert FACTOR_POOL_DICT["ma_trend_state"]["group"] == "均线趋势"
    assert FACTOR_POOL_DICT["trend_r2"]["group"] == "趋势强度"
    assert FACTOR_POOL_DICT["donchian_breakout"]["group"] == "突破"
    assert FACTOR_POOL_DICT["trend_residual"]["group"] == "价格结构"
    assert FACTOR_POOL_DICT["volatility"]["group"] == "波动调整趋势"


def test_resolve_factor_name_list_by_group_collects_and_deduplicates():
    factor_name_list = resolve_factor_name_list_by_group(["趋势强度", "突破", "趋势强度"])
    assert factor_name_list[0] == "momentum"
    assert "donchian_breakout" in factor_name_list
    assert len(factor_name_list) == len(set(factor_name_list))


def test_build_single_factor_series_returns_named_series():
    price_series = pd.Series(
        [1.0, 1.01, 1.02, 1.04, 1.03, 1.05, 1.08, 1.09, 1.10, 1.11],
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )
    factor_series = build_single_factor_series(
        price_series=price_series,
        factor_name="momentum",
        strategy_params=build_sample_strategy_params(),
    )
    assert factor_series.name == "momentum"
    assert factor_series.index.equals(price_series.index)


def test_build_factor_search_param_spec_dict_collects_enabled_factor_ranges():
    search_param_spec_dict = build_factor_search_param_spec_dict(
        strategy_params=build_sample_strategy_params()
    )
    assert search_param_spec_dict["momentum"]["window"] == (10, 120, 5)
    assert search_param_spec_dict["ma_slope"]["window"] == (10, 60, 5)
    assert search_param_spec_dict["ma_slope"]["lookback"] == (2, 20, 1)
    assert search_param_spec_dict["trend_r2"]["window"] == (10, 60, 5)
    assert search_param_spec_dict["donchian_breakout"]["window"] == (10, 60, 5)
    assert search_param_spec_dict["risk_adjusted_momentum"]["window"] == (10, 60, 5)


def test_resolve_factor_param_dict_merges_defaults_and_overrides():
    factor_param_dict = resolve_factor_param_dict(
        strategy_params={
            "enabled_factor_list": ["momentum", "ma_slope"],
            "factor_weight_dict": {
                "momentum": 0.5,
                "ma_slope": 0.5,
            },
            "factor_param_dict": {
                "momentum": {"window": 25},
                "ma_slope": {"lookback": 2},
            },
        }
    )
    assert factor_param_dict["momentum"]["window"] == 25
    assert factor_param_dict["ma_slope"]["window"] == 20
    assert factor_param_dict["ma_slope"]["lookback"] == 2


def test_resolve_factor_param_dict_raises_for_unknown_factor_name():
    try:
        resolve_factor_param_dict(
            strategy_params={
                "enabled_factor_list": ["momentum"],
                "factor_weight_dict": {"momentum": 1.0},
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
                "enabled_factor_list": ["momentum"],
                "factor_weight_dict": {"momentum": 1.0},
                "factor_param_dict": {
                    "momentum": {"unknown_param": 20},
                },
            }
        )
        raise AssertionError("预期应抛出 ValueError")
    except ValueError as exc:
        assert "momentum 存在未定义参数" in str(exc)


def test_build_factor_search_param_spec_dict_collects_all_searchable_params():
    search_param_spec_dict = build_factor_search_param_spec_dict(
        strategy_params={
            "enabled_factor_list": ["momentum", "ma_slope"],
            "factor_weight_dict": {"momentum": 1.0, "ma_slope": 1.0},
        }
    )
    assert search_param_spec_dict["momentum"] == {"window": (10, 120, 5)}
    assert search_param_spec_dict["ma_slope"] == {
        "window": (10, 60, 5),
        "lookback": (2, 20, 1),
    }


def test_build_multi_factor_score_returns_score_series():
    price_series = pd.Series(
        [1.0, 1.01, 1.02, 1.04, 1.03, 1.05, 1.08, 1.09, 1.10, 1.11, 1.13, 1.15],
        index=pd.date_range("2024-01-01", periods=12, freq="D"),
    )
    factor_df, score_series = build_multi_factor_score(
        price_series=price_series,
        strategy_params=build_sample_strategy_params(),
    )
    assert len(score_series) == len(price_series)
    assert score_series.name == "multi_factor_score"
    assert len(factor_df) == len(price_series)
