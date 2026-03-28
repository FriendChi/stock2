import pandas as pd

from tradition.factor_engine import build_factor_table, build_multi_factor_score, rolling_zscore


def test_rolling_zscore_returns_zero_for_constant_series():
    series = pd.Series([1.0] * 10)
    zscore = rolling_zscore(series, window=3)
    assert (zscore == 0.0).all()


def test_build_factor_table_contains_expected_columns():
    price_series = pd.Series(
        [1.0, 1.01, 1.02, 1.04, 1.03, 1.05, 1.08, 1.09, 1.10, 1.11],
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )
    factor_df = build_factor_table(
        price_series=price_series,
        strategy_params={
            "momentum_window_short": 2,
            "momentum_window_long": 3,
            "volatility_window": 2,
            "drawdown_window": 3,
            "score_window": 3,
            "factor_weight_dict": {
                "momentum_20": 0.3,
                "momentum_60": 0.3,
                "volatility_20": 0.2,
                "drawdown_60": 0.2,
            },
            "entry_threshold": 0.2,
            "exit_threshold": 0.0,
        },
    )
    expected_cols = {"momentum_20", "momentum_60", "volatility_20", "drawdown_60"}
    assert expected_cols.issubset(set(factor_df.columns))


def test_build_multi_factor_score_returns_score_series():
    price_series = pd.Series(
        [1.0, 1.01, 1.02, 1.04, 1.03, 1.05, 1.08, 1.09, 1.10, 1.11],
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )
    factor_df, score_series = build_multi_factor_score(
        price_series=price_series,
        strategy_params={
            "momentum_window_short": 2,
            "momentum_window_long": 3,
            "volatility_window": 2,
            "drawdown_window": 3,
            "score_window": 3,
            "factor_weight_dict": {
                "momentum_20": 0.3,
                "momentum_60": 0.3,
                "volatility_20": 0.2,
                "drawdown_60": 0.2,
            },
            "entry_threshold": 0.2,
            "exit_threshold": 0.0,
        },
    )
    assert len(score_series) == len(price_series)
    assert score_series.name == "multi_factor_score"
    assert len(factor_df) == len(price_series)
