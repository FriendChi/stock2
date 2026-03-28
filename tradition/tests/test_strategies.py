import numpy as np
import pandas as pd
import pytest

from tradition.strategies import calculate_momentum, calculate_sma, generate_signals, get_strategy_params


def test_calculate_sma_matches_rolling_mean():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = calculate_sma(series, window=2)
    assert np.isnan(result.iloc[0])
    assert result.iloc[-1] == 3.5


def test_calculate_momentum_uses_pct_change():
    series = pd.Series([1.0, 1.1, 1.21])
    result = calculate_momentum(series, window=1)
    assert pytest.approx(result.iloc[-1], rel=1e-6) == 0.1


def test_get_strategy_params_merges_override():
    params = get_strategy_params("ma_cross", {"fast": 3, "slow": 10})
    assert params == {"fast": 3, "slow": 10}


def test_get_strategy_params_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        get_strategy_params("unknown")


def test_generate_signals_buy_and_hold_enters_once():
    price_series = pd.Series([1.0, 1.1, 1.2], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    entries, exits, params = generate_signals(price_series, "buy_and_hold")
    assert params == {}
    assert entries.sum() == 1
    assert bool(entries.iloc[0]) is True
    assert exits.sum() == 0


def test_generate_signals_ma_cross_returns_boolean_series():
    price_series = pd.Series([1, 2, 3, 4, 3, 2, 3, 4, 5, 4], index=pd.date_range("2024-01-01", periods=10, freq="D"), dtype=float)
    entries, exits, _ = generate_signals(price_series, "ma_cross", {"fast": 2, "slow": 3})
    assert entries.dtype == bool
    assert exits.dtype == bool
    assert entries.sum() >= 1


def test_generate_signals_momentum_returns_boolean_series():
    price_series = pd.Series([1.0, 0.9, 0.8, 1.0, 1.2, 1.3], index=pd.date_range("2024-01-01", periods=6, freq="D"))
    entries, exits, _ = generate_signals(price_series, "momentum", {"window": 2})
    assert entries.dtype == bool
    assert exits.dtype == bool
    assert entries.sum() >= 1
