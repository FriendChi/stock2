import numpy as np
import pandas as pd
import pytest

from tradition.strategies import calculate_momentum, calculate_sma, get_strategy_params


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
