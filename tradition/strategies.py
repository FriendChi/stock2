from copy import deepcopy

import pandas as pd

from tradition.config import DEFAULT_STRATEGY_PARAM_DICT
from tradition.factor_engine import build_multi_factor_score


def calculate_sma(series, window):
    # 优先调用第三方指标库，若环境尚未安装则退回 pandas 原生实现，保持测试和开发可用性。
    try:
        import pandas_ta_classic as ta

        sma = ta.sma(series, length=int(window))
        if sma is not None:
            return pd.Series(sma, index=series.index)
    except ImportError:
        pass
    return series.rolling(int(window)).mean()


def calculate_momentum(series, window):
    # 动量定义固定为窗口收益率，避免策略层和测试层对同一概念采用不同口径。
    return series.pct_change(int(window))


def get_strategy_params(strategy_name, strategy_params=None):
    # 策略默认参数集中在这里做合并，runner 不再关心每个策略的参数细节。
    strategy_name = str(strategy_name).lower()
    if strategy_name not in DEFAULT_STRATEGY_PARAM_DICT:
        raise ValueError(f"不支持的 strategy_name: {strategy_name}")
    merged_params = deepcopy(DEFAULT_STRATEGY_PARAM_DICT[strategy_name])
    if strategy_params is not None:
        if not isinstance(strategy_params, dict):
            raise ValueError("strategy_params 必须为dict。")
        for key, value in strategy_params.items():
            if isinstance(value, dict) and isinstance(merged_params.get(key), dict):
                nested_dict = dict(merged_params[key])
                nested_dict.update(value)
                merged_params[key] = nested_dict
                continue
            merged_params[key] = value
    return merged_params


def _build_edge_signals(entry_raw, exit_raw):
    # 用边沿检测把连续条件压成信号点，避免同一区间内重复入场和离场。
    entry_raw = entry_raw.astype(bool)
    exit_raw = exit_raw.astype(bool)
    prev_entry = entry_raw.shift(1, fill_value=False).astype(bool)
    prev_exit = exit_raw.shift(1, fill_value=False).astype(bool)
    entries = (entry_raw & (~prev_entry)).astype(bool)
    exits = (exit_raw & (~prev_exit)).astype(bool)
    return entries, exits


def generate_signals(price_series, strategy_name, strategy_params=None):
    # 统一输出 entries/exits 布尔序列，让执行层只关心信号，不再关心策略类细节。
    if len(price_series) == 0:
        raise ValueError("price_series 为空，无法生成信号。")
    price_series = pd.Series(price_series, copy=True).astype(float)
    strategy_name = str(strategy_name).lower()
    params = get_strategy_params(strategy_name=strategy_name, strategy_params=strategy_params)

    if strategy_name == "buy_and_hold":
        entries = pd.Series(False, index=price_series.index, dtype=bool)
        exits = pd.Series(False, index=price_series.index, dtype=bool)
        entries.iloc[0] = True
        return entries, exits, params

    if strategy_name == "ma_cross":
        fast = int(params["fast"])
        slow = int(params["slow"])
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError(f"均线参数非法，要求 fast < slow 且均为正整数，当前 fast={fast}, slow={slow}")
        fast_ma = calculate_sma(price_series, window=fast)
        slow_ma = calculate_sma(price_series, window=slow)
        entry_raw = (fast_ma > slow_ma).fillna(False)
        exit_raw = (fast_ma < slow_ma).fillna(False)
        entries, exits = _build_edge_signals(entry_raw=entry_raw, exit_raw=exit_raw)
        return entries, exits, params

    if strategy_name == "momentum":
        window = int(params["window"])
        if window <= 0:
            raise ValueError(f"动量窗口必须为正整数，当前 window={window}")
        momentum = calculate_momentum(price_series, window=window)
        entry_raw = (momentum > 0).fillna(False)
        exit_raw = (momentum <= 0).fillna(False)
        entries, exits = _build_edge_signals(entry_raw=entry_raw, exit_raw=exit_raw)
        return entries, exits, params

    if strategy_name == "multi_factor_score":
        factor_df, score_series = build_multi_factor_score(price_series=price_series, strategy_params=params)
        entry_threshold = float(params["entry_threshold"])
        exit_threshold = float(params["exit_threshold"])
        entry_raw = (score_series > entry_threshold).fillna(False)
        exit_raw = (score_series <= exit_threshold).fillna(False)
        entries, exits = _build_edge_signals(entry_raw=entry_raw, exit_raw=exit_raw)
        return entries, exits, params

    raise ValueError(f"不支持的 strategy_name: {strategy_name}")
