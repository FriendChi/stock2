import pandas as pd


def load_pandas_ta_module():
    # 技术指标优先复用 pandas-ta-classic，若环境缺失则回退到本地实现。
    try:
        import pandas_ta_classic as ta

        return ta
    except ImportError:
        return None


def calculate_factor_momentum(series, window):
    # 动量因子固定使用窗口收益率定义，保持与策略层 momentum 语义一致。
    price_series = pd.Series(series, copy=True).astype(float)
    ta = load_pandas_ta_module()
    if ta is not None:
        momentum = ta.percent_return(price_series, length=int(window))
        if momentum is not None:
            return pd.Series(momentum, index=price_series.index, dtype=float)
    return price_series.pct_change(int(window))


def calculate_volatility(series, window):
    # 波动率继续使用收益率口径，但优先复用 pandas-ta-classic 的 stdev 实现。
    returns = pd.Series(series, copy=True).astype(float).pct_change()
    ta = load_pandas_ta_module()
    if ta is not None:
        volatility = ta.stdev(returns, length=int(window))
        if volatility is not None:
            return pd.Series(volatility, index=returns.index, dtype=float)
    return returns.rolling(int(window)).std()


def calculate_drawdown(series, window):
    # 回撤固定相对滚动窗口内最高净值计算，值越小表示回撤越深。
    price_series = pd.Series(series, copy=True).astype(float)
    rolling_max = price_series.rolling(int(window)).max()
    return price_series / rolling_max - 1.0


def calculate_ma_trend_state(series, window):
    # 均线趋势状态优先复用 pandas-ta-classic 的均线实现，避免和策略层口径漂移。
    price_series = pd.Series(series, copy=True).astype(float)
    ta = load_pandas_ta_module()
    if ta is not None:
        moving_average = ta.sma(price_series, length=int(window))
        if moving_average is not None:
            moving_average = pd.Series(moving_average, index=price_series.index, dtype=float)
        else:
            moving_average = price_series.rolling(int(window)).mean()
    else:
        moving_average = price_series.rolling(int(window)).mean()
    return price_series / moving_average - 1.0


def calculate_ma_slope(series, ma_window, lookback_window):
    # 均线斜率沿用均线变化率定义，但底层均线优先复用技术指标库。
    price_series = pd.Series(series, copy=True).astype(float)
    ta = load_pandas_ta_module()
    if ta is not None:
        moving_average = ta.sma(price_series, length=int(ma_window))
        if moving_average is not None:
            moving_average = pd.Series(moving_average, index=price_series.index, dtype=float)
        else:
            moving_average = price_series.rolling(int(ma_window)).mean()
    else:
        moving_average = price_series.rolling(int(ma_window)).mean()
    return moving_average.pct_change(int(lookback_window))


def calculate_price_position(series, window):
    # 区间位置因子衡量价格位于近期区间的相对高低，越接近 1 表示越靠近阶段高点。
    price_series = pd.Series(series, copy=True).astype(float)
    rolling_min = price_series.rolling(int(window)).min()
    rolling_max = price_series.rolling(int(window)).max()
    position = (price_series - rolling_min) / (rolling_max - rolling_min)
    return position.replace([float("inf"), -float("inf")], float("nan"))


def calculate_breakout_strength(series, window):
    # 突破强度因子用价格相对前期高点偏离表示，值大于 0 说明发生了向上突破。
    price_series = pd.Series(series, copy=True).astype(float)
    previous_high = price_series.rolling(int(window)).max().shift(1)
    return price_series / previous_high - 1.0


def calculate_donchian_breakout(series, window):
    # 唐奇安突破因子使用是否突破前期高点的二值信号，突出趋势确认而非偏离幅度。
    price_series = pd.Series(series, copy=True).astype(float)
    previous_high = price_series.rolling(int(window)).max().shift(1)
    return (price_series > previous_high).astype(float)


def calculate_risk_adjusted_momentum(series, window):
    # 风险调整动量统一以窗口收益除以同窗口收益波动，避免把高波动上涨误当成同等质量趋势。
    momentum_series = calculate_factor_momentum(series, window=window)
    volatility_series = calculate_volatility(series, window=window)
    return momentum_series / volatility_series.replace(0.0, float("nan"))


def calculate_sharpe_like_trend(series, window):
    # Sharpe-like 趋势因子使用滚动均值收益除以滚动波动，统一提供风险收益比视角。
    returns = pd.Series(series, copy=True).astype(float).pct_change()
    rolling_mean = returns.rolling(int(window)).mean()
    rolling_std = returns.rolling(int(window)).std(ddof=0)
    return rolling_mean / rolling_std.replace(0.0, float("nan"))


def calculate_trend_r2(series, window):
    # 趋势拟合度只基于单价序列滚动线性回归的 R2，刻画价格路径的线性趋势稳定性。
    price_series = pd.Series(series, copy=True).astype(float)
    x_series = pd.Series(range(int(window)), dtype=float)

    def compute_r2(window_values):
        # 滚动窗口回归统一按位置计算，避免日期索引和位置索引对齐后把乘积算成全 NaN。
        window_series = pd.Series(window_values, dtype=float).reset_index(drop=True)
        if window_series.isna().any():
            return float("nan")
        if len(window_series) < 2:
            return float("nan")
        y_mean = float(window_series.mean())
        centered_x = x_series - float(x_series.mean())
        centered_y = window_series - y_mean
        denominator = float((centered_x ** 2).sum())
        if denominator <= 0.0:
            return float("nan")
        slope = float((centered_x * centered_y).sum() / denominator)
        intercept = y_mean - slope * float(x_series.mean())
        fitted = intercept + slope * x_series
        residual = window_series - fitted
        ss_total = float(((window_series - y_mean) ** 2).sum())
        if ss_total <= 0.0:
            return 0.0
        ss_residual = float((residual ** 2).sum())
        return float(max(0.0, 1.0 - ss_residual / ss_total))

    return price_series.rolling(int(window)).apply(compute_r2, raw=False)


def calculate_trend_tvalue(series, window):
    # 趋势 t 值使用滚动回归斜率除以标准误，突出趋势方向的统计显著性。
    price_series = pd.Series(series, copy=True).astype(float)
    x_series = pd.Series(range(int(window)), dtype=float)

    def compute_tvalue(window_values):
        # 滚动窗口回归统一按位置计算，避免日期索引和位置索引对齐后把乘积算成全 NaN。
        window_series = pd.Series(window_values, dtype=float).reset_index(drop=True)
        if window_series.isna().any():
            return float("nan")
        if len(window_series) < 3:
            return float("nan")
        centered_x = x_series - float(x_series.mean())
        centered_y = window_series - float(window_series.mean())
        sxx = float((centered_x ** 2).sum())
        if sxx <= 0.0:
            return float("nan")
        slope = float((centered_x * centered_y).sum() / sxx)
        intercept = float(window_series.mean()) - slope * float(x_series.mean())
        fitted = intercept + slope * x_series
        residual = window_series - fitted
        degrees_of_freedom = len(window_series) - 2
        if degrees_of_freedom <= 0:
            return float("nan")
        residual_variance = float((residual ** 2).sum() / degrees_of_freedom)
        slope_std = float((residual_variance / sxx) ** 0.5)
        if slope_std <= 0.0:
            return 0.0
        return float(slope / slope_std)

    return price_series.rolling(int(window)).apply(compute_tvalue, raw=False)


def calculate_trend_residual(series, window):
    # 价格结构简版因子使用滚动趋势线残差衡量价格相对主趋势的偏离程度。
    price_series = pd.Series(series, copy=True).astype(float)
    x_series = pd.Series(range(int(window)), dtype=float)

    def compute_residual(window_values):
        # 滚动窗口回归统一按位置计算，避免日期索引和位置索引对齐后把乘积算成全 NaN。
        window_series = pd.Series(window_values, dtype=float).reset_index(drop=True)
        if window_series.isna().any():
            return float("nan")
        if len(window_series) < 2:
            return float("nan")
        centered_x = x_series - float(x_series.mean())
        centered_y = window_series - float(window_series.mean())
        sxx = float((centered_x ** 2).sum())
        if sxx <= 0.0:
            return float("nan")
        slope = float((centered_x * centered_y).sum() / sxx)
        intercept = float(window_series.mean()) - slope * float(x_series.mean())
        fitted_last = intercept + slope * float(x_series.iloc[-1])
        return float(window_series.iloc[-1] / fitted_last - 1.0) if abs(fitted_last) > 1e-12 else float("nan")

    return price_series.rolling(int(window)).apply(compute_residual, raw=False)


def build_factor_pool_dict():
    # 轻量因子库集中定义分类、参数和搜索范围，策略层只决定启用哪些因子。
    return {
        "momentum": {
            "group": "趋势强度",
            "param_spec": {
                "window": {"default": 40, "search_space": (10, 120, 5)},
            },
        },
        "ma_trend_state": {
            "group": "均线趋势",
            "param_spec": {
                "window": {"default": 60, "search_space": (20, 120, 5)},
            },
        },
        "ma_slope": {
            "group": "均线趋势",
            "param_spec": {
                "window": {"default": 20, "search_space": (10, 60, 5)},
                "lookback": {"default": 5, "search_space": (2, 20, 1)},
            },
        },
        "trend_r2": {
            "group": "趋势强度",
            "param_spec": {
                "window": {"default": 20, "search_space": (10, 60, 5)},
            },
        },
        "trend_tvalue": {
            "group": "趋势强度",
            "param_spec": {
                "window": {"default": 20, "search_space": (10, 60, 5)},
            },
        },
        "price_position": {
            "group": "突破",
            "param_spec": {
                "window": {"default": 60, "search_space": (20, 120, 5)},
            },
        },
        "breakout_strength": {
            "group": "突破",
            "param_spec": {
                "window": {"default": 60, "search_space": (20, 120, 5)},
            },
        },
        "donchian_breakout": {
            "group": "突破",
            "param_spec": {
                "window": {"default": 20, "search_space": (10, 60, 5)},
            },
        },
        "trend_residual": {
            "group": "价格结构",
            "param_spec": {
                "window": {"default": 20, "search_space": (10, 60, 5)},
            },
        },
        "volatility": {
            "group": "波动调整趋势",
            "param_spec": {
                "window": {"default": 20, "search_space": (10, 40, 5)},
            },
        },
        "drawdown": {
            "group": "波动调整趋势",
            "param_spec": {
                "window": {"default": 60, "search_space": (20, 120, 5)},
            },
        },
        "risk_adjusted_momentum": {
            "group": "波动调整趋势",
            "param_spec": {
                "window": {"default": 20, "search_space": (10, 60, 5)},
            },
        },
        "sharpe_like_trend": {
            "group": "波动调整趋势",
            "param_spec": {
                "window": {"default": 20, "search_space": (10, 60, 5)},
            },
        },
    }


FACTOR_POOL_DICT = build_factor_pool_dict()


def resolve_factor_name_list_by_group(factor_group_list):
    # 因子筛选入口允许按因子族传入，并从因子库中展开去重，避免 runner 自己感知因子库细节。
    if not isinstance(factor_group_list, list) or len(factor_group_list) == 0:
        raise ValueError("factor_group_list 不能为空。")

    normalized_group_list = [str(group_name).strip() for group_name in factor_group_list if str(group_name).strip()]
    if len(normalized_group_list) == 0:
        raise ValueError("factor_group_list 不能为空。")

    resolved_factor_name_list = []
    unknown_group_list = []
    for group_name in normalized_group_list:
        matched_factor_name_list = [
            factor_name
            for factor_name, factor_meta in FACTOR_POOL_DICT.items()
            if str(factor_meta.get("group", "")).strip() == group_name
        ]
        if len(matched_factor_name_list) == 0:
            unknown_group_list.append(group_name)
            continue
        resolved_factor_name_list.extend(matched_factor_name_list)
    if len(unknown_group_list) > 0:
        raise ValueError(f"存在未定义因子族: {unknown_group_list}")

    deduplicated_factor_name_list = list(dict.fromkeys(resolved_factor_name_list))
    if len(deduplicated_factor_name_list) == 0:
        raise ValueError("按因子族展开后无可用因子。")
    return deduplicated_factor_name_list


def build_raw_factor_series(price_series, factor_name, factor_param_dict):
    # 原始因子统一在这里派发，避免因子表构建阶段堆叠大量 if/else。
    factor_name = str(factor_name)
    factor_params = dict(factor_param_dict[factor_name])
    if factor_name == "momentum":
        return calculate_factor_momentum(price_series, window=int(factor_params["window"]))
    if factor_name == "ma_trend_state":
        return calculate_ma_trend_state(price_series, window=int(factor_params["window"]))
    if factor_name == "ma_slope":
        return calculate_ma_slope(
            price_series,
            ma_window=int(factor_params["window"]),
            lookback_window=int(factor_params["lookback"]),
        )
    if factor_name == "trend_r2":
        return calculate_trend_r2(price_series, window=int(factor_params["window"]))
    if factor_name == "trend_tvalue":
        return calculate_trend_tvalue(price_series, window=int(factor_params["window"]))
    if factor_name == "price_position":
        return calculate_price_position(price_series, window=int(factor_params["window"]))
    if factor_name == "breakout_strength":
        return calculate_breakout_strength(price_series, window=int(factor_params["window"]))
    if factor_name == "donchian_breakout":
        return calculate_donchian_breakout(price_series, window=int(factor_params["window"]))
    if factor_name == "trend_residual":
        return calculate_trend_residual(price_series, window=int(factor_params["window"]))
    if factor_name == "volatility":
        return calculate_volatility(price_series, window=int(factor_params["window"]))
    if factor_name == "drawdown":
        return calculate_drawdown(price_series, window=int(factor_params["window"]))
    if factor_name == "risk_adjusted_momentum":
        return calculate_risk_adjusted_momentum(price_series, window=int(factor_params["window"]))
    if factor_name == "sharpe_like_trend":
        return calculate_sharpe_like_trend(price_series, window=int(factor_params["window"]))
    raise ValueError(f"当前未定义因子: {factor_name}")
