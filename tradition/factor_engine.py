import pandas as pd


def load_pandas_ta_module():
    # 技术指标优先复用 pandas-ta-classic，若环境缺失则回退到本地实现。
    try:
        import pandas_ta_classic as ta

        return ta
    except ImportError:
        return None


def calculate_factor_momentum(series, window):
    # 因子层动量优先复用技术指标库的 percent_return，保持窗口收益率语义不变。
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


def rolling_zscore(series, window):
    # 用滚动均值和标准差做时序标准化，避免直接使用全样本统计带来未来信息污染。
    series = pd.Series(series, copy=True).astype(float)
    rolling_mean = series.rolling(int(window)).mean()
    rolling_std = series.rolling(int(window)).std(ddof=0)
    zscore = (series - rolling_mean) / rolling_std.replace(0.0, float("nan"))
    return zscore.fillna(0.0).astype(float)


def build_factor_pool_dict():
    # 轻量因子池集中定义分类、参数和搜索范围，策略层只决定启用哪些因子。
    return {
        "momentum_short": {
            "group": "趋势/动量",
            "param_spec": {
                "window": {"default": 20, "search_space": (10, 30, 5)},
            },
        },
        "momentum_mid": {
            "group": "趋势/动量",
            "param_spec": {
                "window": {"default": 40, "search_space": (30, 60, 5)},
            },
        },
        "momentum_long": {
            "group": "趋势/动量",
            "param_spec": {
                "window": {"default": 60, "search_space": (60, 120, 5)},
            },
        },
        "ma_trend_state": {
            "group": "趋势/动量",
            "param_spec": {
                "window": {"default": 60, "search_space": (40, 120, 5)},
            },
        },
        "ma_slope": {
            "group": "趋势/动量",
            "param_spec": {
                "window": {"default": 20, "search_space": None},
                "lookback": {"default": 5, "search_space": None},
            },
        },
        "price_position": {
            "group": "趋势/动量",
            "param_spec": {
                "window": {"default": 60, "search_space": (30, 120, 5)},
            },
        },
        "breakout_strength": {
            "group": "趋势/动量",
            "param_spec": {
                "window": {"default": 60, "search_space": (30, 120, 5)},
            },
        },
        "volatility": {
            "group": "波动",
            "param_spec": {
                "window": {"default": 20, "search_space": (10, 40, 5)},
            },
        },
        "drawdown": {
            "group": "波动",
            "param_spec": {
                "window": {"default": 60, "search_space": (30, 120, 5)},
            },
        },
    }


FACTOR_POOL_DICT = build_factor_pool_dict()


def validate_multi_factor_config(strategy_params, enabled_factor_list=None, require_factor_weight_dict=False):
    # 因子配置校验集中在因子层，尽早拦截未知因子、未知参数和不可搜索参数。
    factor_weight_dict = dict(strategy_params.get("factor_weight_dict", {}))
    if enabled_factor_list is None:
        configured_enabled_factor_list = strategy_params.get("enabled_factor_list")
        if configured_enabled_factor_list is None:
            configured_enabled_factor_list = list(factor_weight_dict.keys())
        enabled_factor_list = [str(factor_name) for factor_name in configured_enabled_factor_list]

    unknown_weight_factor_list = [factor_name for factor_name in factor_weight_dict if factor_name not in FACTOR_POOL_DICT]
    if len(unknown_weight_factor_list) > 0:
        raise ValueError(f"factor_weight_dict 中存在未定义因子: {unknown_weight_factor_list}")

    configured_factor_param_dict = dict(strategy_params.get("factor_param_dict", {}))
    unknown_param_factor_list = [factor_name for factor_name in configured_factor_param_dict if factor_name not in FACTOR_POOL_DICT]
    if len(unknown_param_factor_list) > 0:
        raise ValueError(f"factor_param_dict 中存在未定义因子: {unknown_param_factor_list}")
    for factor_name, param_dict in configured_factor_param_dict.items():
        factor_param_spec = dict(FACTOR_POOL_DICT[str(factor_name)]["param_spec"])
        unknown_param_name_list = [param_name for param_name in dict(param_dict) if str(param_name) not in factor_param_spec]
        if len(unknown_param_name_list) > 0:
            raise ValueError(f"{factor_name} 存在未定义参数: {unknown_param_name_list}")

    if require_factor_weight_dict:
        missing_weight_factor_list = [factor_name for factor_name in enabled_factor_list if factor_name not in factor_weight_dict]
        if len(missing_weight_factor_list) > 0:
            raise ValueError(f"enabled_factor_list 中存在缺失权重的因子: {missing_weight_factor_list}")

    search_factor_param_name_dict = {
        str(factor_name): [str(param_name) for param_name in param_name_list]
        for factor_name, param_name_list in dict(strategy_params.get("search_factor_param_name_dict", {})).items()
    }
    for factor_name, param_name_list in search_factor_param_name_dict.items():
        if factor_name not in enabled_factor_list:
            raise ValueError(f"search_factor_param_name_dict 中存在未启用因子: {factor_name}")
        factor_param_spec = dict(FACTOR_POOL_DICT[factor_name]["param_spec"])
        for param_name in param_name_list:
            if param_name not in factor_param_spec:
                raise ValueError(f"{factor_name} 存在未定义搜索参数: {param_name}")
            if factor_param_spec[param_name].get("search_space") is None:
                raise ValueError(f"{factor_name}.{param_name} 未定义搜索范围。")


def resolve_factor_param_dict(strategy_params):
    # 因子参数统一按因子名做一层字典隔离，避免外层扁平参数名不断膨胀。
    enabled_factor_list = resolve_enabled_factor_list(strategy_params=strategy_params)
    validate_multi_factor_config(
        strategy_params=strategy_params,
        enabled_factor_list=enabled_factor_list,
    )
    configured_factor_param_dict = dict(strategy_params.get("factor_param_dict", {}))
    resolved_factor_param_dict = {}
    for factor_name in enabled_factor_list:
        resolved_param_dict = {
            str(param_name): spec["default"]
            for param_name, spec in dict(FACTOR_POOL_DICT[factor_name]["param_spec"]).items()
        }
        override_param_dict = dict(configured_factor_param_dict.get(factor_name, {}))
        for param_name, value in override_param_dict.items():
            resolved_param_dict[str(param_name)] = value
        resolved_factor_param_dict[factor_name] = resolved_param_dict
    return resolved_factor_param_dict


def build_raw_factor_series(price_series, factor_name, factor_param_dict):
    # 原始因子统一在这里派发，避免因子表构建阶段堆叠大量 if/else。
    factor_params = dict(factor_param_dict[factor_name])
    if factor_name == "momentum_short":
        return calculate_factor_momentum(price_series, window=int(factor_params["window"]))
    if factor_name == "momentum_mid":
        return calculate_factor_momentum(price_series, window=int(factor_params["window"]))
    if factor_name == "momentum_long":
        return calculate_factor_momentum(price_series, window=int(factor_params["window"]))
    if factor_name == "ma_trend_state":
        return calculate_ma_trend_state(price_series, window=int(factor_params["window"]))
    if factor_name == "ma_slope":
        return calculate_ma_slope(
            price_series,
            ma_window=int(factor_params["window"]),
            lookback_window=int(factor_params["lookback"]),
        )
    if factor_name == "price_position":
        return calculate_price_position(price_series, window=int(factor_params["window"]))
    if factor_name == "breakout_strength":
        return calculate_breakout_strength(price_series, window=int(factor_params["window"]))
    if factor_name == "volatility":
        return calculate_volatility(price_series, window=int(factor_params["window"]))
    if factor_name == "drawdown":
        return calculate_drawdown(price_series, window=int(factor_params["window"]))
    raise ValueError(f"当前未定义因子: {factor_name}")


def normalize_factor_series(raw_factor_series, factor_name, score_window):
    # 标准化和同向化在统一入口处理，保证新增因子复用同一套评分口径。
    normalized_factor = rolling_zscore(raw_factor_series, window=int(score_window))
    if factor_name == "volatility":
        return -normalized_factor
    return normalized_factor


def resolve_enabled_factor_list(strategy_params):
    # 因子启用列表优先由策略配置决定，未显式配置时退回到权重表中的因子名。
    enabled_factor_list = strategy_params.get("enabled_factor_list")
    if enabled_factor_list is None:
        enabled_factor_list = list(dict(strategy_params["factor_weight_dict"]).keys())
    resolved_factor_list = [str(factor_name) for factor_name in enabled_factor_list]
    if len(resolved_factor_list) == 0:
        raise ValueError("enabled_factor_list 为空，无法构建多因子评分。")
    unknown_factor_list = [factor_name for factor_name in resolved_factor_list if factor_name not in FACTOR_POOL_DICT]
    if len(unknown_factor_list) > 0:
        raise ValueError(f"存在未定义因子: {unknown_factor_list}")
    return resolved_factor_list


def build_factor_search_param_spec_dict(strategy_params):
    # 因子相关搜索范围统一从因子池收集，避免优化层和因子池重复维护参数边界。
    enabled_factor_list = resolve_enabled_factor_list(strategy_params=strategy_params)
    validate_multi_factor_config(
        strategy_params=strategy_params,
        enabled_factor_list=enabled_factor_list,
    )
    search_param_spec_dict = {}
    for factor_name in enabled_factor_list:
        param_spec = dict(FACTOR_POOL_DICT[factor_name]["param_spec"])
        search_param_spec_dict[factor_name] = {}
        for param_name, spec in param_spec.items():
            search_space = spec.get("search_space")
            if search_space is None:
                continue
            search_param_spec_dict[factor_name][param_name] = search_space
    return search_param_spec_dict


def build_factor_table(price_series, strategy_params):
    # 因子表按启用因子列表构建，后续新增因子只需扩充因子池而不改主流程。
    score_window = int(strategy_params["score_window"])
    enabled_factor_list = resolve_enabled_factor_list(strategy_params=strategy_params)
    factor_param_dict = resolve_factor_param_dict(strategy_params=strategy_params)

    factor_df = pd.DataFrame(index=price_series.index)
    for factor_name in enabled_factor_list:
        raw_col = f"{factor_name}_raw"
        raw_factor_series = build_raw_factor_series(
            price_series=price_series,
            factor_name=factor_name,
            factor_param_dict=factor_param_dict,
        )
        factor_df[raw_col] = raw_factor_series
        factor_df[factor_name] = normalize_factor_series(
            raw_factor_series=raw_factor_series,
            factor_name=factor_name,
            score_window=score_window,
        )
    return factor_df.fillna(0.0)


def build_multi_factor_score(price_series, strategy_params):
    # 综合分数只消费启用因子的标准化列，权重缺失时直接报错，避免策略配置静默失效。
    validate_multi_factor_config(
        strategy_params=strategy_params,
        require_factor_weight_dict=True,
    )
    factor_df = build_factor_table(price_series=price_series, strategy_params=strategy_params)
    factor_weight_dict = dict(strategy_params["factor_weight_dict"])
    enabled_factor_list = resolve_enabled_factor_list(strategy_params=strategy_params)
    score_series = pd.Series(0.0, index=price_series.index, dtype=float)
    for factor_name in enabled_factor_list:
        if factor_name not in factor_weight_dict:
            raise ValueError(f"未找到因子权重: {factor_name}")
        score_series = score_series + factor_df[factor_name] * float(factor_weight_dict[factor_name])
    score_series.name = "multi_factor_score"
    return factor_df, score_series
