import pandas as pd


def calculate_factor_momentum(series, window):
    # 因子层动量定义与策略层保持一致，但独立实现以避免模块循环依赖。
    return pd.Series(series, copy=True).astype(float).pct_change(int(window))


def calculate_volatility(series, window):
    # 波动率固定定义为窗口收益率标准差，和动量因子共享收益率口径。
    returns = pd.Series(series, copy=True).astype(float).pct_change()
    return returns.rolling(int(window)).std()


def calculate_drawdown(series, window):
    # 回撤固定相对滚动窗口内最高净值计算，值越小表示回撤越深。
    price_series = pd.Series(series, copy=True).astype(float)
    rolling_max = price_series.rolling(int(window)).max()
    return price_series / rolling_max - 1.0


def rolling_zscore(series, window):
    # 用滚动均值和标准差做时序标准化，避免直接使用全样本统计带来未来信息污染。
    series = pd.Series(series, copy=True).astype(float)
    rolling_mean = series.rolling(int(window)).mean()
    rolling_std = series.rolling(int(window)).std(ddof=0)
    zscore = (series - rolling_mean) / rolling_std.replace(0.0, float("nan"))
    return zscore.fillna(0.0).astype(float)


def build_factor_table(price_series, strategy_params):
    # 多因子版本第一阶段固定输出 4 个因子，后续扩展仍复用统一表结构。
    momentum_window_short = int(strategy_params["momentum_window_short"])
    momentum_window_long = int(strategy_params["momentum_window_long"])
    volatility_window = int(strategy_params["volatility_window"])
    drawdown_window = int(strategy_params["drawdown_window"])
    score_window = int(strategy_params["score_window"])

    factor_df = pd.DataFrame(index=price_series.index)
    factor_df["momentum_20_raw"] = calculate_factor_momentum(price_series, window=momentum_window_short)
    factor_df["momentum_60_raw"] = calculate_factor_momentum(price_series, window=momentum_window_long)
    factor_df["volatility_20_raw"] = calculate_volatility(price_series, window=volatility_window)
    factor_df["drawdown_60_raw"] = calculate_drawdown(price_series, window=drawdown_window)

    # 标准化后统一同向化：动量越大越好，波动率和深回撤越大越差。
    factor_df["momentum_20"] = rolling_zscore(factor_df["momentum_20_raw"], window=score_window)
    factor_df["momentum_60"] = rolling_zscore(factor_df["momentum_60_raw"], window=score_window)
    factor_df["volatility_20"] = -rolling_zscore(factor_df["volatility_20_raw"], window=score_window)
    factor_df["drawdown_60"] = rolling_zscore(factor_df["drawdown_60_raw"], window=score_window)
    return factor_df.fillna(0.0)


def build_multi_factor_score(price_series, strategy_params):
    # 综合分数只消费标准化后的因子列，避免权重层直接耦合原始因子量纲。
    factor_df = build_factor_table(price_series=price_series, strategy_params=strategy_params)
    factor_weight_dict = dict(strategy_params["factor_weight_dict"])
    score_series = pd.Series(0.0, index=price_series.index, dtype=float)
    for factor_name, weight in factor_weight_dict.items():
        if factor_name not in factor_df.columns:
            raise ValueError(f"未找到因子列: {factor_name}")
        score_series = score_series + factor_df[factor_name] * float(weight)
    score_series.name = "multi_factor_score"
    return factor_df, score_series
