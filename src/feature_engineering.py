import pandas as pd


def build_aligned_price_df(data, fund_list):
    # 先过滤基金池，再透视成多基金宽表并按日期排序
    filtered = data[data["code"].isin(fund_list)].copy()
    # 过滤后为空时，直接给出可读错误，避免后续nan切片异常
    if filtered.empty:
        raise ValueError(
            "过滤后无数据：请检查code类型或fund_list取值。"
            f" data中示例code={data['code'].astype(str).head(5).tolist()}"
        )
    price_df = filtered.pivot(index="date", columns="code", values="nav").sort_index()
    price_df = price_df.dropna(how="all")
    # 透视后为空也提前失败，避免进入公共区间计算
    if price_df.empty:
        raise ValueError("透视后无有效净值数据，无法构建宽表。")

    # 对齐到所有基金都有数据的公共时间区间
    start_dates = price_df.apply(lambda s: s.first_valid_index())
    end_dates = price_df.apply(lambda s: s.last_valid_index())
    common_start = start_dates.max()
    common_end = end_dates.min()
    print("各基金公共起点:", common_start)
    print("各基金公共终点:", common_end)
    # 公共区间不可用时，显式报错并给出基金列信息
    if pd.isna(common_start) or pd.isna(common_end):
        raise ValueError(
            "基金公共区间为空，无法对齐。"
            f" 当前基金列={price_df.columns.astype(str).tolist()}"
        )

    price_df_aligned = price_df.loc[common_start:common_end].copy()
    price_df_aligned = price_df_aligned.dropna(how="any")
    # 对齐后为空时直接提示，便于定位是区间过窄还是缺失过多
    if price_df_aligned.empty:
        raise ValueError("对齐并去缺失后无数据，请调整基金池或时间范围。")
    print(price_df_aligned.head())
    print(price_df_aligned.shape)
    return price_df_aligned


def build_feature_dict(price_df):
    feature_dict = {}

    # 基础收益类特征
    ret_1 = price_df.pct_change(1)
    ret_5 = price_df.pct_change(5)
    ret_10 = price_df.pct_change(10)
    ret_20 = price_df.pct_change(20)
    ret_60 = price_df.pct_change(60)
    feature_dict["ret_1"] = ret_1
    feature_dict["ret_5"] = ret_5
    feature_dict["ret_10"] = ret_10
    feature_dict["ret_20"] = ret_20
    feature_dict["ret_60"] = ret_60

    # 动量类特征
    momentum_5 = price_df / price_df.shift(5) - 1
    momentum_20 = price_df / price_df.shift(20) - 1
    momentum_60 = price_df / price_df.shift(60) - 1
    momentum_120 = price_df / price_df.shift(120) - 1
    feature_dict["momentum_5"] = momentum_5
    feature_dict["momentum_20"] = momentum_20
    feature_dict["momentum_60"] = momentum_60
    feature_dict["momentum_120"] = momentum_120

    # 波动率类特征
    daily_ret = price_df.pct_change()
    vol_5 = daily_ret.rolling(5).std()
    vol_10 = daily_ret.rolling(10).std()
    vol_20 = daily_ret.rolling(20).std()
    vol_60 = daily_ret.rolling(60).std()
    feature_dict["vol_5"] = vol_5
    feature_dict["vol_10"] = vol_10
    feature_dict["vol_20"] = vol_20
    feature_dict["vol_60"] = vol_60

    # 均线/趋势类特征
    ma_5 = price_df.rolling(5).mean()
    ma_10 = price_df.rolling(10).mean()
    ma_20 = price_df.rolling(20).mean()
    ma_60 = price_df.rolling(60).mean()
    ma_120 = price_df.rolling(120).mean()
    feature_dict["ma_ratio_5"] = price_df / ma_5 - 1
    feature_dict["ma_ratio_10"] = price_df / ma_10 - 1
    feature_dict["ma_ratio_20"] = price_df / ma_20 - 1
    feature_dict["ma_ratio_60"] = price_df / ma_60 - 1
    feature_dict["ma_ratio_120"] = price_df / ma_120 - 1
    feature_dict["ma_cross_5_20"] = ma_5 / ma_20 - 1
    feature_dict["ma_cross_10_60"] = ma_10 / ma_60 - 1
    feature_dict["ma_cross_20_120"] = ma_20 / ma_120 - 1

    # 回撤/位置类特征
    rolling_max_20 = price_df.rolling(20).max()
    rolling_max_60 = price_df.rolling(60).max()
    rolling_max_120 = price_df.rolling(120).max()
    drawdown_20 = price_df / rolling_max_20 - 1
    drawdown_60 = price_df / rolling_max_60 - 1
    drawdown_120 = price_df / rolling_max_120 - 1
    feature_dict["drawdown_20"] = drawdown_20
    feature_dict["drawdown_60"] = drawdown_60
    feature_dict["drawdown_120"] = drawdown_120

    rolling_min_20 = price_df.rolling(20).min()
    rolling_min_60 = price_df.rolling(60).min()
    feature_dict["position_20"] = (price_df - rolling_min_20) / (rolling_max_20 - rolling_min_20)
    feature_dict["position_60"] = (price_df - rolling_min_60) / (rolling_max_60 - rolling_min_60)
    return feature_dict
