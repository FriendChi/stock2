import pandas as pd


def strategy_buy_and_hold(price_df, **params):
    """第一天买入，之后一直持有"""
    entries = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
    exits = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
    if len(price_df) > 0:
        entries.iloc[0, :] = True
    return entries.astype(bool), exits.astype(bool)


def strategy_ma_cross(price_df, fast=5, slow=20, t_plus_one=False, **params):
    """多基金双均线策略"""
    fast_ma = price_df.rolling(fast).mean()
    slow_ma = price_df.rolling(slow).mean()
    entry_raw = fast_ma > slow_ma
    exit_raw = fast_ma < slow_ma

    # 用shift的fill_value避免fillna在object列上的隐式降类型告警
    prev_entry = entry_raw.shift(1, fill_value=False).astype(bool)
    # exit信号同样使用fill_value保持布尔语义稳定
    prev_exit = exit_raw.shift(1, fill_value=False).astype(bool)
    entries = entry_raw & (~prev_entry)
    exits = exit_raw & (~prev_exit)

    if t_plus_one:
        # T+1时序平移后直接填False，避免后续fillna告警
        entries = entries.shift(1, fill_value=False).astype(bool)
        # exits与entries保持同一处理策略
        exits = exits.shift(1, fill_value=False).astype(bool)
    return entries.astype(bool), exits.astype(bool)


def strategy_momentum(price_df, window=20, t_plus_one=False, **params):
    """多基金简单动量策略：过去window天收益率>0则持有"""
    momentum = price_df.pct_change(window)
    entry_raw = momentum > 0
    exit_raw = momentum <= 0

    # 用shift的fill_value避免fillna在object列上的隐式降类型告警
    prev_entry = entry_raw.shift(1, fill_value=False).astype(bool)
    # exit信号同样使用fill_value保持布尔语义稳定
    prev_exit = exit_raw.shift(1, fill_value=False).astype(bool)
    entries = entry_raw & (~prev_entry)
    exits = exit_raw & (~prev_exit)

    if t_plus_one:
        # T+1时序平移后直接填False，避免后续fillna告警
        entries = entries.shift(1, fill_value=False).astype(bool)
        # exits与entries保持同一处理策略
        exits = exits.shift(1, fill_value=False).astype(bool)
    return entries.astype(bool), exits.astype(bool)


strategy_dict = {
    "buy_and_hold": strategy_buy_and_hold,
    "ma_cross": strategy_ma_cross,
    "momentum": strategy_momentum,
}
