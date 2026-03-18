import pandas as pd


class BuyAndHoldStrategy:
    @classmethod
    def generate_signals(cls, price_df, **params):
        # 第一日全量买入，后续不再产生入场与离场信号
        entries = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
        exits = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
        if len(price_df) > 0:
            entries.iloc[0, :] = True
        return entries.astype(bool), exits.astype(bool)


class MACrossStrategy:
    @classmethod
    def generate_signals(cls, price_df, fast=5, slow=20, t_plus_one=False, **params):
        # 计算快慢均线并生成原始入场与离场条件
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


class MomentumStrategy:
    @classmethod
    def generate_signals(cls, price_df, window=20, t_plus_one=False, **params):
        # 基于窗口收益率构造动量入场与离场条件
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
    "buy_and_hold": BuyAndHoldStrategy.generate_signals,
    "ma_cross": MACrossStrategy.generate_signals,
    "momentum": MomentumStrategy.generate_signals,
}
