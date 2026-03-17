import vectorbt as vbt

from strategies import strategy_dict


def run_backtest(price_df, strategy_name, strategy_params=None, init_cash=10000, fees=0.0):
    # 统一策略入口，按策略名获取信号生成函数
    if strategy_params is None:
        strategy_params = {}
    strategy_func = strategy_dict[strategy_name]
    entries, exits = strategy_func(price_df, **strategy_params)

    # 使用统一参数构建组合对象，便于多策略横向比较
    pf = vbt.Portfolio.from_signals(
        close=price_df,
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=fees,
        slippage=0.0,
        freq="1D",
    )
    return pf, entries, exits
