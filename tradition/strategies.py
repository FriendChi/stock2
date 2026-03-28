from copy import deepcopy

import numpy as np
import pandas as pd

from tradition.config import DEFAULT_STRATEGY_PARAM_DICT


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
        merged_params.update(strategy_params)
    return merged_params


def _sma_array(values, window):
    series = pd.Series(values, dtype=float)
    return calculate_sma(series=series, window=window).to_numpy(dtype=float)


def _momentum_array(values, window):
    series = pd.Series(values, dtype=float)
    return calculate_momentum(series=series, window=window).to_numpy(dtype=float)


def build_strategy_class(strategy_name, strategy_params=None):
    # 动态生成回测策略类，既维持单一注册入口，也避免模块导入时强依赖第三方回测库。
    strategy_name = str(strategy_name).lower()
    params = get_strategy_params(strategy_name=strategy_name, strategy_params=strategy_params)

    from backtesting import Strategy
    from backtesting.lib import crossover

    if strategy_name == "buy_and_hold":

        class BuyAndHoldStrategy(Strategy):
            def init(self):
                # buy_and_hold 无额外指标缓存，但仍需实现抽象接口以满足回测库约束。
                return None

            def next(self):
                # 仅在首次进入 next 时建仓，后续保持持有直到回测结束。
                if not self.position:
                    self.buy()

        return BuyAndHoldStrategy, params

    if strategy_name == "ma_cross":
        fast = int(params["fast"])
        slow = int(params["slow"])
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError(f"均线参数非法，要求 fast < slow 且均为正整数，当前 fast={fast}, slow={slow}")

        class MACrossStrategy(Strategy):
            def init(self):
                # 在 init 中缓存均线指标，减少 next 中重复计算。
                self.fast_ma = self.I(_sma_array, self.data.Close, self.fast)
                self.slow_ma = self.I(_sma_array, self.data.Close, self.slow)

            def next(self):
                # 金叉开仓、死叉平仓，不叠加杠杆和复杂仓位逻辑。
                if crossover(self.fast_ma, self.slow_ma):
                    self.buy()
                elif crossover(self.slow_ma, self.fast_ma) and self.position:
                    self.position.close()

        MACrossStrategy.fast = fast
        MACrossStrategy.slow = slow
        return MACrossStrategy, params

    if strategy_name == "momentum":
        window = int(params["window"])
        if window <= 0:
            raise ValueError(f"动量窗口必须为正整数，当前 window={window}")

        class MomentumStrategy(Strategy):
            def init(self):
                # 固定使用窗口收益率作为动量指标，正值持有，非正值空仓。
                self.momentum = self.I(_momentum_array, self.data.Close, self.window)

            def next(self):
                current_momentum = self.momentum[-1]
                if np.isnan(current_momentum):
                    return
                if current_momentum > 0 and not self.position:
                    self.buy()
                elif current_momentum <= 0 and self.position:
                    self.position.close()

        MomentumStrategy.window = window
        return MomentumStrategy, params

    raise ValueError(f"不支持的 strategy_name: {strategy_name}")
