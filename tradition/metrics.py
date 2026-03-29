from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def align_rf_series_to_returns(returns_index, rf_series, annual_factor=252.0):
    # 无风险利率统一按年化收益率序列传入，并在这里对齐到策略收益频率后折算成日频收益。
    if rf_series is None:
        return None
    rf_series = pd.Series(rf_series, copy=True)
    rf_series.index = pd.to_datetime(rf_series.index, errors="coerce")
    rf_series = pd.to_numeric(rf_series, errors="coerce")
    rf_series = rf_series.dropna().sort_index()
    if rf_series.empty:
        raise ValueError("rf_series 为空，无法对齐无风险利率。")

    aligned_index = rf_series.index.union(pd.DatetimeIndex(returns_index)).sort_values()
    aligned_rf = rf_series.reindex(aligned_index).ffill().reindex(pd.DatetimeIndex(returns_index)).bfill()
    if aligned_rf.isna().any():
        raise ValueError("rf_series 对齐后仍存在缺失值。")
    return (1.0 + aligned_rf.astype(float)).pow(1.0 / annual_factor) - 1.0


def compute_return_metrics(equity_curve, rf_series=None):
    # 优先用量化统计库补充指标，Sharpe 统一在本地按对齐后的无风险利率序列计算，避免默认 rf=0。
    if len(equity_curve) == 0:
        raise ValueError("equity_curve 为空，无法计算指标。")

    equity_series = pd.Series(equity_curve, dtype=float).dropna()
    if equity_series.empty:
        raise ValueError("equity_curve 全为空，无法计算指标。")
    returns = equity_series.pct_change().dropna()
    cumulative_return = float(equity_series.iloc[-1] / equity_series.iloc[0] - 1.0)
    if len(returns) == 0:
        return {
            "cumulative_return": cumulative_return,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    annual_factor = 252.0
    annual_return = float((1.0 + cumulative_return) ** (annual_factor / len(returns)) - 1.0)
    annual_volatility = float(returns.std(ddof=0) * np.sqrt(annual_factor))
    daily_rf = align_rf_series_to_returns(returns.index, rf_series=rf_series, annual_factor=annual_factor)
    excess_returns = returns if daily_rf is None else returns - daily_rf
    sharpe = 0.0
    excess_volatility = float(excess_returns.std(ddof=0) * np.sqrt(annual_factor))
    if excess_volatility > 0:
        sharpe = float(excess_returns.mean() / excess_returns.std(ddof=0) * np.sqrt(annual_factor))
    drawdown = equity_series / equity_series.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    try:
        import quantstats as qs

        annual_return = float(qs.stats.cagr(returns))
        annual_volatility = float(qs.stats.volatility(returns))
        max_drawdown = float(qs.stats.max_drawdown(returns))
    except ImportError:
        pass

    return {
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def save_equity_curve_plot(equity_curve, output_path, title):
    # 图像输出固定为静态 PNG，便于第一阶段快速查看策略与基线表现。
    equity_series = pd.Series(equity_curve, dtype=float).dropna()
    if equity_series.empty:
        raise ValueError("equity_curve 为空，无法保存图像。")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_series.index, equity_series.values)
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("equity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
