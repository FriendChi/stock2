from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_return_metrics(equity_curve):
    # 优先用量化统计库补充指标，若不可用则回退到本地公式，保证最小功能可运行。
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
    sharpe = 0.0
    if annual_volatility > 0:
        sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(annual_factor))
    drawdown = equity_series / equity_series.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    try:
        import quantstats as qs

        annual_return = float(qs.stats.cagr(returns))
        annual_volatility = float(qs.stats.volatility(returns))
        sharpe = float(qs.stats.sharpe(returns))
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
