import pandas as pd
import pytest


@pytest.fixture
def sample_fund_df():
    # 用小规模净值样本覆盖常见时序择时路径，避免单元测试依赖真实缓存和网络。
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "code": ["007301"] * 8,
            "fund": ["半导体"] * 8,
            "nav": [1.0, 1.02, 1.01, 1.05, 1.08, 1.04, 1.10, 1.12],
            "cumulative_nav": [1.0, 1.02, 1.03, 1.07, 1.10, 1.06, 1.12, 1.15],
            "daily_growth_rate": [0.0, 2.0, -0.98, 3.96, 2.86, -3.70, 5.77, 1.82],
        }
    )
