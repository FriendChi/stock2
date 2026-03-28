import sys
import types

import pandas as pd

from tradition.data_fetcher import build_cache_path, fetch_fund_data_with_cache


class FakeAkshareModule:
    def __init__(self):
        self.calls = []

    def fund_open_fund_info_em(self, symbol, indicator):
        self.calls.append((symbol, indicator))
        if indicator == "单位净值走势":
            return pd.DataFrame(
                {
                    "净值日期": ["2024-01-01", "2024-01-02"],
                    "单位净值": [1.0, 1.1],
                    "日增长率": [0.0, 10.0],
                }
            )
        if indicator == "累计净值走势":
            return pd.DataFrame(
                {
                    "净值日期": ["2024-01-01", "2024-01-02"],
                    "累计净值": [1.0, 1.2],
                }
            )
        raise AssertionError("unexpected indicator")


def test_build_cache_path_uses_prefix_and_date(tmp_path):
    cache_path = build_cache_path(tmp_path, cache_prefix="tradition_fund", trade_date="2024-01-01")
    assert cache_path.name == "tradition_fund_2024-01-01.csv"


def test_fetch_fund_data_with_cache_writes_cache(tmp_path, monkeypatch):
    fake_module = FakeAkshareModule()
    monkeypatch.setitem(sys.modules, "akshare", fake_module)

    data = fetch_fund_data_with_cache(
        code_dict={"007301": "半导体"},
        cache_dir=tmp_path,
        force_refresh=True,
        cache_prefix="tradition_fund",
        trade_date="2024-01-01",
    )

    assert set(["date", "code", "fund", "nav", "cumulative_nav", "daily_growth_rate"]).issubset(set(data.columns))
    assert len(fake_module.calls) == 2
    assert (tmp_path / "tradition_fund_2024-01-01.csv").exists()


def test_fetch_fund_data_with_cache_hits_existing_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "tradition_fund_2024-01-01.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "code": ["007301"],
            "fund": ["半导体"],
            "nav": [1.0],
        }
    ).to_csv(cache_path, index=False)

    fake_module = types.SimpleNamespace(fund_open_fund_info_em=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not call akshare")))
    monkeypatch.setitem(sys.modules, "akshare", fake_module)

    data = fetch_fund_data_with_cache(
        code_dict={"007301": "半导体"},
        cache_dir=tmp_path,
        force_refresh=False,
        cache_prefix="tradition_fund",
        trade_date="2024-01-01",
    )

    assert data.iloc[0]["code"] == "007301"
