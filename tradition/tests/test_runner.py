import pandas as pd

from tradition import runner


class FakePortfolioTrades:
    def count(self):
        return 2


class FakePortfolio:
    def __init__(self, close, entries, exits, init_cash, fees, slippage, freq):
        self.close = close
        self.entries = entries
        self.exits = exits
        self.init_cash = init_cash
        self.fees = fees
        self.slippage = slippage
        self.freq = freq
        self.trades = FakePortfolioTrades()

    def value(self):
        return pd.Series([10000.0, 10100.0, 10200.0], index=pd.date_range("2024-01-01", periods=3, freq="D"))


class FakePortfolioFactory:
    @staticmethod
    def from_signals(close, entries, exits, init_cash, fees, slippage, freq):
        return FakePortfolio(close, entries, exits, init_cash, fees, slippage, freq)


class FakeVectorbt:
    Portfolio = FakePortfolioFactory


def test_build_cli_override_collects_strategy_params():
    parser = runner.build_arg_parser()
    args = parser.parse_args([
        "--fund-code",
        "123",
        "--strategy-name",
        "ma_cross",
        "--force-refresh",
        "--ma-fast",
        "3",
        "--ma-slow",
        "9",
    ])

    override = runner.build_cli_override(args)

    assert override["default_fund_code"] == "000123"
    assert override["default_strategy_name"] == "ma_cross"
    assert override["force_refresh"] is True
    assert override["strategy_param_dict"]["ma_cross"] == {"fast": 3, "slow": 9}


def test_merge_strategy_params_overrides_single_strategy():
    merged = runner.merge_strategy_params(
        default_param_dict={"buy_and_hold": {}, "ma_cross": {"fast": 5, "slow": 20}},
        override_param_dict={"ma_cross": {"fast": 3}},
    )
    assert merged["ma_cross"] == {"fast": 3, "slow": 20}
    assert merged["buy_and_hold"] == {}


def test_run_single_fund_strategy_returns_summary(monkeypatch, sample_fund_df, tmp_path):
    monkeypatch.setattr(
        runner,
        "build_tradition_config",
        lambda config_override=None: {
            "code_dict": {"007301": "半导体"},
            "data_dir": tmp_path,
            "output_dir": tmp_path,
            "force_refresh": False,
            "cache_prefix": "tradition_fund",
            "default_fund_code": "007301",
            "default_strategy_name": "buy_and_hold",
            "strategy_param_dict": {"buy_and_hold": {}},
            "init_cash": 10000.0,
            "fees": 0.001,
        },
    )
    monkeypatch.setattr(runner, "fetch_fund_data_with_cache", lambda **kwargs: sample_fund_df)
    monkeypatch.setattr(runner, "normalize_fund_data", lambda data: data)
    monkeypatch.setattr(runner, "filter_single_fund", lambda data, fund_code: data)
    monkeypatch.setattr(
        runner,
        "adapt_to_price_series",
        lambda fund_df: (
            pd.Series([1.0, 1.1, 1.2], index=pd.date_range("2024-01-01", periods=3, freq="D"), name="price"),
            "nav_price_series",
        ),
    )
    monkeypatch.setattr(
        runner,
        "generate_signals",
        lambda price_series, strategy_name, strategy_params=None: (
            pd.Series([True, False, False], index=price_series.index),
            pd.Series([False, False, False], index=price_series.index),
            strategy_params or {},
        ),
    )
    monkeypatch.setattr(runner, "load_vectorbt_module", lambda: FakeVectorbt)
    monkeypatch.setattr(
        runner,
        "compute_return_metrics",
        lambda equity_curve: {
            "cumulative_return": 0.02,
            "annual_return": 0.5,
            "annual_volatility": 0.1,
            "sharpe": 1.2,
            "max_drawdown": -0.03,
        },
    )
    monkeypatch.setattr(runner, "save_equity_curve_plot", lambda equity_curve, output_path, title: output_path)

    result = runner.run_single_fund_strategy()

    assert result["fund_code"] == "007301"
    assert result["strategy_name"] == "buy_and_hold"
    assert result["trade_count"] == 2
    assert result["data_mode"] == "nav_price_series"
