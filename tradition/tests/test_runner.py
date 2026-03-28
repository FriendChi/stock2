import pandas as pd

from tradition import runner


class FakeBacktest:
    def __init__(self, data, strategy_cls, cash, commission, exclusive_orders):
        self.data = data
        self.strategy_cls = strategy_cls
        self.cash = cash
        self.commission = commission
        self.exclusive_orders = exclusive_orders

    def run(self):
        return {
            "_equity_curve": pd.DataFrame({"Equity": [10000.0, 10100.0, 10200.0]}),
            "# Trades": 2,
        }


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
        "adapt_to_backtesting_ohlc",
        lambda fund_df: (
            pd.DataFrame(
                {
                    "Open": [1.0, 1.1, 1.2],
                    "High": [1.0, 1.1, 1.2],
                    "Low": [1.0, 1.1, 1.2],
                    "Close": [1.0, 1.1, 1.2],
                    "Volume": [1.0, 1.0, 1.0],
                },
                index=pd.date_range("2024-01-01", periods=3, freq="D"),
            ),
            "synthetic_from_nav",
        ),
    )
    monkeypatch.setattr(runner, "build_strategy_class", lambda strategy_name, strategy_params=None: (object, strategy_params or {}))
    monkeypatch.setattr(runner, "load_backtest_class", lambda: FakeBacktest)
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
