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


def test_build_cli_override_collects_batch_codes():
    parser = runner.build_arg_parser()
    args = parser.parse_args(["--batch-run", "--fund-codes", "7301,12832"])
    override = runner.build_cli_override(args)
    assert override["batch_run"] is True
    assert override["fund_code_list"] == ["007301", "012832"]


def test_build_cli_override_collects_compare_all_flag():
    parser = runner.build_arg_parser()
    args = parser.parse_args(["--compare-all", "--fund-code", "7301"])
    override = runner.build_cli_override(args)
    assert override["compare_all"] is True
    assert override["default_fund_code"] == "007301"


def test_build_cli_override_collects_optimize_flag_and_trials():
    parser = runner.build_arg_parser()
    args = parser.parse_args(["--optimize", "--n-trials", "12", "--fund-code", "7301"])
    override = runner.build_cli_override(args)
    assert override["optimize"] is True
    assert override["optimization_config"]["n_trials"] == 12
    assert override["default_fund_code"] == "007301"


def test_merge_strategy_params_overrides_single_strategy():
    merged = runner.merge_strategy_params(
        default_param_dict={"buy_and_hold": {}, "ma_cross": {"fast": 5, "slow": 20}},
        override_param_dict={"ma_cross": {"fast": 3}},
    )
    assert merged["ma_cross"] == {"fast": 3, "slow": 20}
    assert merged["buy_and_hold"] == {}


def test_merge_optimization_config_overrides_trials():
    merged = runner.merge_optimization_config(
        default_optimization_config={"n_trials": 30, "target_metric": "sharpe"},
        override_optimization_config={"n_trials": 12},
    )
    assert merged["n_trials"] == 12
    assert merged["target_metric"] == "sharpe"


def test_build_summary_table_sorts_by_sharpe():
    summary_df = runner.build_summary_table(
        [
            {
                "fund_code": "A",
                "strategy_name": "buy_and_hold",
                "sample_start": pd.Timestamp("2024-01-01"),
                "sample_end": pd.Timestamp("2024-01-02"),
                "data_mode": "nav_price_series",
                "trade_count": 1,
                "output_path": "a.png",
                "stats": {
                    "cumulative_return": 0.1,
                    "annual_return": 0.1,
                    "annual_volatility": 0.2,
                    "sharpe": 0.5,
                    "max_drawdown": -0.1,
                },
            },
            {
                "fund_code": "B",
                "strategy_name": "buy_and_hold",
                "sample_start": pd.Timestamp("2024-01-01"),
                "sample_end": pd.Timestamp("2024-01-02"),
                "data_mode": "nav_price_series",
                "trade_count": 1,
                "output_path": "b.png",
                "stats": {
                    "cumulative_return": 0.2,
                    "annual_return": 0.2,
                    "annual_volatility": 0.2,
                    "sharpe": 1.0,
                    "max_drawdown": -0.2,
                },
            },
        ]
    )
    assert summary_df.iloc[0]["fund_code"] == "B"



def test_build_param_signature_supports_nested_dict():
    signature = runner.build_param_signature({
        "entry_threshold": 0.2,
        "factor_weight_dict": {
            "momentum_20": 0.3,
            "drawdown_60": 0.2,
        },
    })
    assert isinstance(signature, tuple)
    assert signature[1][0] == "factor_weight_dict"


def test_run_single_price_series_strategy_returns_summary(monkeypatch, tmp_path):
    config = {
        "output_dir": tmp_path,
        "init_cash": 10000.0,
        "fees": 0.001,
    }
    price_series = pd.Series([1.0, 1.1, 1.2], index=pd.date_range("2024-01-01", periods=3, freq="D"))
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

    result = runner.run_single_price_series_strategy(
        price_series=price_series,
        config=config,
        fund_code="007301",
        strategy_name="buy_and_hold",
        strategy_params={},
        print_summary=False,
    )

    assert result["fund_code"] == "007301"
    assert result["strategy_name"] == "buy_and_hold"
    assert result["trade_count"] == 2
    assert result["data_mode"] == "nav_price_series"


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
            "optimization_config": {"default_target_strategy_name": "multi_factor_score"},
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
        "run_single_price_series_strategy",
        lambda price_series, config, fund_code, strategy_name, strategy_params=None, print_summary=True, save_plot=True, data_mode="nav_price_series", output_name=None: {
            "fund_code": fund_code,
            "strategy_name": strategy_name,
            "strategy_params": strategy_params or {},
            "stats": {
                "cumulative_return": 0.02,
                "annual_return": 0.5,
                "annual_volatility": 0.1,
                "sharpe": 1.2,
                "max_drawdown": -0.03,
            },
            "equity_curve": pd.Series([1.0, 1.1]),
            "output_path": tmp_path / "out.png",
            "data_mode": data_mode,
            "sample_start": pd.Timestamp("2024-01-01"),
            "sample_end": pd.Timestamp("2024-01-02"),
            "trade_count": 2,
        },
    )

    result = runner.run_single_fund_strategy()

    assert result["fund_code"] == "007301"
    assert result["strategy_name"] == "buy_and_hold"
    assert result["trade_count"] == 2
    assert result["data_mode"] == "nav_price_series"


def test_run_multi_fund_strategy_returns_summary(monkeypatch, tmp_path):
    fake_config = {
        "code_dict": {"007301": "半导体", "012832": "新能源"},
        "data_dir": tmp_path,
        "output_dir": tmp_path,
        "force_refresh": False,
        "cache_prefix": "tradition_fund",
        "default_strategy_name": "momentum",
        "strategy_param_dict": {"momentum": {"window": 20}},
        "optimization_config": {"default_target_strategy_name": "multi_factor_score"},
        "init_cash": 10000.0,
        "fees": 0.001,
    }
    monkeypatch.setattr(runner, "build_tradition_config", lambda config_override=None: dict(fake_config, **(config_override or {})))
    monkeypatch.setattr(runner, "fetch_fund_data_with_cache", lambda **kwargs: pd.DataFrame({"date": [], "code": [], "fund": [], "nav": []}))
    monkeypatch.setattr(runner, "normalize_fund_data", lambda data: data)
    monkeypatch.setattr(
        runner,
        "run_single_fund_strategy_from_data",
        lambda normalized_data, config, fund_code, print_summary=False: {
            "fund_code": fund_code,
            "strategy_name": config["default_strategy_name"],
            "strategy_params": config["strategy_param_dict"][config["default_strategy_name"]],
            "stats": {
                "cumulative_return": 0.1 if fund_code == "007301" else 0.2,
                "annual_return": 0.1 if fund_code == "007301" else 0.2,
                "annual_volatility": 0.2,
                "sharpe": 0.5 if fund_code == "007301" else 1.0,
                "max_drawdown": -0.1,
            },
            "equity_curve": pd.Series([1.0, 1.1]),
            "output_path": tmp_path / "{}.png".format(fund_code),
            "data_mode": "nav_price_series",
            "sample_start": pd.Timestamp("2024-01-01"),
            "sample_end": pd.Timestamp("2024-01-02"),
            "trade_count": 2,
        },
    )

    result = runner.run_multi_fund_strategy()

    assert list(result["summary_df"]["fund_code"]) == ["012832", "007301"]
    assert result["summary_path"].exists()


def test_run_compare_all_strategies_returns_summary(monkeypatch, tmp_path):
    fake_config = {
        "code_dict": {"007301": "半导体"},
        "data_dir": tmp_path,
        "output_dir": tmp_path,
        "force_refresh": False,
        "cache_prefix": "tradition_fund",
        "default_fund_code": "007301",
        "default_strategy_name": "buy_and_hold",
        "strategy_param_dict": {
            "buy_and_hold": {},
            "ma_cross": {"fast": 5, "slow": 20},
            "momentum": {"window": 20},
            "multi_factor_score": {
                "entry_threshold": 0.2,
                "exit_threshold": 0.0,
                "factor_weight_dict": {},
                "momentum_window_short": 20,
                "momentum_window_long": 60,
                "volatility_window": 20,
                "drawdown_window": 60,
                "score_window": 60,
            },
        },
        "optimization_config": {"default_target_strategy_name": "multi_factor_score"},
        "init_cash": 10000.0,
        "fees": 0.001,
    }
    monkeypatch.setattr(runner, "build_tradition_config", lambda config_override=None: dict(fake_config, **(config_override or {})))
    monkeypatch.setattr(runner, "fetch_fund_data_with_cache", lambda **kwargs: pd.DataFrame({"date": [], "code": [], "fund": [], "nav": []}))
    monkeypatch.setattr(runner, "normalize_fund_data", lambda data: data)

    def fake_run_single_fund_strategy_from_data(normalized_data, config, fund_code, print_summary=False):
        strategy_name = config["default_strategy_name"]
        return {
            "fund_code": fund_code,
            "strategy_name": strategy_name,
            "strategy_params": config["strategy_param_dict"][strategy_name],
            "stats": {
                "cumulative_return": 0.1,
                "annual_return": 0.1 if strategy_name != "momentum" else 0.2,
                "annual_volatility": 0.2,
                "sharpe": 0.5 if strategy_name != "momentum" else 1.0,
                "max_drawdown": -0.1,
            },
            "equity_curve": pd.Series([1.0, 1.1]),
            "output_path": tmp_path / "{}.png".format(strategy_name),
            "data_mode": "nav_price_series",
            "sample_start": pd.Timestamp("2024-01-01"),
            "sample_end": pd.Timestamp("2024-01-02"),
            "trade_count": 2,
        }

    monkeypatch.setattr(runner, "run_single_fund_strategy_from_data", fake_run_single_fund_strategy_from_data)

    result = runner.run_compare_all_strategies()

    assert result["fund_code"] == "007301"
    assert list(result["summary_df"]["strategy_name"])[0] == "momentum"
    assert result["summary_path"].exists()


def test_run_optimize_single_fund_strategy_returns_summary(monkeypatch, sample_fund_df, tmp_path):
    fake_config = {
        "code_dict": {"007301": "半导体"},
        "data_dir": tmp_path,
        "output_dir": tmp_path,
        "force_refresh": False,
        "cache_prefix": "tradition_fund",
        "default_fund_code": "007301",
        "default_strategy_name": "buy_and_hold",
        "strategy_param_dict": {
            "buy_and_hold": {},
            "multi_factor_score": {
                "momentum_window_short": 20,
                "momentum_window_long": 60,
                "volatility_window": 20,
                "drawdown_window": 60,
                "score_window": 60,
                "factor_weight_dict": {
                    "momentum_20": 0.30,
                    "momentum_60": 0.30,
                    "volatility_20": 0.20,
                    "drawdown_60": 0.20,
                },
                "entry_threshold": 0.2,
                "exit_threshold": 0.0,
            },
        },
        "data_split_dict": {"train_ratio": 0.6, "valid_ratio": 0.2, "test_ratio": 0.2, "min_segment_size": 2},
        "optimization_config": {
            "default_target_strategy_name": "multi_factor_score",
            "n_trials": 3,
            "study_direction": "maximize",
            "study_name_prefix": "tradition_optuna",
            "target_metric": "sharpe",
            "penalty_weight": 0.2,
        },
        "init_cash": 10000.0,
        "fees": 0.001,
    }
    monkeypatch.setattr(runner, "build_tradition_config", lambda config_override=None: dict(fake_config, **(config_override or {})))
    monkeypatch.setattr(runner, "fetch_fund_data_with_cache", lambda **kwargs: sample_fund_df)
    monkeypatch.setattr(runner, "normalize_fund_data", lambda data: data)
    monkeypatch.setattr(runner, "filter_single_fund", lambda data, fund_code: data)
    monkeypatch.setattr(
        runner,
        "adapt_to_price_series",
        lambda fund_df: (
            pd.Series(range(12), index=pd.date_range("2024-01-01", periods=12, freq="D"), dtype=float),
            "nav_price_series",
        ),
    )
    monkeypatch.setattr(
        runner,
        "split_time_series_by_ratio",
        lambda price_series, split_config: {
            "train": price_series.iloc[:6],
            "valid": price_series.iloc[6:9],
            "test": price_series.iloc[9:],
        },
    )
    monkeypatch.setattr(
        runner,
        "optimize_strategy_params",
        lambda strategy_name, base_params, evaluate_params_fn, optimization_config: {
            "best_params": {
                "momentum_window_short": 30,
                "momentum_window_long": 90,
                "entry_threshold": 0.3,
                "exit_threshold": -0.1,
            },
            "best_value": 0.95,
            "top_k_params_list": [
                {
                    "trial_number": 0,
                    "train_objective": 0.95,
                    "params": {
                        "momentum_window_short": 30,
                        "momentum_window_long": 90,
                        "entry_threshold": 0.3,
                        "exit_threshold": -0.1,
                    },
                    "source": "top_k",
                }
            ],
            "improving_best_params_list": [
                {
                    "trial_number": 1,
                    "train_objective": 0.90,
                    "params": {
                        "momentum_window_short": 25,
                        "momentum_window_long": 80,
                        "entry_threshold": 0.2,
                        "exit_threshold": 0.0,
                    },
                    "source": "improving_best",
                }
            ],
            "trial_df": pd.DataFrame([{"value": 0.95}]),
        },
    )
    monkeypatch.setattr(
        runner,
        "run_single_price_series_strategy",
        lambda price_series, config, fund_code, strategy_name, strategy_params=None, print_summary=False, save_plot=False, data_mode="nav_price_series", output_name=None: {
            "fund_code": fund_code,
            "strategy_name": strategy_name,
            "strategy_params": strategy_params or {},
            "stats": {
                "cumulative_return": 0.1,
                "annual_return": 0.2,
                "annual_volatility": 0.1,
                "sharpe": 0.8,
                "max_drawdown": -0.2,
            },
            "equity_curve": pd.Series([1.0, 1.1]),
            "output_path": tmp_path / (output_name or "test.png"),
            "data_mode": data_mode,
            "sample_start": price_series.index.min(),
            "sample_end": price_series.index.max(),
            "trade_count": 2,
        },
    )

    result = runner.run_optimize_single_fund_strategy()

    assert result["fund_code"] == "007301"
    assert result["strategy_name"] == "multi_factor_score"
    assert result["best_params"]["momentum_window_short"] == 30
    assert len(result["candidate_result_list"]) == 2
    assert result["best_candidate_source"] in {"top_k", "improving_best", "improving_best,top_k", "top_k,improving_best"}
    assert result["trial_path"].exists()
