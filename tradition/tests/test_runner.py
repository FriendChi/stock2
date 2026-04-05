import pandas as pd
import pytest

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


def test_build_cli_override_collects_walk_forward_config():
    parser = runner.build_arg_parser()
    args = parser.parse_args(["--walk-forward", "--wf-window-size", "600", "--wf-step-size", "50"])
    override = runner.build_cli_override(args)
    assert override["walk_forward"] is True
    assert override["walk_forward_config"] == {"window_size": 600, "step_size": 50}


def test_build_cli_override_collects_factor_selection_args():
    parser = runner.build_arg_parser()
    args = parser.parse_args(
        [
            "--factor-select",
            "--fund-code",
            "7301",
            "--factor-groups",
            "趋势/动量,波动",
            "--train-min-spearman-ic",
            "0.01",
            "--train-min-spearman-icir",
            "0.2",
        ]
    )
    override = runner.build_cli_override(args)
    assert override["factor_select"] is True
    assert override["default_fund_code"] == "007301"
    assert override["factor_group_list"] == ["趋势/动量", "波动"]
    assert override["train_min_spearman_ic"] == 0.01
    assert override["train_min_spearman_icir"] == 0.2


def test_build_cli_override_collects_single_factor_stability_analysis_args():
    parser = runner.build_arg_parser()
    args = parser.parse_args(
        [
            "--single-factor-stability-analysis",
            "--factor-selection-path",
            "/tmp/factor_selection_007301_2026-04-05.json",
        ]
    )
    override = runner.build_cli_override(args)
    assert override["single_factor_stability_analysis"] is True
    assert override["factor_selection_path"] == "/tmp/factor_selection_007301_2026-04-05.json"


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


def test_merge_walk_forward_config_overrides_sizes():
    merged = runner.merge_walk_forward_config(
        default_walk_forward_config={"window_size": 700, "step_size": 60},
        override_walk_forward_config={"step_size": 30},
    )
    assert merged["window_size"] == 700
    assert merged["step_size"] == 30


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
            "momentum_short": 0.3,
            "drawdown": 0.2,
        },
    })
    assert isinstance(signature, tuple)
    assert signature[1][0] == "factor_weight_dict"



def test_extract_segment_equity_curves_keeps_previous_day():
    equity_curve = pd.Series([100.0, 101.0, 103.0, 102.0], index=pd.date_range("2024-01-01", periods=4, freq="D"))
    metric_curve, display_curve = runner.extract_segment_equity_curves(
        equity_curve=equity_curve,
        segment_index=equity_curve.index[2:],
    )
    assert list(display_curve.index) == list(equity_curve.index[2:])
    assert list(metric_curve.index) == list(equity_curve.index[1:])


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
        lambda equity_curve, rf_series=None: {
            "cumulative_return": 0.02,
            "annual_return": 0.5,
            "annual_volatility": 0.1,
            "sharpe": 1.2,
            "max_drawdown": -0.03,
        },
    )
    captured_plot_kwargs = {}

    def fake_save_equity_curve_plot(equity_curve, output_path, title, benchmark_curve=None):
        captured_plot_kwargs["equity_curve"] = equity_curve
        captured_plot_kwargs["benchmark_curve"] = benchmark_curve
        return output_path

    monkeypatch.setattr(runner, "save_equity_curve_plot", fake_save_equity_curve_plot)

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
    assert captured_plot_kwargs["benchmark_curve"].equals(price_series)


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
                "enabled_factor_list": [
                    "momentum_short",
                    "momentum_mid",
                    "momentum_long",
                    "ma_trend_state",
                    "ma_slope",
                    "price_position",
                    "breakout_strength",
                    "volatility",
                    "drawdown",
                ],
                "search_strategy_param_name_list": ["entry_threshold", "exit_threshold"],
                "entry_threshold": 0.2,
                "exit_threshold": 0.0,
                "factor_weight_dict": {
                    "momentum_short": 0.12,
                    "momentum_mid": 0.12,
                    "momentum_long": 0.12,
                    "ma_trend_state": 0.15,
                    "ma_slope": 0.12,
                    "price_position": 0.10,
                    "breakout_strength": 0.10,
                    "volatility": 0.10,
                    "drawdown": 0.07,
                },
                "factor_param_dict": {
                    "momentum_short": {"window": 20},
                    "momentum_mid": {"window": 40},
                    "momentum_long": {"window": 60},
                    "ma_trend_state": {"window": 60},
                    "ma_slope": {"window": 20, "lookback": 5},
                    "price_position": {"window": 60},
                    "breakout_strength": {"window": 60},
                    "volatility": {"window": 20},
                    "drawdown": {"window": 60},
                },
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
                "enabled_factor_list": [
                    "momentum_short",
                    "momentum_mid",
                    "momentum_long",
                    "ma_trend_state",
                    "ma_slope",
                    "price_position",
                    "breakout_strength",
                    "volatility",
                    "drawdown",
                ],
                "search_strategy_param_name_list": ["entry_threshold", "exit_threshold"],
                "factor_param_dict": {
                    "momentum_short": {"window": 20},
                    "momentum_mid": {"window": 40},
                    "momentum_long": {"window": 60},
                    "ma_trend_state": {"window": 60},
                    "ma_slope": {"window": 20, "lookback": 5},
                    "price_position": {"window": 60},
                    "breakout_strength": {"window": 60},
                    "volatility": {"window": 20},
                    "drawdown": {"window": 60},
                },
                "score_window": 60,
                "factor_weight_dict": {
                    "momentum_short": 0.12,
                    "momentum_mid": 0.12,
                    "momentum_long": 0.12,
                    "ma_trend_state": 0.15,
                    "ma_slope": 0.12,
                    "price_position": 0.10,
                    "breakout_strength": 0.10,
                    "volatility": 0.20,
                    "drawdown": 0.07,
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
                "factor_param_dict": {
                    "momentum_short": {"window": 30},
                    "momentum_mid": {"window": 55},
                    "momentum_long": {"window": 90},
                },
                "entry_threshold": 0.3,
                "exit_threshold": -0.1,
            },
            "best_value": 0.95,
            "top_k_params_list": [
                {
                    "trial_number": 0,
                    "train_objective": 0.95,
                    "params": {
                        "factor_param_dict": {
                            "momentum_short": {"window": 30},
                            "momentum_mid": {"window": 55},
                            "momentum_long": {"window": 90},
                        },
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
                        "factor_param_dict": {
                            "momentum_short": {"window": 25},
                            "momentum_mid": {"window": 50},
                            "momentum_long": {"window": 80},
                        },
                        "entry_threshold": 0.2,
                        "exit_threshold": 0.0,
                    },
                    "source": "improving_best",
                }
            ],
            "trial_df": pd.DataFrame([{"value": 0.95}]),
        },
    )
    execution_call_lengths = []
    metric_segment_starts = []

    def fake_execute_price_series_strategy(price_series, config, strategy_name, strategy_params=None):
        execution_call_lengths.append(len(price_series))
        return {
            "price_series": price_series,
            "entries": pd.Series(False, index=price_series.index),
            "exits": pd.Series(False, index=price_series.index),
            "resolved_params": strategy_params or {},
            "portfolio": FakePortfolio(price_series, None, None, config["init_cash"], config["fees"], 0.0, "1D"),
            "equity_curve": pd.Series(range(100, 100 + len(price_series)), index=price_series.index, dtype=float),
        }

    def fake_build_result_from_execution(
        execution_result,
        fund_code,
        strategy_name,
        data_mode="nav_price_series",
        save_plot=False,
        output_dir=None,
        output_name=None,
        title=None,
        print_summary=False,
        segment_index=None,
        rf_series=None,
    ):
        if segment_index is None:
            segment_index = execution_result["price_series"].index
        segment_index = pd.Index(segment_index)
        metric_segment_starts.append(segment_index.min())
        short_window = (
            (execution_result["resolved_params"] or {})
            .get("factor_param_dict", {})
            .get("momentum_short", {})
            .get("window", 0)
        )
        sharpe = 0.8
        if segment_index.min() == pd.Timestamp("2024-01-07"):
            sharpe = 1.0 if short_window == 30 else 0.6
        elif segment_index.min() == pd.Timestamp("2024-01-10"):
            sharpe = 0.7
        return {
            "fund_code": fund_code,
            "strategy_name": strategy_name,
            "strategy_params": execution_result["resolved_params"],
            "stats": {
                "cumulative_return": 0.1,
                "annual_return": 0.2,
                "annual_volatility": 0.1,
                "sharpe": sharpe,
                "max_drawdown": -0.2,
            },
            "equity_curve": execution_result["equity_curve"].loc[segment_index],
            "output_path": tmp_path / (output_name or "test.png"),
            "data_mode": data_mode,
            "sample_start": segment_index.min(),
            "sample_end": segment_index.max(),
            "trade_count": 0,
        }

    monkeypatch.setattr(runner, "execute_price_series_strategy", fake_execute_price_series_strategy)
    monkeypatch.setattr(runner, "build_result_from_execution", fake_build_result_from_execution)

    result = runner.run_optimize_single_fund_strategy()

    assert result["fund_code"] == "007301"
    assert result["strategy_name"] == "multi_factor_score"
    assert result["best_params"]["factor_param_dict"]["momentum_short"]["window"] == 30
    assert len(result["candidate_result_list"]) == 2
    assert result["best_candidate_source"] in {"top_k", "improving_best", "improving_best,top_k", "top_k,improving_best"}
    assert result["trial_path"].exists()
    assert execution_call_lengths == [12, 12]
    assert metric_segment_starts == [pd.Timestamp("2024-01-07"), pd.Timestamp("2024-01-07"), pd.Timestamp("2024-01-10")]


def test_run_walk_forward_single_fund_strategy_collects_fold_results(monkeypatch, tmp_path):
    sample_index = pd.date_range("2024-01-01", periods=1000, freq="D")
    sample_df = pd.DataFrame({
        "date": sample_index,
        "code": ["007301"] * len(sample_index),
        "fund": ["半导体"] * len(sample_index),
        "nav": range(1000, 2000),
    })
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
            "strategy_param_dict": {"multi_factor_score": {}},
            "optimization_config": {
                "default_target_strategy_name": "multi_factor_score",
                "n_trials": 5,
                "study_direction": "maximize",
                "study_name_prefix": "tradition_optuna",
                "target_metric": "sharpe",
                "penalty_weight": 0.2,
                "top_k": 5,
            },
            "walk_forward_config": {
                "window_size": 700,
                "step_size": 60,
                "min_fold_count": 1,
            },
            "data_split_dict": {
                "train_ratio": 0.6,
                "valid_ratio": 0.2,
                "test_ratio": 0.2,
                "min_segment_size": 60,
            },
            "rf_config": {"enabled": False},
            "init_cash": 10000.0,
            "fees": 0.001,
            "walk_forward": True,
        },
    )
    monkeypatch.setattr(runner, "fetch_fund_data_with_cache", lambda **kwargs: sample_df)
    monkeypatch.setattr(runner, "normalize_fund_data", lambda data: data)
    monkeypatch.setattr(runner, "filter_single_fund", lambda data, fund_code: data)
    monkeypatch.setattr(runner, "adapt_to_price_series", lambda fund_df: (pd.Series(fund_df["nav"].values, index=pd.to_datetime(fund_df["date"]), dtype=float), "nav_price_series"))
    monkeypatch.setattr(runner, "load_rf_series_for_price_series", lambda config, price_series: None)
    monkeypatch.setattr(
        runner,
        "execute_optimization_fold",
        lambda price_series, config, fund_code, strategy_name, data_mode, rf_series, split_dict: {
            "best_params": {"entry_threshold": 0.2},
            "best_value": 0.8,
            "top_k_params_list": [],
            "improving_best_params_list": [],
            "candidate_result_list": [],
            "best_candidate_source": "top_k",
            "valid_result": {"stats": {"cumulative_return": 0.05, "sharpe": 0.6}},
            "best_execution_result": {
                "price_series": price_series,
                "entries": pd.Series(False, index=price_series.index),
                "exits": pd.Series(False, index=price_series.index),
                "resolved_params": {"entry_threshold": 0.2},
                "portfolio": FakePortfolio(price_series, None, None, config["init_cash"], config["fees"], 0.0, "1D"),
                "equity_curve": pd.Series(range(100, 100 + len(price_series)), index=price_series.index, dtype=float),
            },
            "trial_df": pd.DataFrame([{"value": 0.8}]),
        },
    )
    monkeypatch.setattr(
        runner,
        "build_result_from_execution",
        lambda execution_result, fund_code, strategy_name, data_mode="nav_price_series", save_plot=False, output_dir=None, output_name=None, title=None, print_summary=False, segment_index=None, rf_series=None: {
            "fund_code": fund_code,
            "strategy_name": strategy_name,
            "strategy_params": execution_result["resolved_params"],
            "stats": {
                "cumulative_return": 0.1,
                "annual_return": 0.2,
                "annual_volatility": 0.1,
                "sharpe": 0.7,
                "max_drawdown": -0.15,
            },
            "equity_curve": execution_result["equity_curve"].loc[segment_index],
            "output_path": None,
            "data_mode": data_mode,
            "sample_start": pd.Index(segment_index).min(),
            "sample_end": pd.Index(segment_index).max(),
            "trade_count": 0,
        },
    )

    result = runner.run_walk_forward_single_fund_strategy()

    assert result["fund_code"] == "007301"
    assert len(result["fold_result_list"]) == 6
    assert result["summary_df"].shape[0] == 6
    assert "test_cumulative_return" in result["summary_df"].columns
    assert "test_mean_daily_return" in result["summary_df"].columns
    assert "walk_forward_efficiency" in result["summary_dict"]
    assert abs(result["summary_dict"]["test_sharpe_mean"] - 0.7) < 1e-12
    assert result["summary_dict"]["positive_test_return_ratio"] == 1.0
    assert result["summary_path"].exists()


def test_main_dispatches_factor_selection_mode(monkeypatch):
    called = {}
    monkeypatch.setattr(runner, "run_factor_selection_single_fund", lambda config_override=None: called.setdefault("factor_select", config_override))
    runner.main(["--factor-select", "--factor-groups", "趋势/动量"])
    assert "factor_select" in called


def test_main_dispatches_single_factor_stability_analysis_mode(monkeypatch):
    called = {}
    monkeypatch.setattr(
        runner,
        "run_single_factor_stability_analysis",
        lambda config_override=None: called.setdefault("single_factor_stability_analysis", config_override),
    )
    runner.main(["--single-factor-stability-analysis", "--factor-selection-path", "/tmp/factor_selection_007301_2026-04-05.json"])
    assert "single_factor_stability_analysis" in called
