import json
from pathlib import Path

import numpy as np
import pandas as pd

from tradition import factor_analysis
def test_build_target_position_series_stays_in_zero_to_one():
    score_series = pd.Series([-2.0, -0.5, 0.0, 0.5, 2.0], index=pd.date_range("2024-01-01", periods=5, freq="D"))
    sigmoid_position_series = factor_analysis.build_target_position_series(
        score_series=score_series,
        position_function_name="sigmoid",
        function_param_dict={"center": 0.0, "slope": 2.0},
        ema_span=3,
        trade_gate=0.0,
    )
    piecewise_position_series = factor_analysis.build_target_position_series(
        score_series=score_series,
        position_function_name="piecewise_linear",
        function_param_dict={"lower": -1.0, "upper": 1.0},
        ema_span=2,
        trade_gate=0.1,
    )
    assert bool(((sigmoid_position_series >= 0.0) & (sigmoid_position_series <= 1.0)).all()) is True
    assert bool(((piecewise_position_series >= 0.0) & (piecewise_position_series <= 1.0)).all()) is True

def test_select_best_strategy_trial_summary_uses_target_segment_first():
    best_summary = factor_analysis.select_best_strategy_trial_summary(
        [
            {
                "position_function_name": "sigmoid",
                "valid_result": {"stats": {"sharpe": 0.8, "annual_return": 0.2, "max_drawdown": -0.1}},
                "test_result": {"stats": {"sharpe": 0.7, "annual_return": 0.3, "max_drawdown": -0.2}},
            },
            {
                "position_function_name": "piecewise_linear",
                "valid_result": {"stats": {"sharpe": 0.7, "annual_return": 0.25, "max_drawdown": -0.12}},
                "test_result": {"stats": {"sharpe": 0.9, "annual_return": 0.25, "max_drawdown": -0.15}},
            },
        ],
        segment_name="test",
    )
    assert best_summary["position_function_name"] == "piecewise_linear"

def test_run_strategy_backtest_outputs_independent_json(monkeypatch, tmp_path):
    factor_combination_path = tmp_path / "factor_combination_007301_2026-04-09.json"
    preprocess_path = tmp_path / "feature_preprocess_007301_2026-04-09_a00000_checked.csv"
    sample_price_series = pd.Series([1.0, 1.02, 1.03, 1.01, 1.05, 1.07, 1.08, 1.10], index=pd.date_range("2024-01-01", periods=8, freq="D"))
    preprocess_df = pd.DataFrame(
        {
            "date": sample_price_series.index,
            "007301__cumulative_nav": sample_price_series.values,
            "momentum(window=10)": np.linspace(0.0, 1.0, len(sample_price_series)),
            "trend_tvalue(window=15)": np.linspace(1.0, 0.0, len(sample_price_series)),
        }
    )
    preprocess_df.to_csv(preprocess_path, index=False)
    factor_combination_payload = {
        "path_code": "abcd0",
        "factor_combination_output": {
            "fund_code": "007301",
            "preprocess_path": str(preprocess_path),
            "target_nav_column": "007301__cumulative_nav",
            "factor_candidate_record_dict": {
                "momentum(window=10)": {
                    "candidate_label": "momentum(window=10)",
                    "factor_name": "momentum",
                    "factor_param_dict": {"window": 10},
                    "factor_group": "趋势/动量",
                },
                "trend_tvalue(window=15)": {
                    "candidate_label": "trend_tvalue(window=15)",
                    "factor_name": "trend_tvalue",
                    "factor_param_dict": {"window": 15},
                    "factor_group": "趋势强度",
                },
            },
            "best_combination_selection_summary": {
                "candidate_label_list": ["momentum(window=10)", "trend_tvalue(window=15)"],
                "selected_method": "equal_weight",
                "candidate_weight_dict": {
                    "momentum(window=10)": 0.5,
                    "trend_tvalue(window=15)": 0.5,
                },
            },
        },
    }
    factor_combination_path.write_text(json.dumps(factor_combination_payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(
        factor_analysis.backtest,
        "build_tradition_config",
        lambda config_override=None: {
            "code_dict": {"007301": "半导体"},
            "data_dir": tmp_path,
            "output_dir": tmp_path,
            "force_refresh": False,
            "cache_prefix": "tradition_fund",
            "factor_combination_path": str(factor_combination_path),
            "strategy_param_dict": {"multi_factor_score": {}},
            "walk_forward_config": {
                "window_size": 4,
                "min_fold_count": 1,
            },
            "data_split_dict": {
                "train_ratio": 0.5,
                "valid_ratio": 0.25,
                "test_ratio": 0.25,
                "min_segment_size": 2,
            },
            "init_cash": 10000.0,
            "fees": 0.001,
        },
    )
    monkeypatch.setattr(
        factor_analysis.backtest,
        "run_position_function_search",
        lambda position_function_config, score_series, fold_list, init_cash, fees: {
            "n_trials": 100,
            "best_valid_trial_summary": {
                "trial_number": 1 if position_function_config["name"] == "sigmoid" else 2,
                "position_function_name": position_function_config["name"],
                "position_function_params": {"center": 0.0, "slope": 1.0} if position_function_config["name"] != "piecewise_linear" else {"lower": -0.1, "upper": 0.1},
                "ema_span": 3,
                "trade_gate": 0.05,
                "train_result": {"stats": {"sharpe": 1.0, "annual_return": 0.2, "max_drawdown": -0.1}},
                "valid_result": {"stats": {"sharpe": 1.1 if position_function_config["name"] == "sigmoid" else 0.9, "annual_return": 0.21, "max_drawdown": -0.09}},
            },
        },
    )
    monkeypatch.setattr(
        factor_analysis.backtest,
        "build_walk_forward_dev_fold_list",
        lambda price_series, walk_forward_config, split_config: [
            {
                "fold_id": 1,
                "train": price_series.iloc[:4],
                "valid": price_series.iloc[4:6],
                "dynamic_step_size": 2,
                "dynamic_tail_size": 0,
                "dynamic_step_select_mode": "exact_divisor",
            },
            {
                "fold_id": 2,
                "train": price_series.iloc[1:5],
                "valid": price_series.iloc[5:7],
                "dynamic_step_size": 2,
                "dynamic_tail_size": 0,
                "dynamic_step_select_mode": "exact_divisor",
            },
        ],
    )
    monkeypatch.setattr(
        factor_analysis.backtest,
        "build_backtest_result",
        lambda price_series, score_series, segment_series, position_function_name, function_param_dict, ema_span, trade_gate, init_cash, fees: {
            "sample_start": pd.Index(segment_series.index).min(),
            "sample_end": pd.Index(segment_series.index).max(),
            "trade_count": 3,
            "stats": {
                "cumulative_return": 0.1,
                "annual_return": 0.2,
                "annual_volatility": 0.1,
                "sharpe": (
                    0.7
                    if (len(pd.Index(segment_series.index)) <= 2 and position_function_name == "sigmoid")
                    else (2.0 if len(pd.Index(segment_series.index)) <= 2 else (1.2 if position_function_name == "sigmoid" else 0.8))
                ),
                "max_drawdown": -0.05,
            },
            "equity_curve": pd.Series([10000.0, 10100.0], index=pd.Index(segment_series.index[:2])),
            "position_series": pd.Series([0.5, 0.6], index=pd.Index(segment_series.index[:2])),
        },
    )
    captured_plot_kwargs = {}

    def fake_save_equity_curve_plot(
        equity_curve,
        output_path,
        title,
        benchmark_curve=None,
        highlight_start=None,
        highlight_end=None,
        highlight_label=None,
    ):
        captured_plot_kwargs["highlight_start"] = highlight_start
        captured_plot_kwargs["highlight_end"] = highlight_end
        captured_plot_kwargs["highlight_label"] = highlight_label
        return output_path

    monkeypatch.setattr(
        factor_analysis.backtest,
        "save_equity_curve_plot",
        fake_save_equity_curve_plot,
    )

    result = factor_analysis.run_strategy_backtest(config_override={})

    assert result["fund_code"] == "007301"
    output_path = result["summary_path"]
    assert output_path.name.startswith("strategy_backtest_007301_")
    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_payload["input_ref"]["fund_code"] == "007301"
    assert len(output_payload["path_code"]) == 6
    assert str(output_payload["path_code"]).isalnum()
    assert output_path.stem.split("_")[-1] == output_payload["path_code"]
    assert "strategy_backtest_output" in output_payload
    assert output_payload["strategy_backtest_output"]["best_strategy_valid_summary"]["position_function_name"] == "sigmoid"
    assert output_payload["strategy_backtest_output"]["best_strategy_test_summary"]["position_function_name"] == "sigmoid"
    assert output_payload["strategy_backtest_output"]["selected_by"] == "valid_wf"
    assert "best_function_valid_summary" not in output_payload["strategy_backtest_output"]
    assert "position_function_search_output" not in output_payload["strategy_backtest_output"]
    assert Path(output_payload["strategy_backtest_output"]["plot_path"]).stem.split("_")[-1] == output_payload["path_code"]
    assert pd.Timestamp(captured_plot_kwargs["highlight_start"]) == sample_price_series.index[6]
    assert pd.Timestamp(captured_plot_kwargs["highlight_end"]) == sample_price_series.index[7]
    assert captured_plot_kwargs["highlight_label"] == "test"


def test_execute_continuous_position_backtest_logic():
    # 逻辑块：构造基础价格序列（线性增长）
    dates = pd.date_range("2024-01-01", periods=5)
    price_series = pd.Series([10.0, 11.0, 12.0, 11.0, 12.0], index=dates)

    # 逻辑块：构造全仓持有策略
    target_position = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=dates)

    result = factor_analysis.backtest.execute_continuous_position_backtest(
        price_series=price_series,
        target_position_series=target_position,
        init_cash=10000.0,
        fees=0.001,
    )

    # 逻辑块：验证权益曲线计算。注意：held_position 是 shift(1)，第一天会有建仓手续费
    assert result["equity_curve"].iloc[0] == 9990.0
    assert result["equity_curve"].iloc[1] > 10000.0
    assert result["trade_count"] >= 1  # 初始买入


def test_apply_position_change_gate():
    # 逻辑块：验证交易门槛（Gate）对调仓的抑制作用
    positions = pd.Series([0.0, 0.04, 0.10, 0.08, 0.20])

    # 门槛 0.05：0.04 不触发，0.10 触发（相对于0.0），0.08 不触发，0.20 触发
    gated = factor_analysis.backtest.apply_position_change_gate(positions, trade_gate=0.05)

    assert gated.iloc[1] == 0.0  # 0.04 < 0.05
    assert gated.iloc[2] == 0.10  # abs(0.10 - 0.0) > 0.05
    assert gated.iloc[3] == 0.10  # abs(0.08 - 0.10) < 0.05
    assert gated.iloc[4] == 0.20  # abs(0.20 - 0.10) > 0.05
