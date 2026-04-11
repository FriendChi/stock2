from datetime import datetime

import numpy as np
import pandas as pd

from tradition.config import build_tradition_config
from tradition.data_adapter import adapt_to_price_series
from tradition.data_fetcher import fetch_fund_data_with_cache
from tradition.data_loader import filter_single_fund, normalize_fund_data
from tradition.factor_engine import build_single_factor_series
from tradition.metrics import compute_return_metrics, save_equity_curve_plot
from tradition.optimizer import load_optuna_module
from tradition.splitter import split_time_series_by_ratio

from .common import build_weighted_instance_combination_score
from .io import (
    load_factor_combination_input,
    print_strategy_backtest_summary,
    resolve_fund_code_from_factor_combination_input,
    save_strategy_backtest_output,
)


def build_strategy_score_series(factor_candidate_list, factor_series_dict, candidate_weight_dict):
    _, score_series = build_weighted_instance_combination_score(
        factor_candidate_list=factor_candidate_list,
        factor_series_dict=factor_series_dict,
        candidate_weight_dict=candidate_weight_dict,
    )
    return score_series.astype(float)


def build_sigmoid_position_series(score_series, center, slope):
    score_series = pd.Series(score_series, copy=True).astype(float)
    position_series = 1.0 / (1.0 + np.exp(-float(slope) * (score_series - float(center))))
    return pd.Series(position_series, index=score_series.index, dtype=float).clip(lower=0.0, upper=1.0)


def build_tanh_rescaled_position_series(score_series, center, slope):
    score_series = pd.Series(score_series, copy=True).astype(float)
    position_series = 0.5 * (np.tanh(float(slope) * (score_series - float(center))) + 1.0)
    return pd.Series(position_series, index=score_series.index, dtype=float).clip(lower=0.0, upper=1.0)


def build_piecewise_linear_position_series(score_series, lower, upper):
    score_series = pd.Series(score_series, copy=True).astype(float)
    lower = float(lower)
    upper = float(upper)
    if upper <= lower:
        upper = lower + 1e-6
    position_series = (score_series - lower) / (upper - lower)
    return pd.Series(position_series, index=score_series.index, dtype=float).clip(lower=0.0, upper=1.0)


def build_raw_position_series(score_series, position_function_name, function_param_dict):
    position_function_name = str(position_function_name)
    function_param_dict = dict(function_param_dict)
    if position_function_name == "sigmoid":
        return build_sigmoid_position_series(
            score_series=score_series,
            center=function_param_dict["center"],
            slope=function_param_dict["slope"],
        )
    if position_function_name == "tanh_rescaled":
        return build_tanh_rescaled_position_series(
            score_series=score_series,
            center=function_param_dict["center"],
            slope=function_param_dict["slope"],
        )
    if position_function_name == "piecewise_linear":
        return build_piecewise_linear_position_series(
            score_series=score_series,
            lower=function_param_dict["lower"],
            upper=function_param_dict["upper"],
        )
    raise ValueError(f"未定义的仓位函数: {position_function_name}")


def build_smoothed_position_series(position_series, ema_span):
    position_series = pd.Series(position_series, copy=True).astype(float)
    ema_span = max(1, int(ema_span))
    return position_series.ewm(span=ema_span, adjust=False).mean().clip(lower=0.0, upper=1.0)


def apply_position_change_gate(position_series, trade_gate):
    position_series = pd.Series(position_series, copy=True).astype(float).clip(lower=0.0, upper=1.0)
    trade_gate = float(trade_gate)
    gated_position_list = []
    previous_position = 0.0
    for position_value in position_series.tolist():
        position_value = float(position_value)
        if len(gated_position_list) == 0:
            gated_position_list.append(position_value)
            previous_position = position_value
            continue
        if abs(position_value - previous_position) < trade_gate:
            gated_position_list.append(previous_position)
            continue
        gated_position_list.append(position_value)
        previous_position = position_value
    return pd.Series(gated_position_list, index=position_series.index, dtype=float).clip(lower=0.0, upper=1.0)


def build_target_position_series(score_series, position_function_name, function_param_dict, ema_span, trade_gate):
    raw_position_series = build_raw_position_series(
        score_series=score_series,
        position_function_name=position_function_name,
        function_param_dict=function_param_dict,
    )
    smoothed_position_series = build_smoothed_position_series(
        position_series=raw_position_series,
        ema_span=ema_span,
    )
    return apply_position_change_gate(
        position_series=smoothed_position_series,
        trade_gate=trade_gate,
    )


def execute_continuous_position_backtest(price_series, target_position_series, init_cash, fees):
    price_series = pd.Series(price_series, copy=True).astype(float).dropna()
    target_position_series = pd.Series(target_position_series, copy=True).astype(float).reindex(price_series.index).ffill().fillna(0.0)
    asset_return_series = price_series.pct_change().fillna(0.0)
    held_position_series = target_position_series.shift(1).fillna(0.0)
    turnover_series = target_position_series.diff().abs().fillna(target_position_series.abs())
    strategy_return_series = held_position_series * asset_return_series - float(fees) * turnover_series
    equity_curve = float(init_cash) * (1.0 + strategy_return_series).cumprod()
    trade_count = int((turnover_series > 1e-12).sum())
    return {
        "equity_curve": equity_curve.astype(float),
        "position_series": target_position_series.astype(float),
        "trade_count": trade_count,
    }


def build_backtest_result(price_series, score_series, segment_series, position_function_name, function_param_dict, ema_span, trade_gate, init_cash, fees):
    segment_price_series = pd.Series(segment_series, copy=True).astype(float).dropna()
    segment_score_series = pd.Series(score_series, copy=True).reindex(segment_price_series.index)
    target_position_series = build_target_position_series(
        score_series=segment_score_series,
        position_function_name=position_function_name,
        function_param_dict=function_param_dict,
        ema_span=ema_span,
        trade_gate=trade_gate,
    )
    execution_result = execute_continuous_position_backtest(
        price_series=segment_price_series,
        target_position_series=target_position_series,
        init_cash=init_cash,
        fees=fees,
    )
    return {
        "sample_start": segment_price_series.index.min(),
        "sample_end": segment_price_series.index.max(),
        "trade_count": int(execution_result["trade_count"]),
        "stats": compute_return_metrics(execution_result["equity_curve"]),
        "equity_curve": execution_result["equity_curve"],
        "position_series": execution_result["position_series"],
    }


def build_serializable_backtest_result(backtest_result):
    backtest_result = dict(backtest_result)
    return {
        "sample_start": pd.Timestamp(backtest_result["sample_start"]).strftime("%Y-%m-%d"),
        "sample_end": pd.Timestamp(backtest_result["sample_end"]).strftime("%Y-%m-%d"),
        "trade_count": int(backtest_result["trade_count"]),
        "stats": {
            metric_name: float(metric_value)
            for metric_name, metric_value in dict(backtest_result["stats"]).items()
        },
    }


def build_position_function_config_list(score_series):
    score_series = pd.Series(score_series, copy=True).astype(float).dropna()
    if score_series.empty:
        raise ValueError("score_series 为空，无法构造仓位函数搜索空间。")
    score_min = float(score_series.min())
    score_max = float(score_series.max())
    score_q10 = float(score_series.quantile(0.10))
    score_q50 = float(score_series.quantile(0.50))
    score_q90 = float(score_series.quantile(0.90))
    if score_max <= score_min:
        score_max = score_min + 1e-6
    if score_q90 <= score_q10:
        score_q90 = score_q10 + 1e-6
    return [
        {
            "name": "sigmoid",
            "param_space": {
                "center": {"type": "float", "low": score_q10, "high": score_q90},
                "slope": {"type": "float", "low": 0.5, "high": 20.0},
            },
        },
        {
            "name": "tanh_rescaled",
            "param_space": {
                "center": {"type": "float", "low": score_q10, "high": score_q90},
                "slope": {"type": "float", "low": 0.5, "high": 20.0},
            },
        },
        {
            "name": "piecewise_linear",
            "param_space": {
                "lower": {"type": "float", "low": score_min, "high": score_q50},
                "upper": {"type": "float", "low": score_q50, "high": score_max},
            },
        },
    ]


def sample_position_function_param_dict(trial, position_function_config):
    position_function_config = dict(position_function_config)
    function_param_dict = {}
    for param_name, param_config in dict(position_function_config["param_space"]).items():
        if str(param_config["type"]) == "float":
            function_param_dict[param_name] = float(
                trial.suggest_float(
                    f"{position_function_config['name']}__{param_name}",
                    float(param_config["low"]),
                    float(param_config["high"]),
                )
            )
            continue
        raise ValueError(f"未支持的仓位函数参数类型: {param_config['type']}")
    return function_param_dict


def select_best_strategy_trial_summary(summary_list, segment_name):
    summary_list = [dict(summary) for summary in summary_list]
    if len(summary_list) == 0:
        return None
    segment_name = str(segment_name)
    sorted_summary_list = sorted(
        summary_list,
        key=lambda summary: (
            -float(summary[f"{segment_name}_result"]["stats"]["sharpe"]),
            -float(summary[f"{segment_name}_result"]["stats"]["annual_return"]),
            -float(summary[f"{segment_name}_result"]["stats"]["max_drawdown"]),
            -float(summary["valid_result"]["stats"]["sharpe"]) if "valid_result" in summary else 0.0,
        ),
    )
    return dict(sorted_summary_list[0])


def run_position_function_search(position_function_config, score_series, split_dict, init_cash, fees):
    optuna_module = load_optuna_module()
    position_function_config = dict(position_function_config)
    train_trial_summary_list = []

    def objective(trial):
        function_param_dict = sample_position_function_param_dict(
            trial=trial,
            position_function_config=position_function_config,
        )
        ema_span = int(trial.suggest_int(f"{position_function_config['name']}__ema_span", 1, 30))
        trade_gate = float(trial.suggest_float(f"{position_function_config['name']}__trade_gate", 0.0, 0.2))
        train_result = build_backtest_result(
            price_series=split_dict["train"],
            score_series=score_series,
            segment_series=split_dict["train"],
            position_function_name=position_function_config["name"],
            function_param_dict=function_param_dict,
            ema_span=ema_span,
            trade_gate=trade_gate,
            init_cash=init_cash,
            fees=fees,
        )
        train_trial_summary_list.append(
            {
                "trial_number": int(trial.number),
                "position_function_name": str(position_function_config["name"]),
                "position_function_params": dict(function_param_dict),
                "ema_span": int(ema_span),
                "trade_gate": float(trade_gate),
                "train_result": train_result,
            }
        )
        return float(train_result["stats"]["sharpe"])

    study = optuna_module.create_study(
        direction="maximize",
        sampler=optuna_module.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=100)
    sorted_train_trial_summary_list = sorted(
        train_trial_summary_list,
        key=lambda summary: (
            -float(summary["train_result"]["stats"]["sharpe"]),
            -float(summary["train_result"]["stats"]["annual_return"]),
            -float(summary["train_result"]["stats"]["max_drawdown"]),
            int(summary["trial_number"]),
        ),
    )
    top_train_trial_summary_list = []
    for summary in sorted_train_trial_summary_list[:50]:
        serializable_summary = dict(summary)
        serializable_summary["train_result"] = build_serializable_backtest_result(summary["train_result"])
        top_train_trial_summary_list.append(serializable_summary)
    valid_trial_summary_list = []
    for train_summary in sorted_train_trial_summary_list[:50]:
        valid_result = build_backtest_result(
            price_series=split_dict["valid"],
            score_series=score_series,
            segment_series=split_dict["valid"],
            position_function_name=position_function_config["name"],
            function_param_dict=dict(train_summary["position_function_params"]),
            ema_span=int(train_summary["ema_span"]),
            trade_gate=float(train_summary["trade_gate"]),
            init_cash=init_cash,
            fees=fees,
        )
        valid_summary = dict(train_summary)
        valid_summary["valid_result"] = build_serializable_backtest_result(valid_result)
        valid_summary["train_result"] = build_serializable_backtest_result(train_summary["train_result"])
        valid_trial_summary_list.append(valid_summary)
    best_valid_trial_summary = select_best_strategy_trial_summary(
        summary_list=valid_trial_summary_list,
        segment_name="valid",
    )
    return {
        "n_trials": 100,
        "train_top_trial_summary_list": top_train_trial_summary_list,
        "best_valid_trial_summary": best_valid_trial_summary,
    }


def run_strategy_backtest(config_override=None):
    config = build_tradition_config(config_override=config_override)
    factor_combination_path = config.get("factor_combination_path")
    if factor_combination_path is None:
        raise ValueError("strategy_backtest 模式必须提供 factor_combination_path。")
    factor_combination_input, resolved_factor_combination_path = load_factor_combination_input(factor_combination_path)
    factor_combination_output = dict(factor_combination_input["factor_combination_output"])
    best_combination_selection_summary = dict(factor_combination_output["best_combination_selection_summary"])
    factor_candidate_record_dict = {
        str(candidate_label): dict(record)
        for candidate_label, record in dict(factor_combination_output["factor_candidate_record_dict"]).items()
    }
    input_candidate_label_list = [str(candidate_label) for candidate_label in best_combination_selection_summary["candidate_label_list"]]
    if len(input_candidate_label_list) == 0:
        raise ValueError("factor_combination 结果中的 best_combination_selection_summary 为空组合。")
    fund_code = resolve_fund_code_from_factor_combination_input(
        factor_combination_input=factor_combination_input,
        factor_combination_path=resolved_factor_combination_path,
    )

    raw_data = fetch_fund_data_with_cache(
        code_dict=config["code_dict"],
        cache_dir=config["data_dir"],
        force_refresh=bool(config["force_refresh"]),
        cache_prefix=config["cache_prefix"],
    )
    normalized_data = normalize_fund_data(raw_data)
    fund_df = filter_single_fund(normalized_data, fund_code=fund_code)
    price_series, data_mode = adapt_to_price_series(fund_df=fund_df)
    split_dict = split_time_series_by_ratio(
        price_series=price_series,
        split_config=config["data_split_dict"],
    )
    base_multi_factor_params = dict(config["strategy_param_dict"]["multi_factor_score"])
    factor_candidate_list = [dict(factor_candidate_record_dict[candidate_label]) for candidate_label in input_candidate_label_list]
    factor_series_dict = {}
    for factor_candidate in factor_candidate_list:
        candidate_label = str(factor_candidate["candidate_label"])
        factor_series_dict[candidate_label] = build_single_factor_series(
            price_series=price_series,
            factor_name=factor_candidate["factor_name"],
            strategy_params=base_multi_factor_params,
            factor_param_override=dict(factor_candidate["factor_param_dict"]),
        )
    score_series = build_strategy_score_series(
        factor_candidate_list=factor_candidate_list,
        factor_series_dict=factor_series_dict,
        candidate_weight_dict=dict(best_combination_selection_summary["candidate_weight_dict"]),
    )
    position_function_config_list = build_position_function_config_list(
        score_series=score_series.reindex(split_dict["train"].index).dropna(),
    )
    position_function_search_output = {}
    function_trial_summary_list = []
    for position_function_config in position_function_config_list:
        function_name = str(position_function_config["name"])
        function_search_output = run_position_function_search(
            position_function_config=position_function_config,
            score_series=score_series,
            split_dict=split_dict,
            init_cash=float(config["init_cash"]),
            fees=float(config["fees"]),
        )
        best_valid_trial_summary = dict(function_search_output["best_valid_trial_summary"])
        test_result = build_backtest_result(
            price_series=split_dict["test"],
            score_series=score_series,
            segment_series=split_dict["test"],
            position_function_name=function_name,
            function_param_dict=dict(best_valid_trial_summary["position_function_params"]),
            ema_span=int(best_valid_trial_summary["ema_span"]),
            trade_gate=float(best_valid_trial_summary["trade_gate"]),
            init_cash=float(config["init_cash"]),
            fees=float(config["fees"]),
        )
        function_summary = dict(best_valid_trial_summary)
        function_summary["candidate_label_list"] = list(input_candidate_label_list)
        function_summary["candidate_weight_dict"] = dict(best_combination_selection_summary["candidate_weight_dict"])
        function_summary["test_result"] = build_serializable_backtest_result(test_result)
        function_trial_summary_list.append(function_summary)
        position_function_search_output[function_name] = {
            "n_trials": int(function_search_output["n_trials"]),
            "best_valid_trial_summary": best_valid_trial_summary,
            "test_summary": build_serializable_backtest_result(test_result),
        }
    best_strategy_test_summary = select_best_strategy_trial_summary(
        summary_list=function_trial_summary_list,
        segment_name="test",
    )
    final_plot_result = build_backtest_result(
        price_series=price_series,
        score_series=score_series,
        segment_series=price_series,
        position_function_name=str(best_strategy_test_summary["position_function_name"]),
        function_param_dict=dict(best_strategy_test_summary["position_function_params"]),
        ema_span=int(best_strategy_test_summary["ema_span"]),
        trade_gate=float(best_strategy_test_summary["trade_gate"]),
        init_cash=float(config["init_cash"]),
        fees=float(config["fees"]),
    )
    plot_path = save_equity_curve_plot(
        equity_curve=final_plot_result["equity_curve"],
        output_path=config["output_dir"] / f"strategy_backtest_{fund_code}_{datetime.today().strftime('%Y-%m-%d')}.png",
        title=f"{fund_code} strategy_backtest",
        benchmark_curve=price_series,
    )
    strategy_backtest_output = {
        "fund_code": fund_code,
        "factor_combination_path": str(resolved_factor_combination_path),
        "analysis_date": datetime.today().strftime("%Y-%m-%d"),
        "candidate_label_list": input_candidate_label_list,
        "candidate_weight_dict": dict(best_combination_selection_summary["candidate_weight_dict"]),
        "score_build_summary": {
            "selected_method": str(best_combination_selection_summary["selected_method"]),
            "score_name": str(score_series.name),
        },
        "best_strategy_test_summary": best_strategy_test_summary,
        "plot_path": str(plot_path),
    }
    summary_path = save_strategy_backtest_output(
        factor_combination_input=factor_combination_input,
        strategy_backtest_output=strategy_backtest_output,
        output_dir=config["output_dir"],
        fund_code=fund_code,
    )
    result = {
        "fund_code": fund_code,
        "data_mode": data_mode,
        "factor_combination_path": str(resolved_factor_combination_path),
        "best_strategy_test_summary": best_strategy_test_summary,
        "plot_path": plot_path,
        "summary_path": summary_path,
    }
    print_strategy_backtest_summary(result)
    return result
