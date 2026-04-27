from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tradition.config import build_tradition_config
from tradition.factor_engine import normalize_factor_series, rolling_zscore
from tradition.factor_library import build_raw_factor_series
from tradition.metrics import compute_return_metrics, save_equity_curve_plot
from tradition.optimizer import load_optuna_module
from tradition.splitter import build_walk_forward_dev_fold_list, split_time_series_by_ratio

from .common import build_weighted_instance_combination_score
from .io import (
    allocate_strategy_backtest_paths,
    load_factor_combination_input,
    load_strategy_backtest_input,
    print_strategy_backtest_summary,
    print_strategy_advice_summary,
    resolve_fund_code_from_strategy_backtest_input,
    resolve_fund_code_from_factor_combination_input,
    save_strategy_advice_output,
    save_strategy_backtest_output,
)
from .feature_preprocess import (
    _build_single_feature_missing_report,
    _fill_single_feature_missing_by_linear_interpolation,
    _fetch_feature_df_with_cache,
    _fetch_feature_df_by_type,
    _resolve_feature_preprocess_code_type_dict,
    _trim_initial_rows,
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


def build_aggregated_backtest_result(backtest_result_list):
    backtest_result_list = [dict(backtest_result) for backtest_result in list(backtest_result_list)]
    if len(backtest_result_list) == 0:
        raise ValueError("backtest_result_list 为空，无法聚合。")
    metric_name_list = list(dict(backtest_result_list[0]["stats"]).keys())
    stats_dict = {}
    for metric_name in metric_name_list:
        metric_value_list = [float(dict(result["stats"])[metric_name]) for result in backtest_result_list]
        stats_dict[metric_name] = float(np.mean(metric_value_list))
    return {
        "sample_start": min(result["sample_start"] for result in backtest_result_list),
        "sample_end": max(result["sample_end"] for result in backtest_result_list),
        "trade_count": int(sum(int(result["trade_count"]) for result in backtest_result_list)),
        "stats": stats_dict,
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


def run_position_function_search(position_function_config, score_series, fold_list, init_cash, fees):
    optuna_module = load_optuna_module()
    position_function_config = dict(position_function_config)
    train_trial_summary_list = []
    fold_list = [dict(fold) for fold in list(fold_list)]
    if len(fold_list) == 0:
        raise ValueError("fold_list 为空，无法执行仓位函数搜索。")

    def objective(trial):
        function_param_dict = sample_position_function_param_dict(
            trial=trial,
            position_function_config=position_function_config,
        )
        ema_span = int(trial.suggest_int(f"{position_function_config['name']}__ema_span", 1, 30))
        trade_gate = float(trial.suggest_float(f"{position_function_config['name']}__trade_gate", 0.0, 0.2))
        train_fold_result_list = []
        for fold in fold_list:
            train_fold_result_list.append(
                build_backtest_result(
                    price_series=fold["train"],
                    score_series=score_series,
                    segment_series=fold["train"],
                    position_function_name=position_function_config["name"],
                    function_param_dict=function_param_dict,
                    ema_span=ema_span,
                    trade_gate=trade_gate,
                    init_cash=init_cash,
                    fees=fees,
                )
            )
        train_result = build_aggregated_backtest_result(train_fold_result_list)
        train_trial_summary_list.append(
            {
                "trial_number": int(trial.number),
                "position_function_name": str(position_function_config["name"]),
                "position_function_params": dict(function_param_dict),
                "ema_span": int(ema_span),
                "trade_gate": float(trade_gate),
                "train_result": train_result,
                "train_fold_count": int(len(train_fold_result_list)),
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
        valid_fold_result_list = []
        for fold in fold_list:
            valid_fold_result_list.append(
                build_backtest_result(
                    price_series=fold["valid"],
                    score_series=score_series,
                    segment_series=fold["valid"],
                    position_function_name=position_function_config["name"],
                    function_param_dict=dict(train_summary["position_function_params"]),
                    ema_span=int(train_summary["ema_span"]),
                    trade_gate=float(train_summary["trade_gate"]),
                    init_cash=init_cash,
                    fees=fees,
                )
            )
        valid_result = build_aggregated_backtest_result(valid_fold_result_list)
        valid_summary = dict(train_summary)
        valid_summary["valid_result"] = build_serializable_backtest_result(valid_result)
        valid_summary["train_result"] = build_serializable_backtest_result(train_summary["train_result"])
        valid_summary["valid_fold_count"] = int(len(valid_fold_result_list))
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
    if bool(config.get("force_refresh", False)):
        raise ValueError("strategy-backtest 流程禁止 --force-refresh，请先运行流程0 data-preprocess。")
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
    preprocess_path = factor_combination_output.get("preprocess_path")
    if preprocess_path is None:
        raise ValueError("factor_combination 结果缺少 preprocess_path，请重新执行流程4。")
    target_nav_column = str(factor_combination_output.get("target_nav_column", "")).strip()
    if len(target_nav_column) == 0:
        raise ValueError("factor_combination 结果缺少 target_nav_column，请重新执行流程4。")
    resolved_preprocess_path = Path(preprocess_path)
    if not resolved_preprocess_path.exists():
        raise FileNotFoundError(f"流程4特征 CSV 不存在: {resolved_preprocess_path}")
    feature_df = pd.read_csv(resolved_preprocess_path)
    required_column_list = ["date", target_nav_column] + input_candidate_label_list
    missing_column_list = [column for column in required_column_list if column not in feature_df.columns]
    if len(missing_column_list) > 0:
        raise ValueError(f"流程4特征 CSV 缺少流程5必需列: {missing_column_list[:10]}")

    # 逻辑块：直接复用流程4承接的特征 CSV，保持流程3/4/5使用同一份固化因子列口径。
    feature_df = feature_df.copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce")
    feature_df = feature_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    feature_df = feature_df.set_index("date")
    if str(fund_code) not in target_nav_column:
        raise ValueError(f"流程4结果中的 target_nav_column 与 fund_code 不匹配: fund_code={fund_code}, target_nav_column={target_nav_column}")
    price_series = pd.Series(feature_df[target_nav_column], copy=True).astype(float)
    if len(price_series.dropna()) < 2:
        raise ValueError("流程4特征 CSV 中目标净值列有效样本不足，无法执行流程5。")
    data_mode = str(factor_combination_output.get("data_mode", "feature_matrix"))
    split_dict = split_time_series_by_ratio(
        price_series=price_series,
        split_config=config["data_split_dict"],
    )
    fold_list = build_walk_forward_dev_fold_list(
        price_series=price_series,
        walk_forward_config=dict(config["walk_forward_config"]),
        split_config=dict(config["data_split_dict"]),
    )
    dev_series = pd.concat([split_dict["train"], split_dict["valid"]]).sort_index()
    factor_candidate_list = [dict(factor_candidate_record_dict[candidate_label]) for candidate_label in input_candidate_label_list]
    factor_series_dict = {}
    for factor_candidate in factor_candidate_list:
        candidate_label = str(factor_candidate["candidate_label"])
        factor_series_dict[candidate_label] = pd.Series(feature_df[candidate_label], copy=True).astype(float)
        if factor_series_dict[candidate_label].dropna().empty:
            raise ValueError(f"流程4特征 CSV 中候选因子列全为空: {candidate_label}")
    score_series = build_strategy_score_series(
        factor_candidate_list=factor_candidate_list,
        factor_series_dict=factor_series_dict,
        candidate_weight_dict=dict(best_combination_selection_summary["candidate_weight_dict"]),
    )
    position_function_config_list = build_position_function_config_list(
        score_series=score_series.reindex(dev_series.index).dropna(),
    )
    position_function_search_output = {}
    function_trial_summary_list = []
    for position_function_config in position_function_config_list:
        function_name = str(position_function_config["name"])
        function_search_output = run_position_function_search(
            position_function_config=position_function_config,
            score_series=score_series,
            fold_list=fold_list,
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
        function_summary["selected_by"] = "valid_wf"
        function_summary["test_result"] = build_serializable_backtest_result(test_result)
        function_trial_summary_list.append(function_summary)
        position_function_search_output[function_name] = {
            "n_trials": int(function_search_output["n_trials"]),
            "best_valid_trial_summary": best_valid_trial_summary,
            "test_summary": build_serializable_backtest_result(test_result),
        }
    best_strategy_valid_summary = select_best_strategy_trial_summary(
        summary_list=function_trial_summary_list,
        segment_name="valid",
    )
    best_strategy_test_summary = dict(best_strategy_valid_summary)
    best_strategy_test_summary["selected_by"] = "valid_wf"
    summary_path, plot_output_path, path_code = allocate_strategy_backtest_paths(
        factor_combination_input=factor_combination_input,
        factor_combination_path=resolved_factor_combination_path,
        output_dir=config["output_dir"],
        fund_code=fund_code,
    )
    final_plot_result = build_backtest_result(
        price_series=price_series,
        score_series=score_series,
        segment_series=price_series,
        position_function_name=str(best_strategy_valid_summary["position_function_name"]),
        function_param_dict=dict(best_strategy_valid_summary["position_function_params"]),
        ema_span=int(best_strategy_valid_summary["ema_span"]),
        trade_gate=float(best_strategy_valid_summary["trade_gate"]),
        init_cash=float(config["init_cash"]),
        fees=float(config["fees"]),
    )
    plot_path = save_equity_curve_plot(
        equity_curve=final_plot_result["equity_curve"],
        output_path=plot_output_path,
        title=f"{fund_code} strategy_backtest",
        benchmark_curve=price_series,
        highlight_start=split_dict["test"].index.min(),
        highlight_end=split_dict["test"].index.max(),
        highlight_label="test",
    )
    strategy_backtest_output = {
        "fund_code": fund_code,
        "preprocess_path": str(resolved_preprocess_path),
        "factor_combination_path": str(resolved_factor_combination_path),
        "analysis_date": datetime.today().strftime("%Y-%m-%d"),
        "candidate_label_list": input_candidate_label_list,
        "candidate_weight_dict": dict(best_combination_selection_summary["candidate_weight_dict"]),
        "score_build_summary": {
            "selected_method": str(best_combination_selection_summary["selected_method"]),
            "score_name": str(score_series.name),
        },
        "dev_walk_forward_summary": {
            "fold_count": int(len(fold_list)),
            "dynamic_step_size": int(fold_list[0]["dynamic_step_size"]) if len(fold_list) > 0 and "dynamic_step_size" in fold_list[0] else None,
            "dynamic_tail_size": int(fold_list[0]["dynamic_tail_size"]) if len(fold_list) > 0 and "dynamic_tail_size" in fold_list[0] else None,
            "dynamic_step_select_mode": str(fold_list[0]["dynamic_step_select_mode"]) if len(fold_list) > 0 and "dynamic_step_select_mode" in fold_list[0] else None,
        },
        "selected_by": "valid_wf",
        "best_strategy_valid_summary": best_strategy_valid_summary,
        "best_strategy_test_summary": best_strategy_test_summary,
        "plot_path": str(plot_path),
    }
    summary_path = save_strategy_backtest_output(
        factor_combination_input=factor_combination_input,
        strategy_backtest_output=strategy_backtest_output,
        output_dir=config["output_dir"],
        fund_code=fund_code,
        output_path=summary_path,
        path_code=path_code,
    )
    result = {
        "fund_code": fund_code,
        "data_mode": data_mode,
        "factor_combination_path": str(resolved_factor_combination_path),
        "best_strategy_valid_summary": best_strategy_valid_summary,
        "best_strategy_test_summary": best_strategy_test_summary,
        "plot_path": plot_path,
        "summary_path": summary_path,
    }
    print_strategy_backtest_summary(result)
    return result


def _parse_candidate_label_config(candidate_label):
    # 逻辑块：流程6需要从流程5保存下来的最终列名反推基础特征、因子名和是否翻转。
    candidate_label = str(candidate_label)
    if not candidate_label.endswith("__zscore"):
        raise ValueError(f"流程6只支持 __zscore 因子列: {candidate_label}")
    column_body = candidate_label[: -len("__zscore")]
    body_part_list = column_body.split("__")
    if len(body_part_list) < 2:
        raise ValueError(f"无法解析因子列来源: {candidate_label}")
    source_column = "__".join(body_part_list[:2])
    if len(body_part_list) == 2:
        return {
            "candidate_label": candidate_label,
            "source_column": source_column,
            "factor_name": "zscore",
            "factor_param_dict": {},
            "flipped": False,
        }
    factor_label = "__".join(body_part_list[2:])
    flipped = False
    if factor_label.startswith("-"):
        flipped = True
        factor_label = factor_label[1:]
    left_paren_position = factor_label.find("(")
    right_paren_position = factor_label.rfind(")")
    if left_paren_position < 0 or right_paren_position < left_paren_position:
        raise ValueError(f"无法解析因子参数: {candidate_label}")
    factor_name = factor_label[:left_paren_position]
    factor_param_text = factor_label[left_paren_position + 1 : right_paren_position]
    factor_param_dict = {}
    if len(factor_param_text.strip()) > 0:
        for param_pair_text in factor_param_text.split(","):
            param_name, param_value = [part.strip() for part in str(param_pair_text).split("=", 1)]
            numeric_value = float(param_value)
            if numeric_value.is_integer():
                numeric_value = int(numeric_value)
            factor_param_dict[str(param_name)] = numeric_value
    return {
        "candidate_label": candidate_label,
        "source_column": source_column,
        "factor_name": str(factor_name),
        "factor_param_dict": factor_param_dict,
        "flipped": bool(flipped),
    }


def _build_latest_wide_feature_df(config, fund_code):
    # 逻辑块：流程6直接复用流程0原始特征抓取与宽表拼接口径，只是不落整套流程0文件。
    code_type_dict = _resolve_feature_preprocess_code_type_dict(config=config)
    normalized_fund_code = str(fund_code).zfill(6)
    if normalized_fund_code not in code_type_dict:
        raise ValueError(f"流程6缺少主代码类型映射: {normalized_fund_code}")
    import akshare as ak

    feature_df_dict = {}
    for code, code_type in dict(code_type_dict).items():
        normalized_code = str(code).zfill(6)
        feature_df = _fetch_feature_df_with_cache(
            ak_module=ak,
            code=normalized_code,
            code_type=code_type,
            cache_dir=config["data_dir"],
            force_refresh=bool(config.get("force_refresh", False)),
        )
        feature_df_dict[normalized_code] = pd.DataFrame(feature_df, copy=True).set_index("date")
    primary_index = pd.Index(feature_df_dict[normalized_fund_code].index, copy=True)
    wide_feature_df = pd.DataFrame(index=primary_index)
    for code in dict(code_type_dict).keys():
        normalized_code = str(code).zfill(6)
        feature_df = feature_df_dict[normalized_code].reindex(primary_index).copy()
        feature_df.columns = [f"{normalized_code}__{column}" for column in feature_df.columns]
        wide_feature_df = pd.concat([wide_feature_df, feature_df], axis=1)
    wide_feature_df = wide_feature_df[~wide_feature_df.index.duplicated(keep="last")]
    wide_feature_df = wide_feature_df.reset_index().rename(columns={"index": "date"})
    wide_feature_df["date"] = pd.to_datetime(wide_feature_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return wide_feature_df, code_type_dict


def _build_strategy_advice_wide_feature_cache_path(data_dir, fund_code):
    # 逻辑块：流程6把当次增量更新后的原始宽表快照落回 data 目录，便于排查和复用。
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"strategy_advice_latest_wide_feature_{str(fund_code).zfill(6)}.csv"


def _save_strategy_advice_wide_feature_df(wide_feature_df, cache_path):
    wide_feature_df = pd.DataFrame(wide_feature_df, copy=True)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    wide_feature_df.to_csv(cache_path, index=False)
    return cache_path


def _load_strategy_advice_wide_feature_df(cache_path):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"流程6原始宽表快照不存在: {cache_path}")
    return pd.read_csv(cache_path)


def _resolve_strategy_advice_wide_feature_snapshot_date(cache_path):
    # 逻辑块：宽表快照是否可复用，先看快照自身最后日期，避免无意义地重读整张表参与后续判断。
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    wide_feature_df = pd.read_csv(cache_path, usecols=["date"])
    if len(wide_feature_df) == 0 or "date" not in wide_feature_df.columns:
        return None
    snapshot_date_series = pd.to_datetime(wide_feature_df["date"], errors="coerce").dropna()
    if len(snapshot_date_series) == 0:
        return None
    return pd.Timestamp(snapshot_date_series.max())


def _resolve_remote_feature_latest_date_dict(code_type_dict):
    # 逻辑块：快照复用判据改为“网上当前可拉取到的最新日期”，不再参考本地原始缓存日期。
    import akshare as ak

    latest_date_dict = {}
    for code, code_type in dict(code_type_dict).items():
        feature_df = _fetch_feature_df_by_type(
            ak_module=ak,
            code=str(code).zfill(6),
            code_type=code_type,
            last_cached_date=None,
        )
        if len(feature_df) == 0 or "date" not in feature_df.columns:
            latest_date_dict[str(code).zfill(6)] = None
            continue
        remote_date_series = pd.to_datetime(feature_df["date"], errors="coerce").dropna()
        latest_date_dict[str(code).zfill(6)] = None if len(remote_date_series) == 0 else pd.Timestamp(remote_date_series.max())
    return latest_date_dict


def _should_reuse_strategy_advice_wide_feature_snapshot(config, fund_code, code_type_dict):
    # 逻辑块：只有当宽表快照不落后于网上当前可拉取数据时才直接复用，否则继续走增量更新并重建。
    if bool(config.get("force_refresh", False)):
        return False, None, None, _build_strategy_advice_wide_feature_cache_path(data_dir=config["data_dir"], fund_code=fund_code)
    snapshot_path = _build_strategy_advice_wide_feature_cache_path(
        data_dir=config["data_dir"],
        fund_code=fund_code,
    )
    snapshot_latest_date = _resolve_strategy_advice_wide_feature_snapshot_date(cache_path=snapshot_path)
    if snapshot_latest_date is None:
        return False, snapshot_path, None, {}
    remote_latest_date_dict = _resolve_remote_feature_latest_date_dict(code_type_dict=code_type_dict)
    comparable_latest_date_list = [date_value for date_value in remote_latest_date_dict.values() if date_value is not None]
    if len(comparable_latest_date_list) == 0:
        return False, snapshot_path, snapshot_latest_date, remote_latest_date_dict
    if snapshot_latest_date >= max(comparable_latest_date_list):
        return True, snapshot_path, snapshot_latest_date, remote_latest_date_dict
    return False, snapshot_path, snapshot_latest_date, remote_latest_date_dict


def _build_latest_checked_source_feature_df(wide_feature_df, required_source_column_list):
    # 逻辑块：流程6按流程0单列缺失治理规则处理基础特征，只保留最终组合需要的基础特征列。
    checked_feature_df = pd.DataFrame(wide_feature_df, copy=True)
    required_source_column_set = set(str(column) for column in required_source_column_list)
    feature_quality_report_list = []
    dropped_source_column_list = []
    for feature_column in [column for column in checked_feature_df.columns if column != "date"]:
        feature_quality_report = _build_single_feature_missing_report(
            checked_feature_df=checked_feature_df,
            feature_column=feature_column,
        )
        if bool(feature_quality_report["dropped"]):
            dropped_source_column_list.append(str(feature_column))
            feature_quality_report["filled"] = False
            feature_quality_report["filled_missing_count"] = 0
            feature_quality_report_list.append(feature_quality_report)
            continue
        if int(feature_quality_report["missing_count"]) > 0:
            checked_feature_df = _fill_single_feature_missing_by_linear_interpolation(
                checked_feature_df=checked_feature_df,
                feature_column=feature_column,
            )
            feature_quality_report["filled"] = True
            feature_quality_report["filled_missing_count"] = int(feature_quality_report["missing_count"])
        else:
            feature_quality_report["filled"] = False
            feature_quality_report["filled_missing_count"] = 0
        feature_quality_report_list.append(feature_quality_report)
    available_source_column_set = set(column for column in checked_feature_df.columns if column != "date") - set(dropped_source_column_list)
    missing_required_source_column_list = sorted(list(required_source_column_set - available_source_column_set))
    if len(missing_required_source_column_list) > 0:
        raise ValueError(f"流程6所需基础特征列因缺失质量问题不可用: {missing_required_source_column_list}")
    kept_column_list = ["date"] + [column for column in checked_feature_df.columns if column in required_source_column_set]
    return checked_feature_df[kept_column_list].copy(), feature_quality_report_list


def _build_latest_required_factor_df(checked_source_feature_df, candidate_label_config_list, score_window):
    # 逻辑块：流程6只重建流程5最终组合需要的因子列，并保持与流程0一致的标准化与翻转语义。
    checked_source_feature_df = pd.DataFrame(checked_source_feature_df, copy=True)
    factor_df = checked_source_feature_df[["date"]].copy()
    source_column_config_dict = {}
    for candidate_label_config in candidate_label_config_list:
        source_column_config_dict.setdefault(str(candidate_label_config["source_column"]), []).append(dict(candidate_label_config))
    for source_column, source_config_list in source_column_config_dict.items():
        feature_series = pd.Series(checked_source_feature_df[source_column], copy=True).astype(float)
        factor_df[f"{source_column}__zscore"] = rolling_zscore(feature_series, window=int(score_window))
        for source_config in source_config_list:
            if str(source_config["factor_name"]) == "zscore":
                continue
            raw_factor_series = pd.Series(
                build_raw_factor_series(
                    price_series=feature_series,
                    factor_name=str(source_config["factor_name"]),
                    factor_param_dict={
                        str(source_config["factor_name"]): dict(source_config["factor_param_dict"]),
                    },
                ),
                copy=True,
            ).astype(float)
            raw_factor_series = raw_factor_series.replace([float("inf"), -float("inf")], float("nan"))
            normalized_factor_series = normalize_factor_series(
                raw_factor_series=raw_factor_series,
                factor_name=str(source_config["factor_name"]),
                score_window=int(score_window),
            )
            output_series = pd.Series(normalized_factor_series, copy=True).astype(float)
            output_column = str(source_config["candidate_label"])
            if bool(source_config["flipped"]):
                output_series = -output_series
            factor_df[output_column] = output_series
    required_candidate_column_list = [str(config["candidate_label"]) for config in candidate_label_config_list]
    kept_column_list = ["date"] + required_candidate_column_list
    factor_df = factor_df[[column for column in kept_column_list if column in factor_df.columns]].copy()
    factor_df = _trim_initial_rows(feature_df=factor_df)
    return factor_df


def run_strategy_advice(config_override=None):
    config = build_tradition_config(config_override=config_override)
    strategy_backtest_path = config.get("strategy_backtest_path")
    if strategy_backtest_path is None:
        raise ValueError("strategy-advice 模式必须提供 strategy_backtest_path。")
    strategy_backtest_input, resolved_strategy_backtest_path = load_strategy_backtest_input(strategy_backtest_path)
    strategy_backtest_output = dict(strategy_backtest_input["strategy_backtest_output"])
    fund_code = resolve_fund_code_from_strategy_backtest_input(
        strategy_backtest_input=strategy_backtest_input,
        strategy_backtest_path=resolved_strategy_backtest_path,
    )
    candidate_label_list = [str(candidate_label) for candidate_label in strategy_backtest_output.get("candidate_label_list", [])]
    if len(candidate_label_list) == 0:
        raise ValueError("strategy_backtest 结果缺少 candidate_label_list，请重新执行流程5。")
    candidate_weight_dict = {
        str(candidate_label): float(weight_value)
        for candidate_label, weight_value in dict(strategy_backtest_output.get("candidate_weight_dict", {})).items()
    }
    best_strategy_summary = dict(strategy_backtest_output.get("best_strategy_test_summary", {}))
    if len(best_strategy_summary) == 0:
        raise ValueError("strategy_backtest 结果缺少 best_strategy_test_summary，请重新执行流程5。")
    required_strategy_key_list = ["position_function_name", "position_function_params", "ema_span", "trade_gate"]
    missing_strategy_key_list = [key for key in required_strategy_key_list if key not in best_strategy_summary]
    if len(missing_strategy_key_list) > 0:
        raise ValueError(f"strategy_backtest 结果缺少流程6必需字段: {missing_strategy_key_list}")

    factor_combination_path = strategy_backtest_output.get("factor_combination_path")
    if factor_combination_path is None:
        raise ValueError("strategy_backtest 结果缺少 factor_combination_path，请重新执行流程5。")
    factor_combination_input, resolved_factor_combination_path = load_factor_combination_input(factor_combination_path)
    factor_combination_output = dict(factor_combination_input["factor_combination_output"])
    score_window = int(dict(config["strategy_param_dict"]["multi_factor_score"])["score_window"])
    candidate_label_config_list = [_parse_candidate_label_config(candidate_label) for candidate_label in candidate_label_list]
    required_source_column_list = sorted(
        set(str(candidate_label_config["source_column"]) for candidate_label_config in candidate_label_config_list)
    )

    # 逻辑块：流程6主动回源拉最新数据，再按流程0口径处理基础特征并重建最终组合所需因子。
    code_type_dict = _resolve_feature_preprocess_code_type_dict(config=config)
    should_reuse_snapshot, latest_wide_feature_path, snapshot_latest_date, remote_feature_latest_date_dict = _should_reuse_strategy_advice_wide_feature_snapshot(
        config=config,
        fund_code=fund_code,
        code_type_dict=code_type_dict,
    )
    if bool(should_reuse_snapshot):
        wide_feature_df = _load_strategy_advice_wide_feature_df(cache_path=latest_wide_feature_path)
    else:
        wide_feature_df, _ = _build_latest_wide_feature_df(config=config, fund_code=fund_code)
        latest_wide_feature_path = _save_strategy_advice_wide_feature_df(
            wide_feature_df=wide_feature_df,
            cache_path=latest_wide_feature_path,
        )
        wide_feature_df = _load_strategy_advice_wide_feature_df(cache_path=latest_wide_feature_path)
        snapshot_latest_date = _resolve_strategy_advice_wide_feature_snapshot_date(cache_path=latest_wide_feature_path)
        remote_feature_latest_date_dict = _resolve_remote_feature_latest_date_dict(code_type_dict=code_type_dict)
    checked_source_feature_df, feature_quality_report_list = _build_latest_checked_source_feature_df(
        wide_feature_df=wide_feature_df,
        required_source_column_list=required_source_column_list,
    )
    factor_feature_df = _build_latest_required_factor_df(
        checked_source_feature_df=checked_source_feature_df,
        candidate_label_config_list=candidate_label_config_list,
        score_window=score_window,
    )
    factor_feature_df["date"] = pd.to_datetime(factor_feature_df["date"], errors="coerce")
    factor_feature_df = factor_feature_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    factor_feature_df = factor_feature_df.set_index("date")
    missing_candidate_column_list = [column for column in candidate_label_list if column not in factor_feature_df.columns]
    if len(missing_candidate_column_list) > 0:
        raise ValueError(f"流程6未能重建最终组合所需因子列: {missing_candidate_column_list}")
    factor_candidate_list = [
        dict(dict(factor_combination_output.get("factor_candidate_record_dict", {})).get(candidate_label, {"candidate_label": candidate_label}))
        for candidate_label in candidate_label_list
    ]
    factor_series_dict = {
        candidate_label: pd.Series(factor_feature_df[candidate_label], copy=True).astype(float)
        for candidate_label in candidate_label_list
    }
    score_series = build_strategy_score_series(
        factor_candidate_list=factor_candidate_list,
        factor_series_dict=factor_series_dict,
        candidate_weight_dict=candidate_weight_dict,
    )
    target_position_series = build_target_position_series(
        score_series=score_series,
        position_function_name=str(best_strategy_summary["position_function_name"]),
        function_param_dict=dict(best_strategy_summary["position_function_params"]),
        ema_span=int(best_strategy_summary["ema_span"]),
        trade_gate=float(best_strategy_summary["trade_gate"]),
    )
    target_position_series = pd.Series(target_position_series, copy=True).dropna()
    score_series = pd.Series(score_series, copy=True).reindex(target_position_series.index)
    if len(target_position_series) < 2:
        raise ValueError("流程6至少需要两个有效交易日样本，当前样本不足。")

    latest_date = pd.Timestamp(target_position_series.index[-1])
    previous_date = pd.Timestamp(target_position_series.index[-2])
    latest_target_position = float(target_position_series.iloc[-1])
    previous_target_position = float(target_position_series.iloc[-2])
    position_delta = float(latest_target_position - previous_target_position)
    trade_gate = float(best_strategy_summary["trade_gate"])
    if position_delta > 1e-12:
        action = "加仓"
    elif position_delta < -1e-12:
        action = "减仓"
    else:
        action = "持有"

    # 逻辑块：拆解最新一日因子贡献，输出正负贡献最大的因子，便于解释今日动作来源。
    latest_contribution_record_list = []
    for candidate_label in candidate_label_list:
        factor_value = float(factor_series_dict[candidate_label].reindex([latest_date]).iloc[0])
        weight_value = float(candidate_weight_dict[candidate_label])
        contribution_value = float(factor_value * weight_value)
        latest_contribution_record_list.append(
            {
                "candidate_label": candidate_label,
                "factor_value": factor_value,
                "weight": weight_value,
                "contribution": contribution_value,
            }
        )
    positive_contribution_record_list = [
        dict(record)
        for record in sorted(
            [record for record in latest_contribution_record_list if float(record["contribution"]) > 0.0],
            key=lambda record: (-float(record["contribution"]), str(record["candidate_label"])),
        )[:3]
    ]
    negative_contribution_record_list = [
        dict(record)
        for record in sorted(
            [record for record in latest_contribution_record_list if float(record["contribution"]) < 0.0],
            key=lambda record: (float(record["contribution"]), str(record["candidate_label"])),
        )[:3]
    ]

    strategy_advice_output = {
        "fund_code": fund_code,
        "analysis_date": datetime.today().strftime("%Y-%m-%d"),
        "strategy_backtest_path": str(resolved_strategy_backtest_path),
        "factor_combination_path": str(resolved_factor_combination_path),
        "latest_wide_feature_path": str(latest_wide_feature_path),
        "used_cached_wide_feature_snapshot": bool(should_reuse_snapshot),
        "latest_wide_feature_snapshot_date": None if snapshot_latest_date is None else pd.Timestamp(snapshot_latest_date).strftime("%Y-%m-%d"),
        "remote_feature_latest_date_dict": {
            str(code): None if date_value is None else pd.Timestamp(date_value).strftime("%Y-%m-%d")
            for code, date_value in dict(remote_feature_latest_date_dict).items()
        },
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "previous_date": previous_date.strftime("%Y-%m-%d"),
        "latest_score": float(score_series.loc[latest_date]),
        "previous_score": float(score_series.loc[previous_date]),
        "latest_target_position": latest_target_position,
        "previous_target_position": previous_target_position,
        "position_delta": position_delta,
        "trade_gate": trade_gate,
        "trade_triggered": bool(abs(position_delta) >= trade_gate),
        "action": action,
        "position_function_name": str(best_strategy_summary["position_function_name"]),
        "position_function_params": dict(best_strategy_summary["position_function_params"]),
        "ema_span": int(best_strategy_summary["ema_span"]),
        "candidate_label_list": list(candidate_label_list),
        "candidate_weight_dict": dict(candidate_weight_dict),
        "required_source_column_list": required_source_column_list,
        "feature_quality_report_list": feature_quality_report_list,
        "top_positive_contributors": positive_contribution_record_list,
        "top_negative_contributors": negative_contribution_record_list,
    }
    summary_path = save_strategy_advice_output(
        strategy_backtest_input=strategy_backtest_input,
        strategy_advice_output=strategy_advice_output,
        output_dir=config["output_dir"],
        fund_code=fund_code,
    )
    result = {
        **strategy_advice_output,
        "summary_path": summary_path,
    }
    print_strategy_advice_summary(result)
    return result
