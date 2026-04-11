from datetime import datetime

from tradition.config import build_tradition_config, resolve_effective_code_dict
from tradition.data_adapter import adapt_to_price_series
from tradition.data_fetcher import fetch_fund_data_with_cache
from tradition.data_loader import filter_single_fund, normalize_fund_data
from tradition.factor_engine import build_single_factor_series
from tradition.optimizer import load_optuna_module
from tradition.splitter import build_walk_forward_dev_fold_list

from .common import (
    build_forward_return_series,
    build_ic_aggregation_config,
    build_metric_summary,
    build_spearman_metric_summary,
    build_weighted_instance_combination_score,
    compute_segment_correlation_metrics,
    evaluate_single_factor_train_icir,
)
from .io import (
    load_dedup_selection_input,
    print_factor_combination_summary,
    resolve_fund_code_from_dedup_selection_input,
    save_factor_combination_output,
)


def build_equal_weight_dict(factor_candidate_list):
    if len(factor_candidate_list) == 0:
        return {}
    weight_value = 1.0 / float(len(factor_candidate_list))
    return {
        str(factor_candidate["candidate_label"]): weight_value
        for factor_candidate in factor_candidate_list
    }


def build_train_icir_weight_dict(factor_candidate_list, factor_series_dict, forward_return_series, fold_list, candidate_train_icir_dict=None, ic_aggregation_config=None):
    resolved_train_icir_dict = {}
    for factor_candidate in factor_candidate_list:
        candidate_label = str(factor_candidate["candidate_label"])
        if candidate_train_icir_dict is not None and candidate_label in candidate_train_icir_dict:
            resolved_train_icir_dict[candidate_label] = float(candidate_train_icir_dict[candidate_label])
            continue
        resolved_train_icir_dict[candidate_label] = evaluate_single_factor_train_icir(
            factor_series=factor_series_dict[candidate_label],
            forward_return_series=forward_return_series,
            fold_list=fold_list,
            ic_aggregation_config=ic_aggregation_config,
        )
    nonnegative_weight_dict = {
        candidate_label: max(float(train_icir), 0.0)
        for candidate_label, train_icir in resolved_train_icir_dict.items()
    }
    total_weight = float(sum(nonnegative_weight_dict.values()))
    if total_weight <= 1e-12:
        return build_equal_weight_dict(factor_candidate_list)
    return {
        candidate_label: float(weight_value / total_weight)
        for candidate_label, weight_value in nonnegative_weight_dict.items()
    }


def evaluate_combination_summary(
    factor_candidate_list,
    factor_series_dict,
    forward_return_series,
    fold_list,
    method_name,
    candidate_train_icir_dict=None,
    include_valid=True,
    candidate_weight_dict_override=None,
    ic_aggregation_config=None,
):
    if candidate_weight_dict_override is not None:
        candidate_weight_dict = {
            str(candidate_label): float(weight_value)
            for candidate_label, weight_value in candidate_weight_dict_override.items()
        }
    elif str(method_name) == "equal_weight":
        candidate_weight_dict = build_equal_weight_dict(factor_candidate_list)
    elif str(method_name) == "train_icir_weighted":
        candidate_weight_dict = build_train_icir_weight_dict(
            factor_candidate_list=factor_candidate_list,
            factor_series_dict=factor_series_dict,
            forward_return_series=forward_return_series,
            fold_list=fold_list,
            candidate_train_icir_dict=candidate_train_icir_dict,
            ic_aggregation_config=ic_aggregation_config,
        )
    else:
        raise ValueError(f"未定义的组合方式: {method_name}")

    _, score_series = build_weighted_instance_combination_score(
        factor_candidate_list=factor_candidate_list,
        factor_series_dict=factor_series_dict,
        candidate_weight_dict=candidate_weight_dict,
    )
    train_metric_list = []
    valid_metric_list = []
    for fold_dict in fold_list:
        train_metric_list.append(
            compute_segment_correlation_metrics(
                factor_series=score_series,
                forward_return_series=forward_return_series,
                segment_index=fold_dict["train"].index,
            )
        )
        if include_valid:
            valid_metric_list.append(
                compute_segment_correlation_metrics(
                    factor_series=score_series,
                    forward_return_series=forward_return_series,
                    segment_index=fold_dict["valid"].index,
                )
            )
    if ic_aggregation_config is None:
        ic_aggregation_config = {"mode": "classic", "half_life": 3.0}
    train_summary = build_spearman_metric_summary([metric_dict["spearman_ic"] for metric_dict in train_metric_list], ic_aggregation_config=ic_aggregation_config)
    summary = {
        "method_name": str(method_name),
        "candidate_label_list": [str(item["candidate_label"]) for item in factor_candidate_list],
        "factor_count": int(len(factor_candidate_list)),
        "candidate_weight_dict": {
            str(candidate_label): float(weight_value)
            for candidate_label, weight_value in candidate_weight_dict.items()
        },
        "train_spearman_ic_mean": train_summary["mean"],
        "train_spearman_icir": train_summary["icir"],
    }
    if include_valid:
        valid_summary = build_spearman_metric_summary([metric_dict["spearman_ic"] for metric_dict in valid_metric_list], ic_aggregation_config=ic_aggregation_config)
        summary["valid_spearman_ic_mean"] = valid_summary["mean"]
        summary["valid_spearman_icir"] = valid_summary["icir"]
    return summary


def select_best_combination_method_summary(summary_list):
    summary_list = [dict(summary) for summary in summary_list]
    if len(summary_list) == 0:
        return None
    method_priority_dict = {
        "equal_weight": 0,
        "train_icir_weighted": 1,
    }
    sorted_summary_list = sorted(
        summary_list,
        key=lambda summary: (
            -float(summary["valid_spearman_icir"]),
            -float(summary["valid_spearman_ic_mean"]),
            -float(summary["train_spearman_icir"]),
            int(method_priority_dict.get(str(summary["method_name"]), 999)),
        ),
    )
    return dict(sorted_summary_list[0])


def select_best_combination_trial_summary(summary_list):
    summary_list = [dict(summary) for summary in summary_list]
    if len(summary_list) == 0:
        return None
    sorted_summary_list = sorted(
        summary_list,
        key=lambda summary: (
            -float(summary["valid_spearman_icir"]),
            -float(summary["valid_spearman_ic_mean"]),
            -float(summary["train_spearman_icir"]),
            int(summary["trial_number"]),
        ),
    )
    return dict(sorted_summary_list[0])


def normalize_candidate_weight_dict(candidate_weight_dict, fallback_weight_dict):
    nonnegative_weight_dict = {
        str(candidate_label): max(float(weight_value), 0.0)
        for candidate_label, weight_value in candidate_weight_dict.items()
    }
    total_weight = float(sum(nonnegative_weight_dict.values()))
    if total_weight <= 1e-12:
        return {
            str(candidate_label): float(weight_value)
            for candidate_label, weight_value in fallback_weight_dict.items()
        }
    return {
        str(candidate_label): float(weight_value / total_weight)
        for candidate_label, weight_value in nonnegative_weight_dict.items()
    }


def build_weight_search_range_dict(base_weight_dict):
    return {
        str(candidate_label): {
            "low": float(max(0.0, float(weight_value) * 0.95)),
            "high": float(max(0.0, float(weight_value) * 1.05)),
        }
        for candidate_label, weight_value in base_weight_dict.items()
    }


def run_factor_combination_weight_tuning(
    factor_candidate_list,
    factor_series_dict,
    forward_return_series,
    fold_list,
    selected_method_summary,
    ic_aggregation_config,
):
    optuna_module = load_optuna_module()
    selected_method_summary = dict(selected_method_summary)
    base_weight_dict = {
        str(candidate_label): float(weight_value)
        for candidate_label, weight_value in dict(selected_method_summary["candidate_weight_dict"]).items()
    }
    weight_search_range_dict = build_weight_search_range_dict(base_weight_dict=base_weight_dict)
    train_trial_summary_list = []

    def objective(trial):
        raw_weight_dict = {}
        for factor_candidate in factor_candidate_list:
            candidate_label = str(factor_candidate["candidate_label"])
            weight_range = dict(weight_search_range_dict[candidate_label])
            raw_weight_dict[candidate_label] = float(
                trial.suggest_float(
                    f"{candidate_label}__weight",
                    float(weight_range["low"]),
                    float(weight_range["high"]),
                )
            )
        normalized_weight_dict = normalize_candidate_weight_dict(
            candidate_weight_dict=raw_weight_dict,
            fallback_weight_dict=base_weight_dict,
        )
        summary = evaluate_combination_summary(
            factor_candidate_list=factor_candidate_list,
            factor_series_dict=factor_series_dict,
            forward_return_series=forward_return_series,
            fold_list=fold_list,
            method_name=str(selected_method_summary["method_name"]),
            include_valid=False,
            candidate_weight_dict_override=normalized_weight_dict,
            ic_aggregation_config=ic_aggregation_config,
        )
        summary["trial_number"] = int(trial.number)
        summary["candidate_weight_dict"] = dict(normalized_weight_dict)
        train_trial_summary_list.append(dict(summary))
        return float(summary["train_spearman_icir"])

    study = optuna_module.create_study(
        direction="maximize",
        sampler=optuna_module.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=100)
    sorted_trial_summary_list = sorted(
        train_trial_summary_list,
        key=lambda summary: (
            -float(summary["train_spearman_icir"]),
            -float(summary["train_spearman_ic_mean"]),
            int(summary["trial_number"]),
        ),
    )
    top_train_trial_summary_list = [dict(summary) for summary in sorted_trial_summary_list[:50]]
    valid_trial_summary_list = []
    for train_summary in top_train_trial_summary_list:
        valid_summary = evaluate_combination_summary(
            factor_candidate_list=factor_candidate_list,
            factor_series_dict=factor_series_dict,
            forward_return_series=forward_return_series,
            fold_list=fold_list,
            method_name=str(selected_method_summary["method_name"]),
            include_valid=True,
            candidate_weight_dict_override=dict(train_summary["candidate_weight_dict"]),
            ic_aggregation_config=ic_aggregation_config,
        )
        valid_summary["trial_number"] = int(train_summary["trial_number"])
        valid_summary["candidate_weight_dict"] = dict(train_summary["candidate_weight_dict"])
        valid_trial_summary_list.append(valid_summary)
    best_tuned_trial_summary = select_best_combination_trial_summary(valid_trial_summary_list)
    return {
        "enabled": True,
        "selected_method": str(selected_method_summary["method_name"]),
        "n_trials": 100,
        "top_k_valid_eval_count": int(len(valid_trial_summary_list)),
        "base_weight_dict": base_weight_dict,
        "weight_search_range_dict": weight_search_range_dict,
        "best_tuned_trial_summary": best_tuned_trial_summary,
    }


def run_factor_combination(config_override=None):
    config = build_tradition_config(config_override=config_override)
    ic_aggregation_config = build_ic_aggregation_config(config)
    dedup_selection_path = config.get("dedup_selection_path")
    if dedup_selection_path is None:
        raise ValueError("factor_combination 模式必须提供 dedup_selection_path。")
    dedup_selection_input, resolved_dedup_selection_path = load_dedup_selection_input(dedup_selection_path)
    dedup_selection_output = dict(dedup_selection_input["dedup_selection_output"])
    best_final_selection_summary = dict(dedup_selection_output["best_final_selection_summary"])
    input_candidate_label_list = [str(candidate_label) for candidate_label in best_final_selection_summary["candidate_label_list"]]
    if len(input_candidate_label_list) == 0:
        raise ValueError("dedup 结果中的 best_final_selection_summary 为空组合。")
    fund_code = resolve_fund_code_from_dedup_selection_input(
        dedup_selection_input=dedup_selection_input,
        dedup_selection_path=resolved_dedup_selection_path,
    )

    raw_data = fetch_fund_data_with_cache(
        code_dict=resolve_effective_code_dict(config),
        cache_dir=config["data_dir"],
        force_refresh=bool(config["force_refresh"]),
        cache_prefix=config["cache_prefix"],
    )
    normalized_data = normalize_fund_data(raw_data)
    fund_df = filter_single_fund(normalized_data, fund_code=fund_code)
    price_series, data_mode = adapt_to_price_series(fund_df=fund_df)
    fold_list = build_walk_forward_dev_fold_list(
        price_series=price_series,
        walk_forward_config=dict(config["walk_forward_config"]),
        split_config=config["data_split_dict"],
    )
    base_multi_factor_params = dict(config["strategy_param_dict"]["multi_factor_score"])
    forward_return_series = build_forward_return_series(price_series=price_series, forward_window=5)

    candidate_record_lookup = {
        str(candidate_label): dict(record)
        for candidate_label, record in dict(dedup_selection_output["record_dict"]).items()
    }
    factor_candidate_list = [dict(candidate_record_lookup[candidate_label]) for candidate_label in input_candidate_label_list]
    factor_series_dict = {}
    candidate_train_icir_dict = {}
    for factor_candidate in factor_candidate_list:
        candidate_label = str(factor_candidate["candidate_label"])
        factor_series_dict[candidate_label] = build_single_factor_series(
            price_series=price_series,
            factor_name=factor_candidate["factor_name"],
            strategy_params=base_multi_factor_params,
            factor_param_override=dict(factor_candidate["factor_param_dict"]),
        )
        candidate_train_icir_dict[candidate_label] = float(candidate_record_lookup[candidate_label]["train_spearman_icir"])

    combination_summary_list = [
        evaluate_combination_summary(
            factor_candidate_list=factor_candidate_list,
            factor_series_dict=factor_series_dict,
            forward_return_series=forward_return_series,
            fold_list=fold_list,
            method_name="equal_weight",
            candidate_train_icir_dict=candidate_train_icir_dict,
            include_valid=True,
            ic_aggregation_config=ic_aggregation_config,
        ),
        evaluate_combination_summary(
            factor_candidate_list=factor_candidate_list,
            factor_series_dict=factor_series_dict,
            forward_return_series=forward_return_series,
            fold_list=fold_list,
            method_name="train_icir_weighted",
            candidate_train_icir_dict=candidate_train_icir_dict,
            include_valid=True,
            ic_aggregation_config=ic_aggregation_config,
        ),
    ]
    selected_method_summary = select_best_combination_method_summary(combination_summary_list)
    combination_compare_output = {
        "candidate_label_list": input_candidate_label_list,
        "equal_weight_summary": next(summary for summary in combination_summary_list if summary["method_name"] == "equal_weight"),
        "train_icir_weighted_summary": next(summary for summary in combination_summary_list if summary["method_name"] == "train_icir_weighted"),
        "selected_method": str(selected_method_summary["method_name"]),
    }
    weight_tuning_output = run_factor_combination_weight_tuning(
        factor_candidate_list=factor_candidate_list,
        factor_series_dict=factor_series_dict,
        forward_return_series=forward_return_series,
        fold_list=fold_list,
        selected_method_summary=selected_method_summary,
        ic_aggregation_config=ic_aggregation_config,
    )
    best_tuned_trial_summary = dict(weight_tuning_output["best_tuned_trial_summary"])
    best_combination_selection_summary = {
        "candidate_label_list": list(best_tuned_trial_summary["candidate_label_list"]),
        "selected_method": str(selected_method_summary["method_name"]),
        "candidate_weight_dict": dict(best_tuned_trial_summary["candidate_weight_dict"]),
        "train_spearman_ic_mean": float(best_tuned_trial_summary["train_spearman_ic_mean"]),
        "train_spearman_icir": float(best_tuned_trial_summary["train_spearman_icir"]),
        "valid_spearman_ic_mean": float(best_tuned_trial_summary["valid_spearman_ic_mean"]),
        "valid_spearman_icir": float(best_tuned_trial_summary["valid_spearman_icir"]),
        "trial_number": int(best_tuned_trial_summary["trial_number"]),
    }
    factor_candidate_record_dict = {
        str(factor_candidate["candidate_label"]): {
            "candidate_label": str(factor_candidate["candidate_label"]),
            "factor_name": str(factor_candidate["factor_name"]),
            "factor_group": str(factor_candidate["factor_group"]),
            "factor_param_dict": dict(factor_candidate["factor_param_dict"]),
        }
        for factor_candidate in factor_candidate_list
    }
    factor_combination_output = {
        "fund_code": fund_code,
        "dedup_selection_path": str(resolved_dedup_selection_path),
        "analysis_date": datetime.today().strftime("%Y-%m-%d"),
        "input_candidate_label_list": input_candidate_label_list,
        "ic_aggregation_config": dict(ic_aggregation_config),
        "factor_candidate_record_dict": factor_candidate_record_dict,
        "combination_compare_output": combination_compare_output,
        "weight_tuning_output": weight_tuning_output,
        "best_combination_selection_summary": best_combination_selection_summary,
    }
    summary_path = save_factor_combination_output(
        dedup_selection_input=dedup_selection_input,
        factor_combination_output=factor_combination_output,
        output_dir=config["output_dir"],
        fund_code=fund_code,
    )
    result = {
        "fund_code": fund_code,
        "data_mode": data_mode,
        "dedup_selection_path": str(resolved_dedup_selection_path),
        "input_candidate_label_list": input_candidate_label_list,
        "combination_compare_output": combination_compare_output,
        "weight_tuning_output": weight_tuning_output,
        "best_combination_selection_summary": best_combination_selection_summary,
        "summary_path": summary_path,
    }
    print_factor_combination_summary(result)
    return result
