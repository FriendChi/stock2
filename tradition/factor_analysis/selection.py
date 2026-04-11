from datetime import datetime

import pandas as pd

from tradition.config import build_tradition_config
from tradition.data_adapter import adapt_to_price_series
from tradition.data_fetcher import fetch_fund_data_with_cache
from tradition.data_loader import filter_single_fund, normalize_fund_data
from tradition.factor_engine import build_single_factor_series, resolve_factor_name_list_by_group
from tradition.splitter import build_walk_forward_fold_list

from .common import (
    build_candidate_record_dict,
    build_factor_candidate_list,
    build_forward_return_series,
    compute_segment_correlation_metrics,
    build_metric_summary,
    build_positive_ic_ratio,
)
from .io import print_factor_selection_summary, save_factor_selection_table


def build_factor_selection_record(factor_candidate, train_metric_list, valid_metric_list, threshold_config):
    train_spearman_summary = build_metric_summary([metric_dict["spearman_ic"] for metric_dict in train_metric_list])
    train_pearson_summary = build_metric_summary([metric_dict["pearson_ic"] for metric_dict in train_metric_list])
    valid_spearman_summary = build_metric_summary([metric_dict["spearman_ic"] for metric_dict in valid_metric_list])
    valid_pearson_summary = build_metric_summary([metric_dict["pearson_ic"] for metric_dict in valid_metric_list])
    train_spearman_positive_ic_ratio = build_positive_ic_ratio([metric_dict["spearman_ic"] for metric_dict in train_metric_list])
    valid_spearman_positive_ic_ratio = build_positive_ic_ratio([metric_dict["spearman_ic"] for metric_dict in valid_metric_list])

    train_passed = (
        train_spearman_summary["count"] > 0
        and train_spearman_summary["mean"] >= float(threshold_config["train_min_spearman_ic"])
        and train_spearman_summary["icir"] >= float(threshold_config["train_min_spearman_icir"])
        and train_spearman_positive_ic_ratio > 0.5
    )
    valid_passed = (
        valid_spearman_summary["count"] > 0
        and valid_spearman_summary["mean"] > 0.0
        and valid_spearman_summary["icir"] > 0.0
        and valid_spearman_positive_ic_ratio > 0.5
    )
    return {
        "factor_name": str(factor_candidate["factor_name"]),
        "factor_group": str(factor_candidate["factor_group"]),
        "factor_param_dict": dict(factor_candidate["param_dict"]),
        "candidate_label": str(factor_candidate["candidate_label"]),
        "train_sample_fold_count": train_spearman_summary["count"],
        "train_spearman_ic_mean": train_spearman_summary["mean"],
        "train_spearman_ic_std": train_spearman_summary["std"],
        "train_spearman_icir": train_spearman_summary["icir"],
        "train_spearman_positive_ic_ratio": train_spearman_positive_ic_ratio,
        "train_pearson_ic_mean": train_pearson_summary["mean"],
        "train_pearson_ic_std": train_pearson_summary["std"],
        "train_pearson_icir": train_pearson_summary["icir"],
        "train_passed": bool(train_passed),
        "valid_sample_fold_count": valid_spearman_summary["count"],
        "valid_spearman_ic_mean": valid_spearman_summary["mean"],
        "valid_spearman_ic_std": valid_spearman_summary["std"],
        "valid_spearman_icir": valid_spearman_summary["icir"],
        "valid_spearman_positive_ic_ratio": valid_spearman_positive_ic_ratio,
        "valid_pearson_ic_mean": valid_pearson_summary["mean"],
        "valid_pearson_ic_std": valid_pearson_summary["std"],
        "valid_pearson_icir": valid_pearson_summary["icir"],
        "valid_passed": bool(valid_passed),
    }


def run_factor_selection_single_fund(config_override=None):
    config = build_tradition_config(config_override=config_override)
    factor_group_list = list(config.get("factor_group_list", []))
    if len(factor_group_list) == 0:
        raise ValueError("factor_select 模式必须提供 factor_group_list。")
    threshold_config = {
        "train_min_spearman_ic": float(config.get("train_min_spearman_ic", 0.0)),
        "train_min_spearman_icir": float(config.get("train_min_spearman_icir", 0.0)),
    }

    raw_data = fetch_fund_data_with_cache(
        code_dict=config["code_dict"],
        cache_dir=config["data_dir"],
        force_refresh=bool(config["force_refresh"]),
        cache_prefix=config["cache_prefix"],
    )
    normalized_data = normalize_fund_data(raw_data)
    fund_code = str(config["default_fund_code"]).zfill(6)
    fund_df = filter_single_fund(normalized_data, fund_code=fund_code)
    price_series, data_mode = adapt_to_price_series(fund_df=fund_df)

    candidate_factor_name_list = resolve_factor_name_list_by_group(factor_group_list=factor_group_list)
    fold_list = build_walk_forward_fold_list(
        price_series=price_series,
        walk_forward_config=dict(config["walk_forward_config"]),
        split_config=config["data_split_dict"],
    )
    multi_factor_params = dict(config["strategy_param_dict"]["multi_factor_score"])
    candidate_factor_list = build_factor_candidate_list(
        candidate_factor_name_list=candidate_factor_name_list,
        strategy_params=multi_factor_params,
    )
    forward_return_series = build_forward_return_series(price_series=price_series, forward_window=5)

    factor_record_list = []
    for factor_candidate in candidate_factor_list:
        factor_series = build_single_factor_series(
            price_series=price_series,
            factor_name=factor_candidate["factor_name"],
            strategy_params=multi_factor_params,
            factor_param_override=factor_candidate["param_dict"],
        )

        train_metric_list = []
        valid_metric_list = []
        for fold_dict in fold_list:
            train_metric_list.append(
                compute_segment_correlation_metrics(
                    factor_series=factor_series,
                    forward_return_series=forward_return_series,
                    segment_index=fold_dict["train"].index,
                )
            )
            valid_metric_list.append(
                compute_segment_correlation_metrics(
                    factor_series=factor_series,
                    forward_return_series=forward_return_series,
                    segment_index=fold_dict["valid"].index,
                )
            )

        factor_record_list.append(
            build_factor_selection_record(
                factor_candidate=factor_candidate,
                train_metric_list=train_metric_list,
                valid_metric_list=valid_metric_list,
                threshold_config=threshold_config,
            )
        )

    summary_df = pd.DataFrame(factor_record_list)
    summary_df = summary_df.sort_values(
        ["train_passed", "valid_passed", "valid_spearman_icir", "valid_spearman_ic_mean", "train_spearman_icir", "candidate_label"],
        ascending=[False, False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    selected_mask = summary_df["train_passed"].astype(bool) & summary_df["valid_passed"].astype(bool)
    summary_df["final_rank"] = pd.Series([None] * len(summary_df), dtype=object)
    if bool(selected_mask.any()):
        summary_df.loc[selected_mask, "final_rank"] = list(range(1, int(selected_mask.sum()) + 1))
    summary_df["selected"] = selected_mask
    selected_summary_df = summary_df[selected_mask].reset_index(drop=True)
    factor_selection_output = {
        "fund_code": fund_code,
        "data_mode": data_mode,
        "analysis_date": datetime.today().strftime("%Y-%m-%d"),
        "factor_group_list": factor_group_list,
        "candidate_factor_name_list": candidate_factor_name_list,
        "selected_factor_name_list": selected_summary_df["factor_name"].tolist(),
        "selected_candidate_label_list": selected_summary_df["candidate_label"].tolist(),
        "threshold_config": threshold_config,
        "candidate_count": int(len(summary_df)),
        "selected_count": int(selected_mask.sum()),
        "record_dict": build_candidate_record_dict(
            summary_df=selected_summary_df,
            keep_field_list=[
                "candidate_label",
                "factor_group",
                "factor_name",
                "factor_param_dict",
                "selected",
            ],
        ),
    }
    summary_path = save_factor_selection_table(
        factor_selection_output=factor_selection_output,
        output_dir=config["output_dir"],
        fund_code=fund_code,
    )
    result = {
        "fund_code": fund_code,
        "data_mode": data_mode,
        "factor_group_list": factor_group_list,
        "candidate_factor_name_list": candidate_factor_name_list,
        "candidate_factor_list": candidate_factor_list,
        "selected_factor_name_list": selected_summary_df["factor_name"].tolist(),
        "selected_candidate_label_list": selected_summary_df["candidate_label"].tolist(),
        "threshold_config": threshold_config,
        "fold_list": fold_list,
        "factor_selection_output": factor_selection_output,
        "summary_df": summary_df,
        "selected_summary_df": selected_summary_df,
        "summary_path": summary_path,
    }
    print_factor_selection_summary(result)
    return result
