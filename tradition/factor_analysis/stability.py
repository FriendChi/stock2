from datetime import datetime

import pandas as pd

from tradition.config import build_tradition_config
from tradition.factor_engine import build_single_factor_series
from tradition.splitter import build_walk_forward_dev_fold_list

from .common import (
    build_candidate_record_dict,
    build_forward_return_series,
    build_ic_aggregation_config,
    build_ic_flip_count,
    build_metric_summary,
    build_positive_ic_ratio,
    build_spearman_metric_summary,
    build_tail_reject_mask,
    build_trimmed_ic_mean,
    compute_segment_correlation_metrics,
    load_preprocess_price_series,
)
from .io import (
    load_factor_selection_input,
    print_single_factor_stability_analysis_summary,
    resolve_fund_code_from_factor_selection_input,
    save_single_factor_stability_analysis_output,
)


def build_single_factor_stability_record(factor_candidate, train_metric_list, valid_metric_list, ic_aggregation_config):
    train_spearman_value_list = [metric_dict["spearman_ic"] for metric_dict in train_metric_list]
    valid_spearman_value_list = [metric_dict["spearman_ic"] for metric_dict in valid_metric_list]
    train_spearman_summary = build_spearman_metric_summary(train_spearman_value_list, ic_aggregation_config=ic_aggregation_config)
    valid_spearman_summary = build_spearman_metric_summary(valid_spearman_value_list, ic_aggregation_config=ic_aggregation_config)
    train_trimmed_ic_mean = build_trimmed_ic_mean(train_spearman_value_list, trim_ratio=0.1)
    valid_trimmed_ic_mean = build_trimmed_ic_mean(valid_spearman_value_list, trim_ratio=0.1)
    return {
        "factor_name": str(factor_candidate["factor_name"]),
        "factor_group": str(factor_candidate["factor_group"]),
        "factor_param_dict": dict(factor_candidate["factor_param_dict"]),
        "candidate_label": str(factor_candidate["candidate_label"]),
        "train_sample_fold_count": train_spearman_summary["count"],
        "train_spearman_ic_mean": train_spearman_summary["mean"],
        "train_spearman_ic_std": train_spearman_summary["std"],
        "train_spearman_icir": train_spearman_summary["icir"],
        "train_positive_ic_ratio": build_positive_ic_ratio(train_spearman_value_list),
        "train_trimmed_spearman_ic_mean": train_trimmed_ic_mean,
        "train_trimmed_ic_gap": float(train_trimmed_ic_mean - train_spearman_summary["mean"]) if pd.notna(train_trimmed_ic_mean) and pd.notna(train_spearman_summary["mean"]) else float("nan"),
        "train_ic_flip_count": build_ic_flip_count(train_spearman_value_list),
        "valid_sample_fold_count": valid_spearman_summary["count"],
        "valid_spearman_ic_mean": valid_spearman_summary["mean"],
        "valid_spearman_ic_std": valid_spearman_summary["std"],
        "valid_spearman_icir": valid_spearman_summary["icir"],
        "valid_positive_ic_ratio": build_positive_ic_ratio(valid_spearman_value_list),
        "valid_trimmed_spearman_ic_mean": valid_trimmed_ic_mean,
        "valid_trimmed_ic_gap": float(valid_trimmed_ic_mean - valid_spearman_summary["mean"]) if pd.notna(valid_trimmed_ic_mean) and pd.notna(valid_spearman_summary["mean"]) else float("nan"),
        "valid_ic_flip_count": build_ic_flip_count(valid_spearman_value_list),
        "train_valid_ic_mean_gap": float(valid_spearman_summary["mean"] - train_spearman_summary["mean"]) if pd.notna(train_spearman_summary["mean"]) and pd.notna(valid_spearman_summary["mean"]) else float("nan"),
    }


def run_single_factor_stability_analysis(config_override=None):
    config = build_tradition_config(config_override=config_override)
    if bool(config.get("force_refresh", False)):
        raise ValueError("stability 流程禁止 --force-refresh，请先运行流程0 data-preprocess。")
    ic_aggregation_config = build_ic_aggregation_config(config)
    factor_selection_path = config.get("factor_selection_path")
    if factor_selection_path is None:
        raise ValueError("single_factor_stability_analysis 模式必须提供 factor_selection_path。")
    factor_selection_input, resolved_factor_selection_path = load_factor_selection_input(factor_selection_path)
    factor_selection_output = dict(factor_selection_input["factor_selection_output"])
    selected_factor_input_list = [
        record
        for record in factor_selection_output["record_dict"].values()
        if bool(record.get("selected", False))
    ]
    if len(selected_factor_input_list) == 0:
        raise ValueError("factor_select 结果中不存在 selected=true 的最终候选。")
    fund_code = resolve_fund_code_from_factor_selection_input(
        factor_selection_input=factor_selection_input,
        factor_selection_path=resolved_factor_selection_path,
    )
    preprocess_path = factor_selection_output.get("preprocess_path")
    if preprocess_path is None:
        raise ValueError("factor_select 结果缺少 preprocess_path，请重新执行流程1并提供流程0输出。")
    price_series, data_mode, _, resolved_preprocess_path = load_preprocess_price_series(
        preprocess_path=preprocess_path,
        expected_fund_code=fund_code,
    )
    fold_list = build_walk_forward_dev_fold_list(
        price_series=price_series,
        walk_forward_config=dict(config["walk_forward_config"]),
        split_config=config["data_split_dict"],
    )
    multi_factor_params = dict(config["strategy_param_dict"]["multi_factor_score"])
    forward_return_series = build_forward_return_series(price_series=price_series, forward_window=5)

    stability_record_list = []
    for factor_candidate in selected_factor_input_list:
        factor_series = build_single_factor_series(
            price_series=price_series,
            factor_name=factor_candidate["factor_name"],
            strategy_params=multi_factor_params,
            factor_param_override=dict(factor_candidate["factor_param_dict"]),
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
        stability_record_list.append(
            build_single_factor_stability_record(
                factor_candidate=factor_candidate,
                train_metric_list=train_metric_list,
                valid_metric_list=valid_metric_list,
                ic_aggregation_config=ic_aggregation_config,
            )
        )

    summary_df = pd.DataFrame(stability_record_list)
    tail_reject_flag_df = build_tail_reject_mask(summary_df=summary_df, reject_ratio=0.05)
    summary_df = pd.concat([summary_df, tail_reject_flag_df], axis=1)
    summary_df["abs_train_valid_ic_mean_gap"] = summary_df["train_valid_ic_mean_gap"].abs()
    summary_df["abs_valid_trimmed_ic_gap"] = summary_df["valid_trimmed_ic_gap"].abs()
    summary_df = summary_df.sort_values(
        [
            "stability_tail_rejected",
            "valid_spearman_ic_mean",
            "valid_spearman_icir",
            "abs_train_valid_ic_mean_gap",
            "valid_ic_flip_count",
            "abs_valid_trimmed_ic_gap",
            "candidate_label",
        ],
        ascending=[True, False, False, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    selected_mask = ~summary_df["stability_tail_rejected"].astype(bool)
    summary_df["stability_rank"] = pd.Series([None] * len(summary_df), dtype=object)
    if bool(selected_mask.any()):
        summary_df.loc[selected_mask, "stability_rank"] = list(range(1, int(selected_mask.sum()) + 1))
    summary_df["selected"] = selected_mask
    selected_summary_df = summary_df[selected_mask].reset_index(drop=True)
    stability_analysis_output = {
        "fund_code": fund_code,
        "preprocess_path": str(resolved_preprocess_path),
        "factor_selection_path": str(resolved_factor_selection_path),
        "analysis_date": datetime.today().strftime("%Y-%m-%d"),
        "candidate_count": int(len(summary_df)),
        "selected_count": int(selected_mask.sum()),
        "tail_reject_candidate_label_list": summary_df.loc[~selected_mask, "candidate_label"].tolist(),
        "selected_candidate_label_list": selected_summary_df["candidate_label"].tolist(),
        "ic_aggregation_config": dict(ic_aggregation_config),
        "record_dict": build_candidate_record_dict(
            summary_df=selected_summary_df,
            keep_field_list=[
                "candidate_label",
                "factor_group",
                "factor_name",
                "factor_param_dict",
                "selected",
                "train_spearman_icir",
                "valid_spearman_ic_mean",
                "valid_spearman_icir",
            ],
        ),
    }
    summary_path = save_single_factor_stability_analysis_output(
        factor_selection_input=factor_selection_input,
        resolved_factor_selection_path=resolved_factor_selection_path,
        stability_analysis_output=stability_analysis_output,
        output_dir=config["output_dir"],
        fund_code=fund_code,
    )
    result = {
        "fund_code": fund_code,
        "data_mode": data_mode,
        "factor_selection_path": str(resolved_factor_selection_path),
        "selected_factor_input_list": selected_factor_input_list,
        "summary_df": summary_df,
        "selected_summary_df": selected_summary_df,
        "selected_candidate_label_list": selected_summary_df["candidate_label"].tolist(),
        "summary_path": summary_path,
    }
    print_single_factor_stability_analysis_summary(result)
    return result
