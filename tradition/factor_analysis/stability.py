from datetime import datetime
from pathlib import Path

import pandas as pd

from tradition.config import build_tradition_config
from tradition.splitter import build_walk_forward_dev_fold_list

from .common import (
    build_candidate_record_dict,
    build_forward_return_series,
    build_ic_aggregation_config,
    build_ic_flip_count,
    build_positive_ic_ratio,
    build_spearman_metric_summary,
    build_tail_reject_mask,
    build_trimmed_ic_mean,
    compute_segment_correlation_metrics,
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


def load_selected_feature_matrix(factor_selection_output, selected_factor_input_list):
    preprocess_path = factor_selection_output.get("preprocess_path")
    if preprocess_path is None:
        raise ValueError("factor_select 结果缺少 preprocess_path，请重新执行流程1并提供流程0输出。")
    resolved_preprocess_path = Path(preprocess_path)
    if not resolved_preprocess_path.exists():
        raise FileNotFoundError(f"流程0特征 CSV 不存在: {resolved_preprocess_path}")
    target_nav_column = str(factor_selection_output.get("target_nav_column", "")).strip()
    if len(target_nav_column) == 0:
        raise ValueError("factor_select 结果缺少 target_nav_column，请重新执行流程1。")
    selected_candidate_label_list = [str(record["candidate_label"]) for record in selected_factor_input_list]
    feature_df = pd.read_csv(resolved_preprocess_path)
    required_column_list = ["date", target_nav_column] + selected_candidate_label_list
    missing_column_list = [column for column in required_column_list if column not in feature_df.columns]
    if len(missing_column_list) > 0:
        raise ValueError(f"流程0特征 CSV 缺少流程2必需列: {missing_column_list[:10]}")
    # 流程2直接消费流程1筛出的候选列，不再按旧式因子定义重建时序。
    feature_df = feature_df.copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce")
    feature_df = feature_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    feature_df = feature_df.set_index("date")
    return feature_df, target_nav_column, resolved_preprocess_path


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
    data_mode = str(factor_selection_output.get("data_mode", "feature_matrix"))
    feature_df, target_nav_column, resolved_preprocess_path = load_selected_feature_matrix(
        factor_selection_output=factor_selection_output,
        selected_factor_input_list=selected_factor_input_list,
    )
    target_nav_series = pd.Series(feature_df[target_nav_column], copy=True).astype(float)
    fold_list = build_walk_forward_dev_fold_list(
        price_series=target_nav_series,
        walk_forward_config=dict(config["walk_forward_config"]),
        split_config=config["data_split_dict"],
    )
    forward_return_series = build_forward_return_series(price_series=target_nav_series, forward_window=5)

    stability_record_list = []
    for factor_candidate in selected_factor_input_list:
        # 候选因子已经在流程0中固化为特征列，这里直接按 candidate_label 读取对应列。
        factor_series = pd.Series(feature_df[str(factor_candidate["candidate_label"])], copy=True).astype(float)
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
        "preprocess_metadata_path": factor_selection_output.get("preprocess_metadata_path"),
        "factor_selection_path": str(resolved_factor_selection_path),
        "analysis_date": datetime.today().strftime("%Y-%m-%d"),
        "target_nav_column": target_nav_column,
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
