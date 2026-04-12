from datetime import datetime

import pandas as pd

from tradition.config import build_tradition_config, resolve_effective_code_dict
from tradition.data_adapter import adapt_to_price_series
from tradition.data_fetcher import fetch_fund_data_with_cache
from tradition.data_loader import filter_single_fund, normalize_fund_data
from tradition.splitter import build_walk_forward_dev_fold_list

from .common import (
    build_candidate_record_dict,
    build_forward_return_series,
    build_ic_aggregation_config,
    load_feature_preprocess_bundle,
    load_preprocess_price_series,
    compute_segment_correlation_metrics,
    build_metric_summary,
    build_positive_ic_ratio,
    build_spearman_metric_summary,
)
from .io import allocate_stage_csv_output_path, print_factor_selection_summary, save_factor_selection_table


def build_factor_selection_record(factor_candidate, train_metric_list, valid_metric_list, threshold_config, ic_aggregation_config):
    train_spearman_summary = build_spearman_metric_summary([metric_dict["spearman_ic"] for metric_dict in train_metric_list], ic_aggregation_config=ic_aggregation_config)
    train_pearson_summary = build_metric_summary([metric_dict["pearson_ic"] for metric_dict in train_metric_list])
    valid_spearman_summary = build_spearman_metric_summary([metric_dict["spearman_ic"] for metric_dict in valid_metric_list], ic_aggregation_config=ic_aggregation_config)
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


def run_data_preprocess_single_fund(config_override=None):
    config = build_tradition_config(config_override=config_override)
    fund_code = str(config["default_fund_code"]).zfill(6)
    raw_data = fetch_fund_data_with_cache(
        code_dict=resolve_effective_code_dict(config),
        cache_dir=config["data_dir"],
        force_refresh=bool(config["force_refresh"]),
        cache_prefix=config["cache_prefix"],
    )
    normalized_data = normalize_fund_data(raw_data)
    fund_df = filter_single_fund(normalized_data, fund_code=fund_code)
    price_series, data_mode = adapt_to_price_series(fund_df=fund_df)
    output_path, _ = allocate_stage_csv_output_path(
        output_dir=config["output_dir"],
        output_prefix="data_preprocess",
        fund_code=fund_code,
    )
    preprocess_df = pd.DataFrame(
        {
            "date": pd.to_datetime(price_series.index).strftime("%Y-%m-%d"),
            "price": price_series.to_numpy(dtype=float),
            "fund_code": fund_code,
            "data_mode": str(data_mode),
        }
    )
    preprocess_df.to_csv(output_path, index=False)
    result = {
        "fund_code": fund_code,
        "data_mode": data_mode,
        "record_count": int(len(preprocess_df)),
        "summary_path": output_path,
    }
    print("数据预处理结果:")
    print("基金代码:", result["fund_code"])
    print("数据模式:", result["data_mode"])
    print("记录数:", result["record_count"])
    print("输出:", result["summary_path"])
    return result


def run_factor_selection_single_fund(config_override=None):
    config = build_tradition_config(config_override=config_override)
    if bool(config.get("force_refresh", False)):
        raise ValueError("factor_select 流程禁止 --force-refresh，请先运行流程0 data-preprocess。")
    preprocess_path = config.get("preprocess_path")
    if preprocess_path is None:
        raise ValueError("factor_select 模式必须提供 preprocess_path（流程0输出）。")
    preprocess_metadata_path = config.get("preprocess_metadata_path")
    if preprocess_metadata_path is None:
        raise ValueError("factor_select 模式必须提供 preprocess_metadata_path（流程0元信息输出）。")
    threshold_config = {
        "train_min_spearman_ic": float(config.get("train_min_spearman_ic", 0.0)),
        "train_min_spearman_icir": float(config.get("train_min_spearman_icir", 0.0)),
    }
    ic_aggregation_config = build_ic_aggregation_config(config)

    fund_code = str(config["default_fund_code"]).zfill(6)
    feature_df, feature_preprocess_output, resolved_fund_code, resolved_preprocess_path, resolved_preprocess_metadata_path = load_feature_preprocess_bundle(
        preprocess_path=preprocess_path,
        preprocess_metadata_path=preprocess_metadata_path,
        expected_fund_code=fund_code,
    )
    fund_code = resolved_fund_code
    data_mode = str(feature_preprocess_output.get("data_mode", "feature_matrix"))
    target_nav_column = str(feature_preprocess_output["target_nav_column"])
    candidate_factor_name_list = [str(column) for column in list(feature_preprocess_output.get("factor_feature_column_list", []))]
    if len(candidate_factor_name_list) == 0:
        raise ValueError("流程0输出中不存在可用因子特征列。")
    feature_df = feature_df.copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce")
    feature_df = feature_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    feature_df = feature_df.set_index("date")
    target_nav_series = pd.Series(feature_df[target_nav_column], copy=True).astype(float)
    fold_list = build_walk_forward_dev_fold_list(
        price_series=target_nav_series,
        walk_forward_config=dict(config["walk_forward_config"]),
        split_config=config["data_split_dict"],
    )
    candidate_factor_list = [
        {
            "factor_name": str(candidate_label),
            "factor_group": "factor_feature_zscore",
            "param_dict": {},
            "candidate_label": str(candidate_label),
        }
        for candidate_label in candidate_factor_name_list
    ]
    forward_return_series = build_forward_return_series(price_series=target_nav_series, forward_window=5)

    factor_record_list = []
    for factor_candidate in candidate_factor_list:
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

        factor_record_list.append(
            build_factor_selection_record(
                factor_candidate=factor_candidate,
                train_metric_list=train_metric_list,
                valid_metric_list=valid_metric_list,
                threshold_config=threshold_config,
                ic_aggregation_config=ic_aggregation_config,
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
        "preprocess_path": str(resolved_preprocess_path),
        "preprocess_metadata_path": str(resolved_preprocess_metadata_path),
        "analysis_date": datetime.today().strftime("%Y-%m-%d"),
        "target_nav_column": target_nav_column,
        "factor_group_list": [],
        "candidate_factor_name_list": candidate_factor_name_list,
        "selected_factor_name_list": selected_summary_df["factor_name"].tolist(),
        "selected_candidate_label_list": selected_summary_df["candidate_label"].tolist(),
        "threshold_config": threshold_config,
        "ic_aggregation_config": dict(ic_aggregation_config),
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
        inherited_path_code=feature_preprocess_output.get("path_code"),
        stage_index=1,
    )
    result = {
        "fund_code": fund_code,
        "data_mode": data_mode,
        "factor_group_list": [],
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
