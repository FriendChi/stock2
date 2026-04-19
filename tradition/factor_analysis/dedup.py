from datetime import datetime
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from tradition.config import build_tradition_config
from tradition.optimizer import load_optuna_module
from tradition.splitter import build_walk_forward_dev_fold_list

from .common import (
    build_candidate_record_dict,
    build_forward_return_series,
    build_ic_aggregation_config,
    build_instance_combination_score,
    build_metric_summary,
    build_spearman_metric_summary,
    compute_segment_correlation_metrics,
)
from .io import (
    load_stability_analysis_input,
    print_single_factor_dedup_selection_summary,
    resolve_fund_code_from_stability_analysis_input,
    save_single_factor_dedup_selection_output,
)


def choose_weaker_candidate(left_record, right_record):
    compare_key_list = [
        ("train_spearman_icir", True),
        ("valid_spearman_icir", True),
        ("valid_spearman_ic_mean", True),
    ]
    for field_name, higher_is_better in compare_key_list:
        left_value = left_record.get(field_name)
        right_value = right_record.get(field_name)
        if pd.isna(left_value) and pd.isna(right_value):
            continue
        if pd.isna(left_value):
            return str(left_record["candidate_label"])
        if pd.isna(right_value):
            return str(right_record["candidate_label"])
        if float(left_value) == float(right_value):
            continue
        if higher_is_better:
            return str(left_record["candidate_label"]) if float(left_value) < float(right_value) else str(right_record["candidate_label"])
    return max(str(left_record["candidate_label"]), str(right_record["candidate_label"]))


def compute_pair_train_corr(left_factor_series, right_factor_series, fold_list):
    corr_value_list = []
    for fold_dict in fold_list:
        aligned_df = pd.concat(
            [
                pd.Series(left_factor_series, copy=True).reindex(fold_dict["train"].index).rename("left"),
                pd.Series(right_factor_series, copy=True).reindex(fold_dict["train"].index).rename("right"),
            ],
            axis=1,
        ).dropna()
        if len(aligned_df) < 2:
            continue
        corr_value = aligned_df["left"].corr(aligned_df["right"], method="pearson")
        if pd.notna(corr_value):
            corr_value_list.append(float(corr_value))
    if len(corr_value_list) == 0:
        return float("nan")
    return float(pd.Series(corr_value_list, dtype=float).mean())


def _build_mean_train_corr_matrix(candidate_label_list, factor_series_dict, fold_list):
    factor_matrix_df = pd.concat(
        [
            pd.Series(factor_series_dict[str(candidate_label)], copy=True).rename(str(candidate_label))
            for candidate_label in candidate_label_list
        ],
        axis=1,
    )
    factor_matrix_df = factor_matrix_df.reindex(columns=list(candidate_label_list))
    pairwise_corr_sum_matrix = np.zeros((len(candidate_label_list), len(candidate_label_list)), dtype=float)
    pairwise_corr_count_matrix = np.zeros((len(candidate_label_list), len(candidate_label_list)), dtype=int)

    # 按 train fold 分别计算整张相关矩阵，再按当前流程3口径对各 fold 结果做简单均值。
    for fold_dict in fold_list:
        train_factor_df = factor_matrix_df.reindex(fold_dict["train"].index)
        if len(train_factor_df) < 2:
            continue
        corr_df = train_factor_df.corr(method="pearson", min_periods=2).reindex(
            index=list(candidate_label_list),
            columns=list(candidate_label_list),
        )
        corr_matrix = corr_df.to_numpy(dtype=float, copy=False)
        valid_mask = ~np.isnan(corr_matrix)
        pairwise_corr_sum_matrix[valid_mask] += corr_matrix[valid_mask]
        pairwise_corr_count_matrix[valid_mask] += 1

    mean_train_corr_matrix = np.full((len(candidate_label_list), len(candidate_label_list)), float("nan"), dtype=float)
    valid_pair_mask = pairwise_corr_count_matrix > 0
    mean_train_corr_matrix[valid_pair_mask] = (
        pairwise_corr_sum_matrix[valid_pair_mask] / pairwise_corr_count_matrix[valid_pair_mask]
    )
    return mean_train_corr_matrix


def build_corr_dedup_result(selected_summary_df, factor_series_dict, fold_list, corr_threshold=0.90, drop_ratio=0.10, min_drop_count=2):
    summary_df = pd.DataFrame(selected_summary_df, copy=True).reset_index(drop=True)
    if len(summary_df) == 0:
        return summary_df, []
    corr_threshold = float(corr_threshold)
    drop_ratio = float(drop_ratio)
    min_drop_count = int(min_drop_count)
    loss_count_dict = {str(label): 0 for label in summary_df["candidate_label"].tolist()}
    mean_train_corr_max_dict = {str(label): float("nan") for label in summary_df["candidate_label"].tolist()}
    record_by_label = {
        str(record["candidate_label"]): record
        for record in summary_df.to_dict(orient="records")
    }
    candidate_label_list = [str(candidate_label) for candidate_label in summary_df["candidate_label"].tolist()]
    mean_train_corr_matrix = _build_mean_train_corr_matrix(
        candidate_label_list=candidate_label_list,
        factor_series_dict=factor_series_dict,
        fold_list=fold_list,
    )
    total_pair_count = int(len(candidate_label_list) * (len(candidate_label_list) - 1) // 2)
    # 相关值已按 fold 聚合完毕，这里只做现有去冗余规则扫描，并增加终端进度条反馈。
    progress_bar = tqdm(
        total=total_pair_count,
        desc="corr dedup",
        unit="pair",
        disable=not sys.stderr.isatty(),
    )
    try:
        for left_index, right_index in combinations(range(len(candidate_label_list)), 2):
            progress_bar.update(1)
            left_label = candidate_label_list[left_index]
            right_label = candidate_label_list[right_index]
            mean_train_corr = mean_train_corr_matrix[left_index, right_index]
            if pd.isna(mean_train_corr):
                continue
            abs_mean_train_corr = abs(float(mean_train_corr))
            for candidate_label in [left_label, right_label]:
                current_max_corr = mean_train_corr_max_dict[candidate_label]
                if pd.isna(current_max_corr) or abs_mean_train_corr > float(current_max_corr):
                    mean_train_corr_max_dict[candidate_label] = abs_mean_train_corr
            if abs_mean_train_corr < corr_threshold:
                continue
            weaker_candidate_label = choose_weaker_candidate(
                left_record=record_by_label[left_label],
                right_record=record_by_label[right_label],
            )
            loss_count_dict[weaker_candidate_label] += 1
    finally:
        progress_bar.close()

    summary_df["corr_loss_count"] = summary_df["candidate_label"].map(loss_count_dict).astype(int)
    summary_df["mean_train_corr_max"] = summary_df["candidate_label"].map(mean_train_corr_max_dict)
    summary_df["corr_dedup_drop_reason"] = None
    summary_df["corr_dedup_selected"] = True

    loss_summary_df = summary_df[summary_df["corr_loss_count"] > 0].copy()
    drop_count = max(min_drop_count, int(len(summary_df) * drop_ratio))
    drop_count = min(drop_count, int(len(loss_summary_df)))
    if drop_count <= 0 or len(loss_summary_df) == 0:
        return summary_df, []

    loss_summary_df = loss_summary_df.sort_values(
        ["corr_loss_count", "train_spearman_icir", "valid_spearman_icir", "candidate_label"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    dropped_candidate_label_list = loss_summary_df.head(drop_count)["candidate_label"].tolist()
    dropped_candidate_label_set = set(dropped_candidate_label_list)
    summary_df.loc[summary_df["candidate_label"].isin(dropped_candidate_label_set), "corr_dedup_selected"] = False
    summary_df.loc[
        summary_df["candidate_label"].isin(dropped_candidate_label_set),
        "corr_dedup_drop_reason",
    ] = "high_train_corr_lower_train_icir"
    return summary_df, dropped_candidate_label_list


def evaluate_factor_candidate_subset(factor_candidate_list, factor_series_dict, forward_return_series, fold_list, include_valid=True, ic_aggregation_config=None):
    _, score_series = build_instance_combination_score(
        factor_candidate_list=factor_candidate_list,
        factor_series_dict=factor_series_dict,
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
    summary_dict = {
        "candidate_label_list": [str(item["candidate_label"]) for item in factor_candidate_list],
        "factor_count": int(len(factor_candidate_list)),
        "train_spearman_ic_mean": train_summary["mean"],
        "train_spearman_icir": train_summary["icir"],
    }
    if include_valid:
        valid_summary = build_spearman_metric_summary([metric_dict["spearman_ic"] for metric_dict in valid_metric_list], ic_aggregation_config=ic_aggregation_config)
        summary_dict["valid_spearman_ic_mean"] = valid_summary["mean"]
        summary_dict["valid_spearman_icir"] = valid_summary["icir"]
    return summary_dict


def build_candidate_label_signature(candidate_label_list):
    return tuple(sorted([str(candidate_label) for candidate_label in candidate_label_list]))


def load_selected_feature_matrix(stability_analysis_output, selected_stability_candidate_list):
    preprocess_path = stability_analysis_output.get("preprocess_path")
    if preprocess_path is None:
        raise ValueError("stability 结果缺少 preprocess_path，请重新执行流程2。")
    resolved_preprocess_path = Path(preprocess_path)
    if not resolved_preprocess_path.exists():
        raise FileNotFoundError(f"流程0特征 CSV 不存在: {resolved_preprocess_path}")
    target_nav_column = str(stability_analysis_output.get("target_nav_column", "")).strip()
    if len(target_nav_column) == 0:
        raise ValueError("stability 结果缺少 target_nav_column，请重新执行流程2。")
    selected_candidate_label_list = [str(record["candidate_label"]) for record in selected_stability_candidate_list]
    feature_df = pd.read_csv(resolved_preprocess_path)
    required_column_list = ["date", target_nav_column] + selected_candidate_label_list
    missing_column_list = [column for column in required_column_list if column not in feature_df.columns]
    if len(missing_column_list) > 0:
        raise ValueError(f"流程0特征 CSV 缺少流程3必需列: {missing_column_list[:10]}")
    # 流程3直接消费流程2筛出的特征列，不再按旧式因子定义重建时序。
    feature_df = feature_df.copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce")
    feature_df = feature_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    feature_df = feature_df.set_index("date")
    return feature_df, target_nav_column, resolved_preprocess_path


def run_train_forward_selection(candidate_record_list, factor_series_dict, forward_return_series, fold_list, root_topk=3, ic_aggregation_config=None):
    candidate_record_list = list(candidate_record_list)
    if len(candidate_record_list) == 0:
        return []
    root_topk = int(root_topk)
    if root_topk <= 0:
        raise ValueError("root_topk 必须为正整数。")
    sorted_candidate_record_list = sorted(
        candidate_record_list,
        key=lambda record: (
            -float(record["train_spearman_icir"]),
            -float(record["valid_spearman_icir"]),
            -float(record["valid_spearman_ic_mean"]),
            str(record["candidate_label"]),
        ),
    )
    path_summary_dict = {}

    frontier_node_list = []
    for root_candidate_record in sorted_candidate_record_list[:root_topk]:
        root_candidate_list = [dict(root_candidate_record)]
        root_summary = evaluate_factor_candidate_subset(
            factor_candidate_list=root_candidate_list,
            factor_series_dict=factor_series_dict,
            forward_return_series=forward_return_series,
            fold_list=fold_list,
            include_valid=False,
            ic_aggregation_config=ic_aggregation_config,
        )
        root_summary["step"] = 1
        root_signature = build_candidate_label_signature(root_summary["candidate_label_list"])
        path_summary_dict[root_signature] = root_summary
        frontier_node_list.append(
            {
                "candidate_record_list": root_candidate_list,
                "summary": root_summary,
            }
        )

    progress_bar = tqdm(
        total=0,
        desc="forward search",
        unit="candidate",
        disable=not sys.stderr.isatty(),
    )
    try:
        while len(frontier_node_list) > 0:
            # 每一层都要拿 frontier 中每条路径去尝试全量候选扩展，因此总量按层动态补齐。
            progress_bar.total += int(len(frontier_node_list) * len(sorted_candidate_record_list))
            progress_bar.refresh()
            next_frontier_node_list = []
            for frontier_node in frontier_node_list:
                parent_candidate_list = list(frontier_node["candidate_record_list"])
                parent_summary = dict(frontier_node["summary"])
                parent_candidate_label_set = set(parent_summary["candidate_label_list"])
                for candidate_record in sorted_candidate_record_list:
                    progress_bar.update(1)
                    candidate_label = str(candidate_record["candidate_label"])
                    if candidate_label in parent_candidate_label_set:
                        continue
                    child_candidate_list = parent_candidate_list + [dict(candidate_record)]
                    child_signature = build_candidate_label_signature([item["candidate_label"] for item in child_candidate_list])
                    if child_signature in path_summary_dict:
                        continue
                    child_summary = evaluate_factor_candidate_subset(
                        factor_candidate_list=child_candidate_list,
                        factor_series_dict=factor_series_dict,
                        forward_return_series=forward_return_series,
                        fold_list=fold_list,
                        include_valid=False,
                        ic_aggregation_config=ic_aggregation_config,
                    )
                    child_summary["step"] = int(len(child_candidate_list))
                    if pd.isna(child_summary["train_spearman_icir"]) or pd.isna(parent_summary["train_spearman_icir"]):
                        continue
                    if float(child_summary["train_spearman_icir"]) < float(parent_summary["train_spearman_icir"]):
                        continue
                    path_summary_dict[child_signature] = child_summary
                    next_frontier_node_list.append(
                        {
                            "candidate_record_list": child_candidate_list,
                            "summary": child_summary,
                        }
                    )
            frontier_node_list = next_frontier_node_list
    finally:
        progress_bar.close()

    path_summary_list = list(path_summary_dict.values())
    path_summary_list = sorted(
        path_summary_list,
        key=lambda summary: (
            int(summary["factor_count"]),
            tuple(summary["candidate_label_list"]),
        ),
    )
    return path_summary_list


def select_top_train_path_summary_list(train_path_summary_list, top_ratio=0.5):
    train_path_summary_list = list(train_path_summary_list)
    if len(train_path_summary_list) == 0:
        return []
    top_count = max(1, int(len(train_path_summary_list) * float(top_ratio)))
    sorted_path_summary_list = sorted(
        train_path_summary_list,
        key=lambda summary: (
            -float(summary["train_spearman_icir"]),
            -float(summary["train_spearman_ic_mean"]),
            int(summary["factor_count"]),
            int(summary["step"]),
            tuple(summary["candidate_label_list"]),
        ),
    )
    return [dict(summary) for summary in sorted_path_summary_list[:top_count]]


def evaluate_valid_for_path_summary_list(path_summary_list, candidate_record_lookup, factor_series_dict, forward_return_series, fold_list, ic_aggregation_config=None, progress_desc=None):
    evaluated_summary_list = []
    progress_bar = None
    if progress_desc is not None:
        progress_bar = tqdm(
            total=len(path_summary_list),
            desc=str(progress_desc),
            unit="path",
            disable=not sys.stderr.isatty(),
        )
    try:
        for path_summary in path_summary_list:
            factor_candidate_list = [dict(candidate_record_lookup[candidate_label]) for candidate_label in path_summary["candidate_label_list"]]
            evaluated_summary = evaluate_factor_candidate_subset(
                factor_candidate_list=factor_candidate_list,
                factor_series_dict=factor_series_dict,
                forward_return_series=forward_return_series,
                fold_list=fold_list,
                include_valid=True,
                ic_aggregation_config=ic_aggregation_config,
            )
            evaluated_summary["step"] = int(path_summary["step"])
            evaluated_summary_list.append(evaluated_summary)
            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return evaluated_summary_list


def select_best_forward_path_summary(forward_selection_path_summary):
    if len(forward_selection_path_summary) == 0:
        return None
    sorted_path_summary_list = sorted(
        forward_selection_path_summary,
        key=lambda summary: (
            -float(summary["valid_spearman_icir"]),
            -float(summary["valid_spearman_ic_mean"]),
            -float(summary["train_spearman_icir"]),
            int(summary["factor_count"]),
            int(summary["step"]),
        ),
    )
    return dict(sorted_path_summary_list[0])


def run_optuna_extension_search(
    baseline_summary,
    corr_selected_candidate_list,
    candidate_record_lookup,
    factor_series_dict,
    forward_return_series,
    fold_list,
    ic_aggregation_config=None,
):
    baseline_summary = dict(baseline_summary)
    baseline_candidate_label_list = [str(candidate_label) for candidate_label in baseline_summary["candidate_label_list"]]
    baseline_candidate_label_set = set(baseline_candidate_label_list)
    remaining_candidate_record_list = [
        dict(candidate_record)
        for candidate_record in corr_selected_candidate_list
        if str(candidate_record["candidate_label"]) not in baseline_candidate_label_set
    ]
    remaining_candidate_record_list = sorted(
        remaining_candidate_record_list,
        key=lambda record: str(record["candidate_label"]),
    )
    remaining_candidate_label_list = [str(record["candidate_label"]) for record in remaining_candidate_record_list]
    remaining_factor_count = int(len(remaining_candidate_record_list))
    if remaining_factor_count == 0:
        return {
            "enabled": False,
            "baseline_candidate_label_list": baseline_candidate_label_list,
            "baseline_train_spearman_icir": float(baseline_summary["train_spearman_icir"]),
            "baseline_valid_spearman_icir": float(baseline_summary["valid_spearman_icir"]),
            "remaining_candidate_label_list": remaining_candidate_label_list,
            "remaining_factor_count": remaining_factor_count,
            "n_trials": 0,
            "train_improved_candidate_count": 0,
            "train_improved_path_summary_list": [],
            "best_optuna_candidate_summary": None,
            "final_selected_source": "forward_selection",
            "best_final_selection_summary": dict(baseline_summary),
        }

    optuna_module = load_optuna_module()
    train_improved_summary_dict = {}
    baseline_train_icir = float(baseline_summary["train_spearman_icir"])

    def objective(trial):
        selected_candidate_label_list = list(baseline_candidate_label_list)
        for idx, candidate_record in enumerate(remaining_candidate_record_list):
            if int(trial.suggest_int(f"add_{idx}", 0, 1)) == 1:
                selected_candidate_label_list.append(str(candidate_record["candidate_label"]))
        selected_candidate_label_list = sorted(selected_candidate_label_list)
        factor_candidate_list = [dict(candidate_record_lookup[candidate_label]) for candidate_label in selected_candidate_label_list]
        summary = evaluate_factor_candidate_subset(
            factor_candidate_list=factor_candidate_list,
            factor_series_dict=factor_series_dict,
            forward_return_series=forward_return_series,
            fold_list=fold_list,
            include_valid=False,
            ic_aggregation_config=ic_aggregation_config,
        )
        summary["step"] = int(len(summary["candidate_label_list"]))
        if float(summary["train_spearman_icir"]) > baseline_train_icir:
            signature = build_candidate_label_signature(summary["candidate_label_list"])
            cached_summary = train_improved_summary_dict.get(signature)
            if cached_summary is None or float(summary["train_spearman_icir"]) > float(cached_summary["train_spearman_icir"]):
                train_improved_summary_dict[signature] = dict(summary)
        return float(summary["train_spearman_icir"])

    n_trials = int(remaining_factor_count ** 2)
    study = optuna_module.create_study(
        direction="maximize",
        sampler=optuna_module.samplers.TPESampler(seed=42),
    )
    progress_bar = tqdm(
        total=n_trials,
        desc="optuna extension",
        unit="trial",
        disable=not sys.stderr.isatty(),
    )
    try:
        def update_progress_bar(study, trial):
            progress_bar.update(1)

        study.optimize(objective, n_trials=n_trials, callbacks=[update_progress_bar])
    finally:
        progress_bar.close()

    train_improved_path_summary_list = sorted(
        train_improved_summary_dict.values(),
        key=lambda summary: (
            -float(summary["train_spearman_icir"]),
            -float(summary["train_spearman_ic_mean"]),
            int(summary["factor_count"]),
            tuple(summary["candidate_label_list"]),
        ),
    )
    valid_evaluated_summary_list = evaluate_valid_for_path_summary_list(
        path_summary_list=train_improved_path_summary_list,
        candidate_record_lookup=candidate_record_lookup,
        factor_series_dict=factor_series_dict,
        forward_return_series=forward_return_series,
        fold_list=fold_list,
        ic_aggregation_config=ic_aggregation_config,
        progress_desc="optuna valid",
    )
    best_optuna_candidate_summary = select_best_forward_path_summary(valid_evaluated_summary_list)
    if best_optuna_candidate_summary is None:
        return {
            "enabled": True,
            "baseline_candidate_label_list": baseline_candidate_label_list,
            "baseline_train_spearman_icir": baseline_train_icir,
            "baseline_valid_spearman_icir": float(baseline_summary["valid_spearman_icir"]),
            "remaining_candidate_label_list": remaining_candidate_label_list,
            "remaining_factor_count": remaining_factor_count,
            "n_trials": n_trials,
            "train_improved_candidate_count": 0,
            "train_improved_path_summary_list": [],
            "best_optuna_candidate_summary": None,
            "final_selected_source": "forward_selection",
            "best_final_selection_summary": dict(baseline_summary),
        }

    best_final_selection_summary = select_best_forward_path_summary(
        [dict(baseline_summary), dict(best_optuna_candidate_summary)]
    )
    final_selected_source = "forward_selection"
    if tuple(best_final_selection_summary["candidate_label_list"]) == tuple(best_optuna_candidate_summary["candidate_label_list"]):
        final_selected_source = "optuna_extension"
    return {
        "enabled": True,
        "baseline_candidate_label_list": baseline_candidate_label_list,
        "baseline_train_spearman_icir": baseline_train_icir,
        "baseline_valid_spearman_icir": float(baseline_summary["valid_spearman_icir"]),
        "remaining_candidate_label_list": remaining_candidate_label_list,
        "remaining_factor_count": remaining_factor_count,
        "n_trials": n_trials,
        "train_improved_candidate_count": int(len(valid_evaluated_summary_list)),
        "train_improved_path_summary_list": valid_evaluated_summary_list,
        "best_optuna_candidate_summary": best_optuna_candidate_summary,
        "final_selected_source": final_selected_source,
        "best_final_selection_summary": best_final_selection_summary,
    }


def run_single_factor_dedup_selection(config_override=None):
    config = build_tradition_config(config_override=config_override)
    if bool(config.get("force_refresh", False)):
        raise ValueError("dedup 流程禁止 --force-refresh，请先运行流程0 data-preprocess。")
    ic_aggregation_config = build_ic_aggregation_config(config)
    stability_analysis_path = config.get("stability_analysis_path")
    if stability_analysis_path is None:
        raise ValueError("single_factor_dedup_selection 模式必须提供 stability_analysis_path。")
    dedup_root_topk = int(config.get("dedup_root_topk", 3))
    if dedup_root_topk <= 0:
        raise ValueError("dedup_root_topk 必须为正整数。")
    stability_analysis_input, resolved_stability_analysis_path = load_stability_analysis_input(stability_analysis_path)
    stability_analysis_output = dict(stability_analysis_input["stability_analysis_output"])
    selected_stability_candidate_list = [
        record
        for record in stability_analysis_output["record_dict"].values()
        if bool(record.get("selected", False))
    ]
    if len(selected_stability_candidate_list) == 0:
        raise ValueError("稳定性分析结果中不存在 selected=true 的最终候选。")
    fund_code = resolve_fund_code_from_stability_analysis_input(
        stability_analysis_input=stability_analysis_input,
        stability_analysis_path=resolved_stability_analysis_path,
    )
    data_mode = str(stability_analysis_output.get("data_mode", "feature_matrix"))
    feature_df, target_nav_column, resolved_preprocess_path = load_selected_feature_matrix(
        stability_analysis_output=stability_analysis_output,
        selected_stability_candidate_list=selected_stability_candidate_list,
    )
    target_nav_series = pd.Series(feature_df[target_nav_column], copy=True).astype(float)
    fold_list = build_walk_forward_dev_fold_list(
        price_series=target_nav_series,
        walk_forward_config=dict(config["walk_forward_config"]),
        split_config=config["data_split_dict"],
    )
    forward_return_series = build_forward_return_series(price_series=target_nav_series, forward_window=5)

    factor_series_dict = {}
    for factor_candidate in selected_stability_candidate_list:
        # 候选因子已经在流程0中固化为特征列，这里直接按 candidate_label 读取对应列。
        factor_series_dict[str(factor_candidate["candidate_label"])] = pd.Series(
            feature_df[str(factor_candidate["candidate_label"])],
            copy=True,
        ).astype(float)

    corr_summary_df, dropped_candidate_label_list = build_corr_dedup_result(
        selected_summary_df=pd.DataFrame(selected_stability_candidate_list),
        factor_series_dict=factor_series_dict,
        fold_list=fold_list,
        corr_threshold=0.90,
        drop_ratio=0.10,
        min_drop_count=2,
    )
    corr_selected_mask = corr_summary_df["corr_dedup_selected"].astype(bool)
    corr_selected_summary_df = corr_summary_df[corr_selected_mask].reset_index(drop=True)
    corr_selected_candidate_list = corr_selected_summary_df.to_dict(orient="records")

    train_forward_selection_path_summary = run_train_forward_selection(
        candidate_record_list=corr_selected_candidate_list,
        factor_series_dict=factor_series_dict,
        forward_return_series=forward_return_series,
        fold_list=fold_list,
        root_topk=dedup_root_topk,
        ic_aggregation_config=ic_aggregation_config,
    )
    candidate_record_lookup = {
        str(candidate_record["candidate_label"]): dict(candidate_record)
        for candidate_record in corr_selected_candidate_list
    }
    forward_selection_path_summary = select_top_train_path_summary_list(
        train_path_summary_list=train_forward_selection_path_summary,
        top_ratio=0.5,
    )
    forward_selection_path_summary = evaluate_valid_for_path_summary_list(
        path_summary_list=forward_selection_path_summary,
        candidate_record_lookup=candidate_record_lookup,
        factor_series_dict=factor_series_dict,
        forward_return_series=forward_return_series,
        fold_list=fold_list,
        ic_aggregation_config=ic_aggregation_config,
        progress_desc="forward valid",
    )
    best_forward_selection_summary = select_best_forward_path_summary(forward_selection_path_summary)
    if best_forward_selection_summary is None:
        raise ValueError("相关性去冗余后无法构建有效的 forward selection 组合。")
    optuna_extension_output = run_optuna_extension_search(
        baseline_summary=best_forward_selection_summary,
        corr_selected_candidate_list=corr_selected_candidate_list,
        candidate_record_lookup=candidate_record_lookup,
        factor_series_dict=factor_series_dict,
        forward_return_series=forward_return_series,
        fold_list=fold_list,
        ic_aggregation_config=ic_aggregation_config,
    )
    best_final_selection_summary = dict(optuna_extension_output["best_final_selection_summary"])

    forward_selected_candidate_label_set = set(best_final_selection_summary["candidate_label_list"])
    corr_summary_df["forward_selected"] = corr_summary_df["candidate_label"].isin(forward_selected_candidate_label_set)
    corr_summary_df["forward_selection_step"] = pd.Series([None] * len(corr_summary_df), dtype=object)
    for step_idx, candidate_label in enumerate(best_final_selection_summary["candidate_label_list"], start=1):
        corr_summary_df.loc[corr_summary_df["candidate_label"] == candidate_label, "forward_selection_step"] = step_idx

    dedup_record_dict = {}
    for candidate_label in best_final_selection_summary["candidate_label_list"]:
        selected_record = build_candidate_record_dict(
            summary_df=corr_summary_df[corr_summary_df["candidate_label"] == candidate_label]
        )[candidate_label]
        selected_record.pop("factor_name", None)
        selected_record.pop("factor_param_dict", None)
        selected_record.pop("factor_group", None)
        dedup_record_dict[candidate_label] = selected_record

    dedup_selection_output = {
        "fund_code": fund_code,
        "preprocess_path": str(resolved_preprocess_path),
        "preprocess_metadata_path": stability_analysis_output.get("preprocess_metadata_path"),
        "stability_analysis_path": str(resolved_stability_analysis_path),
        "analysis_date": datetime.today().strftime("%Y-%m-%d"),
        "target_nav_column": target_nav_column,
        "dedup_root_topk": dedup_root_topk,
        "input_candidate_count": int(len(selected_stability_candidate_list)),
        "corr_dedup_drop_count": int(len(dropped_candidate_label_list)),
        "corr_dedup_selected_count": int(corr_selected_mask.sum()),
        "train_path_count": int(len(train_forward_selection_path_summary)),
        "valid_eval_count": int(len(forward_selection_path_summary)),
        "valid_eval_ratio": 0.5,
        "ic_aggregation_config": dict(ic_aggregation_config),
        "forward_selected_count": int(len(best_final_selection_summary["candidate_label_list"])),
        "corr_dedup_dropped_candidate_label_list": dropped_candidate_label_list,
        "corr_dedup_selected_candidate_label_list": corr_selected_summary_df["candidate_label"].tolist(),
        "forward_selected_candidate_label_list": list(best_final_selection_summary["candidate_label_list"]),
        "record_dict": dedup_record_dict,
        "best_forward_selection_summary": best_forward_selection_summary,
        "final_selected_source": str(optuna_extension_output["final_selected_source"]),
        "best_final_selection_summary": best_final_selection_summary,
    }
    summary_path = save_single_factor_dedup_selection_output(
        stability_analysis_input=stability_analysis_input,
        dedup_selection_output=dedup_selection_output,
        output_dir=config["output_dir"],
        fund_code=fund_code,
    )
    result = {
        "fund_code": fund_code,
        "data_mode": data_mode,
        "stability_analysis_path": str(resolved_stability_analysis_path),
        "selected_stability_candidate_list": selected_stability_candidate_list,
        "corr_selected_summary_df": corr_selected_summary_df,
        "best_forward_selection_summary": best_forward_selection_summary,
        "final_selected_source": str(optuna_extension_output["final_selected_source"]),
        "best_final_selection_summary": best_final_selection_summary,
        "summary_path": summary_path,
    }
    print_single_factor_dedup_selection_summary(result)
    return result
