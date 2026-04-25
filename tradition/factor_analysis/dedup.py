from datetime import datetime
from itertools import combinations
from pathlib import Path
import sys
import multiprocessing
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
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
from .sqlite_cache import ForwardSelectionSummaryCache, build_forward_selection_cache_path


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


def _build_combination_score_name(candidate_label_list):
    return "|".join([str(candidate_label) for candidate_label in candidate_label_list])


def _build_single_candidate_score_series(candidate_label, factor_series_dict):
    score_series = pd.Series(factor_series_dict[str(candidate_label)], copy=True).astype(float).fillna(0.0)
    score_series.name = _build_combination_score_name([candidate_label])
    return score_series


def _extend_combination_score_series(parent_score_series, candidate_label, factor_series_dict, candidate_label_list):
    candidate_series = pd.Series(factor_series_dict[str(candidate_label)], copy=True).astype(float)
    # 组合分数在当前流程里等于逐日求和，这里直接基于父分数做增量更新，避免重建整张组合表。
    child_score_series = pd.Series(parent_score_series, copy=True).astype(float).add(candidate_series, fill_value=0.0)
    child_score_series = child_score_series.astype(float)
    child_score_series.name = _build_combination_score_name(candidate_label_list)
    return child_score_series


def _evaluate_combination_score_series(score_series, forward_return_series, fold_list, candidate_label_list, include_valid=True, ic_aggregation_config=None):
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
        "candidate_label_list": [str(candidate_label) for candidate_label in candidate_label_list],
        "factor_count": int(len(candidate_label_list)),
        "train_spearman_ic_mean": train_summary["mean"],
        "train_spearman_icir": train_summary["icir"],
    }
    if include_valid:
        valid_summary = build_spearman_metric_summary([metric_dict["spearman_ic"] for metric_dict in valid_metric_list], ic_aggregation_config=ic_aggregation_config)
        summary_dict["valid_spearman_ic_mean"] = valid_summary["mean"]
        summary_dict["valid_spearman_icir"] = valid_summary["icir"]
    return summary_dict


def _build_forward_search_fold_cache(candidate_label_list, factor_series_dict, forward_return_series, fold_list, fold_type="train"):
    """
    通用缓存构建器：针对指定类型（train/valid）的 fold 预计算数据切片与 Spearman 排名组件。
    """
    target_rank_component_list = []
    candidate_segment_dict = {str(candidate_label): [] for candidate_label in candidate_label_list}

    # 逻辑块：预处理收益序列排名
    for fold_dict in fold_list:
        fold_series = fold_dict.get(fold_type)
        if fold_series is None:
            target_rank_component_list.append(None)
            continue
            
        fold_index = fold_series.index
        y_raw = pd.Series(forward_return_series, copy=True).reindex(fold_index).values.astype(float)
        
        valid_mask = ~np.isnan(y_raw)
        y_valid = y_raw[valid_mask]
        
        if len(y_valid) < 2:
            target_rank_component_list.append(None)
        else:
            y_rank = rankdata(y_valid, method="average")
            y_rank_centered = y_rank - np.mean(y_rank)
            y_rank_std = np.std(y_rank)
            if y_rank_std < 1e-12:
                target_rank_component_list.append(None)
            else:
                target_rank_component_list.append({
                    "mask": valid_mask,
                    "y_processed": y_rank_centered / (y_rank_std * len(y_rank)),
                })

    # 逻辑块：预切片候选因子序列并转为 numpy 数组
    for candidate_label in candidate_label_list:
        candidate_series = factor_series_dict[str(candidate_label)]
        for fold_dict in fold_list:
            fold_series = fold_dict.get(fold_type)
            if fold_series is None:
                candidate_segment_dict[str(candidate_label)].append(np.array([], dtype=float))
                continue
                
            fold_index = fold_series.index
            x_raw = pd.Series(candidate_series, copy=True).reindex(fold_index).values.astype(float)
            x_raw[np.isnan(x_raw)] = 0.0
            candidate_segment_dict[str(candidate_label)].append(x_raw)

    return {
        "target_rank_component_list": target_rank_component_list,
        "candidate_segment_dict": candidate_segment_dict,
    }


def _build_single_candidate_train_score_segment_list(candidate_label, candidate_train_segment_dict):
    # 转换为 numpy 数组列表，确保后续的因子组合累加是纯向量运算
    return [np.array(segment, copy=True) for segment in candidate_train_segment_dict[str(candidate_label)]]


def _extend_train_score_segment_list(parent_train_score_segment_list, candidate_train_segment_list, candidate_label_list):
    # 利用 numpy.add 实现极速向量求和。在第三层搜索中，这是最高频调用的操作。
    return [np.add(p, c) for p, c in zip(parent_train_score_segment_list, candidate_train_segment_list)]


def _compute_prealigned_train_spearman_metric_numpy(x_full, target_component):
    if target_component is None:
        return np.nan
    
    # 根据预处理的掩码提取有效样本点
    x_valid = x_full[target_component["mask"]]
    
    # 对当前因子/组合值进行 Rank 处理。由于收益序列已预排位，此处计算退化为简单的点积
    x_rank = rankdata(x_valid, method="average")
    x_rank_centered = x_rank - np.mean(x_rank)
    x_rank_std = np.std(x_rank)
    
    if x_rank_std < 1e-12:
        return np.nan
        
    # 基于 Pearson 等效公式计算 Spearman 相关系数：np.dot( (X-mu)/sigma, Y_processed )
    return np.dot(x_rank_centered / x_rank_std, target_component["y_processed"])


def _evaluate_train_score_segment_list(train_score_segment_list, train_target_rank_component_list, candidate_label_list, ic_aggregation_config=None):
    train_spearman_ic_list = []
    # 遍历所有 fold，在 NumPy 层面完成 Spearman IC 的计算
    for x_segment, target_comp in zip(train_score_segment_list, train_target_rank_component_list):
        train_spearman_ic_list.append(
            _compute_prealigned_train_spearman_metric_numpy(
                x_full=x_segment,
                target_component=target_comp,
            )
        )
    
    if ic_aggregation_config is None:
        ic_aggregation_config = {"mode": "classic", "half_life": 3.0}
        
    # 汇总各 fold 的 IC 结果，计算均值和 ICIR
    train_summary = build_spearman_metric_summary(
        metric_value_list=train_spearman_ic_list,
        ic_aggregation_config=ic_aggregation_config,
    )
    return {
        "candidate_label_list": [str(candidate_label) for candidate_label in candidate_label_list],
        "factor_count": int(len(candidate_label_list)),
        "train_spearman_ic_mean": train_summary["mean"],
        "train_spearman_icir": train_summary["icir"],
    }


# 全局容器：训练集与验证集的特征数据（仅在子进程中有效）
_G_TRAIN_SEGMENTS = {}
_G_VALID_SEGMENTS = {}


def _init_forward_selection_worker(train_segments, valid_segments):
    """
    进程池初始化函数：在子进程启动时挂载全量数据切片。
    由于数据是只读的，在 Linux 下利用写时复制 (CoW) 特性可以极大地节省内存。
    """
    global _G_TRAIN_SEGMENTS, _G_VALID_SEGMENTS
    _G_TRAIN_SEGMENTS = train_segments
    _G_VALID_SEGMENTS = valid_segments


def _worker_evaluate_batch(task_payload):
    """
    前向搜索评估 Worker：接收一个父节点的累加分数值和一批候选因子标签，返回评估结果。
    """
    parent_arrays, batch_payloads, target_rank_components, ic_aggregation_config = task_payload
    results = []
    
    # 逻辑块：执行增量累加与 ICIR 评估
    for child_labels, child_sig, label in batch_payloads:
        child_arrays = [np.add(p, c) for p, c in zip(parent_arrays, _G_TRAIN_SEGMENTS[label])]
        
        train_spearman_ic_list = []
        for x_segment, target_comp in zip(child_arrays, target_rank_components):
            train_spearman_ic_list.append(
                _compute_prealigned_train_spearman_metric_numpy(x_full=x_segment, target_component=target_comp)
            )
            
        train_summary = build_spearman_metric_summary(
            metric_value_list=train_spearman_ic_list,
            ic_aggregation_config=ic_aggregation_config,
        )
        
        results.append((
            label,
            {
                "train_spearman_ic_mean": train_summary["mean"],
                "train_spearman_icir": train_summary["icir"],
            },
            child_labels,
            child_sig,
        ))
        
    return results


def _worker_evaluate_batch_valid(task_payload):
    """
    并行验证集 Batch Worker：对一批因子组合在验证集上执行完整评估。
    利用 NumPy 向量化运算和 Batching 机制显著提升 10w+ 任务的验证速度。
    """
    path_labels_list, target_components, aggregation_config = task_payload
    results = []
    
    # 逻辑块：静默零方差导致的除零警告，提升计算吞吐量
    with np.errstate(divide='ignore', invalid='ignore'):
        for labels in path_labels_list:
            # 重建分数值 (NumPy)
            combined_arrays = None
            for label in labels:
                if combined_arrays is None:
                    combined_arrays = [np.array(s, copy=True) for s in _G_VALID_SEGMENTS[label]]
                else:
                    combined_arrays = [np.add(p, c) for p, c in zip(combined_arrays, _G_VALID_SEGMENTS[label])]
            
            # 计算验证集 Spearman IC
            valid_ic_list = []
            for x_segment, target_comp in zip(combined_arrays, target_components):
                valid_ic_list.append(
                    _compute_prealigned_train_spearman_metric_numpy(x_full=x_segment, target_component=target_comp)
                )
            
            # 逻辑块：纯 NumPy 汇总逻辑，彻底摆脱 Pandas 内部拦截
            ic_array = np.array(valid_ic_list)
            ic_mean = np.nanmean(ic_array)
            ic_std = np.nanstd(ic_array)
            icir = ic_mean / ic_std if ic_std > 1e-12 else 0.0
            
            results.append({
                "candidate_label_list": list(labels),
                "valid_spearman_ic_mean": float(ic_mean),
                "valid_spearman_icir": float(icir),
            })
            
    return results


def evaluate_factor_candidate_subset(factor_candidate_list, factor_series_dict, forward_return_series, fold_list, include_valid=True, ic_aggregation_config=None):
    _, score_series = build_instance_combination_score(
        factor_candidate_list=factor_candidate_list,
        factor_series_dict=factor_series_dict,
    )
    return _evaluate_combination_score_series(
        score_series=score_series,
        forward_return_series=forward_return_series,
        fold_list=fold_list,
        candidate_label_list=[str(item["candidate_label"]) for item in factor_candidate_list],
        include_valid=include_valid,
        ic_aggregation_config=ic_aggregation_config,
    )


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


def run_train_forward_selection(
    candidate_record_list,
    factor_series_dict,
    forward_return_series,
    fold_list,
    root_topk=3,
    summary_cache=None,
    ic_aggregation_config=None,
    n_processes=2,
    layer_time_budget_seconds=600,
):
    """
    前向搜索主逻辑：支持多进程并行评估任务池。
    """
    candidate_record_list = list(candidate_record_list)
    if len(candidate_record_list) == 0:
        return []
    if ic_aggregation_config is None:
        ic_aggregation_config = {"mode": "classic", "half_life": 3.0}

    # 逻辑块：环境准备与根节点同步评估
    # 提前对齐、切片并 Rank 收益序列，将数据转化为纯 NumPy 数组以便高效分发。
    candidate_label_list = [str(record["candidate_label"]) for record in candidate_record_list]
    train_cache = _build_forward_search_fold_cache(
        candidate_label_list=candidate_label_list,
        factor_series_dict=factor_series_dict,
        forward_return_series=forward_return_series,
        fold_list=fold_list,
        fold_type="train"
    )
    sorted_candidate_record_list = sorted(
        candidate_record_list,
        key=lambda r: (-float(r["train_spearman_icir"]), str(r["candidate_label"])),
    )
    path_summary_dict = {}

    # 逻辑块：第一层搜索（根节点评估）
    # 仅使用排序后的 root_topk 个候选作为搜索根节点，保持原有前向搜索空间。
    root_candidate_record_list = sorted_candidate_record_list[:root_topk]
    root_pbar = tqdm(
        total=len(root_candidate_record_list), 
        desc="forward search 第1层", 
        unit="candidate",
        disable=not sys.stderr.isatty()
    )
    root_node_list = []
    for root_candidate_record in root_candidate_record_list:
        root_pbar.update(1)
        root_label = str(root_candidate_record["candidate_label"])
        root_signature = build_candidate_label_signature([root_label])
        root_summary = None if summary_cache is None else summary_cache.get(root_signature)
        root_arrays = _build_single_candidate_train_score_segment_list(root_label, train_cache["candidate_segment_dict"])
        
        if root_summary is None:
            root_summary = _evaluate_train_score_segment_list(
                root_arrays, train_cache["target_rank_component_list"], [root_label], ic_aggregation_config
            )
            root_summary["step"] = 1
            if summary_cache is not None:
                summary_cache.set(root_signature, root_summary)
        
        path_summary_dict[root_signature] = root_summary
        root_node_list.append({
            "candidate_record_list": [dict(root_candidate_record)],
            "summary": dict(root_summary),
            "train_score_segment_list": root_arrays,
        })
    root_pbar.close()

    frontier_node_list = root_node_list

    # 逻辑块：按层启动进程池并执行并行搜索
    # 每层独立创建 Pool，时间预算耗尽后可以终止当前层未返回的任务并进入下一层。
    try:
        step = 2
        layer_time_budget_seconds = int(layer_time_budget_seconds)
        while len(frontier_node_list) > 0:
            layer_start_time = time.monotonic()
            step_pbar = tqdm(
                total=len(frontier_node_list) * len(sorted_candidate_record_list),
                desc=f"forward search 第{step}层",
                unit="candidate",
                disable=not sys.stderr.isatty()
            )
            child_parent_node_dict = {}
            pool = multiprocessing.Pool(
                processes=n_processes,
                initializer=_init_forward_selection_worker,
                initargs=(train_cache["candidate_segment_dict"], {})
            )
            pool_terminated = False
            pending_result_list = []
            max_pending_batch_count = max(1, int(n_processes) * 4)

            def collect_batch_results(batch_results):
                for label, res, child_labels, child_sig in batch_results:
                    step_pbar.update(1)
                    child_summary = {
                        "candidate_label_list": list(child_labels),
                        "factor_count": len(child_labels),
                        "step": step,
                        **res
                    }
                    path_summary_dict[child_sig] = child_summary
                    if summary_cache is not None:
                        summary_cache.set(child_sig, child_summary)

            def drain_pending_results(wait_for_one=False):
                # 逻辑块：只在主线程回收结果并写缓存，避免 SQLite connection 被 Pool 内部线程访问。
                drained_any = False
                remaining_result_list = []
                for result in pending_result_list:
                    if result.ready():
                        collect_batch_results(result.get())
                        drained_any = True
                    else:
                        remaining_result_list.append(result)
                pending_result_list[:] = remaining_result_list
                if wait_for_one and not drained_any and len(pending_result_list) > 0:
                    try:
                        collect_batch_results(pending_result_list.pop(0).get(timeout=0.1))
                        drained_any = True
                    except multiprocessing.TimeoutError:
                        pass
                return drained_any

            def submit_batch(frontier_node, batch_payloads):
                pending_result_list.append(
                    pool.apply_async(
                        _worker_evaluate_batch,
                        args=((
                            frontier_node["train_score_segment_list"],
                            list(batch_payloads),
                            train_cache["target_rank_component_list"],
                            ic_aggregation_config,
                        ),),
                    )
                )

            try:
                # 逻辑块：主线程生成 batch 和访问 SQLite cache，Pool 只接收纯计算任务。
                layer_budget_exhausted = False
                for frontier_node in frontier_node_list:
                    if layer_time_budget_seconds > 0 and time.monotonic() - layer_start_time >= layer_time_budget_seconds:
                        layer_budget_exhausted = True
                        break
                    parent_summary = frontier_node["summary"]
                    parent_labels = set(parent_summary["candidate_label_list"])
                    batch_payloads = []

                    for cand_record in sorted_candidate_record_list:
                        if layer_time_budget_seconds > 0 and time.monotonic() - layer_start_time >= layer_time_budget_seconds:
                            layer_budget_exhausted = True
                            break
                        cand_label = str(cand_record["candidate_label"])
                        if cand_label in parent_labels:
                            step_pbar.update(1)
                            continue

                        child_labels = sorted(list(parent_labels) + [cand_label])
                        child_signature = tuple(child_labels)

                        if child_signature in path_summary_dict:
                            step_pbar.update(1)
                            continue

                        child_summary = None if summary_cache is None else summary_cache.get(child_signature)
                        if child_summary is not None:
                            path_summary_dict[child_signature] = child_summary
                            child_parent_node_dict[child_signature] = (frontier_node, dict(cand_record), child_labels)
                            step_pbar.update(1)
                            continue

                        batch_payloads.append((child_labels, child_signature, cand_label))
                        child_parent_node_dict[child_signature] = (frontier_node, dict(cand_record), child_labels)

                        if len(batch_payloads) >= 20:
                            submit_batch(frontier_node, batch_payloads)
                            batch_payloads = []
                            drain_pending_results()
                            while len(pending_result_list) >= max_pending_batch_count:
                                if layer_time_budget_seconds > 0 and time.monotonic() - layer_start_time >= layer_time_budget_seconds:
                                    layer_budget_exhausted = True
                                    break
                                drain_pending_results(wait_for_one=True)
                            if layer_budget_exhausted:
                                break

                    if batch_payloads and not layer_budget_exhausted:
                        submit_batch(frontier_node, batch_payloads)
                        drain_pending_results()
                    if layer_budget_exhausted:
                        break

                if layer_budget_exhausted:
                    pool.terminate()
                    pool_terminated = True
                else:
                    while len(pending_result_list) > 0:
                        drain_pending_results(wait_for_one=True)
            finally:
                if pool_terminated:
                    pool.join()
                else:
                    pool.close()
                    pool.join()
                step_pbar.close()

            # 逻辑块：层级同步
            # 只基于本层已完成或缓存命中的结果构造下一层，并优先扩展训练 ICIR 更高的路径。
            next_frontier_node_list = []
            for child_sig, (frontier_node, cand_record, child_labels) in child_parent_node_dict.items():
                child_summary = path_summary_dict.get(child_sig)
                if child_summary is None or int(child_summary["step"]) != step:
                    continue
                parent_summary = frontier_node["summary"]
                if float(child_summary["train_spearman_icir"]) >= float(parent_summary["train_spearman_icir"]):
                    next_frontier_node_list.append({
                        "candidate_record_list": frontier_node["candidate_record_list"] + [dict(cand_record)],
                        "summary": child_summary,
                        "train_score_segment_list": _extend_train_score_segment_list(
                            frontier_node["train_score_segment_list"],
                            train_cache["candidate_segment_dict"][str(cand_record["candidate_label"])],
                            child_labels
                        )
                    })

            frontier_node_list = sorted(
                next_frontier_node_list,
                key=lambda node: (
                    -float(node["summary"]["train_spearman_icir"]),
                    int(node["summary"]["factor_count"]),
                    tuple(node["summary"]["candidate_label_list"]),
                ),
            )
            step += 1
    except Exception as e:
        raise e
    finally:
        pass

    return sorted(path_summary_dict.values(), key=lambda x: (int(x["factor_count"]), tuple(x["candidate_label_list"])))


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


def evaluate_valid_for_path_summary_list(
    path_summary_list, 
    candidate_record_lookup, 
    valid_cache,
    ic_aggregation_config=None, 
    progress_desc=None,
    n_processes=2
):
    """
    并行验证集评估：针对候选路径列表，利用 Batching 机制和 NumPy 引擎进行极速验证。
    """
    if not path_summary_list:
        return []
        
    # 逻辑块：任务 Batch 化打包 (BatchSize=100)
    # 显著减少 13w+ 任务产生的 IPC 往返开销。
    batch_size = 100
    task_list = []
    for i in range(0, len(path_summary_list), batch_size):
        chunk = path_summary_list[i : i + batch_size]
        labels_batch = [p["candidate_label_list"] for p in chunk]
        task_list.append((
            labels_batch, 
            valid_cache["target_rank_component_list"], 
            ic_aggregation_config
        ))

    evaluated_summary_list = []
    
    # 逻辑块：并行执行与结果流式收集
    with multiprocessing.Pool(
        processes=n_processes,
        initializer=_init_forward_selection_worker,
        initargs=({}, valid_cache["candidate_segment_dict"])
    ) as pool:
        progress_bar = tqdm(
            total=len(path_summary_list),
            desc=str(progress_desc or "validating paths"),
            unit="path",
            disable=not sys.stderr.isatty(),
        )
        
        # 逻辑块：建立路径查找表，确保合并结果时保留原始的训练集指标 (train_spearman_icir 等)
        path_lookup = {tuple(p["candidate_label_list"]): p for p in path_summary_list}
        
        for batch_results in pool.imap_unordered(_worker_evaluate_batch_valid, task_list):
            for res in batch_results:
                sig = tuple(res["candidate_label_list"])
                # 合并：以原始 summary 为基准，更新入验证集评估出的新指标
                combined_summary = dict(path_lookup[sig])
                combined_summary.update(res)
                evaluated_summary_list.append(combined_summary)
                progress_bar.update(1)
            
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
    train_cache,
    valid_cache,
    ic_aggregation_config=None,
    n_processes=2,
    optuna_time_budget_seconds=600,
):
    """
    Optuna 增强搜索：在多进程和 NumPy 引擎加持下探索最优组合。
    """
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
            "baseline_valid_spearman_icir": float(baseline_summary.get("valid_spearman_icir", 0.0)),
            "remaining_candidate_label_list": remaining_candidate_label_list,
            "remaining_factor_count": remaining_factor_count,
            "n_trials": 0,
            "train_improved_candidate_count": 0,
            "train_improved_path_summary_list": [],
            "best_optuna_candidate_summary": None,
            "final_selected_source": "forward_selection",
            "best_final_selection_summary": dict(baseline_summary),
        }

    # 逻辑块：预计算 Baseline 数组 (NumPy)
    baseline_arrays = None
    for label in baseline_candidate_label_list:
        cand_arrays = _build_single_candidate_train_score_segment_list(label, train_cache["candidate_segment_dict"])
        if baseline_arrays is None:
            baseline_arrays = cand_arrays
        else:
            baseline_arrays = [np.add(p, c) for p, c in zip(baseline_arrays, cand_arrays)]

    optuna_module = load_optuna_module()
    train_improved_summary_dict = {}
    baseline_train_icir = float(baseline_summary["train_spearman_icir"])
    
    def objective(trial):
        selected_labels = list(baseline_candidate_label_list)
        current_arrays = [np.array(arr, copy=True) for arr in baseline_arrays]
        
        for idx, label in enumerate(remaining_candidate_label_list):
            if int(trial.suggest_int(f"add_{idx}", 0, 1)) == 1:
                selected_labels.append(label)
                cand_data = train_cache["candidate_segment_dict"][label]
                current_arrays = [np.add(p, c) for p, c in zip(current_arrays, cand_data)]
        
        summary = _evaluate_train_score_segment_list(
            current_arrays, 
            train_cache["target_rank_component_list"], 
            selected_labels, 
            ic_aggregation_config
        )
        return float(summary["train_spearman_icir"])

    n_trials = int(remaining_factor_count ** 2)
    study = optuna_module.create_study(
        direction="maximize",
        sampler=optuna_module.samplers.TPESampler(seed=42),
    )
    
    progress_bar = tqdm(total=n_trials, desc="optuna extension", unit="trial", disable=not sys.stderr.isatty())
    try:
        def update_progress_bar(study, trial):
            progress_bar.update(1)
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=int(optuna_time_budget_seconds),
            callbacks=[update_progress_bar],
            n_jobs=1,
        )
    finally:
        progress_bar.close()

    # 逻辑块：统一从已完成 trial 重建改进路径，避免并发写共享 summary dict。
    for trial in study.trials:
        if getattr(trial, "state", None) != optuna_module.trial.TrialState.COMPLETE:
            continue
        train_icir = float(trial.value)
        if train_icir <= baseline_train_icir:
            continue

        selected_labels = list(baseline_candidate_label_list)
        current_arrays = [np.array(arr, copy=True) for arr in baseline_arrays]
        for idx, label in enumerate(remaining_candidate_label_list):
            if int(trial.params.get(f"add_{idx}", 0)) == 1:
                selected_labels.append(label)
                cand_data = train_cache["candidate_segment_dict"][label]
                current_arrays = [np.add(p, c) for p, c in zip(current_arrays, cand_data)]

        summary = _evaluate_train_score_segment_list(
            current_arrays,
            train_cache["target_rank_component_list"],
            selected_labels,
            ic_aggregation_config,
        )
        summary["step"] = int(len(selected_labels))
        signature = build_candidate_label_signature(selected_labels)
        cached_summary = train_improved_summary_dict.get(signature)
        if cached_summary is None or float(summary["train_spearman_icir"]) > float(cached_summary["train_spearman_icir"]):
            train_improved_summary_dict[signature] = dict(summary)

    # 逻辑块：对改进路径进行高性能并行验证
    train_improved_path_summary_list = sorted(
        train_improved_summary_dict.values(),
        key=lambda s: (-float(s["train_spearman_icir"]), -float(s["train_spearman_ic_mean"])),
    )
    valid_evaluated_summary_list = evaluate_valid_for_path_summary_list(
        path_summary_list=train_improved_path_summary_list,
        candidate_record_lookup=candidate_record_lookup,
        valid_cache=valid_cache,
        ic_aggregation_config=ic_aggregation_config,
        progress_desc="optuna valid",
        n_processes=n_processes,
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

    # 逻辑块：构造训练集与验证集的 NumPy 缓存
    # 通过预对齐和预排名，将后续所有评估流程（前向搜索、验证、Optuna）切换至极速引擎。
    candidate_labels = [str(r["candidate_label"]) for r in selected_stability_candidate_list]
    train_cache = _build_forward_search_fold_cache(
        candidate_label_list=candidate_labels,
        factor_series_dict=factor_series_dict,
        forward_return_series=forward_return_series,
        fold_list=fold_list,
        fold_type="train"
    )
    valid_cache = _build_forward_search_fold_cache(
        candidate_label_list=candidate_labels,
        factor_series_dict=factor_series_dict,
        forward_return_series=forward_return_series,
        fold_list=fold_list,
        fold_type="valid"
    )

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

    forward_selection_cache_path = build_forward_selection_cache_path(
        output_dir=config["output_dir"],
        fund_code=fund_code,
        resolved_stability_analysis_path=resolved_stability_analysis_path,
        resolved_preprocess_path=resolved_preprocess_path,
        target_nav_column=target_nav_column,
        selected_candidate_label_list=[record["candidate_label"] for record in selected_stability_candidate_list],
        walk_forward_config=config["walk_forward_config"],
        data_split_dict=config["data_split_dict"],
        ic_aggregation_config=ic_aggregation_config,
        dedup_root_topk=dedup_root_topk,
    )
    
    dedup_n_processes = int(config.get("dedup_n_processes", 2))
    
    # 逻辑块：前向并行搜索 (训练集)
    with ForwardSelectionSummaryCache(
        sqlite_path=forward_selection_cache_path,
        memory_cache_size=5000,
    ) as forward_selection_summary_cache:
        train_forward_selection_path_summary = run_train_forward_selection(
            candidate_record_list=corr_selected_candidate_list,
            factor_series_dict=factor_series_dict,
            forward_return_series=forward_return_series,
            fold_list=fold_list,
            root_topk=dedup_root_topk,
            ic_aggregation_config=ic_aggregation_config,
            summary_cache=forward_selection_summary_cache,
            n_processes=dedup_n_processes,
        )
        
    candidate_record_lookup = {
        str(candidate_record["candidate_label"]): dict(candidate_record)
        for candidate_record in corr_selected_candidate_list
    }
    
    # 逻辑块：验证集并行高速评估
    # 利用预计算的 valid_cache 避开 Pandas，直接在多进程中验证搜索出的 Top 组合。
    forward_selection_path_summary = select_top_train_path_summary_list(
        train_path_summary_list=train_forward_selection_path_summary,
        top_ratio=0.5,
    )
    forward_selection_path_summary = evaluate_valid_for_path_summary_list(
        path_summary_list=forward_selection_path_summary,
        candidate_record_lookup=candidate_record_lookup,
        valid_cache=valid_cache,
        ic_aggregation_config=ic_aggregation_config,
        progress_desc="forward valid",
        n_processes=dedup_n_processes,
    )
    
    best_forward_selection_summary = select_best_forward_path_summary(forward_selection_path_summary)
    if best_forward_selection_summary is None:
        raise ValueError("相关性去冗余后无法构建有效的 forward selection 组合。")
        
    # 逻辑块：Optuna 并行增强搜索
    optuna_extension_output = run_optuna_extension_search(
        baseline_summary=best_forward_selection_summary,
        corr_selected_candidate_list=corr_selected_candidate_list,
        candidate_record_lookup=candidate_record_lookup,
        train_cache=train_cache,
        valid_cache=valid_cache,
        ic_aggregation_config=ic_aggregation_config,
        n_processes=dedup_n_processes,
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
