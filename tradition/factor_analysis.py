import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from tradition.config import build_tradition_config
from tradition.data_adapter import adapt_to_price_series
from tradition.data_fetcher import fetch_fund_data_with_cache
from tradition.data_loader import filter_single_fund, normalize_fund_data
from tradition.factor_engine import (
    FACTOR_POOL_DICT,
    build_single_factor_series,
    resolve_factor_name_list_by_group,
)
from tradition.metrics import compute_return_metrics, save_equity_curve_plot
from tradition.optimizer import load_optuna_module
from tradition.splitter import build_walk_forward_fold_list, split_time_series_by_ratio


def build_forward_return_series(price_series, forward_window=5):
    # 因子筛选统一使用未来 5 日简单收益标签，并按 t 时点左对齐到当前因子值。
    series = pd.Series(price_series, copy=True).astype(float).dropna()
    forward_window = int(forward_window)
    if forward_window <= 0:
        raise ValueError("forward_window 必须为正整数。")
    forward_return_series = series.shift(-forward_window) / series - 1.0
    forward_return_series.name = f"forward_return_{forward_window}d"
    return forward_return_series


def compute_segment_correlation_metrics(factor_series, forward_return_series, segment_index):
    # 单基金筛选按时间序列样本计算区间相关性，避免把跨折汇总和单折样本处理耦合在一起。
    aligned_df = pd.DataFrame(
        {
            "factor": pd.Series(factor_series, copy=True).reindex(segment_index),
            "forward_return": pd.Series(forward_return_series, copy=True).reindex(segment_index),
        }
    ).dropna()
    if len(aligned_df) < 2:
        return {
            "sample_size": int(len(aligned_df)),
            "spearman_ic": float("nan"),
            "pearson_ic": float("nan"),
        }
    return {
        "sample_size": int(len(aligned_df)),
        "spearman_ic": float(aligned_df["factor"].corr(aligned_df["forward_return"], method="spearman")),
        "pearson_ic": float(aligned_df["factor"].corr(aligned_df["forward_return"], method="pearson")),
    }


def build_metric_summary(metric_value_list):
    # ICIR 统一按折级 IC 序列做汇总，样本不足或零波动时回落到 0，避免得到伪稳定高分。
    metric_series = pd.Series(metric_value_list, dtype=float).dropna()
    if len(metric_series) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "icir": 0.0,
        }
    metric_std = float(metric_series.std(ddof=0))
    metric_mean = float(metric_series.mean())
    metric_icir = 0.0
    if abs(metric_std) > 1e-12:
        metric_icir = float(metric_mean / metric_std)
    return {
        "count": int(len(metric_series)),
        "mean": metric_mean,
        "std": metric_std,
        "icir": metric_icir,
    }


def build_positive_ic_ratio(metric_value_list):
    # 正 IC 比例用有效折中的正值占比衡量方向稳定性，无有效折时直接记为 0。
    metric_series = pd.Series(metric_value_list, dtype=float).dropna()
    if len(metric_series) == 0:
        return 0.0
    return float((metric_series > 0.0).mean())


def build_trimmed_ic_mean(metric_value_list, trim_ratio=0.1):
    # 截尾稳定性统一在折级 IC 序列上处理，去掉两端极值后只保留均值，不改变原始序列汇总逻辑。
    metric_series = pd.Series(metric_value_list, dtype=float).dropna().sort_values(ignore_index=True)
    if len(metric_series) == 0:
        return float("nan")
    trim_ratio = float(trim_ratio)
    if trim_ratio < 0.0 or trim_ratio >= 0.5:
        raise ValueError(f"trim_ratio 必须满足 0 <= trim_ratio < 0.5，当前 trim_ratio={trim_ratio}")
    trim_count = int(len(metric_series) * trim_ratio)
    if trim_count <= 0 or trim_count * 2 >= len(metric_series):
        return float(metric_series.mean())
    trimmed_series = metric_series.iloc[trim_count:-trim_count]
    return float(trimmed_series.mean())


def build_ic_flip_count(metric_value_list):
    # IC 翻转次数只统计有效且非零折的方向变化，0 和 NaN 统一跳过，避免把无方向样本误算成翻转。
    metric_series = pd.Series(metric_value_list, dtype=float).dropna()
    sign_list = []
    for value in metric_series.tolist():
        if abs(float(value)) <= 1e-12:
            continue
        sign_list.append(1 if float(value) > 0.0 else -1)
    if len(sign_list) < 2:
        return 0
    return int(sum(1 for idx in range(1, len(sign_list)) if sign_list[idx] != sign_list[idx - 1]))


def build_tail_reject_mask(summary_df, reject_ratio=0.05):
    # 稳定性尾部剔除按三个坏指标分别取最差 reject_ratio，并集后统一淘汰。
    summary_df = pd.DataFrame(summary_df, copy=True)
    if len(summary_df) == 0:
        return pd.Series(dtype=bool)
    reject_ratio = float(reject_ratio)
    if reject_ratio < 0.0 or reject_ratio >= 1.0:
        raise ValueError(f"reject_ratio 必须满足 0 <= reject_ratio < 1，当前 reject_ratio={reject_ratio}")
    metric_config_list = [
        ("abs_train_valid_ic_mean_gap", summary_df["train_valid_ic_mean_gap"].abs()),
        ("valid_ic_flip_count", pd.Series(summary_df["valid_ic_flip_count"], copy=True)),
        ("abs_valid_trimmed_ic_gap", summary_df["valid_trimmed_ic_gap"].abs()),
    ]
    rejected_mask = pd.Series(False, index=summary_df.index, dtype=bool)
    for metric_name, metric_series in metric_config_list:
        valid_metric_series = pd.Series(metric_series, copy=True).dropna()
        if len(valid_metric_series) == 0:
            summary_df[f"{metric_name}_tail_rejected"] = False
            continue
        reject_count = int(len(valid_metric_series) * reject_ratio)
        if reject_count <= 0:
            summary_df[f"{metric_name}_tail_rejected"] = False
            continue
        threshold_value = float(valid_metric_series.sort_values(ascending=False).iloc[reject_count - 1])
        metric_rejected_mask = pd.Series(metric_series, copy=True) >= threshold_value
        metric_rejected_mask = metric_rejected_mask.fillna(False).astype(bool)
        summary_df[f"{metric_name}_tail_rejected"] = metric_rejected_mask
        rejected_mask = rejected_mask | metric_rejected_mask
    summary_df["stability_tail_rejected"] = rejected_mask
    return summary_df[
        [
            "abs_train_valid_ic_mean_gap_tail_rejected",
            "valid_ic_flip_count_tail_rejected",
            "abs_valid_trimmed_ic_gap_tail_rejected",
            "stability_tail_rejected",
        ]
    ]


def build_factor_candidate_label(factor_name, factor_param_dict):
    # 参数化候选需要稳定且可读的显示标签，便于终端输出和 JSON 排序复盘。
    factor_name = str(factor_name)
    factor_param_dict = dict(factor_param_dict)
    if len(factor_param_dict) == 0:
        return factor_name
    param_text = ", ".join(
        [f"{param_name}={factor_param_dict[param_name]}" for param_name in sorted(factor_param_dict.keys())]
    )
    return f"{factor_name}({param_text})"


def expand_param_values_from_search_space(search_space):
    # 搜索空间统一按离散候选值展开，保持与现有优化参数边界定义一致。
    low, high, step = search_space
    if step is None:
        raise ValueError(f"当前不支持 step=None 的筛选候选展开: {search_space}")
    low = int(low)
    high = int(high)
    step = int(step)
    if step <= 0:
        raise ValueError(f"search_space.step 必须为正整数，当前 step={step}")
    return list(range(low, high + 1, step))


def resolve_factor_group_name(factor_name):
    # 输出层只消费因子名时仍要补回所属因子族，避免筛选和稳定性结果丢失输入语义。
    resolved_factor_name = str(factor_name)
    if resolved_factor_name not in FACTOR_POOL_DICT:
        raise ValueError(f"未定义因子: {resolved_factor_name}")
    return str(FACTOR_POOL_DICT[resolved_factor_name]["group"])


def factor_pool_dict():
    # 因子分析侧只读消费因子库快照，避免候选展开逻辑在多个模块重复拼装元信息。
    return dict(FACTOR_POOL_DICT)


def build_factor_candidate_list(candidate_factor_name_list, strategy_params):
    # 因子筛选入口将筛选对象从因子名提升到参数化候选，双参数因子按参数组合完整展开。
    factor_param_dict = dict(strategy_params.get("factor_param_dict", {}))
    candidate_list = []
    for factor_name in candidate_factor_name_list:
        factor_name = str(factor_name)
        param_spec = dict(factor_pool_dict()[factor_name]["param_spec"])
        configured_param_dict = dict(factor_param_dict.get(factor_name, {}))
        searchable_param_name_list = []
        candidate_value_list_by_param = {}
        resolved_param_dict = {}
        for param_name, spec in param_spec.items():
            param_name = str(param_name)
            search_space = spec.get("search_space")
            resolved_default_value = configured_param_dict.get(param_name, spec["default"])
            if search_space is None:
                resolved_param_dict[param_name] = resolved_default_value
                continue
            searchable_param_name_list.append(param_name)
            candidate_value_list_by_param[param_name] = expand_param_values_from_search_space(search_space)
        if len(searchable_param_name_list) == 0:
            candidate_list.append(
                {
                    "factor_name": factor_name,
                    "factor_group": resolve_factor_group_name(factor_name=factor_name),
                    "param_dict": resolved_param_dict,
                    "candidate_label": build_factor_candidate_label(factor_name=factor_name, factor_param_dict=resolved_param_dict),
                }
            )
            continue

        # 单参数和双参数因子统一按笛卡尔积展开，避免筛选阶段遗漏有效参数组合。
        candidate_param_dict_list = [resolved_param_dict]
        for param_name in searchable_param_name_list:
            next_candidate_param_dict_list = []
            for base_param_dict in candidate_param_dict_list:
                for candidate_value in candidate_value_list_by_param[param_name]:
                    next_param_dict = dict(base_param_dict)
                    next_param_dict[param_name] = candidate_value
                    next_candidate_param_dict_list.append(next_param_dict)
            candidate_param_dict_list = next_candidate_param_dict_list

        for candidate_param_dict in candidate_param_dict_list:
            candidate_list.append(
                {
                    "factor_name": factor_name,
                    "factor_group": resolve_factor_group_name(factor_name=factor_name),
                    "param_dict": candidate_param_dict,
                    "candidate_label": build_factor_candidate_label(factor_name=factor_name, factor_param_dict=candidate_param_dict),
                }
            )
    return candidate_list


def build_factor_selection_record(factor_candidate, train_metric_list, valid_metric_list, threshold_config):
    # 单个参数化候选的训练筛选与验证排序指标集中在这里落成行记录，避免输出列在多个地方重复拼接。
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


def build_candidate_record_dict(summary_df, keep_field_list=None):
    # JSON 输出统一改成以 candidate_label 为键的子字典，并允许在落盘前裁剪为后续流程真正需要的最小字段集合。
    normalized_df = summary_df.where(pd.notna(summary_df), None)
    record_dict = {}
    for record in normalized_df.to_dict(orient="records"):
        if keep_field_list is not None:
            record = {
                field_name: record[field_name]
                for field_name in keep_field_list
                if field_name in record
            }
        candidate_label = str(record["candidate_label"])
        record_dict[candidate_label] = record
    return record_dict


def save_factor_selection_table(factor_selection_output, output_dir, fund_code):
    # 因子筛选结果统一落成顶层字典，并在子字典中保留按 candidate_label 编排的候选明细。
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    output_path = output_dir / f"factor_selection_{str(fund_code).zfill(6)}_{date_str}.json"
    payload = {
        "factor_selection_output": factor_selection_output,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


def print_factor_selection_summary(result):
    # 终端先打印训练和验证双门槛筛选结果，再展示最终通过候选按验证集 ICIR 排序后的结果。
    print("因子筛选结果:")
    print("基金代码:", result["fund_code"])
    print("输入因子族:", ",".join(result["factor_group_list"]))
    print("候选参数化因子数量:", len(result["candidate_factor_list"]))
    print("训练集 Spearman IC 阈值:", result["threshold_config"]["train_min_spearman_ic"])
    print("训练集 Spearman ICIR 阈值:", result["threshold_config"]["train_min_spearman_icir"])
    print("训练通过参数化因子数量:", int(result["summary_df"]["train_passed"].astype(bool).sum()))
    print("验证通过参数化因子数量:", int(result["summary_df"]["valid_passed"].astype(bool).sum()))
    print("筛选后参数化因子列表:", result["selected_candidate_label_list"])
    if len(result["selected_summary_df"]) > 0:
        printable_df = result["selected_summary_df"][
            [
                "final_rank",
                "candidate_label",
                "factor_group",
                "train_spearman_positive_ic_ratio",
                "valid_spearman_ic_mean",
                "valid_spearman_icir",
                "valid_pearson_ic_mean",
                "valid_pearson_icir",
            ]
        ].copy()
        print(printable_df.to_string(index=False))
    print("汇总输出:", result["summary_path"])


def run_factor_selection_single_fund(config_override=None):
    # 独立因子筛选入口复用现有数据准备与 walk-forward 切分，并从因子库中按因子族展开候选因子。
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

        # 训练集负责过门槛筛选，验证集只对通过训练筛选的因子做后续排序汇总。
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

    # 最终结果先过训练门槛，再过验证门槛，最后仅对双通过候选按原验证集排序规则排序。
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


def load_factor_selection_input(factor_selection_path):
    # 稳定性分析直接消费筛选 JSON 顶层字典，并从子字典里读取按 candidate_label 索引的候选结果。
    factor_selection_path = Path(factor_selection_path)
    if not factor_selection_path.exists():
        raise FileNotFoundError(f"factor_select 结果文件不存在: {factor_selection_path}")
    with factor_selection_path.open("r", encoding="utf-8") as input_file:
        factor_selection_input = json.load(input_file)
    if not isinstance(factor_selection_input, dict):
        raise ValueError("factor_select 结果文件必须是顶层字典。")
    factor_selection_output = factor_selection_input.get("factor_selection_output")
    if not isinstance(factor_selection_output, dict):
        raise ValueError("factor_select 结果文件缺少 factor_selection_output 子字典。")
    record_dict = factor_selection_output.get("record_dict")
    if not isinstance(record_dict, dict):
        raise ValueError("factor_select 结果文件缺少 record_dict 子字典。")
    return factor_selection_input, factor_selection_path


def resolve_fund_code_from_factor_selection_path(factor_selection_path):
    # 当前 factor_select 输出文件名已包含基金代码，稳定性分析优先复用这个约定，避免要求输入文件再重复携带元信息。
    factor_selection_path = Path(factor_selection_path)
    path_stem_part_list = factor_selection_path.stem.split("_")
    if len(path_stem_part_list) >= 3 and str(path_stem_part_list[0]) == "factor" and str(path_stem_part_list[1]) == "selection":
        candidate_code = str(path_stem_part_list[2]).strip()
        if len(candidate_code) == 6 and candidate_code.isdigit():
            return candidate_code
    raise ValueError(f"无法从 factor_select 结果文件名解析基金代码: {factor_selection_path.name}")


def resolve_fund_code_from_factor_selection_input(factor_selection_input, factor_selection_path):
    # 稳定性分析优先信任 JSON 内显式保存的 fund_code，只有缺失时才回退到文件名约定。
    factor_selection_output = dict(factor_selection_input.get("factor_selection_output", {}))
    candidate_code = str(factor_selection_output.get("fund_code", "")).strip()
    if len(candidate_code) > 0:
        candidate_code = candidate_code.zfill(6)
        if len(candidate_code) == 6 and candidate_code.isdigit():
            return candidate_code
    return resolve_fund_code_from_factor_selection_path(factor_selection_path)


def build_single_factor_stability_record(factor_candidate, train_metric_list, valid_metric_list):
    # 单候选稳定性记录只面向 train/valid 两段，集中产出后续排序和 JSON 落盘需要的全部指标。
    train_spearman_value_list = [metric_dict["spearman_ic"] for metric_dict in train_metric_list]
    valid_spearman_value_list = [metric_dict["spearman_ic"] for metric_dict in valid_metric_list]
    train_spearman_summary = build_metric_summary(train_spearman_value_list)
    valid_spearman_summary = build_metric_summary(valid_spearman_value_list)
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


def save_single_factor_stability_analysis_output(factor_selection_input, stability_analysis_output, output_dir, fund_code):
    # 稳定性分析输出改为独立落盘，只保留上游输入路径引用，避免把完整结果树继续复制到下游文件。
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    output_path = output_dir / f"single_factor_stability_{str(fund_code).zfill(6)}_{date_str}.json"
    factor_selection_output = dict(factor_selection_input.get("factor_selection_output", {}))
    payload = {
        "input_ref": {
            "factor_selection_path": str(output_dir / f"factor_selection_{str(fund_code).zfill(6)}_{date_str}.json"),
            "fund_code": str(factor_selection_output.get("fund_code", str(fund_code).zfill(6))),
        },
        "stability_analysis_output": stability_analysis_output,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


def load_stability_analysis_input(stability_analysis_path):
    # 去冗余流程直接消费稳定性分析 JSON 顶层字典，并要求其中已有稳定性分析子字典。
    stability_analysis_path = Path(stability_analysis_path)
    if not stability_analysis_path.exists():
        raise FileNotFoundError(f"稳定性分析结果文件不存在: {stability_analysis_path}")
    with stability_analysis_path.open("r", encoding="utf-8") as input_file:
        stability_analysis_input = json.load(input_file)
    if not isinstance(stability_analysis_input, dict):
        raise ValueError("稳定性分析结果文件必须是顶层字典。")
    stability_analysis_output = stability_analysis_input.get("stability_analysis_output")
    if not isinstance(stability_analysis_output, dict):
        raise ValueError("稳定性分析结果文件缺少 stability_analysis_output 子字典。")
    record_dict = stability_analysis_output.get("record_dict")
    if not isinstance(record_dict, dict):
        raise ValueError("稳定性分析结果文件缺少 record_dict 子字典。")
    return stability_analysis_input, stability_analysis_path


def resolve_fund_code_from_stability_analysis_input(stability_analysis_input, stability_analysis_path):
    # 去冗余流程优先复用稳定性分析和因子筛选两层输出中的基金代码，再回退到历史文件名约定。
    stability_analysis_output = dict(stability_analysis_input.get("stability_analysis_output", {}))
    candidate_code = str(stability_analysis_output.get("fund_code", "")).strip()
    if len(candidate_code) > 0:
        candidate_code = candidate_code.zfill(6)
        if len(candidate_code) == 6 and candidate_code.isdigit():
            return candidate_code
    return resolve_fund_code_from_factor_selection_input(
        factor_selection_input=stability_analysis_input,
        factor_selection_path=stability_analysis_path,
    )


def choose_weaker_candidate(left_record, right_record):
    # 高相关候选对的劣者按 train ICIR 优先比较，再用 valid 指标和标签打破并列。
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
    # 两两相关性只在每折 train 上计算，再对跨折相关性做均值汇总。
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


def build_corr_dedup_result(selected_summary_df, factor_series_dict, fold_list, corr_threshold=0.90, drop_ratio=0.10, min_drop_count=2):
    # 相关性去冗余只在稳定性保留候选上工作，并全局严格剔除最差 10% 且至少 2 个高相关劣者。
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

    for left_label, right_label in combinations(summary_df["candidate_label"].tolist(), 2):
        mean_train_corr = compute_pair_train_corr(
            left_factor_series=factor_series_dict[str(left_label)],
            right_factor_series=factor_series_dict[str(right_label)],
            fold_list=fold_list,
        )
        if pd.isna(mean_train_corr):
            continue
        abs_mean_train_corr = abs(float(mean_train_corr))
        for candidate_label in [str(left_label), str(right_label)]:
            current_max_corr = mean_train_corr_max_dict[candidate_label]
            if pd.isna(current_max_corr) or abs_mean_train_corr > float(current_max_corr):
                mean_train_corr_max_dict[candidate_label] = abs_mean_train_corr
        if abs_mean_train_corr < corr_threshold:
            continue
        weaker_candidate_label = choose_weaker_candidate(
            left_record=record_by_label[str(left_label)],
            right_record=record_by_label[str(right_label)],
        )
        loss_count_dict[weaker_candidate_label] += 1

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


def build_instance_factor_table(factor_candidate_list, factor_series_dict):
    # 实例级组合使用 candidate_label 作为列名，允许同一 factor_name 的不同参数版本同时进入同一组合。
    factor_table = pd.DataFrame()
    for factor_candidate in factor_candidate_list:
        candidate_label = str(factor_candidate["candidate_label"])
        factor_series = pd.Series(factor_series_dict[candidate_label], copy=True).astype(float)
        factor_table[candidate_label] = factor_series
    return factor_table


def build_instance_combination_score(factor_candidate_list, factor_series_dict):
    # 实例级组合评分固定采用等权求和，避免沿用 factor_name 唯一键路径导致参数版本冲突。
    factor_table = build_instance_factor_table(
        factor_candidate_list=factor_candidate_list,
        factor_series_dict=factor_series_dict,
    )
    score_series = factor_table.sum(axis=1).astype(float)
    score_series.name = "|".join([str(item["candidate_label"]) for item in factor_candidate_list])
    return factor_table, score_series


def build_weighted_instance_combination_score(factor_candidate_list, factor_series_dict, candidate_weight_dict):
    # 因子组合流程使用实例级权重求和，允许不同参数版本按 candidate_label 独立赋权。
    factor_table = build_instance_factor_table(
        factor_candidate_list=factor_candidate_list,
        factor_series_dict=factor_series_dict,
    )
    weighted_factor_table = pd.DataFrame(index=factor_table.index)
    for candidate_label in factor_table.columns.tolist():
        weighted_factor_table[candidate_label] = factor_table[candidate_label] * float(candidate_weight_dict[candidate_label])
    score_series = weighted_factor_table.sum(axis=1).astype(float)
    score_series.name = "|".join([str(item["candidate_label"]) for item in factor_candidate_list])
    return factor_table, score_series


def evaluate_single_factor_train_icir(factor_series, forward_return_series, fold_list):
    # 加权组合的单因子 ICIR 统一只在 train 上计算，保证权重来源和训练目标一致。
    train_metric_list = []
    for fold_dict in fold_list:
        train_metric_list.append(
            compute_segment_correlation_metrics(
                factor_series=factor_series,
                forward_return_series=forward_return_series,
                segment_index=fold_dict["train"].index,
            )
        )
    return float(build_metric_summary([metric_dict["spearman_ic"] for metric_dict in train_metric_list])["icir"])


def build_equal_weight_dict(factor_candidate_list):
    # 等权组合固定给每个实例同样权重，并统一归一化到 1。
    if len(factor_candidate_list) == 0:
        return {}
    weight_value = 1.0 / float(len(factor_candidate_list))
    return {
        str(factor_candidate["candidate_label"]): weight_value
        for factor_candidate in factor_candidate_list
    }


def build_train_icir_weight_dict(factor_candidate_list, factor_series_dict, forward_return_series, fold_list, candidate_train_icir_dict=None):
    # 训练集 ICIR 加权先把负值截断到 0，再归一化；若总和为 0，则回退等权。
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
):
    # 因子组合阶段统一输出 train 指标，并按需要补算 valid，避免 train 搜索阶段过早消耗样本外评估成本。
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
    train_summary = build_metric_summary([metric_dict["spearman_ic"] for metric_dict in train_metric_list])
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
        valid_summary = build_metric_summary([metric_dict["spearman_ic"] for metric_dict in valid_metric_list])
        summary["valid_spearman_ic_mean"] = valid_summary["mean"]
        summary["valid_spearman_icir"] = valid_summary["icir"]
    return summary


def select_best_combination_method_summary(summary_list):
    # 组合方式比较优先看 valid ICIR，再看 valid IC 和 train ICIR，完全相同则默认等权。
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
    # 参数微调后的最终参数只在进入 valid 的候选中比较，规则与组合选择保持一致但不依赖 step 字段。
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


def select_best_combination_trial_summary(summary_list):
    # 权重微调后的最终方案只在进入 valid 的 trial 中比较，不依赖 forward selection 的 step 字段。
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
    # Optuna trial 先生成原始权重，再统一归一化后才进入 train 评估，总和过小则回退基准权重。
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
    # 权重微调范围固定为基准权重上下 5%，并保持非负，最终仍需再次归一化。
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
):
    # 权重微调只对优胜组合方式运行，trial 固定 100 轮，再把 train 前 50 个 trial 放到 valid 上比较。
    optuna_module = load_optuna_module()
    selected_method_summary = dict(selected_method_summary)
    base_weight_dict = {
        str(candidate_label): float(weight_value)
        for candidate_label, weight_value in dict(selected_method_summary["candidate_weight_dict"]).items()
    }
    weight_search_range_dict = build_weight_search_range_dict(base_weight_dict=base_weight_dict)
    train_trial_summary_list = []

    # trial 只围绕基准权重上下 5% 生成原始权重，再归一化成真正可执行的组合权重。
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


def save_factor_combination_output(dedup_selection_input, factor_combination_output, output_dir, fund_code):
    # 因子组合流程输出单独落盘，只保留 dedup 输入路径引用，避免继续复制上游完整结果树。
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    output_path = output_dir / f"factor_combination_{str(fund_code).zfill(6)}_{date_str}.json"
    payload = {
        "input_ref": {
            "dedup_selection_path": str(factor_combination_output.get("dedup_selection_path", "")),
            "fund_code": str(factor_combination_output.get("fund_code", str(fund_code).zfill(6))),
        },
        "factor_combination_output": factor_combination_output,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


def print_factor_combination_summary(result):
    # 因子组合流程终端只输出当前使用的组合方式和最终微调后的权重结果，完整 trial 细节落盘保存。
    print("因子组合结果:")
    print("基金代码:", result["fund_code"])
    print("输入去冗余文件:", result["dedup_selection_path"])
    print("输入因子组合:", result["input_candidate_label_list"])
    print("组合对比优胜方法:", result["combination_compare_output"]["selected_method"])
    print("权重微调后组合:", result["best_combination_selection_summary"]["candidate_label_list"])
    print("权重微调后方法:", result["best_combination_selection_summary"]["selected_method"])
    print("汇总输出:", result["summary_path"])


def load_factor_combination_input(factor_combination_path):
    # 策略回测流程直接消费 factor_combination 顶层结果树，并要求最终组合已在前序流程中确定。
    factor_combination_path = Path(factor_combination_path)
    if not factor_combination_path.exists():
        raise FileNotFoundError(f"因子组合结果文件不存在: {factor_combination_path}")
    with factor_combination_path.open("r", encoding="utf-8") as input_file:
        factor_combination_input = json.load(input_file)
    if not isinstance(factor_combination_input, dict):
        raise ValueError("因子组合结果文件必须是顶层字典。")
    factor_combination_output = factor_combination_input.get("factor_combination_output")
    if not isinstance(factor_combination_output, dict):
        raise ValueError("因子组合结果文件缺少 factor_combination_output 子字典。")
    if not isinstance(factor_combination_output.get("best_combination_selection_summary"), dict):
        raise ValueError("因子组合结果文件缺少 factor_combination_output.best_combination_selection_summary。")
    if not isinstance(factor_combination_output.get("factor_candidate_record_dict"), dict):
        raise ValueError("因子组合结果文件缺少 factor_combination_output.factor_candidate_record_dict。")
    return factor_combination_input, factor_combination_path


def resolve_fund_code_from_factor_combination_input(factor_combination_input, factor_combination_path):
    # 策略回测优先复用 factor_combination 输出中的基金代码，再回退到前序结果树和文件名约定。
    factor_combination_output = dict(factor_combination_input.get("factor_combination_output", {}))
    candidate_code = str(factor_combination_output.get("fund_code", "")).strip()
    if len(candidate_code) > 0:
        candidate_code = candidate_code.zfill(6)
        if len(candidate_code) == 6 and candidate_code.isdigit():
            return candidate_code
    return resolve_fund_code_from_dedup_selection_input(
        dedup_selection_input=factor_combination_input,
        dedup_selection_path=factor_combination_path,
    )


def build_strategy_score_series(factor_candidate_list, factor_series_dict, candidate_weight_dict):
    # 策略回测直接复用组合流程已确定的实例级因子集合与权重，避免在回测层重新搜索组合结构。
    _, score_series = build_weighted_instance_combination_score(
        factor_candidate_list=factor_candidate_list,
        factor_series_dict=factor_series_dict,
        candidate_weight_dict=candidate_weight_dict,
    )
    return score_series.astype(float)


def build_sigmoid_position_series(score_series, center, slope):
    # sigmoid 仓位函数适合把组合打分映射为平滑的 0 到 1 仓位。
    score_series = pd.Series(score_series, copy=True).astype(float)
    position_series = 1.0 / (1.0 + np.exp(-float(slope) * (score_series - float(center))))
    return pd.Series(position_series, index=score_series.index, dtype=float).clip(lower=0.0, upper=1.0)


def build_tanh_rescaled_position_series(score_series, center, slope):
    # tanh 仓位函数保留对称性，再统一缩放到 0 到 1 区间。
    score_series = pd.Series(score_series, copy=True).astype(float)
    position_series = 0.5 * (np.tanh(float(slope) * (score_series - float(center))) + 1.0)
    return pd.Series(position_series, index=score_series.index, dtype=float).clip(lower=0.0, upper=1.0)


def build_piecewise_linear_position_series(score_series, lower, upper):
    # 分段线性仓位函数保留最直观的阈值解释，并把中间区间线性映射到 0 到 1。
    score_series = pd.Series(score_series, copy=True).astype(float)
    lower = float(lower)
    upper = float(upper)
    if upper <= lower:
        upper = lower + 1e-6
    position_series = (score_series - lower) / (upper - lower)
    return pd.Series(position_series, index=score_series.index, dtype=float).clip(lower=0.0, upper=1.0)


def build_raw_position_series(score_series, position_function_name, function_param_dict):
    # 不同仓位函数统一在这里收敛成目标仓位序列，便于上层搜索逻辑按函数名切换。
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
    # 连续仓位先对目标仓位做 EMA 平滑，减少高频来回切换引入的无效换手。
    position_series = pd.Series(position_series, copy=True).astype(float)
    ema_span = max(1, int(ema_span))
    return position_series.ewm(span=ema_span, adjust=False).mean().clip(lower=0.0, upper=1.0)


def apply_position_change_gate(position_series, trade_gate):
    # 调仓门槛只作用于仓位变化幅度，小于门槛时沿用上一日仓位，避免微小抖动导致过度交易。
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
    # 连续仓位统一走 仓位函数 -> EMA 平滑 -> 调仓门槛 三段式链路，保持不同函数之间可公平比较。
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
    # 连续仓位回测按前一日持仓赚取当日收益，并按仓位变化量扣减手续费，不依赖布尔开平仓引擎。
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
    # train、valid、test 三段都复用相同的连续仓位执行逻辑，只在这里统一封装指标和样本边界。
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
    # 回测落盘只保留样本边界、交易次数和指标，避免把大体量时序对象直接写进 JSON。
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
    # 各仓位函数都在统一的打分尺度上定义搜索空间，便于后续 train 搜索和跨函数 test 比较。
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
    # 仓位函数参数采样统一按配置驱动，避免把具体搜索空间散落在 objective 分支里。
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
    # 函数内 valid 选优和跨函数 test 选优都收敛到同一比较规则，避免多个阶段口径漂移。
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
    # 每个仓位函数先在 train 上独立用 Optuna 搜索参数，再把 train 前 50 个候选放到 valid 上选出该函数最优方案。
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


def save_strategy_backtest_output(factor_combination_input, strategy_backtest_output, output_dir, fund_code):
    # 策略回测输出独立落盘，只保留 factor_combination 输入路径引用，避免继续携带上游树结构。
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    output_path = output_dir / f"strategy_backtest_{str(fund_code).zfill(6)}_{date_str}.json"
    payload = {
        "input_ref": {
            "factor_combination_path": str(strategy_backtest_output.get("factor_combination_path", "")),
            "fund_code": str(strategy_backtest_output.get("fund_code", str(fund_code).zfill(6))),
        },
        "strategy_backtest_output": strategy_backtest_output,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


def print_strategy_backtest_summary(result):
    # 终端摘要固定展示 test 层最终优胜仓位函数，避免把大量 train/valid 试验细节打印到控制台。
    print("策略回测结果:")
    print("基金代码:", result["fund_code"])
    print("输入因子组合文件:", result["factor_combination_path"])
    print("最终仓位函数:", result["best_strategy_test_summary"]["position_function_name"])
    print("最终组合因子:", result["best_strategy_test_summary"]["candidate_label_list"])
    print("最终 test Sharpe:", result["best_strategy_test_summary"]["test_result"]["stats"]["sharpe"])
    print("图像输出:", result["plot_path"])
    print("汇总输出:", result["summary_path"])


def run_strategy_backtest(config_override=None):
    # 策略回测流程独立消费 factor_combination 结果，并在连续仓位路径上完成 train 搜索、valid 选参与 test 定型。
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
    best_function_valid_summary = select_best_strategy_trial_summary(
        summary_list=function_trial_summary_list,
        segment_name="valid",
    )
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


def evaluate_factor_candidate_subset(factor_candidate_list, factor_series_dict, forward_return_series, fold_list, include_valid=True):
    # 候选组合统一转换成实例级组合分数序列，再按指定阶段复用现有 IC 与 ICIR 汇总逻辑。
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
    train_summary = build_metric_summary([metric_dict["spearman_ic"] for metric_dict in train_metric_list])
    summary_dict = {
        "candidate_label_list": [str(item["candidate_label"]) for item in factor_candidate_list],
        "factor_count": int(len(factor_candidate_list)),
        "train_spearman_ic_mean": train_summary["mean"],
        "train_spearman_icir": train_summary["icir"],
    }
    if include_valid:
        valid_summary = build_metric_summary([metric_dict["spearman_ic"] for metric_dict in valid_metric_list])
        summary_dict["valid_spearman_ic_mean"] = valid_summary["mean"]
        summary_dict["valid_spearman_icir"] = valid_summary["icir"]
    return summary_dict


def build_candidate_label_signature(candidate_label_list):
    # 树形搜索按组合签名去重，避免不同扩展顺序生成同一候选集合。
    return tuple(sorted([str(candidate_label) for candidate_label in candidate_label_list]))


def run_train_forward_selection(candidate_record_list, factor_series_dict, forward_return_series, fold_list, root_topk=3):
    # 组合搜索从 train 上最优的 topk 个单因子根节点出发，只保留 train ICIR 不下降的扩展分支。
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

    # 第一层只从 train 表现最优的 topk 个单因子根节点出发，控制树形扩展的起始分支数量。
    frontier_node_list = []
    for root_candidate_record in sorted_candidate_record_list[:root_topk]:
        root_candidate_list = [dict(root_candidate_record)]
        root_summary = evaluate_factor_candidate_subset(
            factor_candidate_list=root_candidate_list,
            factor_series_dict=factor_series_dict,
            forward_return_series=forward_return_series,
            fold_list=fold_list,
            include_valid=False,
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

    while len(frontier_node_list) > 0:
        next_frontier_node_list = []
        for frontier_node in frontier_node_list:
            parent_candidate_list = list(frontier_node["candidate_record_list"])
            parent_summary = dict(frontier_node["summary"])
            parent_signature = build_candidate_label_signature(parent_summary["candidate_label_list"])
            parent_candidate_label_set = set(parent_summary["candidate_label_list"])
            for candidate_record in sorted_candidate_record_list:
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
    # valid 评估前先按 train ICIR 对组合路径做截断，减少后续样本外评估数量。
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


def evaluate_valid_for_path_summary_list(path_summary_list, candidate_record_lookup, factor_series_dict, forward_return_series, fold_list):
    # 进入 valid 评估的路径只补算 valid 指标，train 指标沿用树搜索阶段的结果。
    evaluated_summary_list = []
    for path_summary in path_summary_list:
        factor_candidate_list = [dict(candidate_record_lookup[candidate_label]) for candidate_label in path_summary["candidate_label_list"]]
        evaluated_summary = evaluate_factor_candidate_subset(
            factor_candidate_list=factor_candidate_list,
            factor_series_dict=factor_series_dict,
            forward_return_series=forward_return_series,
            fold_list=fold_list,
            include_valid=True,
        )
        evaluated_summary["step"] = int(path_summary["step"])
        evaluated_summary_list.append(evaluated_summary)
    return evaluated_summary_list


def select_best_forward_path_summary(forward_selection_path_summary):
    # 最终组合选择只看 valid 表现，再用 train 稳定性和更小组合打破并列。
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
):
    # Optuna 扩展搜索只在前向选择基准之外的剩余因子上做布尔搜索，并且只保留 train 上优于基准的组合。
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

    # 搜索空间只允许在前向选择基准组合上附加剩余因子，避免回头改变已确认的基准结构。
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
    study.optimize(objective, n_trials=n_trials)

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


def save_single_factor_dedup_selection_output(stability_analysis_input, dedup_selection_output, output_dir, fund_code):
    # 去冗余结果独立落盘，只保留稳定性分析输入路径引用，避免继续复制上游完整结果树。
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    output_path = output_dir / f"single_factor_dedup_{str(fund_code).zfill(6)}_{date_str}.json"
    stability_analysis_output = dict(stability_analysis_input.get("stability_analysis_output", {}))
    payload = {
        "input_ref": {
            "stability_analysis_path": str(dedup_selection_output.get("stability_analysis_path", "")),
            "fund_code": str(stability_analysis_output.get("fund_code", str(fund_code).zfill(6))),
        },
        "dedup_selection_output": dedup_selection_output,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


def load_dedup_selection_input(dedup_selection_path):
    # 因子组合流程直接消费 dedup JSON 顶层字典，并要求其中已有 dedup_selection_output 子字典。
    dedup_selection_path = Path(dedup_selection_path)
    if not dedup_selection_path.exists():
        raise FileNotFoundError(f"去冗余结果文件不存在: {dedup_selection_path}")
    with dedup_selection_path.open("r", encoding="utf-8") as input_file:
        dedup_selection_input = json.load(input_file)
    if not isinstance(dedup_selection_input, dict):
        raise ValueError("去冗余结果文件必须是顶层字典。")
    dedup_selection_output = dedup_selection_input.get("dedup_selection_output")
    if not isinstance(dedup_selection_output, dict):
        raise ValueError("去冗余结果文件缺少 dedup_selection_output 子字典。")
    if not isinstance(dedup_selection_output.get("record_dict"), dict):
        raise ValueError("去冗余结果文件缺少 dedup_selection_output.record_dict。")
    if not isinstance(dedup_selection_output.get("best_final_selection_summary"), dict):
        raise ValueError("去冗余结果文件缺少 dedup_selection_output.best_final_selection_summary。")
    return dedup_selection_input, dedup_selection_path


def resolve_fund_code_from_dedup_selection_input(dedup_selection_input, dedup_selection_path):
    # 因子组合流程优先复用 dedup 输出中的基金代码，再回退到前序结果和文件名约定。
    dedup_selection_output = dict(dedup_selection_input.get("dedup_selection_output", {}))
    candidate_code = str(dedup_selection_output.get("fund_code", "")).strip()
    if len(candidate_code) > 0:
        candidate_code = candidate_code.zfill(6)
        if len(candidate_code) == 6 and candidate_code.isdigit():
            return candidate_code
    return resolve_fund_code_from_stability_analysis_input(
        stability_analysis_input=dedup_selection_input,
        stability_analysis_path=dedup_selection_path,
    )


def run_factor_combination(config_override=None):
    # 因子组合流程独立消费 dedup 结果，只比较组合方式并在优胜方法上做权重微调，不再改写前一阶段文件。
    config = build_tradition_config(config_override=config_override)
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
        code_dict=config["code_dict"],
        cache_dir=config["data_dir"],
        force_refresh=bool(config["force_refresh"]),
        cache_prefix=config["cache_prefix"],
    )
    normalized_data = normalize_fund_data(raw_data)
    fund_df = filter_single_fund(normalized_data, fund_code=fund_code)
    price_series, data_mode = adapt_to_price_series(fund_df=fund_df)
    fold_list = build_walk_forward_fold_list(
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
        ),
        evaluate_combination_summary(
            factor_candidate_list=factor_candidate_list,
            factor_series_dict=factor_series_dict,
            forward_return_series=forward_return_series,
            fold_list=fold_list,
            method_name="train_icir_weighted",
            candidate_train_icir_dict=candidate_train_icir_dict,
            include_valid=True,
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


def print_single_factor_dedup_selection_summary(result):
    # 终端摘要只展示去冗余后保留数量和最终正向选择结果，避免把大体量路径信息打印到控制台。
    print("单因子去冗余与正向选择结果:")
    print("基金代码:", result["fund_code"])
    print("输入稳定性文件:", result["stability_analysis_path"])
    print("输入稳定性保留候选数量:", len(result["selected_stability_candidate_list"]))
    print("相关性去冗余后候选数量:", len(result["corr_selected_summary_df"]))
    print("最终组合来源:", result["final_selected_source"])
    print("最终正向选择组合:", result["best_final_selection_summary"]["candidate_label_list"])
    print("汇总输出:", result["summary_path"])


def print_single_factor_stability_analysis_summary(result):
    # 终端只打印最终稳定性排序结果的高信号摘要，完整输入输出仍以 JSON 落盘保存。
    print("单因子稳定性分析结果:")
    print("基金代码:", result["fund_code"])
    print("输入筛选文件:", result["factor_selection_path"])
    print("输入最终候选数量:", len(result["selected_factor_input_list"]))
    print("稳定性分析候选数量:", len(result["summary_df"]))
    print("稳定性尾部剔除后保留数量:", len(result["selected_summary_df"]))
    print("稳定性排序候选列表:", result["selected_candidate_label_list"])
    if len(result["selected_summary_df"]) > 0:
        printable_df = result["selected_summary_df"][
            [
                "stability_rank",
                "candidate_label",
                "factor_group",
                "train_spearman_ic_mean",
                "valid_spearman_ic_mean",
                "train_valid_ic_mean_gap",
                "valid_trimmed_ic_gap",
                "valid_ic_flip_count",
            ]
        ].copy()
        print(printable_df.to_string(index=False))
    print("汇总输出:", result["summary_path"])


def run_single_factor_stability_analysis(config_override=None):
    # 单因子稳定性分析只消费已筛选候选，并严格限制在 train/valid 两段，不进入 test 或交易层风险分析。
    config = build_tradition_config(config_override=config_override)
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

    raw_data = fetch_fund_data_with_cache(
        code_dict=config["code_dict"],
        cache_dir=config["data_dir"],
        force_refresh=bool(config["force_refresh"]),
        cache_prefix=config["cache_prefix"],
    )
    normalized_data = normalize_fund_data(raw_data)
    fund_df = filter_single_fund(normalized_data, fund_code=fund_code)
    price_series, data_mode = adapt_to_price_series(fund_df=fund_df)
    fold_list = build_walk_forward_fold_list(
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
        "factor_selection_path": str(resolved_factor_selection_path),
        "analysis_date": datetime.today().strftime("%Y-%m-%d"),
        "candidate_count": int(len(summary_df)),
        "selected_count": int(selected_mask.sum()),
        "tail_reject_candidate_label_list": summary_df.loc[~selected_mask, "candidate_label"].tolist(),
        "selected_candidate_label_list": selected_summary_df["candidate_label"].tolist(),
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


def run_single_factor_dedup_selection(config_override=None):
    # 去冗余流程只消费稳定性保留候选，先做 train 相关性去冗余，再跑 train forward selection 并在 valid 上挑最优组合。
    config = build_tradition_config(config_override=config_override)
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

    raw_data = fetch_fund_data_with_cache(
        code_dict=config["code_dict"],
        cache_dir=config["data_dir"],
        force_refresh=bool(config["force_refresh"]),
        cache_prefix=config["cache_prefix"],
    )
    normalized_data = normalize_fund_data(raw_data)
    fund_df = filter_single_fund(normalized_data, fund_code=fund_code)
    price_series, data_mode = adapt_to_price_series(fund_df=fund_df)
    fold_list = build_walk_forward_fold_list(
        price_series=price_series,
        walk_forward_config=dict(config["walk_forward_config"]),
        split_config=config["data_split_dict"],
    )
    base_multi_factor_params = dict(config["strategy_param_dict"]["multi_factor_score"])
    forward_return_series = build_forward_return_series(price_series=price_series, forward_window=5)

    factor_series_dict = {}
    for factor_candidate in selected_stability_candidate_list:
        factor_series_dict[str(factor_candidate["candidate_label"])] = build_single_factor_series(
            price_series=price_series,
            factor_name=factor_candidate["factor_name"],
            strategy_params=base_multi_factor_params,
            factor_param_override=dict(factor_candidate["factor_param_dict"]),
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

    train_forward_selection_path_summary = run_train_forward_selection(
        candidate_record_list=corr_selected_candidate_list,
        factor_series_dict=factor_series_dict,
        forward_return_series=forward_return_series,
        fold_list=fold_list,
        root_topk=dedup_root_topk,
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
    )
    best_final_selection_summary = dict(optuna_extension_output["best_final_selection_summary"])

    forward_selected_candidate_label_set = set(best_final_selection_summary["candidate_label_list"])
    corr_summary_df["forward_selected"] = corr_summary_df["candidate_label"].isin(forward_selected_candidate_label_set)
    corr_summary_df["forward_selection_step"] = pd.Series([None] * len(corr_summary_df), dtype=object)
    for step_idx, candidate_label in enumerate(best_final_selection_summary["candidate_label_list"], start=1):
        corr_summary_df.loc[corr_summary_df["candidate_label"] == candidate_label, "forward_selection_step"] = step_idx

    dedup_selection_output = {
        "fund_code": fund_code,
        "stability_analysis_path": str(resolved_stability_analysis_path),
        "analysis_date": datetime.today().strftime("%Y-%m-%d"),
        "dedup_root_topk": dedup_root_topk,
        "input_candidate_count": int(len(selected_stability_candidate_list)),
        "corr_dedup_drop_count": int(len(dropped_candidate_label_list)),
        "corr_dedup_selected_count": int(corr_selected_mask.sum()),
        "train_path_count": int(len(train_forward_selection_path_summary)),
        "valid_eval_count": int(len(forward_selection_path_summary)),
        "valid_eval_ratio": 0.5,
        "forward_selected_count": int(len(best_final_selection_summary["candidate_label_list"])),
        "corr_dedup_dropped_candidate_label_list": dropped_candidate_label_list,
        "corr_dedup_selected_candidate_label_list": corr_selected_summary_df["candidate_label"].tolist(),
        "forward_selected_candidate_label_list": list(best_final_selection_summary["candidate_label_list"]),
        "record_dict": {
            candidate_label: build_candidate_record_dict(
                summary_df=corr_summary_df[corr_summary_df["candidate_label"] == candidate_label]
            )[candidate_label]
            for candidate_label in best_final_selection_summary["candidate_label_list"]
        },
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
