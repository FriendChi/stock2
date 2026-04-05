import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from tradition.config import build_tradition_config
from tradition.data_adapter import adapt_to_price_series
from tradition.data_fetcher import fetch_fund_data_with_cache
from tradition.data_loader import filter_single_fund, normalize_fund_data
from tradition.factor_engine import FACTOR_POOL_DICT, build_single_factor_series, resolve_factor_name_list_by_group
from tradition.splitter import build_walk_forward_fold_list


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


def build_candidate_record_dict(summary_df):
    # JSON 输出统一改成以 candidate_label 为键的子字典，避免后续流程再依赖列表位置索引。
    normalized_df = summary_df.where(pd.notna(summary_df), None)
    record_dict = {}
    for record in normalized_df.to_dict(orient="records"):
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
        "record_dict": build_candidate_record_dict(summary_df=summary_df),
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
    # 稳定性分析输出直接追加到已有顶层字典，形成按流程逐步扩展的统一结果树。
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    output_path = output_dir / f"single_factor_stability_{str(fund_code).zfill(6)}_{date_str}.json"
    payload = dict(factor_selection_input)
    payload["stability_analysis_output"] = stability_analysis_output
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


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
        "record_dict": build_candidate_record_dict(summary_df=summary_df),
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
