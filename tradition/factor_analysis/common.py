import numpy as np
import pandas as pd

from tradition.factor_engine import FACTOR_POOL_DICT, build_raw_factor_series, resolve_factor_name_list_by_group


def build_forward_return_series(price_series, forward_window=5):
    series = pd.Series(price_series, copy=True).astype(float).dropna()
    forward_window = int(forward_window)
    if forward_window <= 0:
        raise ValueError("forward_window 必须为正整数。")
    forward_return_series = series.shift(-forward_window) / series - 1.0
    forward_return_series.name = f"forward_return_{forward_window}d"
    return forward_return_series


def compute_segment_correlation_metrics(factor_series, forward_return_series, segment_index):
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
    metric_series = pd.Series(metric_value_list, dtype=float).dropna()
    if len(metric_series) == 0:
        return 0.0
    return float((metric_series > 0.0).mean())


def build_trimmed_ic_mean(metric_value_list, trim_ratio=0.1):
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
    factor_name = str(factor_name)
    factor_param_dict = dict(factor_param_dict)
    if len(factor_param_dict) == 0:
        return factor_name
    param_text = ", ".join(
        [f"{param_name}={factor_param_dict[param_name]}" for param_name in sorted(factor_param_dict.keys())]
    )
    return f"{factor_name}({param_text})"


def expand_param_values_from_search_space(search_space):
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
    resolved_factor_name = str(factor_name)
    if resolved_factor_name not in FACTOR_POOL_DICT:
        raise ValueError(f"未定义因子: {resolved_factor_name}")
    return str(FACTOR_POOL_DICT[resolved_factor_name]["group"])


def factor_pool_dict():
    return dict(FACTOR_POOL_DICT)


def build_factor_candidate_list(candidate_factor_name_list, strategy_params):
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


def build_candidate_record_dict(summary_df, keep_field_list=None):
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


def build_instance_factor_table(factor_candidate_list, factor_series_dict):
    factor_table = pd.DataFrame()
    for factor_candidate in factor_candidate_list:
        candidate_label = str(factor_candidate["candidate_label"])
        factor_series = pd.Series(factor_series_dict[candidate_label], copy=True).astype(float)
        factor_table[candidate_label] = factor_series
    return factor_table


def build_instance_combination_score(factor_candidate_list, factor_series_dict):
    factor_table = build_instance_factor_table(
        factor_candidate_list=factor_candidate_list,
        factor_series_dict=factor_series_dict,
    )
    score_series = factor_table.sum(axis=1).astype(float)
    score_series.name = "|".join([str(item["candidate_label"]) for item in factor_candidate_list])
    return factor_table, score_series


def build_weighted_instance_combination_score(factor_candidate_list, factor_series_dict, candidate_weight_dict):
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
