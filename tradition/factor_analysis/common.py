from pathlib import Path
import json

import numpy as np
import pandas as pd

from tradition.factor_engine import FACTOR_POOL_DICT, build_raw_factor_series, resolve_factor_name_list_by_group


def load_preprocess_price_series(preprocess_path, expected_fund_code=None):
    # 统一校验流程0输出结构并恢复价格序列，确保后续流程只消费预处理产物。
    if preprocess_path is None:
        raise ValueError("必须提供 preprocess_path。")
    preprocess_path = Path(preprocess_path)
    if not preprocess_path.exists():
        raise FileNotFoundError(f"流程0数据预处理文件不存在: {preprocess_path}")
    preprocess_df = pd.read_csv(preprocess_path, dtype={"fund_code": str})
    required_column_list = ["date", "price", "fund_code", "data_mode"]
    missing_column_list = [column for column in required_column_list if column not in preprocess_df.columns]
    if len(missing_column_list) > 0:
        raise ValueError(f"流程0数据预处理文件缺少字段: {missing_column_list}")
    preprocess_df = preprocess_df[required_column_list].copy()
    preprocess_df["date"] = pd.to_datetime(preprocess_df["date"], errors="coerce")
    preprocess_df["price"] = pd.to_numeric(preprocess_df["price"], errors="coerce")
    preprocess_df["fund_code"] = preprocess_df["fund_code"].astype(str).str.zfill(6)
    preprocess_df["data_mode"] = preprocess_df["data_mode"].astype(str).str.strip()
    preprocess_df = preprocess_df.dropna(subset=["date", "price"])
    if len(preprocess_df) == 0:
        raise ValueError(f"流程0数据预处理文件为空或无有效价格记录: {preprocess_path}")
    fund_code_list = sorted(preprocess_df["fund_code"].dropna().unique().tolist())
    if len(fund_code_list) != 1:
        raise ValueError(f"流程0数据预处理文件基金代码不唯一: {preprocess_path}")
    fund_code = str(fund_code_list[0]).zfill(6)
    if expected_fund_code is not None and fund_code != str(expected_fund_code).zfill(6):
        raise ValueError(f"流程0数据预处理文件基金代码不匹配: expected={str(expected_fund_code).zfill(6)} actual={fund_code}")
    data_mode_list = [value for value in preprocess_df["data_mode"].dropna().unique().tolist() if len(str(value).strip()) > 0]
    if len(data_mode_list) != 1:
        raise ValueError(f"流程0数据预处理文件 data_mode 不唯一: {preprocess_path}")
    data_mode = str(data_mode_list[0])
    price_series = pd.Series(preprocess_df["price"].to_numpy(dtype=float), index=preprocess_df["date"], name="price")
    price_series = price_series.sort_index()
    price_series = price_series[~price_series.index.duplicated(keep="last")]
    return price_series, data_mode, fund_code, preprocess_path


def load_feature_preprocess_bundle(preprocess_path, preprocess_metadata_path, expected_fund_code=None):
    # 新流程0输出必须提供元信息 JSON；若未显式传 CSV，则从 JSON 中回填 csv_path。
    if preprocess_metadata_path is None:
        raise ValueError("必须提供 preprocess_metadata_path。")
    preprocess_metadata_path = Path(preprocess_metadata_path)
    if not preprocess_metadata_path.exists():
        raise FileNotFoundError(f"流程0元信息 JSON 不存在: {preprocess_metadata_path}")
    with preprocess_metadata_path.open("r", encoding="utf-8") as input_file:
        metadata_input = json.load(input_file)
    if not isinstance(metadata_input, dict):
        raise ValueError("流程0元信息 JSON 必须是顶层字典。")
    feature_preprocess_output = metadata_input.get("feature_preprocess_output")
    if not isinstance(feature_preprocess_output, dict):
        raise ValueError("流程0元信息 JSON 缺少 feature_preprocess_output 子字典。")
    fund_code = str(feature_preprocess_output.get("fund_code", "")).zfill(6)
    if expected_fund_code is not None and fund_code != str(expected_fund_code).zfill(6):
        raise ValueError(f"流程0元信息基金代码不匹配: expected={str(expected_fund_code).zfill(6)} actual={fund_code}")
    csv_path_in_metadata = str(feature_preprocess_output.get("csv_path", "")).strip()
    if len(csv_path_in_metadata) == 0:
        raise ValueError("流程0元信息缺少 csv_path。")
    # 仅提供元信息路径时，优先复用流程0固化下来的特征 CSV 绝对路径。
    if preprocess_path is None:
        preprocess_path = csv_path_in_metadata
    preprocess_path = Path(preprocess_path)
    if Path(csv_path_in_metadata).resolve() != preprocess_path.resolve():
        raise ValueError("流程0元信息中的 csv_path 与传入 preprocess_path 不一致。")
    if not preprocess_path.exists():
        raise FileNotFoundError(f"流程0特征 CSV 不存在: {preprocess_path}")
    feature_df = pd.read_csv(preprocess_path)
    required_column_list = ["date"]
    missing_column_list = [column for column in required_column_list if column not in feature_df.columns]
    if len(missing_column_list) > 0:
        raise ValueError(f"流程0特征 CSV 缺少字段: {missing_column_list}")
    target_nav_column = str(feature_preprocess_output.get("target_nav_column", "")).strip()
    target_price_column = str(feature_preprocess_output.get("target_price_column", "")).strip()
    factor_feature_column_list = [str(column) for column in list(feature_preprocess_output.get("factor_feature_column_list", []))]
    if len(target_nav_column) == 0:
        raise ValueError("流程0元信息缺少 target_nav_column。")
    if target_nav_column not in feature_df.columns:
        raise ValueError(f"target_nav_column 不存在于流程0特征 CSV: {target_nav_column}")
    if len(target_price_column) == 0:
        raise ValueError("流程0元信息缺少 target_price_column。")
    if target_price_column not in feature_df.columns:
        raise ValueError(f"target_price_column 不存在于流程0特征 CSV: {target_price_column}")
    missing_factor_column_list = [column for column in factor_feature_column_list if column not in feature_df.columns]
    if len(missing_factor_column_list) > 0:
        raise ValueError(f"流程0元信息中的因子列不存在于 CSV: {missing_factor_column_list[:10]}")
    feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce")
    feature_df = feature_df.dropna(subset=["date"]).copy()
    feature_df = feature_df.sort_values("date").reset_index(drop=True)
    return feature_df, feature_preprocess_output, fund_code, preprocess_path, preprocess_metadata_path


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


def build_ic_aggregation_config(config):
    aggregation_mode = str(config.get("ic_aggregation_mode", "classic")).strip().lower()
    if aggregation_mode not in {"classic", "exp_weighted"}:
        raise ValueError(f"未支持的 ic_aggregation_mode: {aggregation_mode}")
    half_life = float(config.get("ic_exp_weight_half_life", 3.0))
    if aggregation_mode == "exp_weighted" and half_life <= 0.0:
        raise ValueError(f"ic_exp_weight_half_life 必须大于 0，当前值为 {half_life}")
    if half_life <= 0.0:
        half_life = 3.0
    return {
        "mode": aggregation_mode,
        "half_life": half_life,
    }


def build_exp_weighted_metric_summary(metric_value_list, half_life):
    metric_series = pd.Series(metric_value_list, dtype=float).dropna()
    if len(metric_series) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "icir": 0.0,
        }
    half_life = float(half_life)
    if half_life <= 0.0:
        raise ValueError(f"half_life 必须大于 0，当前值为 {half_life}")
    reverse_distance_array = np.arange(len(metric_series) - 1, -1, -1, dtype=float)
    weight_array = np.power(0.5, reverse_distance_array / half_life)
    weight_array = weight_array / weight_array.sum()
    metric_value_array = metric_series.to_numpy(dtype=float)
    metric_mean = float(np.sum(weight_array * metric_value_array))
    metric_std = float(np.sqrt(np.sum(weight_array * np.square(metric_value_array - metric_mean))))
    metric_icir = 0.0
    if abs(metric_std) > 1e-12:
        metric_icir = float(metric_mean / metric_std)
    return {
        "count": int(len(metric_series)),
        "mean": metric_mean,
        "std": metric_std,
        "icir": metric_icir,
    }


def build_spearman_metric_summary(metric_value_list, ic_aggregation_config):
    ic_aggregation_config = dict(ic_aggregation_config)
    aggregation_mode = str(ic_aggregation_config["mode"])
    if aggregation_mode == "classic":
        return build_metric_summary(metric_value_list)
    if aggregation_mode == "exp_weighted":
        return build_exp_weighted_metric_summary(
            metric_value_list=metric_value_list,
            half_life=float(ic_aggregation_config["half_life"]),
        )
    raise ValueError(f"未支持的 Spearman 聚合模式: {aggregation_mode}")


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


def evaluate_single_factor_train_icir(factor_series, forward_return_series, fold_list, ic_aggregation_config=None):
    train_metric_list = []
    for fold_dict in fold_list:
        train_metric_list.append(
            compute_segment_correlation_metrics(
                factor_series=factor_series,
                forward_return_series=forward_return_series,
                segment_index=fold_dict["train"].index,
            )
        )
    if ic_aggregation_config is None:
        ic_aggregation_config = {"mode": "classic", "half_life": 3.0}
    return float(
        build_spearman_metric_summary(
            [metric_dict["spearman_ic"] for metric_dict in train_metric_list],
            ic_aggregation_config=ic_aggregation_config,
        )["icir"]
    )
