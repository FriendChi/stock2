import json

import pandas as pd
import pytest

from tradition import factor_analysis


def test_build_forward_return_series_uses_future_simple_return():
    price_series = pd.Series([1.0, 1.1, 1.21, 1.331, 1.4641, 1.61051], index=pd.date_range("2024-01-01", periods=6, freq="D"))
    forward_return_series = factor_analysis.build_forward_return_series(price_series=price_series, forward_window=5)
    assert abs(float(forward_return_series.iloc[0]) - 0.61051) < 1e-12
    assert pd.isna(forward_return_series.iloc[-1])


def test_build_factor_candidate_list_expands_multi_param_combinations():
    candidate_list = factor_analysis.build_factor_candidate_list(
        candidate_factor_name_list=["momentum", "ma_slope"],
        strategy_params={
            "factor_param_dict": {
                "momentum": {"window": 20},
                "ma_slope": {"window": 20, "lookback": 5},
            }
        },
    )
    momentum_candidate_list = [item for item in candidate_list if item["factor_name"] == "momentum"]
    ma_slope_candidate_list = [item for item in candidate_list if item["factor_name"] == "ma_slope"]
    assert len(momentum_candidate_list) == len(range(10, 121, 5))
    assert len(ma_slope_candidate_list) == len(range(10, 61, 5)) * len(range(2, 21, 1))
    assert ma_slope_candidate_list[0]["candidate_label"].startswith("ma_slope(")


def test_build_metric_summary_returns_zero_icir_for_constant_metric_series():
    summary = factor_analysis.build_metric_summary([0.1, 0.1, 0.1])
    assert abs(summary["mean"] - 0.1) < 1e-12
    assert abs(summary["std"] - 0.0) < 1e-12
    assert abs(summary["icir"] - 0.0) < 1e-12


def test_build_positive_ic_ratio_uses_positive_valid_fold_ratio():
    assert abs(factor_analysis.build_positive_ic_ratio([0.1, -0.2, 0.0, 0.3, float("nan")]) - (2 / 4)) < 1e-12
    assert abs(factor_analysis.build_positive_ic_ratio([float("nan")]) - 0.0) < 1e-12


def test_build_trimmed_ic_mean_and_flip_count_follow_stability_rules():
    assert abs(factor_analysis.build_trimmed_ic_mean([1, 2, 3, 100, -50, 4, 5, 6, 7, 8]) - 4.5) < 1e-12
    assert factor_analysis.build_ic_flip_count([0.1, 0.2, -0.1, 0.0, -0.3, 0.4, float("nan")]) == 2
    assert factor_analysis.build_ic_flip_count([0.0, float("nan")]) == 0


def test_build_tail_reject_mask_rejects_union_of_worst_five_percent():
    summary_df = pd.DataFrame(
        [
            {
                "candidate_label": f"factor_{idx}",
                "train_valid_ic_mean_gap": 0.001 * idx,
                "valid_ic_flip_count": idx,
                "valid_trimmed_ic_gap": 0.002 * idx,
            }
            for idx in range(20)
        ]
    )
    tail_reject_flag_df = factor_analysis.build_tail_reject_mask(summary_df=summary_df, reject_ratio=0.05)
    assert bool(tail_reject_flag_df.loc[19, "abs_train_valid_ic_mean_gap_tail_rejected"]) is True
    assert bool(tail_reject_flag_df.loc[19, "valid_ic_flip_count_tail_rejected"]) is True
    assert bool(tail_reject_flag_df.loc[19, "abs_valid_trimmed_ic_gap_tail_rejected"]) is True
    assert bool(tail_reject_flag_df.loc[19, "stability_tail_rejected"]) is True
    assert int(tail_reject_flag_df["stability_tail_rejected"].sum()) == 1


def test_resolve_factor_group_name_raises_for_unknown_factor():
    with pytest.raises(ValueError, match="未定义因子"):
        factor_analysis.resolve_factor_group_name("unknown_factor")


def test_run_factor_selection_single_fund_returns_ranked_selection(monkeypatch, sample_fund_df, tmp_path):
    sample_index = pd.date_range("2024-01-01", periods=30, freq="D")
    sample_df = pd.DataFrame(
        {
            "date": sample_index,
            "code": ["007301"] * len(sample_index),
            "fund": ["半导体"] * len(sample_index),
            "nav": [1.0 + idx * 0.01 for idx in range(len(sample_index))],
        }
    )
    monkeypatch.setattr(
        factor_analysis,
        "build_tradition_config",
        lambda config_override=None: {
            "code_dict": {"007301": "半导体"},
            "data_dir": tmp_path,
            "output_dir": tmp_path,
            "force_refresh": False,
            "cache_prefix": "tradition_fund",
            "default_fund_code": "007301",
            "strategy_param_dict": {
                "multi_factor_score": {
                    "enabled_factor_list": ["momentum", "ma_slope"],
                    "factor_param_dict": {
                        "momentum": {"window": 20},
                        "ma_slope": {"window": 20, "lookback": 5},
                    },
                    "score_window": 60,
                    "factor_weight_dict": {
                        "momentum": 0.5,
                        "ma_slope": 0.5,
                    },
                }
            },
            "walk_forward_config": {
                "window_size": 20,
                "step_size": 5,
                "min_fold_count": 2,
            },
            "data_split_dict": {
                "train_ratio": 0.5,
                "valid_ratio": 0.25,
                "test_ratio": 0.25,
                "min_segment_size": 5,
            },
            "factor_group_list": ["趋势/动量", "均线趋势"],
            "train_min_spearman_ic": 0.1,
            "train_min_spearman_icir": 0.0,
        },
    )
    monkeypatch.setattr(factor_analysis, "fetch_fund_data_with_cache", lambda **kwargs: sample_df)
    monkeypatch.setattr(factor_analysis, "normalize_fund_data", lambda data: data)
    monkeypatch.setattr(factor_analysis, "filter_single_fund", lambda data, fund_code: data)
    monkeypatch.setattr(
        factor_analysis,
        "adapt_to_price_series",
        lambda fund_df: (
            pd.Series(fund_df["nav"].values, index=pd.to_datetime(fund_df["date"]), dtype=float),
            "nav_price_series",
        ),
    )
    monkeypatch.setattr(
        factor_analysis,
        "build_walk_forward_fold_list",
        lambda price_series, walk_forward_config, split_config: [
            {
                "fold_id": 1,
                "train": price_series.iloc[:10],
                "valid": price_series.iloc[10:15],
                "test": price_series.iloc[15:20],
            },
            {
                "fold_id": 2,
                "train": price_series.iloc[5:15],
                "valid": price_series.iloc[15:20],
                "test": price_series.iloc[20:25],
            },
        ],
    )
    monkeypatch.setattr(
        factor_analysis,
        "build_single_factor_series",
        lambda price_series, factor_name, strategy_params, factor_param_override=None: pd.Series(
            range(len(price_series))
            if factor_name == "momentum" and int(factor_param_override["window"]) == 10
            else list(reversed(range(len(price_series)))),
            index=price_series.index,
            dtype=float,
            name=factor_analysis.build_factor_candidate_label(factor_name=factor_name, factor_param_dict=factor_param_override or {}),
        ),
    )
    monkeypatch.setattr(
        factor_analysis,
        "build_forward_return_series",
        lambda price_series, forward_window=5: pd.Series(range(len(price_series)), index=price_series.index, dtype=float),
    )

    def fake_compute_segment_correlation_metrics(factor_series, forward_return_series, segment_index):
        segment_start = pd.Index(segment_index).min()
        if factor_series.name == "momentum(window=10)":
            if len(segment_index) == 10:
                return {"sample_size": len(segment_index), "spearman_ic": 0.4, "pearson_ic": 0.3}
            if segment_start == pd.Timestamp("2024-01-11"):
                return {"sample_size": len(segment_index), "spearman_ic": 0.4, "pearson_ic": 0.3}
            return {"sample_size": len(segment_index), "spearman_ic": 0.5, "pearson_ic": 0.35}
        if factor_series.name == "momentum(window=15)":
            if len(segment_index) == 10:
                return {"sample_size": len(segment_index), "spearman_ic": 0.3, "pearson_ic": 0.35}
            return {"sample_size": len(segment_index), "spearman_ic": 0.3, "pearson_ic": 0.2}
        return {"sample_size": len(segment_index), "spearman_ic": -0.1, "pearson_ic": -0.1}

    monkeypatch.setattr(factor_analysis, "compute_segment_correlation_metrics", fake_compute_segment_correlation_metrics)

    result = factor_analysis.run_factor_selection_single_fund()

    assert result["fund_code"] == "007301"
    assert result["selected_factor_name_list"] == ["momentum"]
    assert result["selected_candidate_label_list"] == ["momentum(window=10)"]
    assert result["summary_df"].iloc[0]["candidate_label"] == "momentum(window=10)"
    assert result["summary_df"].iloc[0]["factor_param_dict"] == {"window": 10}
    assert bool(result["summary_df"].iloc[0]["train_passed"]) is True
    assert bool(result["summary_df"].iloc[0]["valid_passed"]) is True
    assert abs(float(result["summary_df"].iloc[0]["train_spearman_positive_ic_ratio"]) - 1.0) < 1e-12
    assert result["summary_df"].iloc[0]["final_rank"] == 1
    assert result["summary_df"].iloc[1]["candidate_label"] == "momentum(window=15)"
    assert bool(result["summary_df"].iloc[1]["train_passed"]) is True
    assert bool(result["summary_df"].iloc[1]["valid_passed"]) is False
    assert result["summary_df"].iloc[1]["final_rank"] is None
    assert result["selected_summary_df"].shape[0] == 1
    assert result["summary_path"].exists()
    assert result["summary_path"].suffix == ".json"
    saved_payload = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    assert "factor_selection_output" in saved_payload
    assert saved_payload["factor_selection_output"]["fund_code"] == "007301"
    assert saved_payload["factor_selection_output"]["record_dict"]["momentum(window=10)"]["candidate_label"] == "momentum(window=10)"
    assert saved_payload["factor_selection_output"]["record_dict"]["momentum(window=10)"]["factor_param_dict"] == {"window": 10}


def test_run_single_factor_stability_analysis_outputs_nested_json(monkeypatch, tmp_path):
    sample_index = pd.date_range("2024-01-01", periods=30, freq="D")
    sample_df = pd.DataFrame(
        {
            "date": sample_index,
            "code": ["007301"] * len(sample_index),
            "fund": ["半导体"] * len(sample_index),
            "nav": [1.0 + idx * 0.01 for idx in range(len(sample_index))],
        }
    )
    factor_selection_path = tmp_path / "factor_selection_007301_2026-04-05.json"
    factor_selection_path.write_text(
        json.dumps(
            {
                "factor_selection_output": {
                    "fund_code": "007301",
                    "record_dict": {
                        "momentum(window=10)": {
                            "factor_name": "momentum",
                            "factor_group": "趋势/动量",
                            "factor_param_dict": {"window": 10},
                            "candidate_label": "momentum(window=10)",
                            "selected": True,
                        },
                        "momentum(window=15)": {
                            "factor_name": "momentum",
                            "factor_group": "趋势/动量",
                            "factor_param_dict": {"window": 15},
                            "candidate_label": "momentum(window=15)",
                            "selected": False,
                        },
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        factor_analysis,
        "build_tradition_config",
        lambda config_override=None: {
            "code_dict": {"007301": "半导体"},
            "data_dir": tmp_path,
            "output_dir": tmp_path,
            "force_refresh": False,
            "cache_prefix": "tradition_fund",
            "strategy_param_dict": {
                "multi_factor_score": {
                    "enabled_factor_list": ["momentum"],
                    "factor_param_dict": {
                        "momentum": {"window": 20},
                    },
                    "score_window": 60,
                    "factor_weight_dict": {
                        "momentum": 1.0,
                    },
                }
            },
            "walk_forward_config": {
                "window_size": 20,
                "step_size": 5,
                "min_fold_count": 2,
            },
            "data_split_dict": {
                "train_ratio": 0.5,
                "valid_ratio": 0.25,
                "test_ratio": 0.25,
                "min_segment_size": 5,
            },
            "factor_selection_path": str(factor_selection_path),
        },
    )
    monkeypatch.setattr(factor_analysis, "fetch_fund_data_with_cache", lambda **kwargs: sample_df)
    monkeypatch.setattr(factor_analysis, "normalize_fund_data", lambda data: data)
    monkeypatch.setattr(factor_analysis, "filter_single_fund", lambda data, fund_code: data)
    monkeypatch.setattr(
        factor_analysis,
        "adapt_to_price_series",
        lambda fund_df: (
            pd.Series(fund_df["nav"].values, index=pd.to_datetime(fund_df["date"]), dtype=float),
            "nav_price_series",
        ),
    )
    monkeypatch.setattr(
        factor_analysis,
        "build_walk_forward_fold_list",
        lambda price_series, walk_forward_config, split_config: [
            {
                "fold_id": 1,
                "train": price_series.iloc[:10],
                "valid": price_series.iloc[10:15],
                "test": price_series.iloc[15:20],
            },
            {
                "fold_id": 2,
                "train": price_series.iloc[5:15],
                "valid": price_series.iloc[15:20],
                "test": price_series.iloc[20:25],
            },
        ],
    )
    monkeypatch.setattr(
        factor_analysis,
        "build_single_factor_series",
        lambda price_series, factor_name, strategy_params, factor_param_override=None: pd.Series(
            range(len(price_series)),
            index=price_series.index,
            dtype=float,
            name=factor_analysis.build_factor_candidate_label(factor_name=factor_name, factor_param_dict=factor_param_override or {}),
        ),
    )
    monkeypatch.setattr(
        factor_analysis,
        "build_forward_return_series",
        lambda price_series, forward_window=5: pd.Series(range(len(price_series)), index=price_series.index, dtype=float),
    )

    def fake_compute_segment_correlation_metrics(factor_series, forward_return_series, segment_index):
        if len(segment_index) == 10:
            if pd.Index(segment_index).min() == pd.Timestamp("2024-01-01"):
                return {"sample_size": len(segment_index), "spearman_ic": 0.1, "pearson_ic": 0.05}
            return {"sample_size": len(segment_index), "spearman_ic": 0.3, "pearson_ic": 0.15}
        return {"sample_size": len(segment_index), "spearman_ic": 0.2, "pearson_ic": 0.1}

    monkeypatch.setattr(factor_analysis, "compute_segment_correlation_metrics", fake_compute_segment_correlation_metrics)

    result = factor_analysis.run_single_factor_stability_analysis()

    assert result["fund_code"] == "007301"
    assert result["selected_candidate_label_list"] == ["momentum(window=10)"]
    assert result["summary_df"].shape[0] == 1
    assert result["selected_summary_df"].shape[0] == 1
    assert result["summary_df"].iloc[0]["candidate_label"] == "momentum(window=10)"
    assert bool(result["summary_df"].iloc[0]["stability_tail_rejected"]) is False
    assert abs(float(result["summary_df"].iloc[0]["train_spearman_ic_mean"]) - 0.2) < 1e-12
    assert abs(float(result["summary_df"].iloc[0]["valid_spearman_ic_mean"]) - 0.2) < 1e-12
    assert abs(float(result["summary_df"].iloc[0]["train_valid_ic_mean_gap"]) - 0.0) < 1e-12
    assert result["summary_df"].iloc[0]["train_ic_flip_count"] == 0
    assert result["summary_df"].iloc[0]["valid_ic_flip_count"] == 0
    assert result["summary_path"].exists()
    assert result["summary_path"].suffix == ".json"
    saved_payload = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    assert saved_payload["factor_selection_output"]["fund_code"] == "007301"
    assert "stability_analysis_output" in saved_payload
    assert saved_payload["stability_analysis_output"]["fund_code"] == "007301"
    assert saved_payload["stability_analysis_output"]["candidate_count"] == 1
    assert saved_payload["stability_analysis_output"]["selected_count"] == 1
    assert saved_payload["stability_analysis_output"]["record_dict"]["momentum(window=10)"]["candidate_label"] == "momentum(window=10)"


def test_run_single_factor_stability_analysis_prefers_json_fund_code_and_absolute_gap_sort(monkeypatch, tmp_path):
    sample_index = pd.date_range("2024-01-01", periods=30, freq="D")
    sample_df = pd.DataFrame(
        {
            "date": sample_index,
            "code": ["007301"] * len(sample_index),
            "fund": ["半导体"] * len(sample_index),
            "nav": [1.0 + idx * 0.01 for idx in range(len(sample_index))],
        }
    )
    factor_selection_path = tmp_path / "renamed_selection_payload.json"
    factor_selection_path.write_text(
        json.dumps(
            {
                "factor_selection_output": {
                    "fund_code": "007301",
                    "record_dict": {
                        "momentum(window=10)": {
                            "factor_name": "momentum",
                            "factor_group": "趋势/动量",
                            "factor_param_dict": {"window": 10},
                            "candidate_label": "momentum(window=10)",
                            "selected": True,
                        },
                        "momentum(window=15)": {
                            "factor_name": "momentum",
                            "factor_group": "趋势/动量",
                            "factor_param_dict": {"window": 15},
                            "candidate_label": "momentum(window=15)",
                            "selected": True,
                        },
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        factor_analysis,
        "build_tradition_config",
        lambda config_override=None: {
            "code_dict": {"007301": "半导体"},
            "data_dir": tmp_path,
            "output_dir": tmp_path,
            "force_refresh": False,
            "cache_prefix": "tradition_fund",
            "strategy_param_dict": {
                "multi_factor_score": {
                    "enabled_factor_list": ["momentum"],
                    "factor_param_dict": {"momentum": {"window": 20}},
                    "score_window": 60,
                    "factor_weight_dict": {"momentum": 1.0},
                }
            },
            "walk_forward_config": {
                "window_size": 20,
                "step_size": 5,
                "min_fold_count": 2,
            },
            "data_split_dict": {
                "train_ratio": 0.5,
                "valid_ratio": 0.25,
                "test_ratio": 0.25,
                "min_segment_size": 5,
            },
            "factor_selection_path": str(factor_selection_path),
        },
    )
    monkeypatch.setattr(factor_analysis, "fetch_fund_data_with_cache", lambda **kwargs: sample_df)
    monkeypatch.setattr(factor_analysis, "normalize_fund_data", lambda data: data)
    monkeypatch.setattr(factor_analysis, "filter_single_fund", lambda data, fund_code: data)
    monkeypatch.setattr(
        factor_analysis,
        "adapt_to_price_series",
        lambda fund_df: (
            pd.Series(fund_df["nav"].values, index=pd.to_datetime(fund_df["date"]), dtype=float),
            "nav_price_series",
        ),
    )
    monkeypatch.setattr(
        factor_analysis,
        "build_walk_forward_fold_list",
        lambda price_series, walk_forward_config, split_config: [
            {
                "fold_id": 1,
                "train": price_series.iloc[:10],
                "valid": price_series.iloc[10:15],
                "test": price_series.iloc[15:20],
            },
            {
                "fold_id": 2,
                "train": price_series.iloc[5:15],
                "valid": price_series.iloc[15:20],
                "test": price_series.iloc[20:25],
            },
        ],
    )
    monkeypatch.setattr(
        factor_analysis,
        "build_single_factor_series",
        lambda price_series, factor_name, strategy_params, factor_param_override=None: pd.Series(
            range(len(price_series)),
            index=price_series.index,
            dtype=float,
            name=factor_analysis.build_factor_candidate_label(
                factor_name=factor_name,
                factor_param_dict=factor_param_override or {},
            ),
        ),
    )
    monkeypatch.setattr(
        factor_analysis,
        "build_forward_return_series",
        lambda price_series, forward_window=5: pd.Series(range(len(price_series)), index=price_series.index, dtype=float),
    )

    def fake_compute_segment_correlation_metrics(factor_series, forward_return_series, segment_index):
        if factor_series.name == "momentum(window=10)":
            if len(segment_index) == 10:
                return {"sample_size": len(segment_index), "spearman_ic": 0.15, "pearson_ic": 0.05}
            return {"sample_size": len(segment_index), "spearman_ic": 0.151, "pearson_ic": 0.05}
        if len(segment_index) == 10:
            return {"sample_size": len(segment_index), "spearman_ic": 0.20, "pearson_ic": 0.05}
        return {"sample_size": len(segment_index), "spearman_ic": 0.12, "pearson_ic": 0.05}

    monkeypatch.setattr(factor_analysis, "compute_segment_correlation_metrics", fake_compute_segment_correlation_metrics)

    result = factor_analysis.run_single_factor_stability_analysis()

    assert result["fund_code"] == "007301"
    assert result["summary_df"].iloc[0]["candidate_label"] == "momentum(window=10)"
    assert abs(float(result["summary_df"].iloc[0]["abs_train_valid_ic_mean_gap"]) - 0.001) < 1e-12
    assert abs(float(result["summary_df"].iloc[1]["abs_train_valid_ic_mean_gap"]) - 0.08) < 1e-12
