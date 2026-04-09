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


def test_build_corr_dedup_result_drops_global_worst_ten_percent_with_min_two_and_cap(monkeypatch):
    selected_summary_df = pd.DataFrame(
        [
            {
                "candidate_label": f"factor_{idx:02d}",
                "factor_name": f"factor_{idx:02d}",
                "factor_group": "趋势/动量",
                "factor_param_dict": {"window": idx},
                "train_spearman_icir": 1.0 - idx * 0.01,
                "valid_spearman_icir": 0.5 - idx * 0.01,
                "valid_spearman_ic_mean": 0.2 - idx * 0.001,
            }
            for idx in range(20)
        ]
    )
    monkeypatch.setattr(
        factor_analysis,
        "compute_pair_train_corr",
        lambda left_factor_series, right_factor_series, fold_list: 0.95
        if {str(left_factor_series.name), str(right_factor_series.name)} in [{"factor_19", "factor_18"}, {"factor_17", "factor_16"}]
        else float("nan"),
    )
    factor_series_dict = {
        f"factor_{idx:02d}": pd.Series([idx, idx + 1], name=f"factor_{idx:02d}", dtype=float)
        for idx in range(20)
    }
    summary_df, dropped_candidate_label_list = factor_analysis.build_corr_dedup_result(
        selected_summary_df=selected_summary_df,
        factor_series_dict=factor_series_dict,
        fold_list=[{"train": pd.Series([1.0, 2.0], index=[0, 1])}],
        corr_threshold=0.90,
        drop_ratio=0.10,
        min_drop_count=2,
    )
    assert dropped_candidate_label_list == ["factor_19", "factor_17"]
    assert bool(summary_df.loc[summary_df["candidate_label"] == "factor_19", "corr_dedup_selected"].iloc[0]) is False
    assert bool(summary_df.loc[summary_df["candidate_label"] == "factor_18", "corr_dedup_selected"].iloc[0]) is True
    assert bool(summary_df.loc[summary_df["candidate_label"] == "factor_17", "corr_dedup_selected"].iloc[0]) is False


def test_run_train_forward_selection_builds_tree_path_from_topk_singleton_roots(monkeypatch):
    candidate_record_list = [
        {
            "candidate_label": "factor_a",
            "factor_name": "momentum",
            "factor_param_dict": {"window": 10},
            "train_spearman_icir": 0.6,
            "valid_spearman_icir": 0.4,
            "valid_spearman_ic_mean": 0.1,
        },
        {
            "candidate_label": "factor_b",
            "factor_name": "momentum",
            "factor_param_dict": {"window": 15},
            "train_spearman_icir": 0.5,
            "valid_spearman_icir": 0.3,
            "valid_spearman_ic_mean": 0.1,
        },
        {
            "candidate_label": "factor_c",
            "factor_name": "trend_tvalue",
            "factor_param_dict": {"window": 15},
            "train_spearman_icir": 0.4,
            "valid_spearman_icir": 0.2,
            "valid_spearman_ic_mean": 0.1,
        },
    ]

    def fake_evaluate_factor_candidate_subset(factor_candidate_list, factor_series_dict, forward_return_series, fold_list, include_valid=True):
        label_tuple = tuple(item["candidate_label"] for item in factor_candidate_list)
        metric_dict = {
            ("factor_a",): {"train_spearman_ic_mean": 0.1, "train_spearman_icir": 0.6},
            ("factor_b",): {"train_spearman_ic_mean": 0.09, "train_spearman_icir": 0.55},
            ("factor_c",): {"train_spearman_ic_mean": 0.08, "train_spearman_icir": 0.50},
            ("factor_a", "factor_b"): {"train_spearman_ic_mean": 0.12, "train_spearman_icir": 0.8},
            ("factor_a", "factor_c"): {"train_spearman_ic_mean": 0.11, "train_spearman_icir": 0.7},
            ("factor_b", "factor_c"): {"train_spearman_ic_mean": 0.10, "train_spearman_icir": 0.60},
            ("factor_a", "factor_b", "factor_c"): {"train_spearman_ic_mean": 0.10, "train_spearman_icir": 0.80},
        }
        selected_metric_dict = dict(metric_dict[label_tuple])
        selected_metric_dict["candidate_label_list"] = list(label_tuple)
        selected_metric_dict["factor_count"] = len(label_tuple)
        return selected_metric_dict

    monkeypatch.setattr(factor_analysis, "evaluate_factor_candidate_subset", fake_evaluate_factor_candidate_subset)
    path_summary_list = factor_analysis.run_train_forward_selection(
        candidate_record_list=candidate_record_list,
        factor_series_dict={record["candidate_label"]: pd.Series([1.0, 2.0], dtype=float) for record in candidate_record_list},
        forward_return_series=pd.Series([0.1, 0.2], dtype=float),
        fold_list=[],
        root_topk=2,
    )
    assert [item["candidate_label_list"] for item in path_summary_list] == [
        ["factor_a"],
        ["factor_b"],
        ["factor_a", "factor_b"],
        ["factor_a", "factor_c"],
        ["factor_b", "factor_c"],
        ["factor_a", "factor_b", "factor_c"],
    ]
    assert path_summary_list[-1]["train_spearman_icir"] == 0.8


def test_select_top_train_path_summary_list_keeps_top_half():
    selected_path_summary_list = factor_analysis.select_top_train_path_summary_list(
        [
            {"candidate_label_list": ["a"], "factor_count": 1, "step": 1, "train_spearman_ic_mean": 0.10, "train_spearman_icir": 0.90},
            {"candidate_label_list": ["b"], "factor_count": 1, "step": 1, "train_spearman_ic_mean": 0.09, "train_spearman_icir": 0.70},
            {"candidate_label_list": ["c"], "factor_count": 1, "step": 1, "train_spearman_ic_mean": 0.08, "train_spearman_icir": 0.60},
            {"candidate_label_list": ["d"], "factor_count": 1, "step": 1, "train_spearman_ic_mean": 0.07, "train_spearman_icir": 0.50},
            {"candidate_label_list": ["e"], "factor_count": 1, "step": 1, "train_spearman_ic_mean": 0.06, "train_spearman_icir": 0.40},
            {"candidate_label_list": ["f"], "factor_count": 1, "step": 1, "train_spearman_ic_mean": 0.05, "train_spearman_icir": 0.30},
        ],
        top_ratio=0.5,
    )
    assert [item["candidate_label_list"] for item in selected_path_summary_list] == [["a"], ["b"], ["c"]]


def test_select_best_forward_path_summary_uses_valid_first():
    best_summary = factor_analysis.select_best_forward_path_summary(
        [
            {
                "step": 1,
                "candidate_label_list": ["factor_a"],
                "factor_count": 1,
                "train_spearman_ic_mean": 0.10,
                "train_spearman_icir": 0.90,
                "valid_spearman_ic_mean": 0.12,
                "valid_spearman_icir": 0.50,
            },
            {
                "step": 2,
                "candidate_label_list": ["factor_a", "factor_b"],
                "factor_count": 2,
                "train_spearman_ic_mean": 0.11,
                "train_spearman_icir": 1.20,
                "valid_spearman_ic_mean": 0.11,
                "valid_spearman_icir": 0.55,
            },
        ]
    )
    assert best_summary["step"] == 2
    assert best_summary["candidate_label_list"] == ["factor_a", "factor_b"]


def test_run_optuna_extension_search_uses_remaining_factor_square_trials_and_can_replace_baseline(monkeypatch):
    baseline_summary = {
        "candidate_label_list": ["factor_a"],
        "factor_count": 1,
        "train_spearman_ic_mean": 0.10,
        "train_spearman_icir": 0.90,
        "valid_spearman_ic_mean": 0.08,
        "valid_spearman_icir": 0.50,
        "step": 1,
    }
    corr_selected_candidate_list = [
        {"candidate_label": "factor_a", "factor_name": "momentum", "factor_param_dict": {"window": 10}},
        {"candidate_label": "factor_b", "factor_name": "trend_tvalue", "factor_param_dict": {"window": 15}},
        {"candidate_label": "factor_c", "factor_name": "trend_r2", "factor_param_dict": {"window": 20}},
    ]
    candidate_record_lookup = {
        item["candidate_label"]: dict(item)
        for item in corr_selected_candidate_list
    }

    class FakeTrial:
        def __init__(self, value_dict):
            self.value_dict = dict(value_dict)

        def suggest_int(self, name, low, high):
            return int(self.value_dict[name])

    class FakeStudy:
        def __init__(self, trial_value_list):
            self.trial_value_list = list(trial_value_list)
            self.optimize_calls = []

        def optimize(self, objective, n_trials):
            self.optimize_calls.append(int(n_trials))
            for trial_value_dict in self.trial_value_list[:n_trials]:
                objective(FakeTrial(trial_value_dict))

    class FakeSampler:
        def __init__(self, seed=None):
            self.seed = seed

    fake_study = FakeStudy(
        [
            {"add_0": 1, "add_1": 0},
            {"add_0": 0, "add_1": 1},
            {"add_0": 1, "add_1": 1},
            {"add_0": 0, "add_1": 0},
        ]
    )
    fake_optuna_module = type(
        "FakeOptuna",
        (),
        {
            "create_study": staticmethod(lambda direction, sampler: fake_study),
            "samplers": type("Samplers", (), {"TPESampler": FakeSampler}),
        },
    )
    monkeypatch.setattr(factor_analysis, "load_optuna_module", lambda: fake_optuna_module)

    def fake_evaluate_factor_candidate_subset(factor_candidate_list, factor_series_dict, forward_return_series, fold_list, include_valid=True):
        label_tuple = tuple(sorted(item["candidate_label"] for item in factor_candidate_list))
        metric_dict = {
            ("factor_a",): {"train_spearman_ic_mean": 0.10, "train_spearman_icir": 0.90, "valid_spearman_ic_mean": 0.08, "valid_spearman_icir": 0.50},
            ("factor_a", "factor_b"): {"train_spearman_ic_mean": 0.11, "train_spearman_icir": 1.00, "valid_spearman_ic_mean": 0.09, "valid_spearman_icir": 0.48},
            ("factor_a", "factor_c"): {"train_spearman_ic_mean": 0.12, "train_spearman_icir": 1.10, "valid_spearman_ic_mean": 0.11, "valid_spearman_icir": 0.60},
            ("factor_a", "factor_b", "factor_c"): {"train_spearman_ic_mean": 0.13, "train_spearman_icir": 1.05, "valid_spearman_ic_mean": 0.10, "valid_spearman_icir": 0.55},
        }
        selected_metric_dict = dict(metric_dict[label_tuple])
        selected_metric_dict["candidate_label_list"] = list(label_tuple)
        selected_metric_dict["factor_count"] = len(label_tuple)
        if not include_valid:
            selected_metric_dict.pop("valid_spearman_ic_mean")
            selected_metric_dict.pop("valid_spearman_icir")
        return selected_metric_dict

    monkeypatch.setattr(factor_analysis, "evaluate_factor_candidate_subset", fake_evaluate_factor_candidate_subset)

    result = factor_analysis.run_optuna_extension_search(
        baseline_summary=baseline_summary,
        corr_selected_candidate_list=corr_selected_candidate_list,
        candidate_record_lookup=candidate_record_lookup,
        factor_series_dict={},
        forward_return_series=pd.Series(dtype=float),
        fold_list=[],
    )

    assert fake_study.optimize_calls == [4]
    assert result["remaining_factor_count"] == 2
    assert result["n_trials"] == 4
    assert result["train_improved_candidate_count"] == 3
    assert result["best_optuna_candidate_summary"]["candidate_label_list"] == ["factor_a", "factor_c"]
    assert result["final_selected_source"] == "optuna_extension"
    assert result["best_final_selection_summary"]["candidate_label_list"] == ["factor_a", "factor_c"]


def test_build_weight_search_range_dict_uses_five_percent_window():
    weight_search_range_dict = factor_analysis.build_weight_search_range_dict(
        {
            "factor_a": 0.5,
            "factor_b": 0.0,
        }
    )
    assert abs(float(weight_search_range_dict["factor_a"]["low"]) - 0.475) < 1e-12
    assert abs(float(weight_search_range_dict["factor_a"]["high"]) - 0.525) < 1e-12
    assert abs(float(weight_search_range_dict["factor_b"]["low"]) - 0.0) < 1e-12
    assert abs(float(weight_search_range_dict["factor_b"]["high"]) - 0.0) < 1e-12


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


def test_run_single_factor_dedup_selection_outputs_nested_json(monkeypatch, tmp_path):
    sample_index = pd.date_range("2024-01-01", periods=30, freq="D")
    sample_df = pd.DataFrame(
        {
            "date": sample_index,
            "code": ["007301"] * len(sample_index),
            "fund": ["半导体"] * len(sample_index),
            "nav": [1.0 + idx * 0.01 for idx in range(len(sample_index))],
        }
    )
    stability_analysis_path = tmp_path / "single_factor_stability_007301_2026-04-05.json"
    stability_analysis_path.write_text(
        json.dumps(
            {
                "factor_selection_output": {
                    "fund_code": "007301",
                    "record_dict": {},
                },
                "stability_analysis_output": {
                    "fund_code": "007301",
                    "record_dict": {
                        "momentum(window=10)": {
                            "factor_name": "momentum",
                            "factor_group": "趋势/动量",
                            "factor_param_dict": {"window": 10},
                            "candidate_label": "momentum(window=10)",
                            "train_spearman_icir": 0.8,
                            "valid_spearman_icir": 0.6,
                            "valid_spearman_ic_mean": 0.11,
                            "selected": True,
                        },
                        "ma_slope(lookback=5, window=20)": {
                            "factor_name": "ma_slope",
                            "factor_group": "均线趋势",
                            "factor_param_dict": {"window": 20, "lookback": 5},
                            "candidate_label": "ma_slope(lookback=5, window=20)",
                            "train_spearman_icir": 0.7,
                            "valid_spearman_icir": 0.5,
                            "valid_spearman_ic_mean": 0.10,
                            "selected": True,
                        },
                    },
                },
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
                    "enabled_factor_list": ["momentum", "ma_slope"],
                    "factor_param_dict": {
                        "momentum": {"window": 20},
                        "ma_slope": {"window": 20, "lookback": 5},
                    },
                    "score_window": 60,
                    "factor_weight_dict": {"momentum": 1.0, "ma_slope": 1.0},
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
            "stability_analysis_path": str(stability_analysis_path),
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
    monkeypatch.setattr(factor_analysis, "compute_pair_train_corr", lambda left_factor_series, right_factor_series, fold_list: 0.2)

    def fake_compute_segment_correlation_metrics(factor_series, forward_return_series, segment_index):
        segment_start = pd.Index(segment_index).min()
        value_dict = {
            ("momentum(window=10)", pd.Timestamp("2024-01-01")): 0.20,
            ("momentum(window=10)", pd.Timestamp("2024-01-06")): 0.00,
            ("momentum(window=10)", pd.Timestamp("2024-01-11")): 0.17,
            ("momentum(window=10)", pd.Timestamp("2024-01-16")): 0.07,
            ("ma_slope(lookback=5, window=20)", pd.Timestamp("2024-01-01")): 0.10,
            ("ma_slope(lookback=5, window=20)", pd.Timestamp("2024-01-06")): 0.02,
            ("ma_slope(lookback=5, window=20)", pd.Timestamp("2024-01-11")): 0.09,
            ("ma_slope(lookback=5, window=20)", pd.Timestamp("2024-01-16")): 0.03,
            ("momentum(window=10)|ma_slope(lookback=5, window=20)", pd.Timestamp("2024-01-01")): 0.19,
            ("momentum(window=10)|ma_slope(lookback=5, window=20)", pd.Timestamp("2024-01-06")): 0.03,
            ("momentum(window=10)|ma_slope(lookback=5, window=20)", pd.Timestamp("2024-01-11")): 0.22,
            ("momentum(window=10)|ma_slope(lookback=5, window=20)", pd.Timestamp("2024-01-16")): 0.10,
        }
        signed_value = value_dict[(factor_series.name, segment_start)]
        return {
            "sample_size": len(segment_index),
            "spearman_ic": signed_value,
            "pearson_ic": signed_value,
        }

    monkeypatch.setattr(factor_analysis, "compute_segment_correlation_metrics", fake_compute_segment_correlation_metrics)

    result = factor_analysis.run_single_factor_dedup_selection()

    assert result["fund_code"] == "007301"
    assert result["best_forward_selection_summary"]["candidate_label_list"] == ["ma_slope(lookback=5, window=20)"]
    assert result["summary_path"].exists()
    saved_payload = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    assert "factor_selection_output" in saved_payload
    assert "stability_analysis_output" in saved_payload
    assert "dedup_selection_output" in saved_payload
    assert "optuna_extension_output" in saved_payload["dedup_selection_output"]
    assert saved_payload["dedup_selection_output"]["train_path_count"] == 3
    assert saved_payload["dedup_selection_output"]["valid_eval_count"] == 1
    assert saved_payload["dedup_selection_output"]["valid_eval_ratio"] == 0.5
    assert len(saved_payload["dedup_selection_output"]["train_forward_selection_path_summary"]) == 3
    assert len(saved_payload["dedup_selection_output"]["forward_selection_path_summary"]) == 1
    assert saved_payload["dedup_selection_output"]["forward_selected_candidate_label_list"] == ["ma_slope(lookback=5, window=20)"]
    assert saved_payload["dedup_selection_output"]["best_final_selection_summary"]["candidate_label_list"] == ["ma_slope(lookback=5, window=20)"]


def test_run_factor_combination_outputs_independent_json(monkeypatch, tmp_path):
    sample_index = pd.date_range("2024-01-01", periods=30, freq="D")
    sample_df = pd.DataFrame(
        {
            "date": sample_index,
            "code": ["007301"] * len(sample_index),
            "fund": ["半导体"] * len(sample_index),
            "nav": [1.0 + idx * 0.01 for idx in range(len(sample_index))],
        }
    )
    dedup_selection_path = tmp_path / "single_factor_dedup_007301_2026-04-09.json"
    dedup_selection_path.write_text(
        json.dumps(
            {
                "factor_selection_output": {"fund_code": "007301", "record_dict": {}},
                "stability_analysis_output": {"fund_code": "007301", "record_dict": {}},
                "dedup_selection_output": {
                    "fund_code": "007301",
                    "record_dict": {
                        "momentum(window=10)": {
                            "candidate_label": "momentum(window=10)",
                            "factor_name": "momentum",
                            "factor_group": "趋势/动量",
                            "factor_param_dict": {"window": 10},
                            "train_spearman_icir": 0.8,
                        },
                        "ma_slope(lookback=5, window=20)": {
                            "candidate_label": "ma_slope(lookback=5, window=20)",
                            "factor_name": "ma_slope",
                            "factor_group": "均线趋势",
                            "factor_param_dict": {"window": 20, "lookback": 5},
                            "train_spearman_icir": 0.6,
                        },
                    },
                    "best_final_selection_summary": {
                        "candidate_label_list": ["momentum(window=10)", "ma_slope(lookback=5, window=20)"],
                    },
                },
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
                    "enabled_factor_list": ["momentum", "ma_slope"],
                    "factor_param_dict": {
                        "momentum": {"window": 20},
                        "ma_slope": {"window": 20, "lookback": 5},
                    },
                    "score_window": 60,
                    "factor_weight_dict": {"momentum": 1.0, "ma_slope": 1.0},
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
            "dedup_selection_path": str(dedup_selection_path),
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
            range(len(price_series)) if factor_name == "momentum" else list(reversed(range(len(price_series)))),
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
        segment_start = pd.Index(segment_index).min()
        value_dict = {
            ("momentum(window=10)|ma_slope(lookback=5, window=20)", pd.Timestamp("2024-01-01")): 0.12,
            ("momentum(window=10)|ma_slope(lookback=5, window=20)", pd.Timestamp("2024-01-06")): 0.10,
            ("momentum(window=10)|ma_slope(lookback=5, window=20)", pd.Timestamp("2024-01-11")): 0.09,
            ("momentum(window=10)|ma_slope(lookback=5, window=20)", pd.Timestamp("2024-01-16")): 0.07,
        }
        return {
            "sample_size": len(segment_index),
            "spearman_ic": value_dict[(factor_series.name, segment_start)],
            "pearson_ic": 0.0,
        }

    monkeypatch.setattr(factor_analysis, "compute_segment_correlation_metrics", fake_compute_segment_correlation_metrics)
    monkeypatch.setattr(
        factor_analysis,
        "run_factor_combination_weight_tuning",
        lambda factor_candidate_list, factor_series_dict, forward_return_series, fold_list, selected_method_summary: {
            "enabled": True,
            "selected_method": selected_method_summary["method_name"],
            "n_trials": 100,
            "top_k_valid_eval_count": 50,
            "base_weight_dict": {"momentum(window=10)": 0.5, "ma_slope(lookback=5, window=20)": 0.5},
            "weight_search_range_dict": {},
            "train_top_trial_summary_list": [],
            "best_tuned_trial_summary": {
                "candidate_label_list": [item["candidate_label"] for item in factor_candidate_list],
                "candidate_weight_dict": {
                    item["candidate_label"]: 0.5
                    for item in factor_candidate_list
                },
                "train_spearman_ic_mean": 0.11,
                "train_spearman_icir": 1.2,
                "valid_spearman_ic_mean": 0.08,
                "valid_spearman_icir": 0.6,
                "trial_number": 7,
            },
        },
    )

    result = factor_analysis.run_factor_combination()

    assert result["fund_code"] == "007301"
    assert result["combination_compare_output"]["selected_method"] == "equal_weight"
    assert result["best_combination_selection_summary"]["selected_method"] == "equal_weight"
    assert result["summary_path"].exists()
    saved_payload = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    assert "factor_selection_output" in saved_payload
    assert "stability_analysis_output" in saved_payload
    assert "dedup_selection_output" in saved_payload
    assert "factor_combination_output" in saved_payload
    assert saved_payload["factor_combination_output"]["combination_compare_output"]["selected_method"] == "equal_weight"
    assert saved_payload["factor_combination_output"]["weight_tuning_output"]["selected_method"] == "equal_weight"
    assert saved_payload["factor_combination_output"]["best_combination_selection_summary"]["candidate_weight_dict"]["momentum(window=10)"] == 0.5
    assert saved_payload["factor_combination_output"]["best_combination_selection_summary"]["trial_number"] == 7
