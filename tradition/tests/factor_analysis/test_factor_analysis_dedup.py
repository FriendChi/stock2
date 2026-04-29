import json

import numpy as np
import pandas as pd

from tradition import factor_analysis
from tradition.factor_analysis import io as factor_analysis_io
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
    factor_series_dict = {
        f"factor_{idx:02d}": pd.Series([idx, idx + 1], name=f"factor_{idx:02d}", dtype=float)
        for idx in range(20)
    }
    corr_matrix = np.full((20, 20), float("nan"), dtype=float)
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix[19, 18] = corr_matrix[18, 19] = 0.95
    corr_matrix[17, 16] = corr_matrix[16, 17] = 0.95
    monkeypatch.setattr(
        factor_analysis.dedup,
        "_build_mean_train_corr_matrix",
        lambda candidate_label_list, factor_series_dict, fold_list: corr_matrix,
    )
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

def test_build_mean_train_corr_matrix_matches_pairwise_fold_mean_with_pairwise_dropna():
    sample_index = pd.date_range("2024-01-01", periods=6, freq="D")
    factor_series_dict = {
        "factor_a": pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], index=sample_index, dtype=float),
        "factor_b": pd.Series([2.0, 4.0, 6.0, 8.0, np.nan, 12.0], index=sample_index, dtype=float),
        "factor_c": pd.Series([1.0, np.nan, 1.5, 2.0, 2.5, 3.0], index=sample_index, dtype=float),
    }
    fold_list = [
        {
            "train": pd.Series([0, 1, 2, 3], index=sample_index[:4], dtype=float),
        },
        {
            "train": pd.Series([0, 1, 2, 3], index=sample_index[2:], dtype=float),
        },
    ]

    mean_train_corr_matrix = factor_analysis.dedup._build_mean_train_corr_matrix(
        candidate_label_list=["factor_a", "factor_b", "factor_c"],
        factor_series_dict=factor_series_dict,
        fold_list=fold_list,
    )
    expected_ab = factor_analysis.compute_pair_train_corr(
        left_factor_series=factor_series_dict["factor_a"],
        right_factor_series=factor_series_dict["factor_b"],
        fold_list=fold_list,
    )
    expected_ac = factor_analysis.compute_pair_train_corr(
        left_factor_series=factor_series_dict["factor_a"],
        right_factor_series=factor_series_dict["factor_c"],
        fold_list=fold_list,
    )

    assert abs(float(mean_train_corr_matrix[0, 1]) - float(expected_ab)) < 1e-12
    assert abs(float(mean_train_corr_matrix[0, 2]) - float(expected_ac)) < 1e-12
    assert abs(float(mean_train_corr_matrix[1, 0]) - float(expected_ab)) < 1e-12

def test_incremental_combination_score_matches_original_sum_with_missing_values():
    sample_index = pd.date_range("2024-01-01", periods=5, freq="D")
    factor_series_dict = {
        "factor_a": pd.Series([1.0, np.nan, 2.0, 1.5, np.nan], index=sample_index, dtype=float),
        "factor_b": pd.Series([np.nan, 0.5, 1.0, np.nan, 2.0], index=sample_index, dtype=float),
        "factor_c": pd.Series([0.2, 0.3, np.nan, 0.4, 0.5], index=sample_index, dtype=float),
    }

    _, original_score_series = factor_analysis.build_instance_combination_score(
        factor_candidate_list=[
            {"candidate_label": "factor_a"},
            {"candidate_label": "factor_b"},
            {"candidate_label": "factor_c"},
        ],
        factor_series_dict=factor_series_dict,
    )
    incremental_score_series = factor_analysis.dedup._build_single_candidate_score_series(
        candidate_label="factor_a",
        factor_series_dict=factor_series_dict,
    )
    incremental_score_series = factor_analysis.dedup._extend_combination_score_series(
        parent_score_series=incremental_score_series,
        candidate_label="factor_b",
        factor_series_dict=factor_series_dict,
        candidate_label_list=["factor_a", "factor_b"],
    )
    incremental_score_series = factor_analysis.dedup._extend_combination_score_series(
        parent_score_series=incremental_score_series,
        candidate_label="factor_c",
        factor_series_dict=factor_series_dict,
        candidate_label_list=["factor_a", "factor_b", "factor_c"],
    )

    pd.testing.assert_series_equal(original_score_series, incremental_score_series)

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

    def fake_evaluate_train_score_segment_list(
        train_score_segment_list,
        train_target_rank_component_list,
        candidate_label_list,
        ic_aggregation_config=None,
    ):
        label_tuple = tuple(sorted(candidate_label_list))
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

    def fake_worker_evaluate_batch(task_payload):
        _, batch_payloads, _, _ = task_payload
        metric_dict = {
            ("factor_a", "factor_b"): {"train_spearman_ic_mean": 0.12, "train_spearman_icir": 0.8},
            ("factor_a", "factor_c"): {"train_spearman_ic_mean": 0.11, "train_spearman_icir": 0.7},
            ("factor_b", "factor_c"): {"train_spearman_ic_mean": 0.10, "train_spearman_icir": 0.60},
            ("factor_a", "factor_b", "factor_c"): {"train_spearman_ic_mean": 0.10, "train_spearman_icir": 0.80},
        }
        result_list = []
        for child_labels, child_sig, label in batch_payloads:
            result_list.append(
                (
                    label,
                    dict(metric_dict[tuple(child_labels)]),
                    child_labels,
                    child_sig,
                )
            )
        return result_list

    class FakeAsyncResult:
        def __init__(self, func, args):
            self.func = func
            self.args = args

        def ready(self):
            return True

        def get(self, timeout=None):
            return self.func(*self.args)

    class FakePool:
        def __init__(self, processes=None, initializer=None, initargs=()):
            if initializer is not None:
                initializer(*initargs)

        def apply_async(self, func, args=(), kwds=None):
            assert kwds in (None, {})
            return FakeAsyncResult(func, args)

        def terminate(self):
            return None

        def close(self):
            return None

        def join(self):
            return None

    monkeypatch.setattr(
        factor_analysis.dedup,
        "_evaluate_train_score_segment_list",
        fake_evaluate_train_score_segment_list,
    )
    # 第二层开始实际走批处理 worker，这里改为同步 fake pool，确保测试覆盖当前调度路径。
    monkeypatch.setattr(factor_analysis.dedup, "_worker_evaluate_batch", fake_worker_evaluate_batch)
    monkeypatch.setattr(factor_analysis.dedup.multiprocessing, "Pool", FakePool)
    path_summary_list = factor_analysis.run_train_forward_selection(
        candidate_record_list=candidate_record_list,
        factor_series_dict={record["candidate_label"]: pd.Series([1.0, 2.0], dtype=float) for record in candidate_record_list},
        forward_return_series=pd.Series([0.1, 0.2], dtype=float),
        fold_list=[{"train": pd.Series([0.1, 0.2], index=pd.Index([1, 2]))}],
        root_topk=2,
        n_processes=1
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

def test_run_train_forward_selection_stops_when_train_ic_mean_decreases(monkeypatch):
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
            "factor_name": "trend_tvalue",
            "factor_param_dict": {"window": 15},
            "train_spearman_icir": 0.5,
            "valid_spearman_icir": 0.3,
            "valid_spearman_ic_mean": 0.1,
        },
    ]

    def fake_evaluate_train_score_segment_list(
        train_score_segment_list,
        train_target_rank_component_list,
        candidate_label_list,
        ic_aggregation_config=None,
    ):
        label_tuple = tuple(sorted(candidate_label_list))
        metric_dict = {
            ("factor_a",): {"train_spearman_ic_mean": 0.10, "train_spearman_icir": 0.60},
            ("factor_b",): {"train_spearman_ic_mean": 0.09, "train_spearman_icir": 0.50},
            ("factor_a", "factor_b"): {"train_spearman_ic_mean": 0.08, "train_spearman_icir": 0.70},
        }
        selected_metric_dict = dict(metric_dict[label_tuple])
        selected_metric_dict["candidate_label_list"] = list(label_tuple)
        selected_metric_dict["factor_count"] = len(label_tuple)
        return selected_metric_dict

    def fake_worker_evaluate_batch(task_payload):
        _, batch_payloads, _, _ = task_payload
        metric_dict = {
            ("factor_a", "factor_b"): {"train_spearman_ic_mean": 0.08, "train_spearman_icir": 0.70},
        }
        result_list = []
        for child_labels, child_sig, label in batch_payloads:
            result_list.append(
                (
                    label,
                    dict(metric_dict[tuple(child_labels)]),
                    child_labels,
                    child_sig,
                )
            )
        return result_list

    class FakeAsyncResult:
        def __init__(self, func, args):
            self.func = func
            self.args = args

        def ready(self):
            return True

        def get(self, timeout=None):
            return self.func(*self.args)

    class FakePool:
        def __init__(self, processes=None, initializer=None, initargs=()):
            if initializer is not None:
                initializer(*initargs)

        def apply_async(self, func, args=(), kwds=None):
            assert kwds in (None, {})
            return FakeAsyncResult(func, args)

        def terminate(self):
            return None

        def close(self):
            return None

        def join(self):
            return None

    monkeypatch.setattr(
        factor_analysis.dedup,
        "_evaluate_train_score_segment_list",
        fake_evaluate_train_score_segment_list,
    )
    monkeypatch.setattr(factor_analysis.dedup, "_worker_evaluate_batch", fake_worker_evaluate_batch)
    monkeypatch.setattr(factor_analysis.dedup.multiprocessing, "Pool", FakePool)
    path_summary_list = factor_analysis.run_train_forward_selection(
        candidate_record_list=candidate_record_list,
        factor_series_dict={record["candidate_label"]: pd.Series([1.0, 2.0], dtype=float) for record in candidate_record_list},
        forward_return_series=pd.Series([0.1, 0.2], dtype=float),
        fold_list=[{"train": pd.Series([0.1, 0.2], index=pd.Index([1, 2]))}],
        root_topk=1,
        n_processes=1,
    )
    assert [item["candidate_label_list"] for item in path_summary_list] == [
        ["factor_a"],
        ["factor_a", "factor_b"],
    ]

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
        def __init__(self, number, value_dict):
            self.number = int(number)
            self.value_dict = dict(value_dict)
            self.params = {}
            self.value = None
            self.state = None

        def suggest_int(self, name, low, high):
            selected_value = int(self.value_dict[name])
            self.params[name] = selected_value
            return selected_value

    class FakeStudy:
        def __init__(self, trial_value_list):
            self.trial_value_list = list(trial_value_list)
            self.optimize_calls = []
            self.trials = []

        def optimize(self, objective, n_trials, timeout=None, callbacks=None, n_jobs=1):
            self.optimize_calls.append(int(n_trials))
            callbacks = list(callbacks or [])
            for trial_idx, trial_value_dict in enumerate(self.trial_value_list[:n_trials]):
                trial = FakeTrial(number=trial_idx, value_dict=trial_value_dict)
                trial.value = float(objective(trial))
                trial.state = FakeTrialState.COMPLETE
                self.trials.append(trial)
                for callback in callbacks:
                    callback(self, trial)

    class FakeSampler:
        def __init__(self, seed=None):
            self.seed = seed

    class FakeTrialState:
        COMPLETE = "COMPLETE"

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
            "trial": type("TrialModule", (), {"TrialState": FakeTrialState}),
        },
    )
    monkeypatch.setattr(factor_analysis.dedup, "load_optuna_module", lambda: fake_optuna_module)

    def fake_evaluate_train_score_segment_list(
        train_score_segment_list,
        train_target_rank_component_list,
        candidate_label_list,
        ic_aggregation_config=None,
    ):
        label_tuple = tuple(sorted(candidate_label_list))
        metric_dict = {
            ("factor_a",): {"train_spearman_ic_mean": 0.10, "train_spearman_icir": 0.90},
            ("factor_a", "factor_b"): {"train_spearman_ic_mean": 0.11, "train_spearman_icir": 1.00},
            ("factor_a", "factor_c"): {"train_spearman_ic_mean": 0.12, "train_spearman_icir": 1.10},
            ("factor_a", "factor_b", "factor_c"): {"train_spearman_ic_mean": 0.13, "train_spearman_icir": 1.05},
        }
        selected_metric_dict = dict(metric_dict[label_tuple])
        selected_metric_dict["candidate_label_list"] = list(label_tuple)
        selected_metric_dict["factor_count"] = len(label_tuple)
        return selected_metric_dict

    def fake_evaluate_valid_for_path_summary_list(
        path_summary_list,
        candidate_record_lookup,
        valid_cache,
        ic_aggregation_config=None,
        progress_desc=None,
        n_processes=2,
    ):
        metric_dict = {
            ("factor_a", "factor_b"): {"valid_spearman_ic_mean": 0.09, "valid_spearman_icir": 0.48},
            ("factor_a", "factor_c"): {"valid_spearman_ic_mean": 0.11, "valid_spearman_icir": 0.60},
            ("factor_a", "factor_b", "factor_c"): {"valid_spearman_ic_mean": 0.10, "valid_spearman_icir": 0.55},
        }
        evaluated_summary_list = []
        for summary in path_summary_list:
            updated_summary = dict(summary)
            updated_summary.update(metric_dict[tuple(sorted(summary["candidate_label_list"]))])
            evaluated_summary_list.append(updated_summary)
        return evaluated_summary_list

    monkeypatch.setattr(factor_analysis.dedup, "_evaluate_train_score_segment_list", fake_evaluate_train_score_segment_list)
    monkeypatch.setattr(factor_analysis.dedup, "evaluate_valid_for_path_summary_list", fake_evaluate_valid_for_path_summary_list)

    train_cache = {
        "candidate_segment_dict": {
            "factor_a": [np.array([1.0, 2.0], dtype=float)],
            "factor_b": [np.array([0.5, 1.0], dtype=float)],
            "factor_c": [np.array([1.5, 0.5], dtype=float)],
        },
        "target_rank_component_list": [None],
    }
    valid_cache = {
        "candidate_segment_dict": {
            "factor_a": [np.array([1.0, 2.0], dtype=float)],
            "factor_b": [np.array([0.5, 1.0], dtype=float)],
            "factor_c": [np.array([1.5, 0.5], dtype=float)],
        },
        "target_rank_component_list": [None],
    }

    result = factor_analysis.run_optuna_extension_search(
        baseline_summary=baseline_summary,
        corr_selected_candidate_list=corr_selected_candidate_list,
        candidate_record_lookup=candidate_record_lookup,
        train_cache=train_cache,
        valid_cache=valid_cache,
    )

    assert fake_study.optimize_calls == [4]
    assert result["remaining_factor_count"] == 2
    assert result["n_trials"] == 4
    assert result["train_improved_candidate_count"] == 3
    assert result["best_optuna_candidate_summary"]["candidate_label_list"] == ["factor_a", "factor_c"]
    assert result["final_selected_source"] == "optuna_extension"
    assert result["best_final_selection_summary"]["candidate_label_list"] == ["factor_a", "factor_c"]

def test_run_single_factor_dedup_selection_outputs_nested_json(monkeypatch, tmp_path):
    sample_index = pd.date_range("2024-01-01", periods=30, freq="D")
    feature_csv_path = tmp_path / "feature_preprocess_007301_2026-04-05_c00000_checked.csv"
    pd.DataFrame(
        {
            "date": sample_index,
            "007301__cumulative_nav": [1.0 + idx * 0.01 for idx in range(len(sample_index))],
            "momentum(window=10)": list(range(len(sample_index))),
            "ma_slope(lookback=5, window=20)": list(reversed(range(len(sample_index)))),
        }
    ).to_csv(feature_csv_path, index=False)
    stability_analysis_path = tmp_path / "single_factor_stability_007301_2026-04-05.json"
    stability_analysis_path.write_text(
        json.dumps(
            {
                "path_code": "aB2000",
                "stability_analysis_output": {
                    "fund_code": "007301",
                    "data_mode": "feature_matrix",
                    "preprocess_path": str(feature_csv_path),
                    "preprocess_metadata_path": str(tmp_path / "feature_preprocess_007301_2026-04-05_c00000.json"),
                    "target_nav_column": "007301__cumulative_nav",
                    "record_dict": {
                        "momentum(window=10)": {
                            "candidate_label": "momentum(window=10)",
                            "train_spearman_icir": 0.8,
                            "valid_spearman_icir": 0.6,
                            "valid_spearman_ic_mean": 0.11,
                            "selected": True,
                        },
                        "ma_slope(lookback=5, window=20)": {
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
        factor_analysis.dedup,
        "build_tradition_config",
        lambda config_override=None: {
            "code_dict": {"007301": "半导体"},
            "data_dir": tmp_path,
            "output_dir": tmp_path,
            "force_refresh": False,
            "cache_prefix": "tradition_fund",
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
    monkeypatch.setattr(
        factor_analysis.dedup,
        "build_walk_forward_dev_fold_list",
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
        factor_analysis.dedup,
        "build_forward_return_series",
        lambda price_series, forward_window=5: pd.Series(range(len(price_series)), index=price_series.index, dtype=float),
    )
    monkeypatch.setattr(
        factor_analysis.dedup,
        "_build_mean_train_corr_matrix",
        lambda candidate_label_list, factor_series_dict, fold_list: np.array(
            [
                [1.0, 0.2],
                [0.2, 1.0],
            ],
            dtype=float,
        ),
    )

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

    monkeypatch.setattr(factor_analysis.dedup, "compute_segment_correlation_metrics", fake_compute_segment_correlation_metrics)

    result = factor_analysis.run_single_factor_dedup_selection()

    assert result["fund_code"] == "007301"
    assert result["best_forward_selection_summary"]["candidate_label_list"] == ["momentum(window=10)"]
    assert result["summary_path"].exists()
    saved_payload = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    assert saved_payload["input_ref"]["fund_code"] == "007301"
    assert len(saved_payload["path_code"]) == 6
    assert saved_payload["path_code"][:3] == "aB2"
    assert saved_payload["path_code"][3] in factor_analysis_io.PATH_CODE_ALPHABET
    assert saved_payload["path_code"][4:] == "00"
    assert result["summary_path"].stem.split("_")[-1] == saved_payload["path_code"]
    assert "dedup_selection_output" in saved_payload
    assert saved_payload["dedup_selection_output"]["preprocess_path"] == str(feature_csv_path)
    assert saved_payload["dedup_selection_output"]["preprocess_metadata_path"] == str(tmp_path / "feature_preprocess_007301_2026-04-05_c00000.json")
    assert saved_payload["dedup_selection_output"]["target_nav_column"] == "007301__cumulative_nav"
    assert saved_payload["dedup_selection_output"]["train_path_count"] == 3
    assert saved_payload["dedup_selection_output"]["valid_eval_count"] == 1
    assert saved_payload["dedup_selection_output"]["valid_eval_ratio"] == 0.5
    assert saved_payload["dedup_selection_output"]["ic_aggregation_config"]["mode"] == "classic"
    assert "train_forward_selection_path_summary" not in saved_payload["dedup_selection_output"]
    assert "forward_selection_path_summary" not in saved_payload["dedup_selection_output"]
    assert "optuna_extension_output" not in saved_payload["dedup_selection_output"]
    assert saved_payload["dedup_selection_output"]["final_selected_source"] == "forward_selection"
    assert saved_payload["dedup_selection_output"]["forward_selected_candidate_label_list"] == ["momentum(window=10)"]
    assert saved_payload["dedup_selection_output"]["best_final_selection_summary"]["candidate_label_list"] == ["momentum(window=10)"]
    selected_record = saved_payload["dedup_selection_output"]["record_dict"]["momentum(window=10)"]
    assert "factor_name" not in selected_record
    assert "factor_param_dict" not in selected_record
    assert "factor_group" not in selected_record
