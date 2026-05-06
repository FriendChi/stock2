import json

import pandas as pd

from tradition import factor_analysis


def test_run_factor_combination_rejects_csv_feature_format(monkeypatch, tmp_path):
    preprocess_path = tmp_path / "feature_preprocess_007301_2026-04-09_c00000_checked.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "007301__cumulative_nav": [1.0, 1.01, 1.02],
            "momentum(window=10)": [0.1, 0.2, 0.3],
        }
    ).to_csv(preprocess_path, index=False)
    preprocess_metadata_path = tmp_path / "feature_preprocess_007301_2026-04-09_c00000.json"
    preprocess_metadata_path.write_text(
        json.dumps(
            {
                "feature_preprocess_output": {
                    "feature_path": str(preprocess_path),
                    "feature_format": "csv",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    dedup_selection_path = tmp_path / "single_factor_dedup_007301_2026-04-09.json"
    dedup_selection_path.write_text(
        json.dumps(
            {
                "dedup_selection_output": {
                    "fund_code": "007301",
                    "preprocess_path": str(preprocess_path),
                    "preprocess_metadata_path": str(preprocess_metadata_path),
                    "target_nav_column": "007301__cumulative_nav",
                    "record_dict": {
                        "momentum(window=10)": {
                            "candidate_label": "momentum(window=10)",
                            "train_spearman_icir": 0.8,
                        }
                    },
                    "best_final_selection_summary": {
                        "candidate_label_list": ["momentum(window=10)"],
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        factor_analysis.combination,
        "build_tradition_config",
        lambda config_override=None: {
            "code_dict": {"007301": "半导体"},
            "output_dir": tmp_path,
            "force_refresh": False,
            "walk_forward_config": {"window_size": 20, "step_size": 5, "min_fold_count": 2},
            "data_split_dict": {"train_ratio": 0.5, "valid_ratio": 0.25, "test_ratio": 0.25, "min_segment_size": 5},
            "dedup_selection_path": str(dedup_selection_path),
        },
    )

    try:
        factor_analysis.run_factor_combination()
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "仅支持 parquet 特征文件" in str(exc)
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

def test_run_factor_combination_outputs_independent_json(monkeypatch, tmp_path):
    sample_index = pd.date_range("2024-01-01", periods=30, freq="D")
    preprocess_path = tmp_path / "feature_preprocess_007301_2026-04-09_c00000_checked.parquet"
    preprocess_df = pd.DataFrame(
        {
            "date": sample_index,
            "007301__cumulative_nav": [1.0 + idx * 0.01 for idx in range(len(sample_index))],
            "momentum(window=10)": list(range(len(sample_index))),
            "ma_slope(lookback=5, window=20)": list(reversed(range(len(sample_index)))),
        }
    )
    preprocess_df.to_parquet(preprocess_path, index=False)
    preprocess_metadata_path = tmp_path / "feature_preprocess_007301_2026-04-09_c00000.json"
    preprocess_metadata_path.write_text(
        json.dumps(
            {
                "feature_preprocess_output": {
                    "feature_path": str(preprocess_path),
                    "feature_format": "parquet",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    dedup_selection_path = tmp_path / "single_factor_dedup_007301_2026-04-09.json"
    dedup_selection_path.write_text(
        json.dumps(
            {
                "dedup_selection_output": {
                    "fund_code": "007301",
                    "data_mode": "feature_matrix",
                    "preprocess_path": str(preprocess_path),
                    "preprocess_metadata_path": str(preprocess_metadata_path),
                    "target_nav_column": "007301__cumulative_nav",
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
        factor_analysis.combination,
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
    monkeypatch.setattr(
        factor_analysis.combination,
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
        factor_analysis.combination,
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

    monkeypatch.setattr(factor_analysis.combination, "compute_segment_correlation_metrics", fake_compute_segment_correlation_metrics)
    monkeypatch.setattr(
        factor_analysis.combination,
        "run_factor_combination_weight_tuning",
        lambda factor_candidate_list, factor_series_dict, forward_return_series, fold_list, selected_method_summary, ic_aggregation_config=None: {
            "enabled": True,
            "selected_method": selected_method_summary["method_name"],
            "n_trials": 100,
            "top_k_valid_eval_count": 50,
            "base_weight_dict": {"momentum(window=10)": 0.5, "ma_slope(lookback=5, window=20)": 0.5},
            "weight_search_range_dict": {},
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
    assert saved_payload["input_ref"]["fund_code"] == "007301"
    assert len(saved_payload["path_code"]) == 6
    assert str(saved_payload["path_code"]).isalnum()
    assert result["summary_path"].stem.split("_")[-1] == saved_payload["path_code"]
    assert "factor_combination_output" in saved_payload
    assert saved_payload["factor_combination_output"]["ic_aggregation_config"]["mode"] == "classic"
    assert saved_payload["factor_combination_output"]["preprocess_metadata_path"] == str(preprocess_metadata_path)
    assert saved_payload["factor_combination_output"]["combination_compare_output"]["selected_method"] == "equal_weight"
    assert saved_payload["factor_combination_output"]["weight_tuning_output"]["selected_method"] == "equal_weight"
    assert "train_top_trial_summary_list" not in saved_payload["factor_combination_output"]["weight_tuning_output"]
    assert saved_payload["factor_combination_output"]["best_combination_selection_summary"]["candidate_weight_dict"]["momentum(window=10)"] == 0.5
    assert saved_payload["factor_combination_output"]["best_combination_selection_summary"]["trial_number"] == 7
