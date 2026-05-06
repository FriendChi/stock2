import json

import pandas as pd

from tradition import factor_analysis
from tradition.factor_analysis import io as factor_analysis_io
def test_run_single_factor_stability_analysis_outputs_nested_json(monkeypatch, tmp_path):
    sample_index = pd.date_range("2024-01-01", periods=30, freq="D")
    feature_csv_path = tmp_path / "feature_preprocess_007301_2026-04-05_a00000_checked.parquet"
    pd.DataFrame(
        {
            "date": sample_index,
            "007301__cumulative_nav": [1.0 + idx * 0.01 for idx in range(len(sample_index))],
            "momentum(window=10)": [0.1 * idx for idx in range(len(sample_index))],
            "momentum(window=15)": [0.2 * idx for idx in range(len(sample_index))],
        }
    ).to_parquet(feature_csv_path, index=False)
    preprocess_metadata_path = tmp_path / "feature_preprocess_007301_2026-04-05_a00000.json"
    preprocess_metadata_path.write_text(
        json.dumps(
            {
                "feature_preprocess_output": {
                    "feature_path": str(feature_csv_path),
                    "feature_format": "parquet",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    factor_selection_path = tmp_path / "factor_selection_007301_2026-04-05.json"
    factor_selection_path.write_text(
        json.dumps(
            {
                "path_code": "a10000",
                "factor_selection_output": {
                    "fund_code": "007301",
                    "data_mode": "feature_matrix",
                    "preprocess_path": str(feature_csv_path),
                    "preprocess_metadata_path": str(preprocess_metadata_path),
                    "target_nav_column": "007301__cumulative_nav",
                    "record_dict": {
                        "momentum(window=10)": {
                            "factor_name": "momentum",
                            "factor_group": "趋势/动量",
                            "factor_param_dict": {"window": 10},
                            "candidate_label": "momentum(window=10)",
                            "binding_mode": "single_field",
                            "bound_field_name_list": ["price"],
                            "bound_source_column_list": ["007301__price"],
                            "selected": True,
                        },
                        "momentum(window=15)": {
                            "factor_name": "momentum",
                            "factor_group": "趋势/动量",
                            "factor_param_dict": {"window": 15},
                            "candidate_label": "momentum(window=15)",
                            "binding_mode": "single_field",
                            "bound_field_name_list": ["price"],
                            "bound_source_column_list": ["007301__price"],
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
        factor_analysis.stability,
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
            "factor_selection_path": str(factor_selection_path),
        },
    )
    monkeypatch.setattr(
        factor_analysis.stability,
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
        factor_analysis.stability,
        "build_forward_return_series",
        lambda price_series, forward_window=5: pd.Series(range(len(price_series)), index=price_series.index, dtype=float),
    )

    def fake_compute_segment_correlation_metrics(factor_series, forward_return_series, segment_index):
        if len(segment_index) == 10:
            if pd.Index(segment_index).min() == pd.Timestamp("2024-01-01"):
                return {"sample_size": len(segment_index), "spearman_ic": 0.1, "pearson_ic": 0.05}
            return {"sample_size": len(segment_index), "spearman_ic": 0.3, "pearson_ic": 0.15}
        return {"sample_size": len(segment_index), "spearman_ic": 0.2, "pearson_ic": 0.1}

    monkeypatch.setattr(factor_analysis.stability, "compute_segment_correlation_metrics", fake_compute_segment_correlation_metrics)

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
    assert saved_payload["input_ref"]["fund_code"] == "007301"
    assert saved_payload["input_ref"]["factor_selection_path"] == str(factor_selection_path)
    assert len(saved_payload["path_code"]) == 6
    assert saved_payload["path_code"][:2] == "a1"
    assert saved_payload["path_code"][2] in factor_analysis_io.PATH_CODE_ALPHABET
    assert saved_payload["path_code"][3:] == "000"
    assert result["summary_path"].stem.split("_")[-1] == saved_payload["path_code"]
    assert "stability_analysis_output" in saved_payload
    assert saved_payload["stability_analysis_output"]["fund_code"] == "007301"
    assert saved_payload["stability_analysis_output"]["preprocess_path"] == str(feature_csv_path)
    assert saved_payload["stability_analysis_output"]["preprocess_metadata_path"] == str(preprocess_metadata_path)
    assert saved_payload["stability_analysis_output"]["target_nav_column"] == "007301__cumulative_nav"
    assert saved_payload["stability_analysis_output"]["candidate_count"] == 1
    assert saved_payload["stability_analysis_output"]["selected_count"] == 1
    assert saved_payload["stability_analysis_output"]["ic_aggregation_config"]["mode"] == "classic"
    assert saved_payload["stability_analysis_output"]["record_dict"]["momentum(window=10)"]["candidate_label"] == "momentum(window=10)"
    assert saved_payload["stability_analysis_output"]["record_dict"]["momentum(window=10)"]["binding_mode"] == "single_field"
    assert saved_payload["stability_analysis_output"]["record_dict"]["momentum(window=10)"]["bound_field_name_list"] == ["price"]
    assert saved_payload["stability_analysis_output"]["record_dict"]["momentum(window=10)"]["bound_source_column_list"] == ["007301__price"]

def test_run_single_factor_stability_analysis_prefers_json_fund_code_and_absolute_gap_sort(monkeypatch, tmp_path):
    sample_index = pd.date_range("2024-01-01", periods=30, freq="D")
    feature_csv_path = tmp_path / "feature_preprocess_007301_2026-04-05_b00000_checked.csv"
    pd.DataFrame(
        {
            "date": sample_index,
            "007301__cumulative_nav": [1.0 + idx * 0.01 for idx in range(len(sample_index))],
            "momentum(window=10)": [0.1 * idx for idx in range(len(sample_index))],
            "momentum(window=15)": [0.2 * idx for idx in range(len(sample_index))],
        }
    ).to_csv(feature_csv_path, index=False)
    factor_selection_path = tmp_path / "renamed_selection_payload.json"
    factor_selection_path.write_text(
        json.dumps(
            {
                "path_code": "b10000",
                "factor_selection_output": {
                    "fund_code": "007301",
                    "data_mode": "feature_matrix",
                    "preprocess_path": str(feature_csv_path),
                    "target_nav_column": "007301__cumulative_nav",
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
        factor_analysis.stability,
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
            "factor_selection_path": str(factor_selection_path),
        },
    )
    monkeypatch.setattr(
        factor_analysis.stability,
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
        factor_analysis.stability,
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

    monkeypatch.setattr(factor_analysis.stability, "compute_segment_correlation_metrics", fake_compute_segment_correlation_metrics)

    result = factor_analysis.run_single_factor_stability_analysis()

    assert result["fund_code"] == "007301"
    assert result["summary_df"].iloc[0]["candidate_label"] == "momentum(window=10)"
    assert abs(float(result["summary_df"].iloc[0]["abs_train_valid_ic_mean_gap"]) - 0.001) < 1e-12
    assert abs(float(result["summary_df"].iloc[1]["abs_train_valid_ic_mean_gap"]) - 0.08) < 1e-12
