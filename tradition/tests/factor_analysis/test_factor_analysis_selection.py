import json

import pandas as pd

from tradition import factor_analysis
def test_run_factor_selection_single_fund_returns_ranked_selection(monkeypatch, sample_fund_df, tmp_path):
    sample_index = pd.date_range("2024-01-01", periods=30, freq="D")
    
    # 物理创建特征 CSV 文件，包含必要的因子列和目标列
    preprocess_csv_path = tmp_path / "fake_preprocess.csv"
    pd.DataFrame({
        "date": sample_index,
        "nav": [1.0 + idx * 0.01 for idx in range(len(sample_index))],
        "momentum(window=10)": list(range(len(sample_index))),
        "momentum(window=15)": list(reversed(range(len(sample_index))))
    }).to_csv(preprocess_csv_path, index=False)

    # 物理创建元数据 JSON，补全符合系统校验要求的 feature_preprocess_output 结构
    metadata_path = tmp_path / "fake_metadata.json"
    metadata_path.write_text(
        json.dumps({
            "feature_preprocess_output": {
                "fund_code": "007301",
                "data_mode": "feature_matrix",
                "csv_path": str(preprocess_csv_path.resolve()),
                "target_nav_column": "nav",
                "target_price_column": "nav",
                "factor_feature_column_list": ["momentum(window=10)", "momentum(window=15)"]
            }
        }),
        encoding="utf-8"
    )

    monkeypatch.setattr(
        factor_analysis.selection,
        "build_tradition_config",
        lambda config_override=None: {
            "code_dict": {"007301": "半导体"},
            "data_dir": tmp_path,
            "output_dir": tmp_path,
            "force_refresh": False,
            "cache_prefix": "tradition_fund",
            "default_fund_code": "007301",
            "preprocess_metadata_path": str(metadata_path),
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
            "factor_group_list": ["趋势强度", "均线趋势"],
            "train_min_spearman_ic": 0.1,
            "train_min_spearman_icir": 0.0,
        },
    )
    
    # 移除原有的 fetch_fund_data_with_cache、normalize_fund_data、filter_single_fund 等 Mock
    # 因为 selection.py 已改为直接从 bundle 加载，不再走数据抓取流程

    monkeypatch.setattr(
        factor_analysis.selection,
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
        factor_analysis.selection,
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

    monkeypatch.setattr(factor_analysis.selection, "compute_segment_correlation_metrics", fake_compute_segment_correlation_metrics)

    result = factor_analysis.run_factor_selection_single_fund()

    assert result["fund_code"] == "007301"
    assert result["selected_factor_name_list"] == ["momentum(window=10)"]
    assert result["selected_candidate_label_list"] == ["momentum(window=10)"]
    assert result["summary_df"].iloc[0]["candidate_label"] == "momentum(window=10)"
    assert result["summary_df"].iloc[0]["factor_param_dict"] == {}
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
    assert saved_payload["factor_selection_output"]["ic_aggregation_config"]["mode"] == "classic"
    assert saved_payload["factor_selection_output"]["record_dict"]["momentum(window=10)"]["candidate_label"] == "momentum(window=10)"
    assert saved_payload["factor_selection_output"]["record_dict"]["momentum(window=10)"]["factor_param_dict"] == {}

def test_run_factor_selection_single_fund_supports_csv_path_from_metadata(monkeypatch, tmp_path):
    feature_csv_path = tmp_path / "feature_preprocess_007301_2026-04-12_abcd0.csv"
    metadata_json_path = tmp_path / "feature_preprocess_007301_2026-04-12_abcd0.json"
    sample_index = pd.date_range("2024-01-01", periods=30, freq="D")
    feature_df = pd.DataFrame(
        {
            "date": sample_index,
            "007301__price": [1.0 + idx * 0.01 for idx in range(len(sample_index))],
            "007301__cumulative_nav": [1.0 + idx * 0.01 for idx in range(len(sample_index))],
            "momentum(window=10)": list(range(len(sample_index))),
            "momentum(window=15)": list(reversed(range(len(sample_index)))),
        }
    )
    feature_df.to_csv(feature_csv_path, index=False)
    metadata_json_path.write_text(
        json.dumps(
            {
                "path_code": "abcd0",
                "feature_preprocess_output": {
                    "fund_code": "007301",
                    "data_mode": "feature_matrix",
                    "csv_path": str(feature_csv_path.resolve()),
                    "target_nav_column": "007301__cumulative_nav",
                    "target_price_column": "007301__price",
                    "factor_feature_column_list": ["momentum(window=10)", "momentum(window=15)"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        factor_analysis.selection,
        "build_tradition_config",
        lambda config_override=None: {
            "default_fund_code": "007301",
            "output_dir": tmp_path,
            "force_refresh": False,
            "strategy_param_dict": {
                "multi_factor_score": {},
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
            "train_min_spearman_ic": 0.1,
            "train_min_spearman_icir": 0.0,
            "preprocess_metadata_path": str(metadata_json_path),
            "preprocess_path": None,
        },
    )
    monkeypatch.setattr(
        factor_analysis.selection,
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
        factor_analysis.selection,
        "build_forward_return_series",
        lambda price_series, forward_window=5: pd.Series(range(len(price_series)), index=price_series.index, dtype=float),
    )

    def fake_compute_segment_correlation_metrics(factor_series, forward_return_series, segment_index):
        if factor_series.name == "momentum(window=10)":
            return {"sample_size": len(segment_index), "spearman_ic": 0.4, "pearson_ic": 0.3}
        return {"sample_size": len(segment_index), "spearman_ic": -0.1, "pearson_ic": -0.1}

    monkeypatch.setattr(factor_analysis.selection, "compute_segment_correlation_metrics", fake_compute_segment_correlation_metrics)

    result = factor_analysis.run_factor_selection_single_fund()

    assert result["fund_code"] == "007301"
    assert result["summary_path"].suffix == ".json"
    saved_payload = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    assert saved_payload["factor_selection_output"]["preprocess_path"] == str(feature_csv_path.resolve())
    assert saved_payload["factor_selection_output"]["preprocess_metadata_path"] == str(metadata_json_path)
