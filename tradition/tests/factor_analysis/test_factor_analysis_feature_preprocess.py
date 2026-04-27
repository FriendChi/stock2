import json

import numpy as np
import pandas as pd

from tradition import factor_analysis
from tradition.factor_analysis import feature_preprocess as factor_analysis_feature_preprocess
from tradition.factor_analysis import io as factor_analysis_io
def test_allocate_stage_csv_json_output_path_only_uses_stage_zero_slot(tmp_path):
    csv_path, json_path, path_code = factor_analysis_io.allocate_stage_csv_json_output_path(
        output_dir=tmp_path,
        output_prefix="feature_preprocess",
        fund_code="007301",
    )
    assert csv_path.stem.split("_")[-1] == path_code
    assert json_path.stem.split("_")[-1] == path_code
    assert len(path_code) == factor_analysis_io.PATH_CODE_LENGTH
    assert path_code[0] in factor_analysis_io.PATH_CODE_ALPHABET
    assert path_code[1:] == "0" * (factor_analysis_io.PATH_CODE_LENGTH - 1)

def test_append_flipped_factor_feature_columns_uses_train_metrics_and_writes_negative_series(monkeypatch, tmp_path):
    checked_output_path = tmp_path / "feature_preprocess_checked.csv"
    sample_index = pd.date_range("2024-01-01", periods=8, freq="D")
    checked_feature_df = pd.DataFrame(
        {
            "date": sample_index,
            "007301__price": [1.0 + idx * 0.01 for idx in range(len(sample_index))],
            "007301__cumulative_nav": [1.0 + idx * 0.01 for idx in range(len(sample_index))],
            "512480__price": [2.0 + idx * 0.02 for idx in range(len(sample_index))],
            "512480__price__momentum(window=10)__zscore": [0.2, 0.1, -0.3, -0.2, 0.4, 0.1, -0.2, 0.3],
            "512480__price__ma_slope(window=20,lookback=5)__zscore": [-0.1, 0.2, 0.4, 0.3, -0.2, 0.5, 0.6, -0.1],
        }
    )
    checked_feature_df.to_csv(checked_output_path, index=False)
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "build_walk_forward_dev_fold_list",
        lambda price_series, walk_forward_config, split_config: [
            {
                "fold_id": 1,
                "train": price_series.iloc[:4],
                "valid": price_series.iloc[4:6],
                "test": price_series.iloc[6:8],
            },
            {
                "fold_id": 2,
                "train": price_series.iloc[2:6],
                "valid": price_series.iloc[6:7],
                "test": price_series.iloc[7:8],
            },
        ],
    )
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "build_forward_return_series",
        lambda price_series, forward_window=5: pd.Series(range(len(price_series)), index=price_series.index, dtype=float),
    )

    def fake_compute_segment_correlation_metrics(factor_series, forward_return_series, segment_index):
        if factor_series.name == "512480__price__momentum(window=10)__zscore":
            return {"sample_size": len(segment_index), "spearman_ic": -0.4, "pearson_ic": -0.3}
        return {"sample_size": len(segment_index), "spearman_ic": 0.2, "pearson_ic": 0.1}

    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "compute_segment_correlation_metrics",
        fake_compute_segment_correlation_metrics,
    )
    updated_checked_df, _, flipped_factor_report_list = factor_analysis_feature_preprocess._append_flipped_factor_feature_columns(
        checked_feature_df=checked_feature_df,
        checked_output_path=checked_output_path,
        source_column_list=["007301__price", "007301__cumulative_nav", "512480__price"],
        fund_code="007301",
        config={
            "walk_forward_config": {
                "window_size": 20,
                "step_size": 5,
                "min_fold_count": 2,
            },
            "data_split_dict": {
                "train_ratio": 0.5,
                "valid_ratio": 0.25,
                "test_ratio": 0.25,
                "min_segment_size": 2,
            },
            "ic_aggregation_mode": "classic",
            "ic_exp_weight_half_life": 3.0,
        },
    )

    flipped_column = "512480__price__-momentum(window=10)__zscore"
    assert flipped_column in updated_checked_df.columns
    assert "512480__price__-ma_slope(window=20,lookback=5)__zscore" not in updated_checked_df.columns
    assert updated_checked_df[flipped_column].tolist() == (-checked_feature_df["512480__price__momentum(window=10)__zscore"]).tolist()
    assert flipped_factor_report_list == [
        {
            "original_column": "512480__price__momentum(window=10)__zscore",
            "flipped_column": flipped_column,
            "train_spearman_ic_mean": -0.4,
            "train_sample_fold_count": 2,
            "positive_ic_count": 0,
            "negative_ic_count": 2,
        }
    ]
    saved_df = pd.read_csv(checked_output_path)
    assert flipped_column in saved_df.columns

def test_build_single_feature_factor_df_uses_trimmed_zero_ratio_after_initial_rows(monkeypatch):
    sample_size = factor_analysis_feature_preprocess.INITIAL_TRIM_ROW_COUNT + 5
    feature_series = pd.Series(range(sample_size), dtype=float)
    normalized_series = pd.Series(
        [0.0] * factor_analysis_feature_preprocess.INITIAL_TRIM_ROW_COUNT + [1.0, 2.0, 3.0, 4.0, 5.0],
        dtype=float,
    )
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "build_raw_factor_series",
        lambda price_series, factor_name, factor_param_dict: pd.Series(range(len(price_series)), index=price_series.index, dtype=float),
    )
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "normalize_factor_series",
        lambda raw_factor_series, factor_name, score_window: normalized_series,
    )

    factor_df, dropped_factor_report_list = factor_analysis_feature_preprocess._build_single_feature_factor_df(
        feature_series=feature_series,
        candidate_factor_list=[
            {
                "candidate_label": "mock_factor(window=10)",
                "factor_name": "mock_factor",
                "param_dict": {"window": 10},
            }
        ],
        strategy_params={"score_window": 20},
    )

    assert "mock_factor(window=10)__zscore" in factor_df.columns
    assert dropped_factor_report_list == []

def test_build_single_feature_factor_df_drops_candidate_when_raw_nan_ratio_reaches_ten_percent(monkeypatch):
    feature_series = pd.Series(range(10), dtype=float)
    raw_factor_series = pd.Series([float(idx) for idx in range(9)] + [float("nan")], dtype=float)
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "build_raw_factor_series",
        lambda price_series, factor_name, factor_param_dict: raw_factor_series,
    )
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "normalize_factor_series",
        lambda raw_factor_series, factor_name, score_window: pd.Series(range(len(raw_factor_series)), dtype=float),
    )

    factor_df, dropped_factor_report_list = factor_analysis_feature_preprocess._build_single_feature_factor_df(
        feature_series=feature_series,
        candidate_factor_list=[
            {
                "candidate_label": "mock_factor(window=10)",
                "factor_name": "mock_factor",
                "param_dict": {"window": 10},
            }
        ],
        strategy_params={"score_window": 20},
    )

    assert "mock_factor(window=10)__zscore" not in factor_df.columns
    assert dropped_factor_report_list == [
        {
            "candidate_label": "mock_factor(window=10)",
            "raw_nan_ratio": 0.1,
            "normalized_zero_ratio": 0.0,
            "drop_reason_list": ["raw_nan_ratio_threshold"],
        }
    ]

def test_check_and_fill_wide_feature_table_drops_column_by_missing_ratio(tmp_path):
    raw_output_path = tmp_path / "feature_preprocess_raw.csv"
    checked_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "007301__price": [1.0 + idx * 0.01 for idx in range(10)],
            "512480__price": [2.0, 2.1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    checked_df.to_csv(raw_output_path, index=False)

    updated_df, _, feature_quality_report_list, dropped_source_column_list = factor_analysis_feature_preprocess._check_and_fill_wide_feature_table(
        raw_output_path=raw_output_path,
        code_type_dict={"007301": "fund", "512480": "fund"},
        primary_code="007301",
    )

    assert "512480__price" not in updated_df.columns
    assert dropped_source_column_list == ["512480__price"]
    dropped_report = next(record for record in feature_quality_report_list if record["column"] == "512480__price")
    assert dropped_report["missing_ratio"] >= 0.10
    assert dropped_report["drop_reason_list"] == ["missing_ratio_threshold"]

def test_check_and_fill_wide_feature_table_drops_column_by_consecutive_missing_count(tmp_path):
    raw_output_path = tmp_path / "feature_preprocess_raw.csv"
    checked_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=25, freq="D"),
            "007301__price": [1.0 + idx * 0.01 for idx in range(25)],
            "512480__price": [2.0, 2.1, 2.2, 2.3, 2.4] + [np.nan] * 20,
        }
    )
    checked_df.to_csv(raw_output_path, index=False)

    updated_df, _, feature_quality_report_list, dropped_source_column_list = factor_analysis_feature_preprocess._check_and_fill_wide_feature_table(
        raw_output_path=raw_output_path,
        code_type_dict={"007301": "fund", "512480": "fund"},
        primary_code="007301",
    )

    assert "512480__price" not in updated_df.columns
    assert dropped_source_column_list == ["512480__price"]
    dropped_report = next(record for record in feature_quality_report_list if record["column"] == "512480__price")
    assert dropped_report["max_consecutive_missing_count"] == 20
    assert dropped_report["drop_reason_list"] == ["missing_ratio_threshold", "consecutive_missing_threshold"]

def test_check_and_fill_wide_feature_table_drops_column_before_interpolation_when_missing_ratio_exceeds_threshold(tmp_path):
    raw_output_path = tmp_path / "feature_preprocess_raw.csv"
    checked_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "007301__price": [1.0 + idx * 0.01 for idx in range(12)],
            "512480__price": [np.nan, 10.0, np.nan, np.nan, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, np.nan],
        }
    )
    checked_df.to_csv(raw_output_path, index=False)

    updated_df, _, feature_quality_report_list, dropped_source_column_list = factor_analysis_feature_preprocess._check_and_fill_wide_feature_table(
        raw_output_path=raw_output_path,
        code_type_dict={"007301": "fund", "512480": "fund"},
        primary_code="007301",
    )

    assert "512480__price" not in updated_df.columns
    assert dropped_source_column_list == ["512480__price"]
    dropped_report = next(record for record in feature_quality_report_list if record["column"] == "512480__price")
    assert abs(float(dropped_report["missing_ratio"]) - (4 / 12)) < 1e-12
    assert dropped_report["drop_reason_list"] == ["missing_ratio_threshold"]
    assert bool(dropped_report["filled"]) is False
    assert dropped_report["filled_missing_count"] == 0

def test_build_checked_factor_table_skips_dropped_source_columns(monkeypatch, tmp_path):
    checked_output_path = tmp_path / "feature_preprocess_checked.csv"
    checked_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "007301__price": [1.0, 1.1, 1.2, 1.3, 1.4],
            "512480__price": [2.0, 2.1, 2.2, 2.3, 2.4],
        }
    )
    checked_df.to_csv(checked_output_path, index=False)
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "_build_factor_candidate_config",
        lambda strategy_params: ({"score_window": 3}, []),
    )

    updated_df, _, source_column_list, _, _ = factor_analysis_feature_preprocess._build_checked_factor_table(
        checked_output_path=checked_output_path,
        strategy_params={"score_window": 3},
        dropped_source_column_list=["512480__price"],
    )

    assert source_column_list == ["007301__price"]
    assert "512480__price__zscore" not in updated_df.columns

def test_trim_initial_rows_keeps_date_price_and_factor_columns_aligned():
    sample_size = factor_analysis_feature_preprocess.INITIAL_TRIM_ROW_COUNT + 5
    sample_index = pd.date_range("2024-01-01", periods=sample_size, freq="D")
    checked_feature_df = pd.DataFrame(
        {
            "date": sample_index,
            "007301__price": [1.0 + idx * 0.01 for idx in range(sample_size)],
            "007301__cumulative_nav": [1.2 + idx * 0.01 for idx in range(sample_size)],
            "007301__price__zscore": [float(idx) for idx in range(sample_size)],
            "007301__price__momentum(window=10)__zscore": [float(idx) * 0.5 for idx in range(sample_size)],
        }
    )

    trimmed_df = factor_analysis_feature_preprocess._trim_initial_rows(checked_feature_df)

    assert len(trimmed_df) == 5
    assert str(trimmed_df.iloc[0]["date"])[:10] == str(sample_index[factor_analysis_feature_preprocess.INITIAL_TRIM_ROW_COUNT].date())
    assert trimmed_df["007301__price"].tolist() == checked_feature_df["007301__price"].iloc[-5:].tolist()
    assert trimmed_df["007301__cumulative_nav"].tolist() == checked_feature_df["007301__cumulative_nav"].iloc[-5:].tolist()
    assert trimmed_df["007301__price__momentum(window=10)__zscore"].tolist() == checked_feature_df["007301__price__momentum(window=10)__zscore"].iloc[-5:].tolist()

def test_trim_initial_rows_keeps_binary_factor_semantics():
    sample_size = factor_analysis_feature_preprocess.INITIAL_TRIM_ROW_COUNT + 3
    checked_feature_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=sample_size, freq="D"),
            "007301__price": [1.0 + idx * 0.01 for idx in range(sample_size)],
            "007301__price__donchian_breakout(window=20)__zscore": [0.0] * factor_analysis_feature_preprocess.INITIAL_TRIM_ROW_COUNT + [0.0, 1.0, 1.0],
        }
    )

    trimmed_df = factor_analysis_feature_preprocess._trim_initial_rows(checked_feature_df)

    assert trimmed_df["007301__price__donchian_breakout(window=20)__zscore"].tolist() == [0.0, 1.0, 1.0]

def test_run_feature_preprocess_trims_final_checked_table(monkeypatch, tmp_path):
    sample_size = factor_analysis_feature_preprocess.INITIAL_TRIM_ROW_COUNT + 5
    checked_output_path = tmp_path / "feature_preprocess_checked.csv"
    raw_output_path = tmp_path / "feature_preprocess_raw.csv"
    metadata_output_path = tmp_path / "feature_preprocess.json"
    sample_index = pd.date_range("2024-01-01", periods=sample_size, freq="D")
    initial_checked_df = pd.DataFrame(
        {
            "date": sample_index,
            "007301__price": [1.0 + idx * 0.01 for idx in range(sample_size)],
            "007301__cumulative_nav": [1.5 + idx * 0.01 for idx in range(sample_size)],
            "007301__price__zscore": [float(idx) for idx in range(sample_size)],
            "007301__price__donchian_breakout(window=20)__zscore": [0.0] * factor_analysis_feature_preprocess.INITIAL_TRIM_ROW_COUNT + [0.0, 1.0, 1.0, 0.0, 1.0],
        }
    )

    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "build_tradition_config",
        lambda config_override=None: {
            "default_fund_code": "007301",
            "output_dir": tmp_path,
            "force_refresh": False,
            "data_dir": tmp_path,
            "linked_code_dict": {},
            "code_type_dict": {},
            "strategy_param_dict": {"multi_factor_score": {"score_window": 20}},
            "walk_forward_config": {"window_size": 20, "step_size": 5, "min_fold_count": 1},
            "data_split_dict": {"train_ratio": 0.6, "valid_ratio": 0.2, "test_ratio": 0.2, "min_segment_size": 2},
            "ic_aggregation_mode": "classic",
            "ic_exp_weight_half_life": 3.0,
        },
    )
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "allocate_stage_csv_json_output_path",
        lambda output_dir, output_prefix, fund_code: (raw_output_path, metadata_output_path, "a00000"),
    )
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "build_code_feature_table",
        lambda **kwargs: (pd.DataFrame(), raw_output_path),
    )
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "_check_and_fill_wide_feature_table",
        lambda raw_output_path, code_type_dict, primary_code: (initial_checked_df.copy(), checked_output_path, [], []),
    )
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "_build_checked_factor_table",
        lambda checked_output_path, strategy_params, dropped_source_column_list=None: (
            initial_checked_df.copy(),
            checked_output_path,
            ["007301__price", "007301__cumulative_nav"],
            [],
            [],
        ),
    )
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "_append_flipped_factor_feature_columns",
        lambda checked_feature_df, checked_output_path, source_column_list, fund_code, config: (
            checked_feature_df.copy(),
            checked_output_path,
            [],
        ),
    )

    result = factor_analysis_feature_preprocess.run_feature_preprocess_single_fund()

    saved_df = pd.read_csv(checked_output_path)
    saved_payload = json.loads(metadata_output_path.read_text(encoding="utf-8"))
    assert result["record_count"] == 5
    assert len(saved_df) == 5
    assert str(saved_df.iloc[0]["date"])[:10] == str(sample_index[factor_analysis_feature_preprocess.INITIAL_TRIM_ROW_COUNT].date())
    pd.testing.assert_series_equal(
        saved_df["007301__price"],
        initial_checked_df["007301__price"].iloc[-5:].reset_index(drop=True),
        check_names=False,
        atol=1e-12,
        rtol=0.0,
    )
    pd.testing.assert_series_equal(
        saved_df["007301__cumulative_nav"],
        initial_checked_df["007301__cumulative_nav"].iloc[-5:].reset_index(drop=True),
        check_names=False,
        atol=1e-12,
        rtol=0.0,
    )
    assert saved_df["007301__price__donchian_breakout(window=20)__zscore"].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]
    assert saved_payload["feature_preprocess_output"]["row_count"] == 5
    assert saved_payload["feature_preprocess_output"]["csv_path"] == str(checked_output_path.resolve())

def test_run_feature_preprocess_metadata_records_dropped_source_columns(monkeypatch, tmp_path):
    sample_size = factor_analysis_feature_preprocess.INITIAL_TRIM_ROW_COUNT + 5
    checked_output_path = tmp_path / "feature_preprocess_checked.csv"
    raw_output_path = tmp_path / "feature_preprocess_raw.csv"
    metadata_output_path = tmp_path / "feature_preprocess.json"
    sample_index = pd.date_range("2024-01-01", periods=sample_size, freq="D")
    initial_checked_df = pd.DataFrame(
        {
            "date": sample_index,
            "007301__price": [1.0 + idx * 0.01 for idx in range(sample_size)],
            "007301__cumulative_nav": [1.5 + idx * 0.01 for idx in range(sample_size)],
        }
    )
    feature_quality_report_list = [
        {
            "column": "512480__amount",
            "missing_count": 25,
            "missing_ratio": 0.2,
            "missing_date_list": [],
            "max_consecutive_missing_count": 20,
            "max_consecutive_missing_start_date": "2024-02-01",
            "max_consecutive_missing_end_date": "2024-02-20",
            "dropped": True,
            "drop_reason_list": ["missing_ratio_threshold", "consecutive_missing_threshold"],
            "filled": False,
            "filled_missing_count": 0,
        }
    ]

    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "build_tradition_config",
        lambda config_override=None: {
            "default_fund_code": "007301",
            "output_dir": tmp_path,
            "force_refresh": False,
            "data_dir": tmp_path,
            "linked_code_dict": {},
            "code_type_dict": {},
            "strategy_param_dict": {"multi_factor_score": {"score_window": 20}},
            "walk_forward_config": {"window_size": 20, "step_size": 5, "min_fold_count": 1},
            "data_split_dict": {"train_ratio": 0.6, "valid_ratio": 0.2, "test_ratio": 0.2, "min_segment_size": 2},
            "ic_aggregation_mode": "classic",
            "ic_exp_weight_half_life": 3.0,
        },
    )
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "allocate_stage_csv_json_output_path",
        lambda output_dir, output_prefix, fund_code: (raw_output_path, metadata_output_path, "a00000"),
    )
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "build_code_feature_table",
        lambda **kwargs: (pd.DataFrame(), raw_output_path),
    )
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "_check_and_fill_wide_feature_table",
        lambda raw_output_path, code_type_dict, primary_code: (
            initial_checked_df.copy(),
            checked_output_path,
            feature_quality_report_list,
            ["512480__amount"],
        ),
    )
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "_build_checked_factor_table",
        lambda checked_output_path, strategy_params, dropped_source_column_list=None: (
            initial_checked_df.copy(),
            checked_output_path,
            ["007301__price", "007301__cumulative_nav"],
            [],
            [],
        ),
    )
    monkeypatch.setattr(
        factor_analysis_feature_preprocess,
        "_append_flipped_factor_feature_columns",
        lambda checked_feature_df, checked_output_path, source_column_list, fund_code, config: (
            checked_feature_df.copy(),
            checked_output_path,
            [],
        ),
    )

    factor_analysis_feature_preprocess.run_feature_preprocess_single_fund()

    saved_payload = json.loads(metadata_output_path.read_text(encoding="utf-8"))
    assert saved_payload["feature_preprocess_output"]["quality_summary"]["dropped_source_column_count"] == 1
    assert saved_payload["feature_preprocess_output"]["dropped_source_feature_list"] == [
        {
            "column": "512480__amount",
            "missing_count": 25,
            "missing_ratio": 0.2,
            "max_consecutive_missing_count": 20,
            "max_consecutive_missing_start_date": "2024-02-01",
            "max_consecutive_missing_end_date": "2024-02-20",
            "drop_reason_list": ["missing_ratio_threshold", "consecutive_missing_threshold"],
        }
    ]
