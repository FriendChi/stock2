import pandas as pd
import pytest

from tradition.splitter import build_walk_forward_fold_list, split_time_series_by_ratio


def test_split_time_series_by_ratio_keeps_time_order():
    price_series = pd.Series(range(300), index=pd.date_range("2024-01-01", periods=300, freq="D"), dtype=float)
    split_dict = split_time_series_by_ratio(
        price_series=price_series,
        split_config={
            "train_ratio": 0.6,
            "valid_ratio": 0.2,
            "test_ratio": 0.2,
            "min_segment_size": 30,
        },
    )

    assert len(split_dict["train"]) == 180
    assert len(split_dict["valid"]) == 60
    assert len(split_dict["test"]) == 60
    assert split_dict["train"].index.max() < split_dict["valid"].index.min()
    assert split_dict["valid"].index.max() < split_dict["test"].index.min()


def test_split_time_series_by_ratio_rejects_short_segments():
    price_series = pd.Series(range(90), index=pd.date_range("2024-01-01", periods=90, freq="D"), dtype=float)
    with pytest.raises(ValueError, match="切分后样本不足"):
        split_time_series_by_ratio(
            price_series=price_series,
            split_config={
                "train_ratio": 0.6,
                "valid_ratio": 0.2,
                "test_ratio": 0.2,
                "min_segment_size": 40,
            },
        )


def test_build_walk_forward_fold_list_generates_non_overlapping_test_windows():
    price_series = pd.Series(range(900), index=pd.date_range("2024-01-01", periods=900, freq="D"), dtype=float)
    fold_list = build_walk_forward_fold_list(
        price_series=price_series,
        walk_forward_config={
            "window_size": 700,
            "step_size": 60,
            "min_fold_count": 1,
        },
        split_config={
            "train_ratio": 0.6,
            "valid_ratio": 0.2,
            "test_ratio": 0.2,
            "min_segment_size": 60,
        },
    )

    assert len(fold_list) == 4
    assert len(fold_list[0]["train"]) == 420
    assert len(fold_list[0]["valid"]) == 140
    assert len(fold_list[0]["test"]) == 140
    assert fold_list[1]["train_start"] == price_series.index[60]
    assert fold_list[0]["test_start"] < fold_list[1]["test_start"]


def test_build_walk_forward_fold_list_rejects_insufficient_folds():
    price_series = pd.Series(range(300), index=pd.date_range("2024-01-01", periods=300, freq="D"), dtype=float)
    with pytest.raises(ValueError, match="walk-forward 可用折数不足"):
        build_walk_forward_fold_list(
            price_series=price_series,
            walk_forward_config={
                "window_size": 300,
                "step_size": 40,
                "min_fold_count": 2,
            },
            split_config={
                "train_ratio": 0.6,
                "valid_ratio": 0.2,
                "test_ratio": 0.2,
                "min_segment_size": 60,
            },
        )
