import pandas as pd
import pytest

from tradition.splitter import split_time_series_by_ratio


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
