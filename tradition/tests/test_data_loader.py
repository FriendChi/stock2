import pytest

from tradition.data_loader import filter_single_fund, normalize_fund_data


def test_normalize_fund_data_sorts_and_formats_codes(sample_fund_df):
    shuffled_df = sample_fund_df.sample(frac=1.0, random_state=42).copy()
    normalized = normalize_fund_data(shuffled_df)
    assert normalized["code"].iloc[0] == "007301"
    assert normalized["date"].is_monotonic_increasing


def test_normalize_fund_data_requires_columns(sample_fund_df):
    with pytest.raises(ValueError):
        normalize_fund_data(sample_fund_df.drop(columns=["nav"]))


def test_filter_single_fund_returns_single_code(sample_fund_df):
    filtered = filter_single_fund(sample_fund_df, "007301")
    assert filtered["code"].nunique() == 1


def test_filter_single_fund_raises_for_missing_code(sample_fund_df):
    with pytest.raises(ValueError):
        filter_single_fund(sample_fund_df, "000001")
