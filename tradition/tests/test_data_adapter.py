from tradition.data_adapter import adapt_to_price_series


def test_adapt_to_price_series_returns_nav_series(sample_fund_df):
    price_series, data_mode = adapt_to_price_series(sample_fund_df)
    assert data_mode == "nav_price_series"
    assert price_series.name == "price"
    assert price_series.index.is_monotonic_increasing


def test_adapt_to_price_series_drops_duplicate_dates(sample_fund_df):
    duplicated_df = sample_fund_df.copy()
    duplicated_df.loc[len(duplicated_df)] = duplicated_df.iloc[-1]
    price_series, _ = adapt_to_price_series(duplicated_df)
    assert price_series.index.is_unique
