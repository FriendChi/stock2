from tradition.data_adapter import adapt_to_backtesting_ohlc


def test_adapt_to_backtesting_ohlc_uses_synthetic_nav(sample_fund_df):
    adapted, price_source = adapt_to_backtesting_ohlc(sample_fund_df)
    assert price_source == "synthetic_from_nav"
    assert list(adapted.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_adapt_to_backtesting_ohlc_prefers_real_price_fields(sample_fund_df):
    real_price_df = sample_fund_df.rename(
        columns={
            "nav": "close",
        }
    ).copy()
    real_price_df["open"] = real_price_df["close"] - 0.01
    real_price_df["high"] = real_price_df["close"] + 0.02
    real_price_df["low"] = real_price_df["close"] - 0.02
    real_price_df["nav"] = real_price_df["close"]
    adapted, price_source = adapt_to_backtesting_ohlc(real_price_df)
    assert price_source == "real_price_fields"
    assert adapted["Open"].iloc[0] == real_price_df["open"].iloc[0]
