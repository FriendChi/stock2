import pandas as pd


def adapt_to_backtesting_ohlc(fund_df):
    # 基金历史数据优先使用真实价格列；若不存在，则退化为净值构造的伪 OHLC 结构。
    if fund_df.empty:
        raise ValueError("fund_df 为空，无法适配回测输入。")
    required_cols = {"date", "nav"}
    missing_cols = required_cols.difference(set(fund_df.columns))
    if len(missing_cols) > 0:
        raise ValueError(f"适配回测输入缺少必要列: {sorted(list(missing_cols))}")

    adapted = fund_df.copy()
    adapted["date"] = pd.to_datetime(adapted["date"], errors="coerce")
    adapted = adapted.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    adapted = adapted.set_index("date")

    real_price_cols = {"open", "high", "low", "close"}
    has_real_price_cols = real_price_cols.issubset(set(adapted.columns))
    if has_real_price_cols:
        ohlc_df = adapted.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
            }
        )[["Open", "High", "Low", "Close"]].copy()
        price_source = "real_price_fields"
    else:
        ohlc_df = pd.DataFrame(index=adapted.index)
        ohlc_df["Open"] = adapted["nav"].astype(float)
        ohlc_df["High"] = adapted["nav"].astype(float)
        ohlc_df["Low"] = adapted["nav"].astype(float)
        ohlc_df["Close"] = adapted["nav"].astype(float)
        price_source = "synthetic_from_nav"

    ohlc_df["Volume"] = 1.0
    if ohlc_df.empty:
        raise ValueError("适配后的 OHLC 数据为空。")
    return ohlc_df, price_source
