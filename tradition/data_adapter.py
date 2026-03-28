import pandas as pd


def adapt_to_price_series(fund_df):
    # 基金历史回测统一使用净值序列作为价格输入，避免再伪造 OHLC 结构。
    if fund_df.empty:
        raise ValueError("fund_df 为空，无法适配价格序列。")
    required_cols = {"date", "nav"}
    missing_cols = required_cols.difference(set(fund_df.columns))
    if len(missing_cols) > 0:
        raise ValueError(f"适配价格序列缺少必要列: {sorted(list(missing_cols))}")

    adapted = fund_df.copy()
    adapted["date"] = pd.to_datetime(adapted["date"], errors="coerce")
    adapted["nav"] = pd.to_numeric(adapted["nav"], errors="coerce")
    adapted = adapted.dropna(subset=["date", "nav"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if adapted.empty:
        raise ValueError("适配后的价格序列为空。")

    price_series = adapted.set_index("date")["nav"].astype(float).copy()
    price_series.name = "price"
    return price_series, "nav_price_series"
