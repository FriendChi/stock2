import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "code",
    "fund",
    "nav",
}


def normalize_fund_data(data):
    # 统一数据类型、日期格式和排序规则，避免后续策略和回测层反复做兜底判断。
    missing_cols = REQUIRED_COLUMNS.difference(set(data.columns))
    if len(missing_cols) > 0:
        raise ValueError(f"缺少必要列: {sorted(list(missing_cols))}")

    normalized = data.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["code"] = normalized["code"].astype(str).str.zfill(6)
    normalized["nav"] = pd.to_numeric(normalized["nav"], errors="coerce")
    if "cumulative_nav" in normalized.columns:
        normalized["cumulative_nav"] = pd.to_numeric(normalized["cumulative_nav"], errors="coerce")
    if "daily_growth_rate" in normalized.columns:
        normalized["daily_growth_rate"] = pd.to_numeric(normalized["daily_growth_rate"], errors="coerce")

    normalized = normalized.dropna(subset=["date", "nav"]).copy()
    normalized = normalized.sort_values(["date", "code"]).drop_duplicates(subset=["date", "code"], keep="last")
    normalized = normalized.reset_index(drop=True)
    if normalized.empty:
        raise ValueError("标准化后无可用基金数据。")
    return normalized


def filter_single_fund(data, fund_code):
    # 第一阶段仅支持单基金模式，筛选结果为空时直接失败，避免后续静默跑空。
    fund_code = str(fund_code).zfill(6)
    filtered = data[data["code"] == fund_code].copy()
    if filtered.empty:
        raise ValueError(f"未找到基金数据，fund_code={fund_code}")
    filtered = filtered.sort_values("date").reset_index(drop=True)
    return filtered
