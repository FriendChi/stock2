from datetime import datetime
from pathlib import Path

import pandas as pd


def build_cache_path(cache_dir, cache_prefix="tradition_fund", trade_date=None):
    # 缓存文件名按日期组织，保证同一天重复运行优先命中本地缓存。
    if trade_date is None:
        trade_date = datetime.today().strftime("%Y-%m-%d")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_prefix}_{trade_date}.csv"


def build_range_cache_path(cache_dir, cache_prefix, start_date, end_date):
    # 区间型数据单独按起止日期命名缓存，避免收益率序列和基金净值缓存相互覆盖。
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_prefix}_{start_date}_{end_date}.csv"


def _rename_if_exists(df, rename_map):
    existing_map = {src: dst for src, dst in rename_map.items() if src in df.columns}
    if len(existing_map) == 0:
        return df.copy()
    return df.rename(columns=existing_map)


def _fetch_indicator_df(ak_module, code, fund_name, indicator, rename_map):
    # 每个指标单独拉取并在这里完成字段标准化，避免上层组合逻辑夹杂中文列名判断。
    df = ak_module.fund_open_fund_info_em(symbol=code, indicator=indicator)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date"])
    df = _rename_if_exists(df, rename_map)
    if "date" not in df.columns:
        raise ValueError(f"{indicator} 缺少净值日期列，基金代码={code}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    keep_cols = [col for col in df.columns if col in {"date", "nav", "cumulative_nav", "daily_growth_rate"}]
    df = df[keep_cols].copy()
    if "nav" not in df.columns and indicator == "单位净值走势":
        raise ValueError(f"单位净值走势缺少nav列，基金代码={code}")
    if len(df) == 0:
        return pd.DataFrame(columns=["date"])
    df["code"] = str(code).zfill(6)
    df["fund"] = fund_name
    return df


def _iter_yearly_windows(start_date, end_date, max_window_days=364):
    # 中债收益率接口单次只支持一年内查询，这里拆成连续小区间依次拉取。
    current_start = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    while current_start <= end_ts:
        current_end = min(current_start + pd.Timedelta(days=max_window_days), end_ts)
        yield current_start.strftime("%Y%m%d"), current_end.strftime("%Y%m%d")
        current_start = current_end + pd.Timedelta(days=1)


def fetch_treasury_yield_with_cache(
    cache_dir,
    start_date,
    end_date,
    force_refresh=False,
    cache_prefix="tradition_rf_cn_bond_yield",
    curve_name="中债国债收益率曲线",
    tenor="1年",
):
    # 无风险利率优先命中本地缓存，拉取时按时间窗口拼接中债国债收益率曲线的目标期限列。
    cache_path = build_range_cache_path(
        cache_dir=cache_dir,
        cache_prefix=cache_prefix,
        start_date=str(start_date),
        end_date=str(end_date),
    )
    if cache_path.exists() and not force_refresh:
        cached = pd.read_csv(cache_path)
        cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
        cached["annual_rf"] = pd.to_numeric(cached["annual_rf"], errors="coerce")
        return cached.dropna(subset=["date", "annual_rf"]).sort_values("date").reset_index(drop=True)

    import akshare as ak

    rf_data_list = []
    for window_start, window_end in _iter_yearly_windows(start_date=start_date, end_date=end_date):
        df = ak.bond_china_yield(start_date=window_start, end_date=window_end)
        if df is None or len(df) == 0:
            continue
        if curve_name not in set(df["曲线名称"]):
            raise ValueError(f"未找到目标曲线: {curve_name}")
        if tenor not in df.columns:
            raise ValueError(f"收益率曲线缺少目标期限列: {tenor}")
        filtered = df[df["曲线名称"] == curve_name].copy()
        filtered = filtered[["日期", tenor]].rename(columns={"日期": "date", tenor: "annual_rf"})
        filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
        filtered["annual_rf"] = pd.to_numeric(filtered["annual_rf"], errors="coerce") / 100.0
        filtered = filtered.dropna(subset=["date", "annual_rf"])
        rf_data_list.append(filtered)

    if len(rf_data_list) == 0:
        raise ValueError("未拉取到任何国债收益率数据。")

    rf_df = pd.concat(rf_data_list, ignore_index=True)
    rf_df = rf_df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    rf_df.to_csv(cache_path, index=False)
    return rf_df


def fetch_fund_data_with_cache(code_dict, cache_dir, force_refresh=False, cache_prefix="tradition_fund", trade_date=None):
    # 先命中本地缓存，再回退到 AkShare 拉取，兼顾稳定性与重复实验效率。
    cache_path = build_cache_path(cache_dir=cache_dir, cache_prefix=cache_prefix, trade_date=trade_date)
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path, dtype={"code": str})

    import akshare as ak

    data_list = []
    nav_rename_map = {
        "净值日期": "date",
        "单位净值": "nav",
        "日增长率": "daily_growth_rate",
    }
    cumulative_rename_map = {
        "净值日期": "date",
        "累计净值": "cumulative_nav",
    }

    for code, fund_name in code_dict.items():
        # 先拿单位净值，再尝试补充累计净值等扩展字段，最终按日期做宽表合并。
        nav_df = _fetch_indicator_df(
            ak_module=ak,
            code=code,
            fund_name=fund_name,
            indicator="单位净值走势",
            rename_map=nav_rename_map,
        )
        cumulative_df = _fetch_indicator_df(
            ak_module=ak,
            code=code,
            fund_name=fund_name,
            indicator="累计净值走势",
            rename_map=cumulative_rename_map,
        )
        if "cumulative_nav" in cumulative_df.columns:
            merged_df = nav_df.merge(cumulative_df[["date", "cumulative_nav"]], on="date", how="left")
        else:
            merged_df = nav_df.copy()
        merged_df["code"] = str(code).zfill(6)
        merged_df["fund"] = fund_name
        data_list.append(merged_df)

    if len(data_list) == 0:
        raise ValueError("未拉取到任何基金数据。")
    data = pd.concat(data_list, ignore_index=True)
    data.to_csv(cache_path, index=False)
    return data
