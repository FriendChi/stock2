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
