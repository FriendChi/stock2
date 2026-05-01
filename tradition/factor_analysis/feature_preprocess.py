import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from tradition.config import build_tradition_config
from tradition.factor_analysis.common import (
    build_factor_candidate_list,
    build_forward_return_series,
    build_ic_aggregation_config,
    build_spearman_metric_summary,
    compute_segment_correlation_metrics,
)
from tradition.factor_engine import normalize_factor_series, rolling_zscore
from tradition.factor_library import build_raw_factor_series, resolve_factor_dependency_spec
from tradition.splitter import build_walk_forward_dev_fold_list

from .io import allocate_stage_csv_json_output_path

DEFAULT_START_DATE = "19700101"
DEFAULT_END_DATE = "20500101"
DEFAULT_OUTPUT_NAME = "feature_preprocess_code_feature_table.csv"
CHECKED_OUTPUT_SUFFIX = "_checked"
PARQUET_OUTPUT_SUFFIX = ".parquet"
FEATURE_CACHE_PREFIX = "temp_price_cache"
INCREMENTAL_FETCH_BUFFER_DAYS = 15
OPEN_FUND_PERIOD_BUFFER_DAYS = 15
MAX_FILL_MISSING_ROW_COUNT = 4
MAX_FACTOR_RAW_NAN_RATIO = 0.10
MAX_NORMALIZED_ZERO_RATIO = 1.0
INITIAL_TRIM_ROW_COUNT = 120
MAX_FEATURE_MISSING_RATIO = 0.10
MAX_FEATURE_CONSECUTIVE_MISSING_COUNT = 20


def _resolve_feature_preprocess_code_type_dict(config):
    # 关联标类型映射进入正式配置层，流程0只消费配置结果，不再回退到临时模块常量。
    config = dict(config)
    primary_code = str(config["default_fund_code"]).zfill(6)
    linked_code_list = [str(code).zfill(6) for code in list(dict(config.get("linked_code_dict", {})).get(primary_code, []))]
    declared_code_type_dict = {
        str(code).zfill(6): str(code_type).strip().lower()
        for code, code_type in dict(config.get("code_type_dict", {})).items()
    }
    resolved_code_type_dict = {primary_code: "fund"}
    for linked_code in linked_code_list:
        if linked_code not in declared_code_type_dict:
            raise ValueError(f"流程0缺少 linked code 的类型映射: {linked_code}")
        resolved_code_type_dict[linked_code] = str(declared_code_type_dict[linked_code])
    return resolved_code_type_dict


def _normalize_feature_df(feature_df):
    # 所有原始特征缓存和接口返回都在这里统一类型与去重规则，避免各分支各自做清洗。
    if feature_df is None or len(feature_df) == 0:
        return pd.DataFrame(columns=["date"])
    normalized_df = pd.DataFrame(feature_df, copy=True)
    if "date" not in normalized_df.columns:
        raise ValueError("原始特征数据缺少 date 列。")
    normalized_df["date"] = pd.to_datetime(normalized_df["date"], errors="coerce")
    for column in normalized_df.columns:
        if column == "date":
            continue
        normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce")
    normalized_df = normalized_df.dropna(subset=["date"]).copy()
    feature_col_list = [column for column in normalized_df.columns if column != "date"]
    if len(feature_col_list) == 0:
        return pd.DataFrame(columns=["date"])
    normalized_df = normalized_df.dropna(subset=feature_col_list, how="all").copy()
    normalized_df = normalized_df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return normalized_df


def _standardize_feature_df(df, rename_map, ordered_column_list, code):
    # 各类 AkShare 返回列名不一致，这里统一映射成流程0内部使用的标准字段名。
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date"])
    standardized_df = pd.DataFrame(df, copy=True).rename(columns=rename_map)
    if "date" not in standardized_df.columns:
        raise ValueError(f"原始特征缺少日期列，code={code}")
    existing_column_list = [column for column in ordered_column_list if column in standardized_df.columns]
    standardized_df = standardized_df[existing_column_list].copy()
    return _normalize_feature_df(feature_df=standardized_df)


def _build_price_cache_path(cache_dir, code, code_type):
    # 原始特征缓存按代码和类型拆分，避免不同标的之间相互覆盖。
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{FEATURE_CACHE_PREFIX}_{str(code).zfill(6)}_{str(code_type).strip().lower()}.csv"


def _load_cached_feature_df(cache_path):
    # 缓存命中后仍走统一标准化流程，兼容旧缓存和回读时的类型漂移。
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return pd.DataFrame(columns=["date"])
    cached_df = pd.read_csv(cache_path)
    return _normalize_feature_df(feature_df=cached_df)


def _save_feature_cache(feature_df, cache_path):
    # 缓存只保存标准化后的原始特征字段，后续增量合并直接复用同一结构。
    output_df = _normalize_feature_df(feature_df=feature_df)
    output_df["date"] = pd.to_datetime(output_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    output_df.to_csv(cache_path, index=False)


def _merge_feature_df(cached_feature_df, incoming_feature_df):
    # 增量回源结果与本地缓存按日期合并，重复日期优先保留最新抓取记录。
    merge_df_list = [df for df in (cached_feature_df, incoming_feature_df) if df is not None and len(df) > 0]
    if len(merge_df_list) == 0:
        return pd.DataFrame(columns=["date"])
    merged_df = pd.concat(merge_df_list, ignore_index=True, sort=False)
    return _normalize_feature_df(feature_df=merged_df)


def _resolve_last_cached_date(cached_feature_df):
    # 增量更新只依赖最后一个有效日期，便于理解和排查。
    if cached_feature_df is None or len(cached_feature_df) == 0:
        return None
    return pd.to_datetime(cached_feature_df["date"], errors="coerce").dropna().max()


def _resolve_incremental_start_date(last_cached_date):
    # 支持显式起止时间的接口从缓存尾部向前回看少量天数，兼容源端最近几日修订。
    if last_cached_date is None:
        return DEFAULT_START_DATE
    incremental_start = pd.Timestamp(last_cached_date) - pd.Timedelta(days=INCREMENTAL_FETCH_BUFFER_DAYS)
    incremental_start = max(incremental_start, pd.Timestamp(DEFAULT_START_DATE))
    return incremental_start.strftime("%Y%m%d")


def _resolve_open_fund_period(last_cached_date):
    # 开放式基金接口不支持自定义 start_date，只能按 period 取最近区间做增量回补。
    if last_cached_date is None:
        return "成立来"
    cover_days = max((pd.Timestamp(datetime.today().date()) - pd.Timestamp(last_cached_date)).days, 0)
    cover_days = cover_days + OPEN_FUND_PERIOD_BUFFER_DAYS
    period_config_list = [
        (31, "1月"),
        (92, "3月"),
        (183, "6月"),
        (366, "1年"),
        (366 * 3, "3年"),
        (366 * 5, "5年"),
    ]
    for max_days, period in period_config_list:
        if cover_days <= max_days:
            return period
    return "成立来"


def _fetch_open_fund_feature_df(ak_module, code, period="成立来"):
    # 开放式基金使用单位净值和累计净值两套接口，统一成一张原始特征表。
    nav_df = ak_module.fund_open_fund_info_em(symbol=code, indicator="单位净值走势", period=period)
    nav_feature_df = _standardize_feature_df(
        df=nav_df,
        rename_map={
            "净值日期": "date",
            "单位净值": "price",
            "日增长率": "daily_growth_rate",
        },
        ordered_column_list=["date", "price", "daily_growth_rate"],
        code=code,
    )
    cumulative_df = ak_module.fund_open_fund_info_em(symbol=code, indicator="累计净值走势", period=period)
    cumulative_feature_df = _standardize_feature_df(
        df=cumulative_df,
        rename_map={
            "净值日期": "date",
            "累计净值": "cumulative_nav",
        },
        ordered_column_list=["date", "cumulative_nav"],
        code=code,
    )
    if len(nav_feature_df) == 0:
        return cumulative_feature_df
    if len(cumulative_feature_df) == 0:
        return nav_feature_df
    merged_df = nav_feature_df.merge(cumulative_feature_df, on="date", how="left")
    return _normalize_feature_df(feature_df=merged_df)


def _fetch_etf_feature_df(ak_module, code, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    # ETF 使用日线行情字段，额外补一个 price 列统一表示收盘价。
    etf_df = ak_module.fund_etf_hist_em(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="",
    )
    standardized_df = _standardize_feature_df(
        df=etf_df,
        rename_map={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "change_pct",
            "涨跌额": "change_amount",
            "换手率": "turnover",
        },
        ordered_column_list=[
            "date",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "change_pct",
            "change_amount",
            "turnover",
        ],
        code=code,
    )
    if "close" in standardized_df.columns:
        standardized_df["price"] = standardized_df["close"]
    ordered_column_list = [
        "date",
        "price",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "change_pct",
        "change_amount",
        "turnover",
    ]
    existing_column_list = [column for column in ordered_column_list if column in standardized_df.columns]
    return _normalize_feature_df(feature_df=standardized_df[existing_column_list])


def _fetch_fund_feature_df(ak_module, code, last_cached_date=None):
    # fund 类型兼容开放式基金和场内 ETF，优先基金净值接口，失败时回退 ETF 行情接口。
    exception_list = []
    open_fund_period = _resolve_open_fund_period(last_cached_date=last_cached_date)
    etf_start_date = _resolve_incremental_start_date(last_cached_date=last_cached_date)
    fetch_config_list = [
        (_fetch_open_fund_feature_df, {"period": open_fund_period}),
        (_fetch_etf_feature_df, {"start_date": etf_start_date, "end_date": DEFAULT_END_DATE}),
    ]
    for fetcher, extra_kwargs in fetch_config_list:
        try:
            feature_df = fetcher(ak_module=ak_module, code=code, **extra_kwargs)
        except Exception as exc:
            exception_list.append(exc)
            continue
        if len(feature_df) > 0:
            return feature_df
    if len(exception_list) > 0:
        raise ValueError(f"fund 类型代码未拉取到有效原始特征: {code}") from exception_list[-1]
    raise ValueError(f"fund 类型代码未拉取到有效原始特征: {code}")


def _fetch_index_hist_feature_df(ak_module, code, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    # 指数优先使用东方财富指数历史接口，字段口径与 ETF 日线基本一致。
    try:
        index_df = ak_module.index_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
        )
    except Exception:
        return pd.DataFrame(columns=["date"])
    if index_df is None or "日期" not in getattr(index_df, "columns", []):
        return pd.DataFrame(columns=["date"])
    standardized_df = _standardize_feature_df(
        df=index_df,
        rename_map={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "change_pct",
            "涨跌额": "change_amount",
            "换手率": "turnover",
        },
        ordered_column_list=[
            "date",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "change_pct",
            "change_amount",
            "turnover",
        ],
        code=code,
    )
    if "close" in standardized_df.columns:
        standardized_df["price"] = standardized_df["close"]
    ordered_column_list = [
        "date",
        "price",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "change_pct",
        "change_amount",
        "turnover",
    ]
    existing_column_list = [column for column in ordered_column_list if column in standardized_df.columns]
    return _normalize_feature_df(feature_df=standardized_df[existing_column_list])


def _fetch_prefixed_index_feature_df(ak_module, symbol, code, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    # 带市场前缀的股票指数接口可覆盖部分普通指数接口无法识别的代码。
    try:
        index_df = ak_module.stock_zh_index_daily_em(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception:
        return pd.DataFrame(columns=["date"])
    if index_df is None or "date" not in getattr(index_df, "columns", []):
        return pd.DataFrame(columns=["date"])
    standardized_df = _standardize_feature_df(
        df=index_df,
        rename_map={
            "date": "date",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "volume": "volume",
            "amount": "amount",
        },
        ordered_column_list=["date", "open", "close", "high", "low", "volume", "amount"],
        code=code,
    )
    if "close" in standardized_df.columns:
        standardized_df["price"] = standardized_df["close"]
    ordered_column_list = ["date", "price", "open", "close", "high", "low", "volume", "amount"]
    existing_column_list = [column for column in ordered_column_list if column in standardized_df.columns]
    return _normalize_feature_df(feature_df=standardized_df[existing_column_list])


def _fetch_sina_index_feature_df(ak_module, symbol, code):
    # 新浪指数接口没有 start_date 参数，只在东方财富接口不稳定时作为补充来源。
    try:
        index_df = ak_module.stock_zh_index_daily(symbol=symbol)
    except Exception:
        return pd.DataFrame(columns=["date"])
    if index_df is None or "date" not in getattr(index_df, "columns", []):
        return pd.DataFrame(columns=["date"])
    standardized_df = _standardize_feature_df(
        df=index_df,
        rename_map={
            "date": "date",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "volume": "volume",
        },
        ordered_column_list=["date", "open", "close", "high", "low", "volume"],
        code=code,
    )
    if "close" in standardized_df.columns:
        standardized_df["price"] = standardized_df["close"]
    ordered_column_list = ["date", "price", "open", "close", "high", "low", "volume"]
    existing_column_list = [column for column in ordered_column_list if column in standardized_df.columns]
    return _normalize_feature_df(feature_df=standardized_df[existing_column_list])


def _fetch_tx_index_feature_df(ak_module, symbol, code):
    # 腾讯指数接口作为最后兜底，补足个别指数代码在前两类接口上的缺口。
    try:
        index_df = ak_module.stock_zh_index_daily_tx(symbol=symbol)
    except Exception:
        return pd.DataFrame(columns=["date"])
    if index_df is None or "date" not in getattr(index_df, "columns", []):
        return pd.DataFrame(columns=["date"])
    standardized_df = _standardize_feature_df(
        df=index_df,
        rename_map={
            "date": "date",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "amount": "amount",
        },
        ordered_column_list=["date", "open", "close", "high", "low", "amount"],
        code=code,
    )
    if "close" in standardized_df.columns:
        standardized_df["price"] = standardized_df["close"]
    ordered_column_list = ["date", "price", "open", "close", "high", "low", "amount"]
    existing_column_list = [column for column in ordered_column_list if column in standardized_df.columns]
    return _normalize_feature_df(feature_df=standardized_df[existing_column_list])


def _fetch_csindex_index_feature_df(ak_module, code, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    try:
        index_df = ak_module.stock_zh_index_hist_csindex(
            symbol=code,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception:
        return pd.DataFrame(columns=["date"])
    if index_df is None or "日期" not in getattr(index_df, "columns", []):
        return pd.DataFrame(columns=["date"])
    standardized_df = _standardize_feature_df(
        df=index_df,
        rename_map={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交金额": "amount",
        },
        ordered_column_list=["date", "open", "close", "high", "low", "volume", "amount"],
        code=code,
    )
    if "close" in standardized_df.columns:
        standardized_df["price"] = standardized_df["close"]
    ordered_column_list = ["date", "price", "open", "close", "high", "low", "volume", "amount"]
    existing_column_list = [column for column in ordered_column_list if column in standardized_df.columns]
    return _normalize_feature_df(feature_df=standardized_df[existing_column_list])


def _fetch_index_feature_df(ak_module, code, last_cached_date=None):
    # index 类型先走支持增量时间窗的主接口，再顺序回退到带前缀接口和其他数据源。
    exception_list = []
    incremental_start_date = _resolve_incremental_start_date(last_cached_date=last_cached_date)
    try:
        feature_df = _fetch_index_hist_feature_df(
            ak_module=ak_module,
            code=code,
            start_date=incremental_start_date,
            end_date=DEFAULT_END_DATE,
        )
        if len(feature_df) > 0:
            return feature_df
    except Exception as exc:
        exception_list.append(exc)
    try:
        feature_df = _fetch_csindex_index_feature_df(
            ak_module=ak_module,
            code=code,
            start_date=incremental_start_date,
            end_date=DEFAULT_END_DATE,
        )
        if len(feature_df) > 0:
            return feature_df
    except Exception as exc:
        exception_list.append(exc)
    for prefixed_symbol in (f"sh{code}", f"sz{code}", f"csi{code}"):
        for fetcher in (_fetch_prefixed_index_feature_df, _fetch_sina_index_feature_df, _fetch_tx_index_feature_df):
            try:
                if fetcher is _fetch_tx_index_feature_df and prefixed_symbol.startswith("csi"):
                    continue
                if fetcher is _fetch_prefixed_index_feature_df:
                    feature_df = fetcher(
                        ak_module=ak_module,
                        symbol=prefixed_symbol,
                        code=code,
                        start_date=incremental_start_date,
                        end_date=DEFAULT_END_DATE,
                    )
                else:
                    feature_df = fetcher(ak_module=ak_module, symbol=prefixed_symbol, code=code)
            except Exception as exc:
                exception_list.append(exc)
                continue
            if len(feature_df) > 0:
                return feature_df
    if len(exception_list) > 0:
        raise ValueError(f"index 类型代码未拉取到有效原始特征: {code}") from exception_list[-1]
    raise ValueError(f"index 类型代码未拉取到有效原始特征: {code}")


def _fetch_feature_df_by_type(ak_module, code, code_type, last_cached_date=None):
    # 原始特征抓取只在这一层按类型分流，输出统一标准化后的 DataFrame。
    normalized_code_type = str(code_type).strip().lower()
    if normalized_code_type == "fund":
        return _fetch_fund_feature_df(ak_module=ak_module, code=code, last_cached_date=last_cached_date)
    if normalized_code_type == "index":
        return _fetch_index_feature_df(ak_module=ak_module, code=code, last_cached_date=last_cached_date)
    raise ValueError(f"未支持的代码类型: code={code}, code_type={code_type}")


def _fetch_feature_df_with_cache(ak_module, code, code_type, cache_dir, force_refresh=False):
    # 原始特征数据优先命中本地缓存，仅对尾部区间回源抓取并回写，实现稳定的增量更新。
    cache_path = _build_price_cache_path(cache_dir=cache_dir, code=code, code_type=code_type)
    cached_feature_df = pd.DataFrame(columns=["date"])
    if cache_path.exists() and not force_refresh:
        cached_feature_df = _load_cached_feature_df(cache_path=cache_path)
    last_cached_date = None if force_refresh else _resolve_last_cached_date(cached_feature_df=cached_feature_df)
    incoming_feature_df = _fetch_feature_df_by_type(
        ak_module=ak_module,
        code=code,
        code_type=code_type,
        last_cached_date=last_cached_date,
    )
    merged_feature_df = _merge_feature_df(cached_feature_df=cached_feature_df, incoming_feature_df=incoming_feature_df)
    if len(merged_feature_df) == 0:
        raise ValueError(f"未获取到有效原始特征缓存数据: code={code}, code_type={code_type}")
    _save_feature_cache(feature_df=merged_feature_df, cache_path=cache_path)
    return merged_feature_df


def _build_checked_output_path(raw_output_path):
    # 最终特征宽表主文件切换到 parquet，仅保留与原始宽表稳定可推导的命名关系。
    raw_output_path = Path(raw_output_path)
    return raw_output_path.with_name(f"{raw_output_path.stem}{CHECKED_OUTPUT_SUFFIX}{PARQUET_OUTPUT_SUFFIX}")


def _load_saved_feature_table(table_path):
    # 流程0宽表读写按文件后缀自动分流，兼容原始 csv 和最终 parquet。
    table_path = Path(table_path)
    if not table_path.exists():
        raise FileNotFoundError(f"宽表文件不存在: {table_path}")
    if table_path.suffix.lower() == ".parquet":
        return pd.read_parquet(table_path)
    return pd.read_csv(table_path)


def _save_feature_table(feature_df, table_path):
    # 最终特征宽表优先落 parquet，原始宽表和缓存仍沿用 csv。
    table_path = Path(table_path)
    output_df = pd.DataFrame(feature_df, copy=True)
    if "date" in output_df.columns:
        output_df["date"] = pd.to_datetime(output_df["date"], errors="coerce")
    if table_path.suffix.lower() == ".parquet":
        output_df.to_parquet(table_path, index=False)
        return
    if "date" in output_df.columns:
        output_df["date"] = output_df["date"].dt.strftime("%Y-%m-%d")
    output_df.to_csv(table_path, index=False)


def _load_saved_wide_feature_table(raw_output_path):
    # 检查阶段统一复用按后缀分流的宽表加载逻辑，避免 csv/parquet 分支散落。
    return _load_saved_feature_table(table_path=raw_output_path)


def _validate_wide_feature_table_structure(wide_feature_df, code_type_dict, primary_code, require_all_code_group_columns=True):
    # 结构性问题属于硬错误，必须在补缺前先挡住。
    if len(wide_feature_df) == 0:
        raise ValueError("宽表为空，无法继续检查。")
    if "date" not in wide_feature_df.columns:
        raise ValueError("宽表缺少 date 列。")
    date_series = pd.to_datetime(wide_feature_df["date"], errors="coerce")
    if date_series.isna().any():
        raise ValueError("宽表存在无法解析的 date 值。")
    if bool(date_series.duplicated().any()):
        raise ValueError("宽表存在重复日期。")
    if not bool(date_series.is_monotonic_increasing):
        raise ValueError("宽表日期未按升序排列。")
    if bool(require_all_code_group_columns):
        for code in dict(code_type_dict).keys():
            normalized_code = str(code).zfill(6)
            column_list = [column for column in wide_feature_df.columns if column.startswith(f"{normalized_code}__")]
            if len(column_list) == 0:
                raise ValueError(f"宽表缺少代码列组: {normalized_code}")
    primary_prefix = f"{str(primary_code).zfill(6)}__"
    primary_column_list = [column for column in wide_feature_df.columns if column.startswith(primary_prefix)]
    primary_missing_mask = ~wide_feature_df[primary_column_list].notna().any(axis=1)
    if bool(primary_missing_mask.any()):
        missing_date_list = wide_feature_df.loc[primary_missing_mask, "date"].tolist()
        raise ValueError(f"主代码存在整组缺失行: {missing_date_list}")


def _build_missing_position_list(feature_series):
    # 基础特征缺失治理改为单列口径，便于直接淘汰质量差的列并阻断整条派生链。
    feature_series = pd.Series(feature_series, copy=False)
    return feature_series.index[feature_series.isna()].tolist()


def _compute_max_consecutive_missing_span(position_list):
    # 连续缺失段长度用于直接淘汰长缺口特征，不再依赖整组字段是否同时为空。
    if len(position_list) == 0:
        return 0, None, None
    best_length = 1
    best_start = int(position_list[0])
    best_end = int(position_list[0])
    current_start = int(position_list[0])
    current_end = int(position_list[0])
    for position in position_list[1:]:
        position = int(position)
        if position == current_end + 1:
            current_end = position
        else:
            current_length = current_end - current_start + 1
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
                best_end = current_end
            current_start = position
            current_end = position
    current_length = current_end - current_start + 1
    if current_length > best_length:
        best_length = current_length
        best_start = current_start
        best_end = current_end
    return int(best_length), int(best_start), int(best_end)


def _build_single_feature_missing_report(checked_feature_df, feature_column):
    # 单列缺失报告既服务于阈值淘汰，也服务于末尾警告和元信息落盘。
    feature_series = pd.to_numeric(pd.Series(checked_feature_df[feature_column], copy=True), errors="coerce")
    missing_position_list = _build_missing_position_list(feature_series=feature_series)
    max_consecutive_missing_count, max_start_position, max_end_position = _compute_max_consecutive_missing_span(
        position_list=missing_position_list,
    )
    missing_ratio = float(len(missing_position_list) / len(feature_series)) if len(feature_series) > 0 else 0.0
    drop_reason_list = []
    if missing_ratio >= float(MAX_FEATURE_MISSING_RATIO):
        drop_reason_list.append("missing_ratio_threshold")
    if max_consecutive_missing_count >= int(MAX_FEATURE_CONSECUTIVE_MISSING_COUNT):
        drop_reason_list.append("consecutive_missing_threshold")
    return {
        "column": str(feature_column),
        "missing_count": int(len(missing_position_list)),
        "missing_ratio": missing_ratio,
        "missing_date_list": [str(checked_feature_df.loc[position, "date"]) for position in missing_position_list],
        "max_consecutive_missing_count": int(max_consecutive_missing_count),
        "max_consecutive_missing_start_date": None if max_start_position is None else str(checked_feature_df.loc[max_start_position, "date"]),
        "max_consecutive_missing_end_date": None if max_end_position is None else str(checked_feature_df.loc[max_end_position, "date"]),
        "dropped": bool(len(drop_reason_list) > 0),
        "drop_reason_list": drop_reason_list,
    }


def _fill_single_feature_missing_by_linear_interpolation(checked_feature_df, feature_column):
    # 方案B：先线性插值，再用最近有效值补齐边界缺口，避免首尾残留 NaN。
    filled_df = checked_feature_df.copy()
    feature_series = pd.to_numeric(pd.Series(filled_df[feature_column], copy=True), errors="coerce")
    interpolated_series = feature_series.interpolate(method="linear", limit_direction="both")
    interpolated_series = interpolated_series.ffill().bfill()
    filled_df[feature_column] = interpolated_series.astype(float)
    return filled_df


def _check_and_fill_wide_feature_table(raw_output_path, code_type_dict, primary_code):
    # 基础特征先按单列质量做淘汰和线性补全，再进入流程0后续因子构造。
    raw_feature_df = _load_saved_wide_feature_table(raw_output_path=raw_output_path)
    _validate_wide_feature_table_structure(
        wide_feature_df=raw_feature_df,
        code_type_dict=code_type_dict,
        primary_code=primary_code,
    )
    checked_feature_df = raw_feature_df.copy()
    source_column_list = [column for column in checked_feature_df.columns if column != "date"]
    feature_quality_report_list = []
    dropped_source_column_list = []
    for feature_column in source_column_list:
        feature_quality_report = _build_single_feature_missing_report(
            checked_feature_df=checked_feature_df,
            feature_column=feature_column,
        )
        if bool(feature_quality_report["dropped"]):
            dropped_source_column_list.append(str(feature_column))
            feature_quality_report["filled"] = False
            feature_quality_report["filled_missing_count"] = 0
            feature_quality_report_list.append(feature_quality_report)
            continue
        if int(feature_quality_report["missing_count"]) > 0:
            checked_feature_df = _fill_single_feature_missing_by_linear_interpolation(
                checked_feature_df=checked_feature_df,
                feature_column=feature_column,
            )
            feature_quality_report["filled"] = True
            feature_quality_report["filled_missing_count"] = int(feature_quality_report["missing_count"])
        else:
            feature_quality_report["filled"] = False
            feature_quality_report["filled_missing_count"] = 0
        feature_quality_report_list.append(feature_quality_report)
    if len(dropped_source_column_list) > 0:
        checked_feature_df = checked_feature_df.drop(columns=dropped_source_column_list).copy()
    _validate_wide_feature_table_structure(
        wide_feature_df=checked_feature_df,
        code_type_dict=code_type_dict,
        primary_code=primary_code,
        require_all_code_group_columns=False,
    )
    checked_output_path = _build_checked_output_path(raw_output_path=raw_output_path)
    _save_feature_table(feature_df=checked_feature_df, table_path=checked_output_path)
    return checked_feature_df, checked_output_path, feature_quality_report_list, dropped_source_column_list


def _resolve_factor_source_column_list(checked_feature_df, dropped_source_column_list=None):
    # 因子计算只消费 checked 宽表中的基础特征列，不把 date 当作输入特征。
    checked_feature_df = pd.DataFrame(checked_feature_df, copy=True)
    dropped_source_column_set = set([] if dropped_source_column_list is None else list(dropped_source_column_list))
    return [column for column in checked_feature_df.columns if column != "date" and column not in dropped_source_column_set]


def _build_factor_candidate_config(strategy_params):
    # 因子候选展开复用 selection 流程的参数搜索空间，保证流程0与筛选逻辑同源。
    resolved_strategy_params = dict(strategy_params)
    candidate_factor_name_list = [str(factor_name) for factor_name in resolved_strategy_params["enabled_factor_list"]]
    candidate_factor_list = build_factor_candidate_list(
        candidate_factor_name_list=candidate_factor_name_list,
        strategy_params=resolved_strategy_params,
    )
    return resolved_strategy_params, candidate_factor_list


def _build_mature_sample_mask(index_like, trim_row_count=INITIAL_TRIM_ROW_COUNT):
    # 前段统一裁剪口径同时服务于筛选统计和最终整表输出，避免两处各自维护边界。
    sample_index = pd.Index(index_like)
    mature_sample_mask = pd.Series(True, index=sample_index, dtype=bool)
    trim_row_count = max(int(trim_row_count), 0)
    if trim_row_count <= 0 or len(sample_index) == 0:
        return mature_sample_mask
    mature_sample_mask.iloc[: min(trim_row_count, len(sample_index))] = False
    return mature_sample_mask


def _compute_normalized_zero_ratio_on_mature_samples(normalized_factor_series, trim_row_count=INITIAL_TRIM_ROW_COUNT):
    # 高零占比判定只看成熟样本段，避免初始窗口不足和早期不稳定区间放大零值比例。
    normalized_factor_series = pd.Series(normalized_factor_series, copy=True).astype(float)
    mature_sample_mask = _build_mature_sample_mask(
        index_like=normalized_factor_series.index,
        trim_row_count=trim_row_count,
    )
    mature_sample_series = normalized_factor_series.loc[mature_sample_mask]
    if len(mature_sample_series) == 0:
        return 0.0
    return float((mature_sample_series == 0.0).mean())


def _trim_initial_rows(feature_df, trim_row_count=INITIAL_TRIM_ROW_COUNT):
    # 流程0最终输出按统一行裁剪，保证 date、价格列和全部因子列继续共享同一索引。
    feature_df = pd.DataFrame(feature_df, copy=True)
    trim_row_count = max(int(trim_row_count), 0)
    if trim_row_count <= 0 or len(feature_df) == 0:
        return feature_df.reset_index(drop=True)
    trimmed_feature_df = feature_df.iloc[min(trim_row_count, len(feature_df)) :].copy()
    return trimmed_feature_df.reset_index(drop=True)


def _build_single_feature_factor_df(feature_series, candidate_factor_list, strategy_params):
    # 单个基础特征列生成自身标准化列，并按阈值筛选可保留的标准化因子列。
    feature_series = pd.Series(feature_series, copy=True).astype(float)
    factor_series_dict = {}
    score_window = int(strategy_params["score_window"])
    factor_series_dict["zscore"] = rolling_zscore(feature_series, window=score_window)
    dropped_factor_report_list = []
    for factor_candidate in candidate_factor_list:
        candidate_label = str(factor_candidate["candidate_label"])
        raw_factor_series = pd.Series(
            build_raw_factor_series(
                price_series=feature_series,
                factor_name=str(factor_candidate["factor_name"]),
                factor_param_dict={
                    str(factor_candidate["factor_name"]): dict(factor_candidate["param_dict"]),
                },
            ),
            copy=True,
        ).astype(float)
        raw_factor_series = raw_factor_series.replace([float("inf"), -float("inf")], float("nan"))
        raw_nan_ratio = float(raw_factor_series.isna().mean())
        normalized_factor_series = normalize_factor_series(
            raw_factor_series=raw_factor_series,
            factor_name=str(factor_candidate["factor_name"]),
            score_window=score_window,
        )
        normalized_factor_series = pd.Series(normalized_factor_series, copy=True).astype(float)
        normalized_zero_ratio = _compute_normalized_zero_ratio_on_mature_samples(
            normalized_factor_series=normalized_factor_series,
        )
        drop_reason_list = []
        if raw_nan_ratio >= float(MAX_FACTOR_RAW_NAN_RATIO):
            drop_reason_list.append("raw_nan_ratio_threshold")
        if normalized_zero_ratio >= float(MAX_NORMALIZED_ZERO_RATIO):
            drop_reason_list.append("normalized_zero_ratio_threshold")
        if len(drop_reason_list) > 0:
            dropped_factor_report_list.append(
                {
                    "candidate_label": candidate_label,
                    "raw_nan_ratio": raw_nan_ratio,
                    "normalized_zero_ratio": normalized_zero_ratio,
                    "drop_reason_list": drop_reason_list,
                }
            )
            continue
        factor_series_dict[f"{candidate_label}__zscore"] = normalized_factor_series
    factor_df = pd.DataFrame(factor_series_dict, index=feature_series.index)
    return factor_df.fillna(0.0), dropped_factor_report_list


def _extract_feature_column_record(feature_column):
    # 流程0基础列统一解析成代码和字段两段，后续单输入和跨标的绑定都复用同一份结构。
    feature_column = str(feature_column)
    if "__" not in feature_column:
        raise ValueError(f"基础特征列不符合 code__field 约定: {feature_column}")
    code, field_name = feature_column.split("__", 1)
    return {
        "column": feature_column,
        "code": str(code).zfill(6),
        "field_name": str(field_name),
    }


def _build_feature_column_record_list(source_column_list):
    # checked 宽表只把原始基础列纳入绑定池，避免把已生成的因子列再次当作输入源。
    return [_extract_feature_column_record(feature_column=source_column) for source_column in list(source_column_list)]


def _build_single_input_factor_output_column(source_column, candidate_label):
    # 单输入因子继续沿用旧列名契约，保证流程1-5继续消费已有产物。
    return f"{str(source_column)}__{str(candidate_label)}__zscore"


def _build_multi_field_factor_output_column(asset_code, bound_field_name_list, candidate_label):
    # 同一资产下多字段因子把依赖字段压到列名前缀中，保证输出列名可直接回溯绑定来源。
    field_signature = "-".join([str(field_name) for field_name in list(bound_field_name_list)])
    return f"{str(asset_code).zfill(6)}__{field_signature}__{str(candidate_label)}__zscore"


def _build_feature_column_dict_by_asset_code(source_column_list):
    # 同一资产多字段依赖需要先把基础列聚合到 code -> field_name -> column 的索引结构中。
    feature_column_dict_by_asset_code = {}
    for record in _build_feature_column_record_list(source_column_list=source_column_list):
        asset_code = str(record["code"])
        field_name = str(record["field_name"])
        feature_column_dict_by_asset_code.setdefault(asset_code, {})
        feature_column_dict_by_asset_code[asset_code][field_name] = str(record["column"])
    return feature_column_dict_by_asset_code


def _build_factor_task_list(source_column_list, candidate_factor_list, fund_code):
    # 因子任务按依赖规格展开：单字段因子继续逐列绑定，多字段因子按同一资产内字段齐全性生成任务。
    feature_column_record_list = _build_feature_column_record_list(source_column_list=source_column_list)
    feature_column_dict_by_asset_code = _build_feature_column_dict_by_asset_code(source_column_list=source_column_list)
    target_code = str(fund_code).zfill(6)
    task_list = []
    for factor_candidate in list(candidate_factor_list):
        factor_name = str(factor_candidate["factor_name"])
        candidate_label = str(factor_candidate["candidate_label"])
        dependency_spec = resolve_factor_dependency_spec(factor_name=factor_name)
        binding_mode = str(dependency_spec["binding_mode"])
        if binding_mode == "single_field":
            candidate_field_name_list = [str(field_name) for field_name in list(dependency_spec["candidate_field_name_list"])]
            for record in feature_column_record_list:
                if str(record["field_name"]) not in candidate_field_name_list:
                    continue
                task_list.append(
                    {
                        "dependency_mode": binding_mode,
                        "factor_candidate": dict(factor_candidate),
                        "output_column": _build_single_input_factor_output_column(
                            source_column=record["column"],
                            candidate_label=candidate_label,
                        ),
                        "feature_input_key_to_column_dict": {
                            "source": str(record["column"]),
                        },
                        "binding_record": {
                            "output_column": _build_single_input_factor_output_column(
                                source_column=record["column"],
                                candidate_label=candidate_label,
                            ),
                            "factor_name": factor_name,
                            "candidate_label": candidate_label,
                            "binding_mode": binding_mode,
                            "asset_code": str(record["code"]),
                            "bound_field_name_list": [str(record["field_name"])],
                            "bound_source_column_list": [str(record["column"])],
                            "target_code": target_code,
                            "linked_code": None,
                        },
                    }
                )
            continue
        if binding_mode != "multi_field_same_asset":
            raise ValueError(f"未支持的 binding_mode: {binding_mode}")
        required_field_name_list = [str(field_name) for field_name in list(dependency_spec["required_field_name_list"])]
        for asset_code, field_to_column_dict in feature_column_dict_by_asset_code.items():
            if any(field_name not in field_to_column_dict for field_name in required_field_name_list):
                continue
            feature_input_key_to_column_dict = {
                str(field_name): str(field_to_column_dict[field_name]) for field_name in required_field_name_list
            }
            bound_source_column_list = [str(field_to_column_dict[field_name]) for field_name in required_field_name_list]
            output_column = _build_multi_field_factor_output_column(
                asset_code=asset_code,
                bound_field_name_list=required_field_name_list,
                candidate_label=candidate_label,
            )
            task_list.append(
                {
                    "dependency_mode": binding_mode,
                    "factor_candidate": dict(factor_candidate),
                    "output_column": output_column,
                    "feature_input_key_to_column_dict": feature_input_key_to_column_dict,
                    "binding_record": {
                        "output_column": output_column,
                        "factor_name": factor_name,
                        "candidate_label": candidate_label,
                        "binding_mode": binding_mode,
                        "asset_code": str(asset_code),
                        "bound_field_name_list": list(required_field_name_list),
                        "bound_source_column_list": list(bound_source_column_list),
                        "target_code": target_code,
                        "linked_code": None,
                    },
                }
            )
    return task_list


def _build_bound_feature_input_dict(checked_feature_df, feature_input_key_to_column_dict):
    # 绑定成功后的输入字典只在这里从宽表取列，避免各类因子分支重复处理 Series 对齐。
    return {
        str(input_key): pd.Series(checked_feature_df[str(source_column)], copy=True).astype(float)
        for input_key, source_column in dict(feature_input_key_to_column_dict).items()
    }


def _build_bound_factor_series(feature_input_dict, factor_candidate, strategy_params):
    # 单字段和同资产多字段因子统一在绑定后生成 raw/normalized 结果，过滤规则继续复用流程0现有阈值。
    factor_candidate = dict(factor_candidate)
    factor_name = str(factor_candidate["factor_name"])
    candidate_label = str(factor_candidate["candidate_label"])
    score_window = int(strategy_params["score_window"])
    raw_factor_series = pd.Series(
        build_raw_factor_series(
            price_series=None,
            factor_name=factor_name,
            factor_param_dict={
                factor_name: dict(factor_candidate["param_dict"]),
            },
            feature_input_dict=feature_input_dict,
        ),
        copy=True,
    ).astype(float)
    raw_factor_series = raw_factor_series.replace([float("inf"), -float("inf")], float("nan"))
    raw_nan_ratio = float(raw_factor_series.isna().mean())
    normalized_factor_series = normalize_factor_series(
        raw_factor_series=raw_factor_series,
        factor_name=factor_name,
        score_window=score_window,
    )
    normalized_factor_series = pd.Series(normalized_factor_series, copy=True).astype(float)
    normalized_zero_ratio = _compute_normalized_zero_ratio_on_mature_samples(
        normalized_factor_series=normalized_factor_series,
    )
    drop_reason_list = []
    if raw_nan_ratio >= float(MAX_FACTOR_RAW_NAN_RATIO):
        drop_reason_list.append("raw_nan_ratio_threshold")
    if normalized_zero_ratio >= float(MAX_NORMALIZED_ZERO_RATIO):
        drop_reason_list.append("normalized_zero_ratio_threshold")
    return {
        "candidate_label": candidate_label,
        "raw_nan_ratio": raw_nan_ratio,
        "normalized_zero_ratio": normalized_zero_ratio,
        "drop_reason_list": drop_reason_list,
        "normalized_factor_series": normalized_factor_series.fillna(0.0),
    }


def _build_checked_factor_table(checked_output_path, strategy_params, fund_code, dropped_source_column_list=None):
    # checked 表中的新增列先批量收集再一次性拼接，避免数千次逐列插入导致 DataFrame 高碎片。
    checked_feature_df = _load_saved_wide_feature_table(raw_output_path=checked_output_path)
    resolved_strategy_params, candidate_factor_list = _build_factor_candidate_config(strategy_params=strategy_params)
    source_column_list = _resolve_factor_source_column_list(
        checked_feature_df=checked_feature_df,
        dropped_source_column_list=dropped_source_column_list,
    )
    base_factor_column_dict = {}
    for source_column in list(source_column_list):
        base_factor_column_dict[f"{source_column}__zscore"] = rolling_zscore(
            pd.Series(checked_feature_df[source_column], copy=True).astype(float),
            window=int(resolved_strategy_params["score_window"]),
        )
    factor_task_list = _build_factor_task_list(
        source_column_list=source_column_list,
        candidate_factor_list=candidate_factor_list,
        fund_code=fund_code,
    )
    expected_added_column_count = int(len(source_column_list) + len(factor_task_list))
    accumulated_added_column_count = 0
    dropped_factor_report_list = []
    factor_binding_record_list = []
    accumulated_added_column_count = accumulated_added_column_count + int(len(source_column_list))
    generated_factor_column_dict = {}
    for task_idx, factor_task in enumerate(factor_task_list, start=1):
        feature_input_dict = _build_bound_feature_input_dict(
            checked_feature_df=checked_feature_df,
            feature_input_key_to_column_dict=factor_task["feature_input_key_to_column_dict"],
        )
        factor_result = _build_bound_factor_series(
            feature_input_dict=feature_input_dict,
            factor_candidate=factor_task["factor_candidate"],
            strategy_params=resolved_strategy_params,
        )
        if len(factor_result["drop_reason_list"]) > 0:
            dropped_factor_report = dict(factor_result)
            dropped_factor_report["output_column"] = str(factor_task["output_column"])
            dropped_factor_report["source_column"] = list(factor_task["binding_record"]["bound_source_column_list"])[0]
            dropped_factor_report["bound_source_column_list"] = list(factor_task["binding_record"]["bound_source_column_list"])
            dropped_factor_report_list.append(dropped_factor_report)
            print(
                "删除因子:",
                f"绑定={','.join(dropped_factor_report['bound_source_column_list'])}",
                f"因子={dropped_factor_report['output_column']}",
                f"raw_nan_ratio={factor_result['raw_nan_ratio']:.4f}",
                f"normalized_zero_ratio={factor_result['normalized_zero_ratio']:.4f}",
                "原因=超过阈值",
            )
            continue
        generated_factor_column_dict[str(factor_task["output_column"])] = factor_result["normalized_factor_series"].to_numpy(dtype=float)
        factor_binding_record_list.append(dict(factor_task["binding_record"]))
        accumulated_added_column_count = accumulated_added_column_count + 1
        print(
            f"因子进度: {task_idx}/{len(factor_task_list)}",
            f"输出列={factor_task['output_column']}",
            f"已生成新增列={accumulated_added_column_count}/{expected_added_column_count}",
        )
    factor_df = pd.DataFrame(
        {
            **base_factor_column_dict,
            **generated_factor_column_dict,
        },
        index=checked_feature_df.index,
    )
    extended_checked_df = pd.concat([checked_feature_df.copy(), factor_df], axis=1)
    _save_feature_table(feature_df=extended_checked_df, table_path=checked_output_path)
    return extended_checked_df, checked_output_path, source_column_list, candidate_factor_list, dropped_factor_report_list, factor_binding_record_list


def build_code_feature_table(code_type_dict, output_path=None, primary_code=None, force_refresh=False, config=None):
    # 原始宽表构造必须遵守调用方配置，避免流程0内部再偷偷回退到默认配置。
    if config is None:
        config = build_tradition_config()
    if code_type_dict is None or len(dict(code_type_dict)) == 0:
        raise ValueError("code_type_dict 不能为空。")
    if primary_code is None:
        raise ValueError("primary_code 不能为空。")

    import akshare as ak

    feature_df_dict = {}
    for code, code_type in dict(code_type_dict).items():
        normalized_code = str(code).zfill(6)
        feature_df = _fetch_feature_df_with_cache(
            ak_module=ak,
            code=normalized_code,
            code_type=code_type,
            cache_dir=config["data_dir"],
            force_refresh=bool(force_refresh),
        )
        feature_df_dict[normalized_code] = _normalize_feature_df(feature_df=feature_df).set_index("date")

    if len(feature_df_dict) == 0:
        raise ValueError("未生成任何原始特征表。")
    normalized_primary_code = str(primary_code).zfill(6)
    if normalized_primary_code not in feature_df_dict:
        raise ValueError(f"主代码未包含在 code_type_dict 中: {normalized_primary_code}")

    primary_index = feature_df_dict[normalized_primary_code].index.copy()
    wide_feature_df = pd.DataFrame(index=primary_index)
    for code in dict(code_type_dict).keys():
        normalized_code = str(code).zfill(6)
        feature_df = feature_df_dict[normalized_code].reindex(primary_index).copy()
        feature_df.columns = [f"{normalized_code}__{column}" for column in feature_df.columns]
        wide_feature_df = pd.concat([wide_feature_df, feature_df], axis=1)
    wide_feature_df = wide_feature_df[~wide_feature_df.index.duplicated(keep="last")]
    wide_feature_df.index = pd.to_datetime(wide_feature_df.index).strftime("%Y-%m-%d")
    wide_feature_df.index.name = "date"

    if output_path is None:
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_output_path = output_dir / DEFAULT_OUTPUT_NAME
    else:
        resolved_output_path = Path(output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    wide_feature_df.to_csv(resolved_output_path)
    return wide_feature_df, resolved_output_path


def _resolve_feature_column_group_list(checked_feature_df, source_column_list):
    # 元信息和翻转逻辑都要基于同一套列分组规则，避免同一文件被不同函数解释成不同候选集合。
    checked_feature_df = checked_feature_df.copy()
    raw_feature_column_list = list(source_column_list)
    raw_feature_zscore_column_list = [f"{column}__zscore" for column in raw_feature_column_list if f"{column}__zscore" in checked_feature_df.columns]
    excluded_column_set = set(["date", *raw_feature_column_list, *raw_feature_zscore_column_list])
    factor_feature_column_list = [column for column in checked_feature_df.columns if column not in excluded_column_set]
    return raw_feature_column_list, raw_feature_zscore_column_list, factor_feature_column_list


def _resolve_target_column_name_pair(checked_feature_df, fund_code):
    # 流程0内部凡是需要定义目标列，都统一复用这组规则，保证和下游看到的 JSON 契约一致。
    target_price_column = f"{str(fund_code).zfill(6)}__price"
    target_nav_column = f"{str(fund_code).zfill(6)}__cumulative_nav"
    if target_nav_column not in checked_feature_df.columns:
        target_nav_column = target_price_column
    return target_price_column, target_nav_column


def _build_flipped_factor_feature_column_name(factor_feature_column):
    # 翻转列只在“因子名”位置加负号，保留基础特征来源和 zscore 后缀不变。
    factor_feature_column = str(factor_feature_column)
    factor_suffix = "__zscore"
    if not factor_feature_column.endswith(factor_suffix):
        raise ValueError(f"翻转列名必须以 {factor_suffix} 结尾: {factor_feature_column}")
    column_body = factor_feature_column[: -len(factor_suffix)]
    if "__" not in column_body:
        raise ValueError(f"无法从候选因子列中解析基础特征与因子名: {factor_feature_column}")
    source_column, factor_name = column_body.rsplit("__", 1)
    return f"{source_column}__-{factor_name}{factor_suffix}"


def _build_flipped_factor_report_list(checked_feature_df, factor_feature_column_list, target_nav_column, config):
    # 翻转判定严格复用流程1的训练集口径，只额外统计正负 IC 次数决定是否追加反向候选列。
    feature_df = checked_feature_df.copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce")
    feature_df = feature_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    feature_df = feature_df.set_index("date")
    target_nav_series = pd.Series(feature_df[target_nav_column], copy=True).astype(float)
    fold_list = build_walk_forward_dev_fold_list(
        price_series=target_nav_series,
        walk_forward_config=dict(config["walk_forward_config"]),
        split_config=config["data_split_dict"],
    )
    ic_aggregation_config = build_ic_aggregation_config(config)
    forward_return_series = build_forward_return_series(price_series=target_nav_series, forward_window=5)
    flipped_factor_report_list = []
    for factor_feature_column in list(factor_feature_column_list):
        factor_series = pd.Series(feature_df[str(factor_feature_column)], copy=True).astype(float)
        train_metric_list = []
        for fold_dict in fold_list:
            train_metric_list.append(
                compute_segment_correlation_metrics(
                    factor_series=factor_series,
                    forward_return_series=forward_return_series,
                    segment_index=fold_dict["train"].index,
                )
            )
        train_spearman_value_list = [metric_dict["spearman_ic"] for metric_dict in train_metric_list]
        train_spearman_summary = build_spearman_metric_summary(
            train_spearman_value_list,
            ic_aggregation_config=ic_aggregation_config,
        )
        valid_train_metric_series = pd.Series(train_spearman_value_list, dtype=float).dropna()
        positive_ic_count = int((valid_train_metric_series > 0.0).sum())
        negative_ic_count = int((valid_train_metric_series < 0.0).sum())
        if (
            train_spearman_summary["count"] > 0
            and pd.notna(train_spearman_summary["mean"])
            and float(train_spearman_summary["mean"]) < 0.0
            and negative_ic_count > positive_ic_count
        ):
            flipped_factor_report_list.append(
                {
                    "original_column": str(factor_feature_column),
                    "flipped_column": _build_flipped_factor_feature_column_name(factor_feature_column=factor_feature_column),
                    "train_spearman_ic_mean": float(train_spearman_summary["mean"]),
                    "train_sample_fold_count": int(train_spearman_summary["count"]),
                    "positive_ic_count": positive_ic_count,
                    "negative_ic_count": negative_ic_count,
                }
            )
    return flipped_factor_report_list


def _append_flipped_factor_feature_columns(checked_feature_df, checked_output_path, source_column_list, fund_code, config):
    # 翻转列同样先批量构造成新表，再统一拼接到 checked 宽表，避免继续逐列插入。
    checked_feature_df = checked_feature_df.copy()
    _, _, factor_feature_column_list = _resolve_feature_column_group_list(
        checked_feature_df=checked_feature_df,
        source_column_list=source_column_list,
    )
    _, target_nav_column = _resolve_target_column_name_pair(
        checked_feature_df=checked_feature_df,
        fund_code=fund_code,
    )
    flipped_factor_report_list = _build_flipped_factor_report_list(
        checked_feature_df=checked_feature_df,
        factor_feature_column_list=factor_feature_column_list,
        target_nav_column=target_nav_column,
        config=config,
    )
    flipped_factor_column_dict = {}
    for flipped_factor_report in flipped_factor_report_list:
        original_column = str(flipped_factor_report["original_column"])
        flipped_column = str(flipped_factor_report["flipped_column"])
        flipped_factor_column_dict[flipped_column] = -pd.Series(checked_feature_df[original_column], copy=True).astype(float)
    if len(flipped_factor_column_dict) > 0:
        checked_feature_df = pd.concat(
            [checked_feature_df, pd.DataFrame(flipped_factor_column_dict, index=checked_feature_df.index)],
            axis=1,
        )
    _save_feature_table(feature_df=checked_feature_df, table_path=checked_output_path)
    return checked_feature_df, checked_output_path, flipped_factor_report_list


def _build_feature_preprocess_metadata(
    checked_feature_df,
    feature_path,
    path_code,
    fund_code,
    code_type_dict,
    source_column_list,
    candidate_factor_list,
    feature_quality_report_list,
    dropped_source_column_list,
    dropped_factor_report_list,
    factor_binding_record_list,
    flipped_factor_report_list,
):
    # 元信息 JSON 显式声明目标列和特征列，避免流程1再从命名规则反推接口。
    checked_feature_df = checked_feature_df.copy()
    raw_feature_column_list, raw_feature_zscore_column_list, factor_feature_column_list = _resolve_feature_column_group_list(
        checked_feature_df=checked_feature_df,
        source_column_list=source_column_list,
    )
    target_price_column, target_nav_column = _resolve_target_column_name_pair(
        checked_feature_df=checked_feature_df,
        fund_code=fund_code,
    )
    feature_column_list = list(raw_feature_zscore_column_list) + list(factor_feature_column_list)
    return {
        "fund_code": str(fund_code).zfill(6),
        "primary_code": str(fund_code).zfill(6),
        "analysis_date": datetime.today().strftime("%Y-%m-%d"),
        "data_mode": "feature_matrix",
        "path_code": str(path_code),
        "feature_path": str(Path(feature_path).resolve()),
        "feature_format": str(Path(feature_path).suffix).lstrip(".").lower(),
        "target_price_column": target_price_column,
        "target_nav_column": target_nav_column,
        "feature_column_list": feature_column_list,
        "raw_feature_column_list": raw_feature_column_list,
        "raw_feature_zscore_column_list": raw_feature_zscore_column_list,
        "factor_feature_column_list": factor_feature_column_list,
        "factor_binding_record_list": list(factor_binding_record_list),
        "linked_code_type_dict": {str(code).zfill(6): str(code_type) for code, code_type in dict(code_type_dict).items()},
        "candidate_factor_count": int(len(candidate_factor_list)),
        "row_count": int(len(checked_feature_df)),
        "column_count": int(len(checked_feature_df.columns)),
        "quality_summary": {
            "feature_quality_report_list": list(feature_quality_report_list),
            "dropped_source_column_count": int(len(dropped_source_column_list)),
            "dropped_factor_count": int(len(dropped_factor_report_list)),
            "flipped_factor_count": int(len(flipped_factor_report_list)),
        },
        "dropped_source_feature_list": [
            {
                "column": str(record["column"]),
                "missing_count": int(record["missing_count"]),
                "missing_ratio": float(record["missing_ratio"]),
                "max_consecutive_missing_count": int(record["max_consecutive_missing_count"]),
                "max_consecutive_missing_start_date": record["max_consecutive_missing_start_date"],
                "max_consecutive_missing_end_date": record["max_consecutive_missing_end_date"],
                "drop_reason_list": list(record["drop_reason_list"]),
            }
            for record in list(feature_quality_report_list)
            if bool(record["dropped"])
        ],
        "dropped_feature_list": [
            {
                "source_column": str(record["source_column"]),
                "bound_source_column_list": [str(column) for column in list(record.get("bound_source_column_list", []))],
                "candidate_label": str(record["candidate_label"]),
                "output_column": str(record.get("output_column", f"{str(record['source_column'])}__{str(record['candidate_label'])}__zscore")),
                "raw_nan_ratio": float(record["raw_nan_ratio"]),
                "normalized_zero_ratio": float(record["normalized_zero_ratio"]),
                "drop_reason_list": list(record["drop_reason_list"]),
            }
            for record in list(dropped_factor_report_list)
        ],
        "flipped_feature_list": [
            {
                "original_column": str(record["original_column"]),
                "flipped_column": str(record["flipped_column"]),
                "train_spearman_ic_mean": float(record["train_spearman_ic_mean"]),
                "train_sample_fold_count": int(record["train_sample_fold_count"]),
                "positive_ic_count": int(record["positive_ic_count"]),
                "negative_ic_count": int(record["negative_ic_count"]),
            }
            for record in list(flipped_factor_report_list)
        ],
    }


def _print_feature_quality_report_list(feature_quality_report_list):
    print("基础特征缺失治理情况:")
    if len(feature_quality_report_list) == 0:
        print("无")
        return
    for feature_quality_report in feature_quality_report_list:
        print(
            "基础特征检查:",
            feature_quality_report["column"],
            f"缺失数={feature_quality_report['missing_count']}",
            f"缺失率={feature_quality_report['missing_ratio']:.4f}",
            f"最长连续缺失={feature_quality_report['max_consecutive_missing_count']}",
            f"已填补={int(feature_quality_report['filled_missing_count'])}",
            f"已抛弃={bool(feature_quality_report['dropped'])}",
        )
        if feature_quality_report["max_consecutive_missing_start_date"] is not None:
            print(
                "最长连续缺失区间:",
                f"{feature_quality_report['max_consecutive_missing_start_date']} -> {feature_quality_report['max_consecutive_missing_end_date']}",
            )
        if len(feature_quality_report["drop_reason_list"]) > 0:
            print("抛弃原因:", ",".join(feature_quality_report["drop_reason_list"]))


def _print_dropped_source_column_warning_list(feature_quality_report_list):
    dropped_feature_quality_report_list = [record for record in list(feature_quality_report_list) if bool(record["dropped"])]
    if len(dropped_feature_quality_report_list) == 0:
        return
    print("警告: 以下基础特征列已因缺失质量问题被抛弃")
    for feature_quality_report in dropped_feature_quality_report_list:
        print(
            "抛弃基础特征:",
            feature_quality_report["column"],
            f"缺失率={feature_quality_report['missing_ratio']:.4f}",
            f"最长连续缺失={feature_quality_report['max_consecutive_missing_count']}",
            f"原因={','.join(feature_quality_report['drop_reason_list'])}",
        )


def _print_dropped_factor_report_list(dropped_factor_report_list):
    print("删除因子明细:")
    if len(dropped_factor_report_list) == 0:
        print("无")
        return
    for dropped_factor_report in dropped_factor_report_list:
        print(
            "删除因子:",
            f"基础特征={dropped_factor_report['source_column']}",
            f"因子={dropped_factor_report['candidate_label']}__zscore",
            f"raw_nan_ratio={dropped_factor_report['raw_nan_ratio']:.4f}",
            f"normalized_zero_ratio={dropped_factor_report['normalized_zero_ratio']:.4f}",
            f"原因={','.join(dropped_factor_report['drop_reason_list'])}",
        )


def _print_flipped_factor_report_list(flipped_factor_report_list):
    print("翻转因子明细:")
    if len(flipped_factor_report_list) == 0:
        print("无")
        return
    for flipped_factor_report in flipped_factor_report_list:
        print(
            "翻转因子:",
            f"原列={flipped_factor_report['original_column']}",
            f"新列={flipped_factor_report['flipped_column']}",
            f"train_ic_mean={flipped_factor_report['train_spearman_ic_mean']:.4f}",
            f"negative_ic_count={flipped_factor_report['negative_ic_count']}",
            f"positive_ic_count={flipped_factor_report['positive_ic_count']}",
        )


def _print_feature_preprocess_summary(result, feature_quality_report_list, source_column_list, candidate_factor_list, dropped_factor_report_list, flipped_factor_report_list):
    # 流程0最终摘要集中打印关键产物、补缺、删因子和翻转结果，便于一次性审阅完整候选生成过程。
    print("特征预处理结果:")
    print("基金代码:", result["fund_code"])
    print("path_code:", result["path_code"])
    print("记录数:", result["record_count"])
    print("总列数:", result["column_count"])
    print("基础特征列数:", int(len(source_column_list)))
    print("候选因子数:", int(len(candidate_factor_list)))
    print("删除因子数:", result["dropped_factor_count"])
    print("翻转因子数:", result["flipped_factor_count"])
    print("原始输出:", result["raw_output_path"])
    print("特征输出:", result["summary_path"])
    print("元信息输出:", result["metadata_path"])
    _print_feature_quality_report_list(feature_quality_report_list=feature_quality_report_list)
    _print_dropped_factor_report_list(dropped_factor_report_list=dropped_factor_report_list)
    _print_flipped_factor_report_list(flipped_factor_report_list=flipped_factor_report_list)
    _print_dropped_source_column_warning_list(feature_quality_report_list=feature_quality_report_list)


def run_feature_preprocess_single_fund(config_override=None):
    config = build_tradition_config(config_override=config_override)
    fund_code = str(config["default_fund_code"]).zfill(6)
    code_type_dict = _resolve_feature_preprocess_code_type_dict(config=config)
    raw_output_path, metadata_output_path, path_code = allocate_stage_csv_json_output_path(
        output_dir=config["output_dir"],
        output_prefix="feature_preprocess",
        fund_code=fund_code,
    )
    _, resolved_raw_output_path = build_code_feature_table(
        config=config,
        code_type_dict=code_type_dict,
        output_path=raw_output_path,
        primary_code=fund_code,
        force_refresh=bool(config["force_refresh"]),
    )
    checked_feature_df, checked_output_path, feature_quality_report_list, dropped_source_column_list = _check_and_fill_wide_feature_table(
        raw_output_path=resolved_raw_output_path,
        code_type_dict=code_type_dict,
        primary_code=fund_code,
    )
    checked_feature_df, checked_output_path, source_column_list, candidate_factor_list, dropped_factor_report_list, factor_binding_record_list = _build_checked_factor_table(
        checked_output_path=checked_output_path,
        strategy_params=config["strategy_param_dict"]["multi_factor_score"],
        fund_code=fund_code,
        dropped_source_column_list=dropped_source_column_list,
    )
    # 翻转列生成沿用流程1训练集口径，只把稳定负向候选追加成新的 factor_feature 列。
    checked_feature_df, checked_output_path, flipped_factor_report_list = _append_flipped_factor_feature_columns(
        checked_feature_df=checked_feature_df,
        checked_output_path=checked_output_path,
        source_column_list=source_column_list,
        fund_code=fund_code,
        config=config,
    )
    # 流程0最终供后续阶段消费的整张特征表统一裁掉前段不稳定样本，保持全表列对齐。
    checked_feature_df = _trim_initial_rows(feature_df=checked_feature_df)
    _save_feature_table(feature_df=checked_feature_df, table_path=checked_output_path)
    metadata_output = _build_feature_preprocess_metadata(
        checked_feature_df=checked_feature_df,
        feature_path=checked_output_path,
        path_code=path_code,
        fund_code=fund_code,
        code_type_dict=code_type_dict,
        source_column_list=source_column_list,
        candidate_factor_list=candidate_factor_list,
        feature_quality_report_list=feature_quality_report_list,
        dropped_source_column_list=dropped_source_column_list,
        dropped_factor_report_list=dropped_factor_report_list,
        factor_binding_record_list=factor_binding_record_list,
        flipped_factor_report_list=flipped_factor_report_list,
    )
    payload = {
        "path_code": str(path_code),
        "feature_preprocess_output": metadata_output,
    }
    with Path(metadata_output_path).open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    result = {
        "fund_code": fund_code,
        "path_code": path_code,
        "raw_output_path": Path(resolved_raw_output_path),
        "summary_path": Path(checked_output_path),
        "metadata_path": Path(metadata_output_path),
        "record_count": int(len(checked_feature_df)),
        "column_count": int(len(checked_feature_df.columns)),
        "dropped_factor_count": int(len(dropped_factor_report_list)),
        "flipped_factor_count": int(len(flipped_factor_report_list)),
    }
    _print_feature_preprocess_summary(
        result=result,
        feature_quality_report_list=feature_quality_report_list,
        source_column_list=source_column_list,
        candidate_factor_list=candidate_factor_list,
        dropped_factor_report_list=dropped_factor_report_list,
        flipped_factor_report_list=flipped_factor_report_list,
    )
    return result
