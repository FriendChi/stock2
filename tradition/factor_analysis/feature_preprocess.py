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
from tradition.factor_library import build_raw_factor_series
from tradition.splitter import build_walk_forward_dev_fold_list

from .io import allocate_stage_csv_json_output_path

DEFAULT_START_DATE = "19700101"
DEFAULT_END_DATE = "20500101"
DEFAULT_OUTPUT_NAME = "feature_preprocess_code_feature_table.csv"
CHECKED_OUTPUT_SUFFIX = "_checked"
FEATURE_CACHE_PREFIX = "temp_price_cache"
INCREMENTAL_FETCH_BUFFER_DAYS = 15
OPEN_FUND_PERIOD_BUFFER_DAYS = 15
MAX_FILL_MISSING_ROW_COUNT = 4
MAX_RAW_NAN_RATIO = 0.2
MAX_NORMALIZED_ZERO_RATIO = 0.8


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
    index_df = ak_module.index_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
    )
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
    index_df = ak_module.stock_zh_index_daily_em(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )
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
    index_df = ak_module.stock_zh_index_daily(symbol=symbol)
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
    index_df = ak_module.stock_zh_index_daily_tx(symbol=symbol)
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
    for prefixed_symbol in (f"sh{code}", f"sz{code}", f"csi{code}"):
        for fetcher in (_fetch_prefixed_index_feature_df, _fetch_sina_index_feature_df, _fetch_tx_index_feature_df):
            try:
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
    # 检查后文件与原始文件并存，命名上保留稳定可推导关系。
    raw_output_path = Path(raw_output_path)
    return raw_output_path.with_name(f"{raw_output_path.stem}{CHECKED_OUTPUT_SUFFIX}{raw_output_path.suffix}")


def _load_saved_wide_feature_table(raw_output_path):
    # 检查阶段直接读取刚保存的原始文件，确保后续处理基于落盘结果而不是内存态对象。
    raw_output_path = Path(raw_output_path)
    if not raw_output_path.exists():
        raise FileNotFoundError(f"原始宽表文件不存在: {raw_output_path}")
    return pd.read_csv(raw_output_path)


def _validate_wide_feature_table_structure(wide_feature_df, code_type_dict, primary_code):
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


def _resolve_group_missing_row_position_list(wide_feature_df, feature_column_list):
    # 缺失行按“该代码整组字段全部为空”定义，而不是按单列缺失定义。
    if len(feature_column_list) == 0:
        return []
    missing_mask = ~wide_feature_df[feature_column_list].notna().any(axis=1)
    return wide_feature_df.index[missing_mask].tolist()


def _is_non_consecutive_position_list(position_list):
    # 只有离散零散缺口才允许用局部均值修补，连续缺口直接保留为问题。
    if len(position_list) <= 1:
        return True
    return all(int(position_list[idx]) - int(position_list[idx - 1]) > 1 for idx in range(1, len(position_list)))


def _fill_group_missing_rows_by_neighbor_average(wide_feature_df, feature_column_list, missing_row_position_list):
    # 满足条件的缺失行按列取前后相邻有效值平均，避免跨代码或跨列混算。
    filled_df = wide_feature_df.copy()
    filled_date_list = []
    unresolved_date_list = []
    for position in missing_row_position_list:
        if position <= 0 or position >= len(filled_df) - 1:
            unresolved_date_list.append(str(filled_df.loc[position, "date"]))
            continue
        previous_row = filled_df.loc[position - 1, feature_column_list]
        next_row = filled_df.loc[position + 1, feature_column_list]
        if bool(previous_row.isna().any()) or bool(next_row.isna().any()):
            unresolved_date_list.append(str(filled_df.loc[position, "date"]))
            continue
        filled_df.loc[position, feature_column_list] = (previous_row.astype(float) + next_row.astype(float)) / 2.0
        filled_date_list.append(str(filled_df.loc[position, "date"]))
    return filled_df, filled_date_list, unresolved_date_list


def _check_and_fill_wide_feature_table(raw_output_path, code_type_dict, primary_code):
    # 检查与补缺都基于原始文件执行，补后的结果另存为 checked 文件。
    raw_feature_df = _load_saved_wide_feature_table(raw_output_path=raw_output_path)
    _validate_wide_feature_table_structure(
        wide_feature_df=raw_feature_df,
        code_type_dict=code_type_dict,
        primary_code=primary_code,
    )
    checked_feature_df = raw_feature_df.copy()
    code_report_list = []
    for code in dict(code_type_dict).keys():
        normalized_code = str(code).zfill(6)
        feature_column_list = [column for column in checked_feature_df.columns if column.startswith(f"{normalized_code}__")]
        missing_row_position_list = _resolve_group_missing_row_position_list(
            wide_feature_df=checked_feature_df,
            feature_column_list=feature_column_list,
        )
        missing_date_list = [str(checked_feature_df.loc[position, "date"]) for position in missing_row_position_list]
        filled_date_list = []
        unresolved_date_list = list(missing_date_list)
        if (
            0 < len(missing_row_position_list) <= int(MAX_FILL_MISSING_ROW_COUNT)
            and _is_non_consecutive_position_list(missing_row_position_list)
        ):
            checked_feature_df, filled_date_list, unresolved_date_list = _fill_group_missing_rows_by_neighbor_average(
                wide_feature_df=checked_feature_df,
                feature_column_list=feature_column_list,
                missing_row_position_list=missing_row_position_list,
            )
        code_report_list.append(
            {
                "code": normalized_code,
                "missing_row_count": int(len(missing_row_position_list)),
                "missing_date_list": missing_date_list,
                "filled_row_count": int(len(filled_date_list)),
                "filled_date_list": filled_date_list,
                "remaining_missing_row_count": int(len(unresolved_date_list)),
                "remaining_missing_date_list": unresolved_date_list,
            }
        )
    _validate_wide_feature_table_structure(
        wide_feature_df=checked_feature_df,
        code_type_dict=code_type_dict,
        primary_code=primary_code,
    )
    checked_output_path = _build_checked_output_path(raw_output_path=raw_output_path)
    checked_feature_df.to_csv(checked_output_path, index=False)
    return checked_feature_df, checked_output_path, code_report_list


def _resolve_factor_source_column_list(checked_feature_df):
    # 因子计算只消费 checked 宽表中的基础特征列，不把 date 当作输入特征。
    checked_feature_df = pd.DataFrame(checked_feature_df, copy=True)
    return [column for column in checked_feature_df.columns if column != "date"]


def _build_factor_candidate_config(strategy_params):
    # 因子候选展开复用 selection 流程的参数搜索空间，保证流程0与筛选逻辑同源。
    resolved_strategy_params = dict(strategy_params)
    candidate_factor_name_list = [str(factor_name) for factor_name in resolved_strategy_params["enabled_factor_list"]]
    candidate_factor_list = build_factor_candidate_list(
        candidate_factor_name_list=candidate_factor_name_list,
        strategy_params=resolved_strategy_params,
    )
    return resolved_strategy_params, candidate_factor_list


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
        normalized_zero_ratio = float((normalized_factor_series == 0.0).mean())
        if raw_nan_ratio > float(MAX_RAW_NAN_RATIO) or normalized_zero_ratio > float(MAX_NORMALIZED_ZERO_RATIO):
            dropped_factor_report_list.append(
                {
                    "candidate_label": candidate_label,
                    "raw_nan_ratio": raw_nan_ratio,
                    "normalized_zero_ratio": normalized_zero_ratio,
                }
            )
            continue
        factor_series_dict[f"{candidate_label}__zscore"] = normalized_factor_series
    factor_df = pd.DataFrame(factor_series_dict, index=feature_series.index)
    return factor_df.fillna(0.0), dropped_factor_report_list


def _build_checked_factor_table(checked_output_path, strategy_params):
    # checked 表在补缺完成后直接扩展标准化原始特征和标准化因子，不再单独落因子文件。
    checked_feature_df = _load_saved_wide_feature_table(raw_output_path=checked_output_path)
    resolved_strategy_params, candidate_factor_list = _build_factor_candidate_config(strategy_params=strategy_params)
    source_column_list = _resolve_factor_source_column_list(checked_feature_df=checked_feature_df)
    extended_checked_df = checked_feature_df.copy()
    total_source_count = int(len(source_column_list))
    expected_added_column_count = int(total_source_count * (1 + len(candidate_factor_list)))
    accumulated_added_column_count = 0
    dropped_factor_report_list = []
    for source_idx, source_column in enumerate(source_column_list, start=1):
        single_feature_factor_df, source_dropped_factor_report_list = _build_single_feature_factor_df(
            feature_series=checked_feature_df[source_column],
            candidate_factor_list=candidate_factor_list,
            strategy_params=resolved_strategy_params,
        )
        single_feature_factor_df.columns = [f"{source_column}__{column}" for column in single_feature_factor_df.columns]
        single_feature_factor_df.index = extended_checked_df.index
        extended_checked_df = pd.concat([extended_checked_df, single_feature_factor_df], axis=1)
        accumulated_added_column_count = accumulated_added_column_count + int(len(single_feature_factor_df.columns))
        for dropped_factor_report in source_dropped_factor_report_list:
            dropped_factor_report = dict(dropped_factor_report)
            dropped_factor_report["source_column"] = source_column
            dropped_factor_report_list.append(dropped_factor_report)
            print(
                "删除因子:",
                f"基础特征={source_column}",
                f"因子={dropped_factor_report['candidate_label']}__zscore",
                f"raw_nan_ratio={dropped_factor_report['raw_nan_ratio']:.4f}",
                f"normalized_zero_ratio={dropped_factor_report['normalized_zero_ratio']:.4f}",
                "原因=超过阈值",
            )
        print(
            f"因子进度: {source_idx}/{total_source_count}",
            f"基础特征={source_column}",
            f"已生成新增列={accumulated_added_column_count}/{expected_added_column_count}",
        )
    extended_checked_df.to_csv(checked_output_path, index=False)
    return extended_checked_df, checked_output_path, source_column_list, candidate_factor_list, dropped_factor_report_list


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
    # 翻转列只基于当前已生成好的候选因子快照执行一次，避免本次流程内递归地对翻转列再次翻转。
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
    for flipped_factor_report in flipped_factor_report_list:
        original_column = str(flipped_factor_report["original_column"])
        flipped_column = str(flipped_factor_report["flipped_column"])
        checked_feature_df[flipped_column] = -pd.Series(checked_feature_df[original_column], copy=True).astype(float)
    checked_feature_df.to_csv(checked_output_path, index=False)
    return checked_feature_df, checked_output_path, flipped_factor_report_list


def _build_feature_preprocess_metadata(
    checked_feature_df,
    csv_path,
    path_code,
    fund_code,
    code_type_dict,
    source_column_list,
    candidate_factor_list,
    code_report_list,
    dropped_factor_report_list,
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
        "csv_path": str(Path(csv_path).resolve()),
        "target_price_column": target_price_column,
        "target_nav_column": target_nav_column,
        "feature_column_list": feature_column_list,
        "raw_feature_column_list": raw_feature_column_list,
        "raw_feature_zscore_column_list": raw_feature_zscore_column_list,
        "factor_feature_column_list": factor_feature_column_list,
        "linked_code_type_dict": {str(code).zfill(6): str(code_type) for code, code_type in dict(code_type_dict).items()},
        "candidate_factor_count": int(len(candidate_factor_list)),
        "row_count": int(len(checked_feature_df)),
        "column_count": int(len(checked_feature_df.columns)),
        "quality_summary": {
            "code_report_list": list(code_report_list),
            "dropped_factor_count": int(len(dropped_factor_report_list)),
            "flipped_factor_count": int(len(flipped_factor_report_list)),
        },
        "dropped_feature_list": [
            {
                "source_column": str(record["source_column"]),
                "candidate_label": str(record["candidate_label"]),
                "output_column": f"{str(record['source_column'])}__{str(record['candidate_label'])}__zscore",
                "raw_nan_ratio": float(record["raw_nan_ratio"]),
                "normalized_zero_ratio": float(record["normalized_zero_ratio"]),
                "drop_reason": "threshold_exceeded",
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


def _print_code_report_list(code_report_list):
    print("缺失值修补情况:")
    if len(code_report_list) == 0:
        print("无")
        return
    for code_report in code_report_list:
        print(
            "代码检查:",
            code_report["code"],
            f"缺失行数={code_report['missing_row_count']}",
            f"已填补={code_report['filled_row_count']}",
            f"剩余缺失={code_report['remaining_missing_row_count']}",
        )
        if len(code_report["filled_date_list"]) > 0:
            print("已填补日期:", ",".join(code_report["filled_date_list"]))
        if len(code_report["remaining_missing_date_list"]) > 0:
            print("剩余缺失日期:", ",".join(code_report["remaining_missing_date_list"]))


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
            "原因=超过阈值",
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


def _print_feature_preprocess_summary(result, code_report_list, source_column_list, candidate_factor_list, dropped_factor_report_list, flipped_factor_report_list):
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
    _print_code_report_list(code_report_list=code_report_list)
    _print_dropped_factor_report_list(dropped_factor_report_list=dropped_factor_report_list)
    _print_flipped_factor_report_list(flipped_factor_report_list=flipped_factor_report_list)


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
    checked_feature_df, checked_output_path, code_report_list = _check_and_fill_wide_feature_table(
        raw_output_path=resolved_raw_output_path,
        code_type_dict=code_type_dict,
        primary_code=fund_code,
    )
    checked_feature_df, checked_output_path, source_column_list, candidate_factor_list, dropped_factor_report_list = _build_checked_factor_table(
        checked_output_path=checked_output_path,
        strategy_params=config["strategy_param_dict"]["multi_factor_score"],
    )
    # 翻转列生成沿用流程1训练集口径，只把稳定负向候选追加成新的 factor_feature 列。
    checked_feature_df, checked_output_path, flipped_factor_report_list = _append_flipped_factor_feature_columns(
        checked_feature_df=checked_feature_df,
        checked_output_path=checked_output_path,
        source_column_list=source_column_list,
        fund_code=fund_code,
        config=config,
    )
    metadata_output = _build_feature_preprocess_metadata(
        checked_feature_df=checked_feature_df,
        csv_path=checked_output_path,
        path_code=path_code,
        fund_code=fund_code,
        code_type_dict=code_type_dict,
        source_column_list=source_column_list,
        candidate_factor_list=candidate_factor_list,
        code_report_list=code_report_list,
        dropped_factor_report_list=dropped_factor_report_list,
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
        code_report_list=code_report_list,
        source_column_list=source_column_list,
        candidate_factor_list=candidate_factor_list,
        dropped_factor_report_list=dropped_factor_report_list,
        flipped_factor_report_list=flipped_factor_report_list,
    )
    return result
