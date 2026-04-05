import pandas as pd

from tradition.factor_library import (
    FACTOR_POOL_DICT,
    build_raw_factor_series,
    calculate_breakout_strength,
    calculate_factor_momentum,
    calculate_ma_slope,
    calculate_ma_trend_state,
    calculate_price_position,
    calculate_drawdown,
    calculate_donchian_breakout,
    calculate_risk_adjusted_momentum,
    calculate_sharpe_like_trend,
    calculate_trend_r2,
    calculate_trend_residual,
    calculate_trend_tvalue,
    calculate_volatility,
    resolve_factor_name_list_by_group,
)


def rolling_zscore(series, window):
    # 用滚动均值和标准差做时序标准化，避免直接使用全样本统计带来未来信息污染。
    series = pd.Series(series, copy=True).astype(float)
    rolling_mean = series.rolling(int(window)).mean()
    rolling_std = series.rolling(int(window)).std(ddof=0)
    zscore = (series - rolling_mean) / rolling_std.replace(0.0, float("nan"))
    return zscore.fillna(0.0).astype(float)


def validate_multi_factor_config(strategy_params, enabled_factor_list=None, require_factor_weight_dict=False):
    # 因子配置校验集中在因子层，尽早拦截未知因子、未知参数和缺失权重配置。
    factor_weight_dict = dict(strategy_params.get("factor_weight_dict", {}))
    if enabled_factor_list is None:
        configured_enabled_factor_list = strategy_params.get("enabled_factor_list")
        if configured_enabled_factor_list is None:
            configured_enabled_factor_list = list(factor_weight_dict.keys())
        enabled_factor_list = [str(factor_name) for factor_name in configured_enabled_factor_list]

    unknown_weight_factor_list = [factor_name for factor_name in factor_weight_dict if factor_name not in FACTOR_POOL_DICT]
    if len(unknown_weight_factor_list) > 0:
        raise ValueError(f"factor_weight_dict 中存在未定义因子: {unknown_weight_factor_list}")

    configured_factor_param_dict = dict(strategy_params.get("factor_param_dict", {}))
    unknown_param_factor_list = [factor_name for factor_name in configured_factor_param_dict if factor_name not in FACTOR_POOL_DICT]
    if len(unknown_param_factor_list) > 0:
        raise ValueError(f"factor_param_dict 中存在未定义因子: {unknown_param_factor_list}")
    for factor_name, param_dict in configured_factor_param_dict.items():
        factor_param_spec = dict(FACTOR_POOL_DICT[str(factor_name)]["param_spec"])
        unknown_param_name_list = [param_name for param_name in dict(param_dict) if str(param_name) not in factor_param_spec]
        if len(unknown_param_name_list) > 0:
            raise ValueError(f"{factor_name} 存在未定义参数: {unknown_param_name_list}")

    if require_factor_weight_dict:
        missing_weight_factor_list = [factor_name for factor_name in enabled_factor_list if factor_name not in factor_weight_dict]
        if len(missing_weight_factor_list) > 0:
            raise ValueError(f"enabled_factor_list 中存在缺失权重的因子: {missing_weight_factor_list}")

def resolve_factor_param_dict(strategy_params):
    # 因子参数统一按因子名做一层字典隔离，避免外层扁平参数名不断膨胀。
    enabled_factor_list = resolve_enabled_factor_list(strategy_params=strategy_params)
    validate_multi_factor_config(
        strategy_params=strategy_params,
        enabled_factor_list=enabled_factor_list,
    )
    configured_factor_param_dict = dict(strategy_params.get("factor_param_dict", {}))
    resolved_factor_param_dict = {}
    for factor_name in enabled_factor_list:
        resolved_param_dict = {
            str(param_name): spec["default"]
            for param_name, spec in dict(FACTOR_POOL_DICT[factor_name]["param_spec"]).items()
        }
        override_param_dict = dict(configured_factor_param_dict.get(factor_name, {}))
        for param_name, value in override_param_dict.items():
            resolved_param_dict[str(param_name)] = value
        resolved_factor_param_dict[factor_name] = resolved_param_dict
    return resolved_factor_param_dict


def normalize_factor_series(raw_factor_series, factor_name, score_window):
    # 标准化和同向化在统一入口处理，保证新增因子复用同一套评分口径。
    normalized_factor = rolling_zscore(raw_factor_series, window=int(score_window))
    if factor_name in {"volatility", "drawdown", "trend_residual"}:
        return -normalized_factor
    return normalized_factor


def resolve_enabled_factor_list(strategy_params):
    # 因子启用列表优先由策略配置决定，未显式配置时退回到权重表中的因子名。
    enabled_factor_list = strategy_params.get("enabled_factor_list")
    if enabled_factor_list is None:
        enabled_factor_list = list(dict(strategy_params["factor_weight_dict"]).keys())
    resolved_factor_list = [str(factor_name) for factor_name in enabled_factor_list]
    if len(resolved_factor_list) == 0:
        raise ValueError("enabled_factor_list 为空，无法构建多因子评分。")
    unknown_factor_list = [factor_name for factor_name in resolved_factor_list if factor_name not in FACTOR_POOL_DICT]
    if len(unknown_factor_list) > 0:
        raise ValueError(f"存在未定义因子: {unknown_factor_list}")
    return resolved_factor_list


def build_factor_search_param_spec_dict(strategy_params):
    # 启用因子的全部可搜索参数统一从因子库收集，避免配置层重复维护参数白名单。
    enabled_factor_list = resolve_enabled_factor_list(strategy_params=strategy_params)
    validate_multi_factor_config(
        strategy_params=strategy_params,
        enabled_factor_list=enabled_factor_list,
    )
    search_param_spec_dict = {}
    for factor_name in enabled_factor_list:
        param_spec = dict(FACTOR_POOL_DICT[factor_name]["param_spec"])
        search_param_spec_dict[factor_name] = {}
        for param_name, spec in param_spec.items():
            search_space = spec.get("search_space")
            if search_space is None:
                continue
            search_param_spec_dict[factor_name][param_name] = search_space
    return search_param_spec_dict


def build_factor_table(price_series, strategy_params):
    # 因子表按启用因子列表构建，后续新增因子只需扩充因子库而不改主流程。
    score_window = int(strategy_params["score_window"])
    enabled_factor_list = resolve_enabled_factor_list(strategy_params=strategy_params)
    factor_param_dict = resolve_factor_param_dict(strategy_params=strategy_params)

    factor_df = pd.DataFrame(index=price_series.index)
    for factor_name in enabled_factor_list:
        raw_col = f"{factor_name}_raw"
        raw_factor_series = build_raw_factor_series(
            price_series=price_series,
            factor_name=factor_name,
            factor_param_dict=factor_param_dict,
        )
        factor_df[raw_col] = raw_factor_series
        factor_df[factor_name] = normalize_factor_series(
            raw_factor_series=raw_factor_series,
            factor_name=factor_name,
            score_window=score_window,
        )
    return factor_df.fillna(0.0)


def build_multi_factor_score(price_series, strategy_params):
    # 综合分数只消费启用因子的标准化列，权重缺失时直接报错，避免策略配置静默失效。
    validate_multi_factor_config(
        strategy_params=strategy_params,
        require_factor_weight_dict=True,
    )
    factor_df = build_factor_table(price_series=price_series, strategy_params=strategy_params)
    factor_weight_dict = dict(strategy_params["factor_weight_dict"])
    enabled_factor_list = resolve_enabled_factor_list(strategy_params=strategy_params)
    score_series = pd.Series(0.0, index=price_series.index, dtype=float)
    for factor_name in enabled_factor_list:
        if factor_name not in factor_weight_dict:
            raise ValueError(f"未找到因子权重: {factor_name}")
        score_series = score_series + factor_df[factor_name] * float(factor_weight_dict[factor_name])
    score_series.name = "multi_factor_score"
    return factor_df, score_series


def build_single_factor_series(price_series, factor_name, strategy_params, factor_param_override=None):
    # 因子筛选场景既支持直接复用默认参数，也支持显式传入参数化候选覆盖项。
    resolved_strategy_params = dict(strategy_params)
    resolved_strategy_params["enabled_factor_list"] = [str(factor_name)]
    if factor_param_override is not None:
        resolved_strategy_params["factor_param_dict"] = {
            str(factor_name): dict(factor_param_override)
        }
    factor_param_dict = resolve_factor_param_dict(strategy_params=resolved_strategy_params)
    raw_factor_series = build_raw_factor_series(
        price_series=price_series,
        factor_name=str(factor_name),
        factor_param_dict=factor_param_dict,
    )
    normalized_factor_series = normalize_factor_series(
        raw_factor_series=raw_factor_series,
        factor_name=str(factor_name),
        score_window=int(resolved_strategy_params["score_window"]),
    )
    normalized_factor_series.name = str(factor_name)
    return normalized_factor_series
