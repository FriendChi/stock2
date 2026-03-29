import argparse
from datetime import datetime

import pandas as pd

from tradition.config import build_tradition_config
from tradition.data_adapter import adapt_to_price_series
from tradition.data_fetcher import fetch_fund_data_with_cache, fetch_treasury_yield_with_cache
from tradition.data_loader import filter_single_fund, normalize_fund_data
from tradition.metrics import compute_return_metrics, save_equity_curve_plot
from tradition.optimizer import optimize_strategy_params
from tradition.splitter import split_time_series_by_ratio
from tradition.strategies import generate_signals


ALL_STRATEGY_NAME_LIST = ["buy_and_hold", "ma_cross", "momentum", "multi_factor_score"]


def load_vectorbt_module():
    # 延迟导入 vectorbt，减少模块导入阶段对环境完整性的强依赖。
    import vectorbt as vbt

    return vbt


def build_cli_override(args):
    # 命令行覆盖项只收敛到当前入门版需要的参数，避免入口层一次性暴露过多实验开关。
    override = {}
    if args.fund_code is not None:
        override["default_fund_code"] = str(args.fund_code).zfill(6)
    if args.fund_codes is not None:
        override["fund_code_list"] = [str(code).strip().zfill(6) for code in str(args.fund_codes).split(",") if str(code).strip()]
    if args.strategy_name is not None:
        override["default_strategy_name"] = str(args.strategy_name).lower()
    if args.force_refresh:
        override["force_refresh"] = True
    if args.init_cash is not None:
        override["init_cash"] = float(args.init_cash)
    if args.fees is not None:
        override["fees"] = float(args.fees)
    if args.batch_run:
        override["batch_run"] = True
    if args.compare_all:
        override["compare_all"] = True
    if args.optimize:
        override["optimize"] = True

    strategy_param_override = {}
    if args.ma_fast is not None:
        strategy_param_override.setdefault("ma_cross", {})["fast"] = int(args.ma_fast)
    if args.ma_slow is not None:
        strategy_param_override.setdefault("ma_cross", {})["slow"] = int(args.ma_slow)
    if args.momentum_window is not None:
        strategy_param_override.setdefault("momentum", {})["window"] = int(args.momentum_window)
    if len(strategy_param_override) > 0:
        override["strategy_param_dict"] = strategy_param_override

    optimization_override = {}
    if args.n_trials is not None:
        optimization_override["n_trials"] = int(args.n_trials)
    if len(optimization_override) > 0:
        override["optimization_config"] = optimization_override
    return override


def build_arg_parser():
    # 第一阶段只开放单基金回测最需要的命令行参数，保持入口简单且可直接试验。
    parser = argparse.ArgumentParser(description="运行 tradition 时序择时入门版单基金或批量回测")
    parser.add_argument("--fund-code", dest="fund_code", help="目标基金代码，例如 007301")
    parser.add_argument("--fund-codes", dest="fund_codes", help="批量模式下使用的基金代码列表，逗号分隔")
    parser.add_argument(
        "--strategy-name",
        dest="strategy_name",
        choices=ALL_STRATEGY_NAME_LIST,
        help="策略名称",
    )
    parser.add_argument("--batch-run", action="store_true", help="按基金池批量运行当前策略并输出汇总表")
    parser.add_argument("--compare-all", action="store_true", help="对单只基金运行全部策略并输出对比表")
    parser.add_argument("--optimize", action="store_true", help="对单只基金执行训练/验证/测试切分后的 Optuna 参数优化")
    parser.add_argument("--force-refresh", action="store_true", help="忽略当天缓存并重新拉取 AkShare 数据")
    parser.add_argument("--init-cash", dest="init_cash", type=float, help="初始资金")
    parser.add_argument("--fees", dest="fees", type=float, help="手续费率")
    parser.add_argument("--ma-fast", dest="ma_fast", type=int, help="ma_cross 策略短均线窗口")
    parser.add_argument("--ma-slow", dest="ma_slow", type=int, help="ma_cross 策略长均线窗口")
    parser.add_argument("--momentum-window", dest="momentum_window", type=int, help="momentum 策略动量窗口")
    parser.add_argument("--n-trials", dest="n_trials", type=int, help="Optuna trial 数量")
    return parser


def merge_strategy_params(default_param_dict, override_param_dict=None):
    # 入口层只对显式传入的策略参数做浅覆盖，保持默认配置来源单一。
    merged = dict(default_param_dict)
    if override_param_dict is None:
        return merged
    for strategy_name, param_dict in override_param_dict.items():
        current_params = dict(merged.get(strategy_name, {}))
        current_params.update(param_dict)
        merged[strategy_name] = current_params
    return merged


def merge_optimization_config(default_optimization_config, override_optimization_config=None):
    # 优化配置独立合并，避免把 Optuna 细节散落到 runner 其余逻辑。
    merged = dict(default_optimization_config)
    if override_optimization_config is None:
        return merged
    merged.update(override_optimization_config)
    return merged


def extract_trade_count(portfolio):
    # 统一把 vectorbt 的交易统计转成标量，避免不同返回类型污染结果结构。
    trade_count = portfolio.trades.count()
    if hasattr(trade_count, "item"):
        return int(trade_count.item())
    return int(trade_count)



def load_rf_series_for_price_series(config, price_series):
    # 无风险利率在评估层按基金日期区间拉取并对齐，避免影响策略信号与回测执行逻辑。
    rf_config = config.get("rf_config")
    if not isinstance(rf_config, dict) or not bool(rf_config.get("enabled", False)):
        return None
    price_index = pd.DatetimeIndex(pd.Series(price_series).dropna().index)
    if len(price_index) == 0:
        return None
    rf_df = fetch_treasury_yield_with_cache(
        cache_dir=config["data_dir"],
        start_date=price_index.min().strftime("%Y-%m-%d"),
        end_date=price_index.max().strftime("%Y-%m-%d"),
        force_refresh=bool(config.get("force_refresh", False)),
        cache_prefix=str(rf_config["cache_prefix"]),
        curve_name=str(rf_config["curve_name"]),
        tenor=str(rf_config["tenor"]),
    )
    return pd.Series(rf_df["annual_rf"].values, index=pd.to_datetime(rf_df["date"]), dtype=float)


def build_summary_record(result):
    # 汇总表固定展开核心指标字段，便于跨基金或跨策略横向比较。
    return {
        "fund_code": result["fund_code"],
        "strategy_name": result["strategy_name"],
        "sample_start": result["sample_start"],
        "sample_end": result["sample_end"],
        "data_mode": result["data_mode"],
        "trade_count": result["trade_count"],
        "cumulative_return": result["stats"]["cumulative_return"],
        "annual_return": result["stats"]["annual_return"],
        "annual_volatility": result["stats"]["annual_volatility"],
        "sharpe": result["stats"]["sharpe"],
        "max_drawdown": result["stats"]["max_drawdown"],
        "output_path": str(result["output_path"]),
    }


def build_summary_table(result_list):
    # 汇总表按 Sharpe 和年化收益降序排序，优先展示风险收益比更好的结果。
    if len(result_list) == 0:
        raise ValueError("result_list 为空，无法构建汇总表。")
    summary_df = pd.DataFrame([build_summary_record(result) for result in result_list])
    summary_df = summary_df.sort_values(["sharpe", "annual_return"], ascending=False).reset_index(drop=True)
    return summary_df


def save_summary_table(summary_df, output_dir, summary_name):
    # 批量模式和单基金多策略模式都用统一 CSV 输出逻辑，便于后续继续分析与比较。
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    output_path = output_dir / f"summary_{summary_name}_{date_str}.csv"
    summary_df.to_csv(output_path, index=False)
    return output_path


def print_summary_table(summary_df, summary_path, title):
    # 终端只打印最关键列，避免汇总模式输出过长难以阅读。
    print(title)
    printable_df = summary_df[["fund_code", "strategy_name", "annual_return", "sharpe", "max_drawdown", "trade_count"]].copy()
    print(printable_df.to_string(index=False))
    print("汇总输出:", summary_path)


def resolve_fund_code_list(config):
    # 批量模式优先使用显式传入的基金列表，否则退回配置中的全部基金代码。
    fund_code_list = config.get("fund_code_list")
    if fund_code_list is not None:
        return [str(code).zfill(6) for code in fund_code_list]
    return sorted([str(code).zfill(6) for code in config["code_dict"].keys()])


def resolve_optimization_strategy_name(config):
    # 优化默认聚焦 multi_factor_score，但允许用户通过 strategy_name 显式覆盖。
    default_strategy_name = str(config["default_strategy_name"]).lower()
    if default_strategy_name != "buy_and_hold":
        return default_strategy_name
    optimization_config = config["optimization_config"]
    return str(optimization_config["default_target_strategy_name"]).lower()



def build_param_signature(params):
    # 候选去重需要支持嵌套字典参数，统一递归转成可哈希签名。
    if isinstance(params, dict):
        return tuple((key, build_param_signature(value)) for key, value in sorted(params.items(), key=lambda item: item[0]))
    if isinstance(params, list):
        return tuple(build_param_signature(item) for item in params)
    return params


def execute_price_series_strategy(price_series, config, strategy_name, strategy_params=None):
    # 执行层只负责在完整价格序列上生成信号与权益曲线，评估区间切片交给上层处理。
    price_series = pd.Series(price_series, copy=True).astype(float)
    entries, exits, resolved_params = generate_signals(
        price_series=price_series,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
    )
    vbt = load_vectorbt_module()
    portfolio = vbt.Portfolio.from_signals(
        close=price_series,
        entries=entries,
        exits=exits,
        init_cash=float(config["init_cash"]),
        fees=float(config["fees"]),
        slippage=0.0,
        freq="1D",
    )
    return {
        "price_series": price_series,
        "entries": entries.astype(bool),
        "exits": exits.astype(bool),
        "resolved_params": resolved_params,
        "portfolio": portfolio,
        "equity_curve": pd.Series(portfolio.value(), dtype=float).dropna(),
    }


def extract_segment_equity_curves(equity_curve, segment_index):
    # 分段评估时保留目标区间上一日权益作为 warm-up，避免首日收益被切片边界吞掉。
    full_equity_curve = pd.Series(equity_curve, dtype=float).dropna()
    segment_index = pd.Index(segment_index)
    segment_equity_curve = full_equity_curve.loc[segment_index].copy()
    if segment_equity_curve.empty:
        raise ValueError("segment_index 对应的权益曲线为空，无法评估分段结果。")

    metric_equity_curve = segment_equity_curve.copy()
    first_loc = full_equity_curve.index.get_loc(segment_equity_curve.index[0])
    if isinstance(first_loc, int) and first_loc > 0:
        metric_equity_curve = pd.concat([full_equity_curve.iloc[[first_loc - 1]], segment_equity_curve])
    return metric_equity_curve, segment_equity_curve


def build_result_from_execution(
    execution_result,
    fund_code,
    strategy_name,
    data_mode="nav_price_series",
    save_plot=False,
    output_dir=None,
    output_name=None,
    title=None,
    print_summary=False,
    segment_index=None,
    rf_series=None,
):
    # 同一份完整执行结果可以按全样本或子区间复用，避免验证集和测试集重复冷启动回测。
    price_series = execution_result["price_series"]
    equity_curve = execution_result["equity_curve"]
    entries = execution_result["entries"]

    sample_index = price_series.index if segment_index is None else pd.Index(segment_index)
    metric_equity_curve = equity_curve
    display_equity_curve = equity_curve
    if segment_index is not None:
        metric_equity_curve, display_equity_curve = extract_segment_equity_curves(
            equity_curve=equity_curve,
            segment_index=sample_index,
        )
    metric_dict = compute_return_metrics(equity_curve=metric_equity_curve, rf_series=rf_series)

    output_path = None
    if save_plot:
        if output_dir is None:
            raise ValueError("save_plot=True 时必须提供 output_dir。")
        date_str = datetime.today().strftime("%Y-%m-%d")
        if output_name is None:
            output_name = f"{fund_code}_{strategy_name}_{date_str}.png"
        output_path = output_dir / output_name
        save_equity_curve_plot(
            equity_curve=display_equity_curve,
            output_path=output_path,
            title=title or f"{fund_code} {strategy_name}",
        )

    trade_count = extract_trade_count(portfolio=execution_result["portfolio"])
    if segment_index is not None:
        trade_count = int(entries.reindex(sample_index, fill_value=False).sum())

    result = {
        "fund_code": str(fund_code).zfill(6),
        "strategy_name": str(strategy_name).lower(),
        "strategy_params": execution_result["resolved_params"],
        "stats": metric_dict,
        "equity_curve": display_equity_curve,
        "output_path": output_path,
        "data_mode": data_mode,
        "sample_start": sample_index.min(),
        "sample_end": sample_index.max(),
        "trade_count": trade_count,
    }
    if print_summary:
        print_run_summary(result=result)
    return result


def run_single_price_series_strategy(
    price_series,
    config,
    fund_code,
    strategy_name,
    strategy_params=None,
    print_summary=True,
    save_plot=True,
    data_mode="nav_price_series",
    output_name=None,
):
    # 单段价格序列普通回测仍复用完整执行结果，但默认按全样本输出指标与图像。
    execution_result = execute_price_series_strategy(
        price_series=price_series,
        config=config,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
    )
    rf_series = load_rf_series_for_price_series(config=config, price_series=price_series)
    return build_result_from_execution(
        execution_result=execution_result,
        fund_code=fund_code,
        strategy_name=strategy_name,
        data_mode=data_mode,
        save_plot=save_plot,
        output_dir=config["output_dir"],
        output_name=output_name,
        title=f"{fund_code} {strategy_name}",
        print_summary=print_summary,
        rf_series=rf_series,
    )


def run_single_fund_strategy_from_data(normalized_data, config, fund_code, print_summary=True):
    # 复用同一份标准化数据执行单基金回测，避免不同模式对同一天缓存重复读取与解析。
    fund_code = str(fund_code).zfill(6)
    fund_df = filter_single_fund(normalized_data, fund_code=fund_code)
    price_series, data_mode = adapt_to_price_series(fund_df=fund_df)
    strategy_name = str(config["default_strategy_name"]).lower()
    return run_single_price_series_strategy(
        price_series=price_series,
        config=config,
        fund_code=fund_code,
        strategy_name=strategy_name,
        strategy_params=config["strategy_param_dict"].get(strategy_name),
        print_summary=print_summary,
        save_plot=True,
        data_mode=data_mode,
    )


def run_single_fund_strategy(config_override=None):
    # 单基金模式在数据准备后复用通用执行函数，保持与其他模式同一回测口径。
    config = build_tradition_config(config_override=config_override)
    raw_data = fetch_fund_data_with_cache(
        code_dict=config["code_dict"],
        cache_dir=config["data_dir"],
        force_refresh=bool(config["force_refresh"]),
        cache_prefix=config["cache_prefix"],
    )
    normalized_data = normalize_fund_data(raw_data)
    fund_code = str(config["default_fund_code"]).zfill(6)
    return run_single_fund_strategy_from_data(
        normalized_data=normalized_data,
        config=config,
        fund_code=fund_code,
        print_summary=True,
    )


def run_multi_fund_strategy(config_override=None):
    # 批量模式统一拉一次数据，再对基金池逐只回测并落汇总结果。
    config = build_tradition_config(config_override=config_override)
    raw_data = fetch_fund_data_with_cache(
        code_dict=config["code_dict"],
        cache_dir=config["data_dir"],
        force_refresh=bool(config["force_refresh"]),
        cache_prefix=config["cache_prefix"],
    )
    normalized_data = normalize_fund_data(raw_data)
    result_list = []
    for fund_code in resolve_fund_code_list(config):
        result = run_single_fund_strategy_from_data(
            normalized_data=normalized_data,
            config=config,
            fund_code=fund_code,
            print_summary=False,
        )
        result_list.append(result)
    summary_df = build_summary_table(result_list=result_list)
    strategy_name = str(config["default_strategy_name"]).lower()
    summary_path = save_summary_table(
        summary_df=summary_df,
        output_dir=config["output_dir"],
        summary_name=strategy_name,
    )
    print_summary_table(summary_df=summary_df, summary_path=summary_path, title="批量回测汇总:")
    return {
        "result_list": result_list,
        "summary_df": summary_df,
        "summary_path": summary_path,
    }


def run_compare_all_strategies(config_override=None):
    # 单基金多策略模式统一使用同一份数据，避免对相同基金重复拉数和重复解析。
    config = build_tradition_config(config_override=config_override)
    raw_data = fetch_fund_data_with_cache(
        code_dict=config["code_dict"],
        cache_dir=config["data_dir"],
        force_refresh=bool(config["force_refresh"]),
        cache_prefix=config["cache_prefix"],
    )
    normalized_data = normalize_fund_data(raw_data)
    fund_code = str(config["default_fund_code"]).zfill(6)
    result_list = []
    for strategy_name in ALL_STRATEGY_NAME_LIST:
        strategy_config = dict(config)
        strategy_config["default_strategy_name"] = strategy_name
        result = run_single_fund_strategy_from_data(
            normalized_data=normalized_data,
            config=strategy_config,
            fund_code=fund_code,
            print_summary=False,
        )
        result_list.append(result)
    summary_df = build_summary_table(result_list=result_list)
    summary_name = f"{fund_code}_all_strategies"
    summary_path = save_summary_table(
        summary_df=summary_df,
        output_dir=config["output_dir"],
        summary_name=summary_name,
    )
    print_summary_table(summary_df=summary_df, summary_path=summary_path, title=f"单基金多策略对比: {fund_code}")
    return {
        "fund_code": fund_code,
        "result_list": result_list,
        "summary_df": summary_df,
        "summary_path": summary_path,
    }


def save_trial_table(trial_df, output_dir, fund_code, strategy_name):
    # 优化试验历史单独落盘，便于后续分析参数敏感性，不和普通回测汇总混在一起。
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    output_path = output_dir / f"optuna_trials_{fund_code}_{strategy_name}_{date_str}.csv"
    trial_df.to_csv(output_path, index=False)
    return output_path


def print_optimization_summary(optimization_result):
    # 参数优化只打印最关键结论，避免 trial 级细节淹没验证集和测试集结果。
    print("参数优化结果:")
    print("基金代码:", optimization_result["fund_code"])
    print("目标策略:", optimization_result["strategy_name"])
    print("训练集最优目标值:", optimization_result["best_value"])
    print("Top-K 候选数量:", len(optimization_result["top_k_params_list"]))
    print("逐次刷新最优候选数量:", len(optimization_result["improving_best_params_list"]))
    print("验证集最优参数:", optimization_result["best_params"])
    print("验证集最优来源:", optimization_result["best_candidate_source"])
    print("验证集 Sharpe:", optimization_result["valid_result"]["stats"]["sharpe"])
    print("测试集 Sharpe:", optimization_result["test_result"]["stats"]["sharpe"])
    print("测试集最大回撤:", optimization_result["test_result"]["stats"]["max_drawdown"])
    print("trial 输出:", optimization_result["trial_path"])


def run_optimize_single_fund_strategy(config_override=None):
    # 优化模式固定使用 train/valid/test 时序切分，训练集搜索，验证集选 Top-K，测试集最终评估。
    config = build_tradition_config(config_override=config_override)
    raw_data = fetch_fund_data_with_cache(
        code_dict=config["code_dict"],
        cache_dir=config["data_dir"],
        force_refresh=bool(config["force_refresh"]),
        cache_prefix=config["cache_prefix"],
    )
    normalized_data = normalize_fund_data(raw_data)
    fund_code = str(config["default_fund_code"]).zfill(6)
    fund_df = filter_single_fund(normalized_data, fund_code=fund_code)
    price_series, data_mode = adapt_to_price_series(fund_df=fund_df)
    split_dict = split_time_series_by_ratio(price_series=price_series, split_config=config["data_split_dict"])
    rf_series = load_rf_series_for_price_series(config=config, price_series=price_series)

    strategy_name = resolve_optimization_strategy_name(config)
    base_params = config["strategy_param_dict"].get(strategy_name)
    optimization_config = dict(config["optimization_config"])

    def evaluate_params_fn(strategy_params):
        # 训练集 objective 改为基于完整历史执行后切片评估，避免分段独立回测导致窗口冷启动。
        execution_result = execute_price_series_strategy(
            price_series=price_series,
            config=config,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
        )
        return build_result_from_execution(
            execution_result=execution_result,
            fund_code=fund_code,
            strategy_name=strategy_name,
            data_mode=data_mode,
            save_plot=False,
            print_summary=False,
            segment_index=split_dict["train"].index,
            rf_series=rf_series,
        )

    optimization_result = optimize_strategy_params(
        strategy_name=strategy_name,
        base_params=base_params,
        evaluate_params_fn=evaluate_params_fn,
        optimization_config=optimization_config,
    )

    # Top-K 与逐次刷新最优集合并行产出后，在验证集阶段统一合并、去重并选优。
    raw_candidate_list = optimization_result["top_k_params_list"] + optimization_result["improving_best_params_list"]
    deduplicated_candidate_dict = {}
    for candidate in raw_candidate_list:
        param_key = build_param_signature(candidate["params"])
        if param_key not in deduplicated_candidate_dict:
            deduplicated_candidate_dict[param_key] = {
                "trial_number": candidate["trial_number"],
                "train_objective": candidate["train_objective"],
                "params": candidate["params"],
                "candidate_source_set": {candidate["source"]},
            }
            continue
        deduplicated_candidate_dict[param_key]["candidate_source_set"].add(candidate["source"])
        if float(candidate["train_objective"]) > float(deduplicated_candidate_dict[param_key]["train_objective"]):
            deduplicated_candidate_dict[param_key]["train_objective"] = candidate["train_objective"]
            deduplicated_candidate_dict[param_key]["trial_number"] = candidate["trial_number"]

    candidate_evaluation_list = []
    for candidate in deduplicated_candidate_dict.values():
        execution_result = execute_price_series_strategy(
            price_series=price_series,
            config=config,
            strategy_name=strategy_name,
            strategy_params=candidate["params"],
        )
        valid_result = build_result_from_execution(
            execution_result=execution_result,
            fund_code=fund_code,
            strategy_name=strategy_name,
            data_mode=data_mode,
            save_plot=False,
            print_summary=False,
            segment_index=split_dict["valid"].index,
            rf_series=rf_series,
        )
        candidate_evaluation_list.append(
            {
                "trial_number": candidate["trial_number"],
                "train_objective": candidate["train_objective"],
                "params": candidate["params"],
                "candidate_source_set": sorted(candidate["candidate_source_set"]),
                "valid_result": valid_result,
                "execution_result": execution_result,
            }
        )

    best_candidate = max(candidate_evaluation_list, key=lambda item: float(item["valid_result"]["stats"]["sharpe"]))
    best_params = dict(best_candidate["params"])
    valid_result = best_candidate["valid_result"]
    best_candidate_source = ",".join(best_candidate["candidate_source_set"])
    candidate_result_list = [
        {
            "trial_number": item["trial_number"],
            "train_objective": item["train_objective"],
            "params": item["params"],
            "candidate_source_set": item["candidate_source_set"],
            "valid_result": item["valid_result"],
        }
        for item in candidate_evaluation_list
    ]

    test_output_name = f"{fund_code}_{strategy_name}_test_best_{datetime.today().strftime('%Y-%m-%d')}.png"
    test_result = build_result_from_execution(
        execution_result=best_candidate["execution_result"],
        fund_code=fund_code,
        strategy_name=strategy_name,
        data_mode=data_mode,
        save_plot=True,
        output_dir=config["output_dir"],
        output_name=test_output_name,
        title=f"{fund_code} {strategy_name}",
        print_summary=False,
        segment_index=split_dict["test"].index,
        rf_series=rf_series,
    )
    trial_path = save_trial_table(
        trial_df=optimization_result["trial_df"],
        output_dir=config["output_dir"],
        fund_code=fund_code,
        strategy_name=strategy_name,
    )
    result = {
        "fund_code": fund_code,
        "strategy_name": strategy_name,
        "split_dict": split_dict,
        "best_params": best_params,
        "best_value": optimization_result["best_value"],
        "top_k_params_list": optimization_result["top_k_params_list"],
        "improving_best_params_list": optimization_result["improving_best_params_list"],
        "candidate_result_list": candidate_result_list,
        "best_candidate_source": best_candidate_source,
        "valid_result": valid_result,
        "test_result": test_result,
        "trial_df": optimization_result["trial_df"],
        "trial_path": trial_path,
    }
    print_optimization_summary(result)
    return result


def print_run_summary(result):
    # 终端摘要统一由这里负责，避免 runner 主链混入大量打印细节。
    print("基金代码:", result["fund_code"])
    print("策略名称:", result["strategy_name"])
    print("策略参数:", result["strategy_params"])
    print("样本区间:", result["sample_start"], result["sample_end"])
    print("数据模式:", result["data_mode"])
    print("交易次数:", result["trade_count"])
    print("累计收益:", result["stats"]["cumulative_return"])
    print("年化收益:", result["stats"]["annual_return"])
    print("年化波动:", result["stats"]["annual_volatility"])
    print("Sharpe:", result["stats"]["sharpe"])
    print("最大回撤:", result["stats"]["max_drawdown"])
    print("图像输出:", result["output_path"])


def main(argv=None):
    # 第一阶段支持单基金、单基金多策略、批量和参数优化模式，命令行只用于覆盖少量默认配置。
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cli_override = build_cli_override(args)
    config_override = None
    if len(cli_override) > 0:
        strategy_param_dict = cli_override.pop("strategy_param_dict", None)
        optimization_override = cli_override.pop("optimization_config", None)
        base_config = build_tradition_config(config_override=cli_override)
        if strategy_param_dict is not None:
            base_config["strategy_param_dict"] = merge_strategy_params(
                default_param_dict=base_config["strategy_param_dict"],
                override_param_dict=strategy_param_dict,
            )
        if optimization_override is not None:
            base_config["optimization_config"] = merge_optimization_config(
                default_optimization_config=base_config["optimization_config"],
                override_optimization_config=optimization_override,
            )
        config_override = base_config
    mode_config = config_override or {}
    if bool(mode_config.get("optimize", False)):
        run_optimize_single_fund_strategy(config_override=config_override)
        return
    if bool(mode_config.get("compare_all", False)):
        run_compare_all_strategies(config_override=config_override)
        return
    if bool(mode_config.get("batch_run", False)):
        run_multi_fund_strategy(config_override=config_override)
        return
    run_single_fund_strategy(config_override=config_override)


if __name__ == "__main__":
    main()
