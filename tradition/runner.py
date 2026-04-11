import argparse
from datetime import datetime

import pandas as pd

from tradition.config import build_tradition_config
from tradition.data_adapter import adapt_to_price_series
from tradition.data_fetcher import fetch_fund_data_with_cache, fetch_treasury_yield_with_cache
from tradition.data_loader import filter_single_fund, normalize_fund_data
from tradition.factor_analysis import (
    run_factor_combination,
    run_factor_selection_single_fund,
    run_single_factor_dedup_selection,
    run_single_factor_stability_analysis,
    run_strategy_backtest,
)
from tradition.metrics import compute_return_metrics, save_equity_curve_plot
from tradition.optimizer import optimize_strategy_params
from tradition.splitter import build_walk_forward_fold_list, split_time_series_by_ratio
from tradition.strategies import generate_signals


ALL_STRATEGY_NAME_LIST = ["buy_and_hold", "ma_cross", "momentum", "multi_factor_score"]


def load_vectorbt_module():
    # 延迟导入 vectorbt，减少模块导入阶段对环境完整性的强依赖。
    import vectorbt as vbt

    return vbt


def add_common_cli_arguments(parser):
    parser.add_argument("--fund-code", dest="fund_code", help="目标基金代码，例如 007301")
    parser.add_argument("--force-refresh", action="store_true", help="忽略当天缓存并重新拉取 AkShare 数据")
    parser.add_argument("--init-cash", dest="init_cash", type=float, help="初始资金")
    parser.add_argument("--fees", dest="fees", type=float, help="手续费率")


def add_strategy_cli_arguments(parser):
    parser.add_argument(
        "--strategy-name",
        dest="strategy_name",
        choices=ALL_STRATEGY_NAME_LIST,
        help="策略名称",
    )
    parser.add_argument("--ma-fast", dest="ma_fast", type=int, help="ma_cross 策略短均线窗口")
    parser.add_argument("--ma-slow", dest="ma_slow", type=int, help="ma_cross 策略长均线窗口")
    parser.add_argument("--momentum-window", dest="momentum_window", type=int, help="momentum 策略动量窗口")


def add_optimization_cli_arguments(parser):
    parser.add_argument("--n-trials", dest="n_trials", type=int, help="Optuna trial 数量")


def add_walk_forward_cli_arguments(parser):
    parser.add_argument("--wf-window-size", dest="wf_window_size", type=int, help="walk-forward 总窗口长度")
    parser.add_argument("--wf-step-size", dest="wf_step_size", type=int, help="walk-forward 滚动步长")


def add_factor_selection_cli_arguments(parser):
    parser.add_argument("--factor-groups", dest="factor_groups", help="因子族名称列表，按因子库分组名输入，逗号分隔")
    parser.add_argument("--train-min-spearman-ic", dest="train_min_spearman_ic", type=float, help="训练集 Spearman IC 最小阈值")
    parser.add_argument("--train-min-spearman-icir", dest="train_min_spearman_icir", type=float, help="训练集 Spearman ICIR 最小阈值")


def add_ic_aggregation_cli_arguments(parser):
    parser.add_argument("--ic-exp-weighted", dest="ic_exp_weighted", action="store_true", help="启用 IC / ICIR 指数加权聚合模式（半衰期使用默认值）")


def add_research_io_cli_arguments(parser):
    parser.add_argument("--factor-selection-path", dest="factor_selection_path", help="factor_select 流程输出的 JSON 文件路径")
    parser.add_argument("--stability-analysis-path", dest="stability_analysis_path", help="single_factor_stability_analysis 流程输出的 JSON 文件路径")
    parser.add_argument("--dedup-root-topk", dest="dedup_root_topk", type=int, help="single_factor_dedup_selection 树形搜索使用的根节点数量，默认 3")
    parser.add_argument("--dedup-selection-path", dest="dedup_selection_path", help="single_factor_dedup_selection 流程输出的 JSON 文件路径")
    parser.add_argument("--factor-combination-path", dest="factor_combination_path", help="factor_combination 流程输出的 JSON 文件路径")


def add_legacy_mode_arguments(parser):
    parser.add_argument("--fund-codes", dest="fund_codes", help="批量模式下使用的基金代码列表，逗号分隔")
    parser.add_argument("--batch-run", action="store_true", help="按基金池批量运行当前策略并输出汇总表")
    parser.add_argument("--compare-all", action="store_true", help="对单只基金运行全部策略并输出对比表")
    parser.add_argument("--optimize", action="store_true", help="对单只基金执行训练/验证/测试切分后的 Optuna 参数优化")
    parser.add_argument("--walk-forward", action="store_true", help="对单只基金执行固定窗口 walk-forward 参数优化")
    parser.add_argument("--factor-select", action="store_true", help="对单只基金执行基于 walk-forward 的因子筛选与评估")
    parser.add_argument("--single-factor-stability-analysis", action="store_true", help="读取 factor_select 结果并执行单因子稳定性分析")
    parser.add_argument("--single-factor-dedup-selection", action="store_true", help="读取稳定性分析结果并执行去冗余与正向选择")
    parser.add_argument("--factor-combination", action="store_true", help="读取 dedup 结果并执行因子组合方式对比与参数微调")
    parser.add_argument("--strategy-backtest", action="store_true", help="读取 factor_combination 结果并执行连续仓位策略回测")


def add_backtest_subparsers(subparsers):
    common_parent = argparse.ArgumentParser(add_help=False)
    add_common_cli_arguments(common_parent)
    strategy_parent = argparse.ArgumentParser(add_help=False)
    add_strategy_cli_arguments(strategy_parent)
    optimization_parent = argparse.ArgumentParser(add_help=False)
    add_optimization_cli_arguments(optimization_parent)
    walk_forward_parent = argparse.ArgumentParser(add_help=False)
    add_walk_forward_cli_arguments(walk_forward_parent)

    single_parser = subparsers.add_parser("single", parents=[common_parent, strategy_parent], help="单基金回测")
    single_parser.set_defaults(command_group="backtest", command_name="single")

    batch_parser = subparsers.add_parser("batch", parents=[common_parent, strategy_parent], help="批量基金回测")
    batch_parser.add_argument("--fund-codes", dest="fund_codes", help="批量模式下使用的基金代码列表，逗号分隔")
    batch_parser.set_defaults(command_group="backtest", command_name="batch")

    compare_parser = subparsers.add_parser("compare", parents=[common_parent], help="单基金多策略对比")
    compare_parser.set_defaults(command_group="backtest", command_name="compare")

    optimize_parser = subparsers.add_parser(
        "optimize",
        parents=[common_parent, strategy_parent, optimization_parent],
        help="单基金参数优化",
    )
    optimize_parser.set_defaults(command_group="backtest", command_name="optimize")

    walk_forward_parser = subparsers.add_parser(
        "walk-forward",
        parents=[common_parent, strategy_parent, optimization_parent, walk_forward_parent],
        help="单基金 walk-forward 优化",
    )
    walk_forward_parser.set_defaults(command_group="backtest", command_name="walk-forward")


def add_research_subparsers(subparsers):
    common_parent = argparse.ArgumentParser(add_help=False)
    add_common_cli_arguments(common_parent)
    factor_selection_parent = argparse.ArgumentParser(add_help=False)
    add_factor_selection_cli_arguments(factor_selection_parent)
    ic_aggregation_parent = argparse.ArgumentParser(add_help=False)
    add_ic_aggregation_cli_arguments(ic_aggregation_parent)
    io_parent = argparse.ArgumentParser(add_help=False)
    add_research_io_cli_arguments(io_parent)

    factor_select_parser = subparsers.add_parser(
        "factor-select",
        parents=[common_parent, factor_selection_parent, ic_aggregation_parent],
        help="流程 1 因子筛选",
    )
    factor_select_parser.set_defaults(command_group="research", command_name="factor-select")

    stability_parser = subparsers.add_parser(
        "stability",
        parents=[io_parent, ic_aggregation_parent],
        help="流程 2 单因子稳定性分析",
    )
    stability_parser.set_defaults(command_group="research", command_name="stability")

    dedup_parser = subparsers.add_parser(
        "dedup",
        parents=[io_parent, ic_aggregation_parent],
        help="流程 3 去冗余与组合选择",
    )
    dedup_parser.set_defaults(command_group="research", command_name="dedup")

    combination_parser = subparsers.add_parser(
        "combination",
        parents=[io_parent, ic_aggregation_parent],
        help="流程 4 因子组合与权重微调",
    )
    combination_parser.set_defaults(command_group="research", command_name="combination")

    strategy_backtest_parser = subparsers.add_parser(
        "strategy-backtest",
        parents=[io_parent],
        help="流程 5 连续仓位策略回测",
    )
    strategy_backtest_parser.set_defaults(command_group="research", command_name="strategy-backtest")


def resolve_legacy_cli_command(args):
    if bool(getattr(args, "strategy_backtest", False)):
        return "research.strategy-backtest"
    if bool(getattr(args, "factor_combination", False)):
        return "research.combination"
    if bool(getattr(args, "single_factor_dedup_selection", False)):
        return "research.dedup"
    if bool(getattr(args, "single_factor_stability_analysis", False)):
        return "research.stability"
    if bool(getattr(args, "factor_select", False)):
        return "research.factor-select"
    if bool(getattr(args, "walk_forward", False)):
        return "backtest.walk-forward"
    if bool(getattr(args, "optimize", False)):
        return "backtest.optimize"
    if bool(getattr(args, "compare_all", False)):
        return "backtest.compare"
    if bool(getattr(args, "batch_run", False)):
        return "backtest.batch"
    return None


def resolve_runner_command(args):
    command_group = getattr(args, "command_group", None)
    command_name = getattr(args, "command_name", None)
    if command_group and command_name:
        return f"{command_group}.{command_name}"
    legacy_command_name = resolve_legacy_cli_command(args)
    if legacy_command_name is not None:
        return legacy_command_name
    return "backtest.single"


def build_common_cli_override(args):
    override = {}
    if args.fund_code is not None:
        override["default_fund_code"] = str(args.fund_code).zfill(6)
    if getattr(args, "fund_codes", None) is not None:
        override["fund_code_list"] = [str(code).strip().zfill(6) for code in str(args.fund_codes).split(",") if str(code).strip()]
    if args.force_refresh:
        override["force_refresh"] = True
    if args.init_cash is not None:
        override["init_cash"] = float(args.init_cash)
    if args.fees is not None:
        override["fees"] = float(args.fees)
    return override


def build_backtest_command_override(args, command_name):
    override = {}
    if args.fund_codes is not None:
        override["fund_code_list"] = [str(code).strip().zfill(6) for code in str(args.fund_codes).split(",") if str(code).strip()]
    if args.strategy_name is not None:
        override["default_strategy_name"] = str(args.strategy_name).lower()
    if command_name == "batch":
        override["batch_run"] = True
    if command_name == "compare":
        override["compare_all"] = True
    if command_name == "optimize":
        override["optimize"] = True
    if command_name == "walk-forward":
        override["walk_forward"] = True

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
    walk_forward_override = {}
    if args.wf_window_size is not None:
        walk_forward_override["window_size"] = int(args.wf_window_size)
    if args.wf_step_size is not None:
        walk_forward_override["step_size"] = int(args.wf_step_size)
    if len(walk_forward_override) > 0:
        override["walk_forward_config"] = walk_forward_override
    return override


def build_research_command_override(args, command_name):
    override = {}
    if command_name == "factor-select":
        override["factor_select"] = True
    elif command_name == "stability":
        override["single_factor_stability_analysis"] = True
    elif command_name == "dedup":
        override["single_factor_dedup_selection"] = True
    elif command_name == "combination":
        override["factor_combination"] = True
    elif command_name == "strategy-backtest":
        override["strategy_backtest"] = True

    if args.factor_groups is not None:
        override["factor_group_list"] = [group_name.strip() for group_name in str(args.factor_groups).split(",") if group_name.strip()]
    if args.train_min_spearman_ic is not None:
        override["train_min_spearman_ic"] = float(args.train_min_spearman_ic)
    if args.train_min_spearman_icir is not None:
        override["train_min_spearman_icir"] = float(args.train_min_spearman_icir)
    if bool(getattr(args, "ic_exp_weighted", False)):
        override["ic_aggregation_mode"] = "exp_weighted"
    if args.factor_selection_path is not None:
        override["factor_selection_path"] = str(args.factor_selection_path)
    if args.stability_analysis_path is not None:
        override["stability_analysis_path"] = str(args.stability_analysis_path)
    if args.dedup_root_topk is not None:
        override["dedup_root_topk"] = int(args.dedup_root_topk)
    if args.dedup_selection_path is not None:
        override["dedup_selection_path"] = str(args.dedup_selection_path)
    if args.factor_combination_path is not None:
        override["factor_combination_path"] = str(args.factor_combination_path)
    return override


def build_arg_parser():
    # 命令行入口改为单入口加子命令分组，旧布尔参数仅保留兼容翻译层。
    parser = argparse.ArgumentParser(description="运行 tradition 时序择时回测与研究流程")
    add_legacy_mode_arguments(parser)
    add_common_cli_arguments(parser)
    add_strategy_cli_arguments(parser)
    add_optimization_cli_arguments(parser)
    add_walk_forward_cli_arguments(parser)
    add_factor_selection_cli_arguments(parser)
    add_ic_aggregation_cli_arguments(parser)
    add_research_io_cli_arguments(parser)
    root_subparsers = parser.add_subparsers(dest="command_group")
    backtest_parser = root_subparsers.add_parser("backtest", help="回测与优化命令组")
    add_backtest_subparsers(backtest_parser.add_subparsers(dest="command_name"))
    research_parser = root_subparsers.add_parser("research", help="研究流程命令组")
    add_research_subparsers(research_parser.add_subparsers(dest="command_name"))
    return parser


def build_cli_override(args):
    # 命令行覆盖项按命令组拆分构造，减少入口层继续膨胀。
    command_name = resolve_runner_command(args)
    override = build_common_cli_override(args)
    command_group, command_leaf = command_name.split(".", 1)
    if command_group == "backtest":
        override.update(build_backtest_command_override(args, command_leaf))
    else:
        override.update(build_research_command_override(args, command_leaf))
    return override


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


def merge_walk_forward_config(default_walk_forward_config, override_walk_forward_config=None):
    # walk-forward 只维护总窗口和步长配置，区间比例复用全局 data_split_dict。
    merged = dict(default_walk_forward_config)
    if override_walk_forward_config is None:
        return merged
    merged.update(override_walk_forward_config)
    return merged


def extract_trade_count(portfolio):
    # 统一把 vectorbt 的交易统计转成标量，避免不同返回类型污染结果结构。
    trade_count = portfolio.trades.count()
    if hasattr(trade_count, "item"):
        return int(trade_count.item())
    return int(trade_count)


def compute_mean_daily_return_from_equity_curve(equity_curve):
    # WFE 使用区间权益曲线的平均日收益率，避免引入新的指标计算入口。
    returns = pd.Series(equity_curve, dtype=float).pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    return float(returns.mean())



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
    display_price_series = price_series
    if segment_index is not None:
        metric_equity_curve, display_equity_curve = extract_segment_equity_curves(
            equity_curve=equity_curve,
            segment_index=sample_index,
        )
        display_price_series = price_series.loc[sample_index].copy()
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
            benchmark_curve=display_price_series,
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


def execute_optimization_fold(
    price_series,
    config,
    fund_code,
    strategy_name,
    data_mode,
    rf_series,
    split_dict,
):
    # 单折执行函数复用当前训练集搜索、验证集选优、测试集评估链路，供单次 optimize 和 walk-forward 共用。
    base_params = config["strategy_param_dict"].get(strategy_name)
    optimization_config = dict(config["optimization_config"])

    def evaluate_params_fn(strategy_params):
        # 训练集 objective 固定基于完整历史执行后切片评估，避免分段独立回测导致窗口冷启动。
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
    return {
        "best_params": best_params,
        "best_value": optimization_result["best_value"],
        "top_k_params_list": optimization_result["top_k_params_list"],
        "improving_best_params_list": optimization_result["improving_best_params_list"],
        "candidate_result_list": candidate_result_list,
        "best_candidate_source": best_candidate_source,
        "valid_result": valid_result,
        "best_execution_result": best_candidate["execution_result"],
        "trial_df": optimization_result["trial_df"],
    }


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
    optimization_result = execute_optimization_fold(
        price_series=price_series,
        config=config,
        fund_code=fund_code,
        strategy_name=strategy_name,
        data_mode=data_mode,
        rf_series=rf_series,
        split_dict=split_dict,
    )

    test_output_name = f"{fund_code}_{strategy_name}_test_best_{datetime.today().strftime('%Y-%m-%d')}.png"
    test_result = build_result_from_execution(
        execution_result=optimization_result["best_execution_result"],
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
        "best_params": optimization_result["best_params"],
        "best_value": optimization_result["best_value"],
        "top_k_params_list": optimization_result["top_k_params_list"],
        "improving_best_params_list": optimization_result["improving_best_params_list"],
        "candidate_result_list": optimization_result["candidate_result_list"],
        "best_candidate_source": optimization_result["best_candidate_source"],
        "valid_result": optimization_result["valid_result"],
        "test_result": test_result,
        "trial_df": optimization_result["trial_df"],
        "trial_path": trial_path,
    }
    print_optimization_summary(result)
    return result


def print_walk_forward_summary(result):
    # walk-forward 汇总集中打印测试窗口分布指标，便于快速判断滚动结果是否稳定。
    print("Walk-forward 结果:")
    print("基金代码:", result["fund_code"])
    print("目标策略:", result["strategy_name"])
    print("折数:", len(result["fold_result_list"]))
    print("测试集累计收益均值:", result["summary_dict"]["test_cumulative_return_mean"])
    print("测试集累计收益中位数:", result["summary_dict"]["test_cumulative_return_median"])
    print("测试集累计收益标准差:", result["summary_dict"]["test_cumulative_return_std"])
    print("测试集累计收益最差值:", result["summary_dict"]["test_cumulative_return_min"])
    print("测试集年化收益均值:", result["summary_dict"]["test_annual_return_mean"])
    print("测试集年化收益中位数:", result["summary_dict"]["test_annual_return_median"])
    print("测试集年化收益标准差:", result["summary_dict"]["test_annual_return_std"])
    print("测试集年化收益最差值:", result["summary_dict"]["test_annual_return_min"])
    print("测试集年化波动均值:", result["summary_dict"]["test_annual_volatility_mean"])
    print("测试集年化波动中位数:", result["summary_dict"]["test_annual_volatility_median"])
    print("测试集年化波动标准差:", result["summary_dict"]["test_annual_volatility_std"])
    print("测试集年化波动最差值:", result["summary_dict"]["test_annual_volatility_max"])
    print("测试集 Sharpe 均值:", result["summary_dict"]["test_sharpe_mean"])
    print("测试集 Sharpe 中位数:", result["summary_dict"]["test_sharpe_median"])
    print("测试集 Sharpe 标准差:", result["summary_dict"]["test_sharpe_std"])
    print("测试集 Sharpe 最差值:", result["summary_dict"]["test_sharpe_min"])
    print("测试集最大回撤均值:", result["summary_dict"]["test_max_drawdown_mean"])
    print("测试集最大回撤中位数:", result["summary_dict"]["test_max_drawdown_median"])
    print("测试集最差回撤:", result["summary_dict"]["test_max_drawdown_min"])
    print("测试集正收益窗口占比:", result["summary_dict"]["positive_test_return_ratio"])
    print("Walk Forward Efficiency:", result["summary_dict"]["walk_forward_efficiency"])
    print("汇总输出:", result["summary_path"])


def run_walk_forward_single_fund_strategy(config_override=None):
    # walk-forward 第一版固定为单基金、固定窗口滚动优化，逐折复用当前单次优化链路。
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
    rf_series = load_rf_series_for_price_series(config=config, price_series=price_series)
    strategy_name = resolve_optimization_strategy_name(config)
    walk_forward_config = dict(config["walk_forward_config"])
    fold_list = build_walk_forward_fold_list(
        price_series=price_series,
        walk_forward_config=walk_forward_config,
        split_config=config["data_split_dict"],
    )

    fold_result_list = []
    for fold_dict in fold_list:
        optimization_result = execute_optimization_fold(
            price_series=price_series,
            config=config,
            fund_code=fund_code,
            strategy_name=strategy_name,
            data_mode=data_mode,
            rf_series=rf_series,
            split_dict=fold_dict,
        )
        train_result = build_result_from_execution(
            execution_result=optimization_result["best_execution_result"],
            fund_code=fund_code,
            strategy_name=strategy_name,
            data_mode=data_mode,
            save_plot=False,
            print_summary=False,
            segment_index=fold_dict["train"].index,
            rf_series=rf_series,
        )
        test_result = build_result_from_execution(
            execution_result=optimization_result["best_execution_result"],
            fund_code=fund_code,
            strategy_name=strategy_name,
            data_mode=data_mode,
            save_plot=False,
            print_summary=False,
            segment_index=fold_dict["test"].index,
            rf_series=rf_series,
        )
        fold_result_list.append(
            {
                "fold_id": fold_dict["fold_id"],
                "train_start": fold_dict["train_start"],
                "train_end": fold_dict["train_end"],
                "valid_start": fold_dict["valid_start"],
                "valid_end": fold_dict["valid_end"],
                "test_start": fold_dict["test_start"],
                "test_end": fold_dict["test_end"],
                "best_params": optimization_result["best_params"],
                "best_value": optimization_result["best_value"],
                "best_candidate_source": optimization_result["best_candidate_source"],
                "train_mean_daily_return": compute_mean_daily_return_from_equity_curve(train_result["equity_curve"]),
                "valid_cumulative_return": optimization_result["valid_result"]["stats"]["cumulative_return"],
                "valid_sharpe": optimization_result["valid_result"]["stats"]["sharpe"],
                "test_cumulative_return": test_result["stats"]["cumulative_return"],
                "test_annual_return": test_result["stats"]["annual_return"],
                "test_annual_volatility": test_result["stats"]["annual_volatility"],
                "test_sharpe": test_result["stats"]["sharpe"],
                "test_max_drawdown": test_result["stats"]["max_drawdown"],
                "test_mean_daily_return": compute_mean_daily_return_from_equity_curve(test_result["equity_curve"]),
            }
        )

    summary_df = pd.DataFrame(fold_result_list)
    summary_name = f"{fund_code}_{strategy_name}_walk_forward"
    summary_path = save_summary_table(summary_df=summary_df, output_dir=config["output_dir"], summary_name=summary_name)
    train_mean_daily_return_mean = float(summary_df["train_mean_daily_return"].mean())
    test_mean_daily_return_mean = float(summary_df["test_mean_daily_return"].mean())
    walk_forward_efficiency = None
    if abs(train_mean_daily_return_mean) > 1e-12:
        walk_forward_efficiency = float(test_mean_daily_return_mean / train_mean_daily_return_mean)
    summary_dict = {
        "test_cumulative_return_mean": float(summary_df["test_cumulative_return"].mean()),
        "test_cumulative_return_median": float(summary_df["test_cumulative_return"].median()),
        "test_cumulative_return_std": float(summary_df["test_cumulative_return"].std(ddof=0)),
        "test_cumulative_return_min": float(summary_df["test_cumulative_return"].min()),
        "test_annual_return_mean": float(summary_df["test_annual_return"].mean()),
        "test_annual_return_median": float(summary_df["test_annual_return"].median()),
        "test_annual_return_std": float(summary_df["test_annual_return"].std(ddof=0)),
        "test_annual_return_min": float(summary_df["test_annual_return"].min()),
        "test_annual_volatility_mean": float(summary_df["test_annual_volatility"].mean()),
        "test_annual_volatility_median": float(summary_df["test_annual_volatility"].median()),
        "test_annual_volatility_std": float(summary_df["test_annual_volatility"].std(ddof=0)),
        "test_annual_volatility_max": float(summary_df["test_annual_volatility"].max()),
        "test_sharpe_mean": float(summary_df["test_sharpe"].mean()),
        "test_sharpe_median": float(summary_df["test_sharpe"].median()),
        "test_sharpe_std": float(summary_df["test_sharpe"].std(ddof=0)),
        "test_sharpe_min": float(summary_df["test_sharpe"].min()),
        "test_max_drawdown_mean": float(summary_df["test_max_drawdown"].mean()),
        "test_max_drawdown_median": float(summary_df["test_max_drawdown"].median()),
        "test_max_drawdown_min": float(summary_df["test_max_drawdown"].min()),
        "positive_test_return_ratio": float((summary_df["test_cumulative_return"] > 0).mean()),
        "walk_forward_efficiency": walk_forward_efficiency,
    }
    result = {
        "fund_code": fund_code,
        "strategy_name": strategy_name,
        "fold_result_list": fold_result_list,
        "summary_df": summary_df,
        "summary_dict": summary_dict,
        "summary_path": summary_path,
    }
    print_walk_forward_summary(result)
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


def dispatch_runner_command(command_name, config_override):
    dispatch_dict = {
        "backtest.single": run_single_fund_strategy,
        "backtest.batch": run_multi_fund_strategy,
        "backtest.compare": run_compare_all_strategies,
        "backtest.optimize": run_optimize_single_fund_strategy,
        "backtest.walk-forward": run_walk_forward_single_fund_strategy,
        "research.factor-select": run_factor_selection_single_fund,
        "research.stability": run_single_factor_stability_analysis,
        "research.dedup": run_single_factor_dedup_selection,
        "research.combination": run_factor_combination,
        "research.strategy-backtest": run_strategy_backtest,
    }
    if command_name not in dispatch_dict:
        raise ValueError(f"未知命令: {command_name}")
    return dispatch_dict[command_name](config_override=config_override)


def main(argv=None):
    # 命令行入口统一解析命令组、构造覆盖配置并通过分发表路由到对应流程。
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    command_name = resolve_runner_command(args)
    cli_override = build_cli_override(args)
    config_override = None
    if len(cli_override) > 0:
        strategy_param_dict = cli_override.pop("strategy_param_dict", None)
        optimization_override = cli_override.pop("optimization_config", None)
        walk_forward_override = cli_override.pop("walk_forward_config", None)
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
        if walk_forward_override is not None:
            base_config["walk_forward_config"] = merge_walk_forward_config(
                default_walk_forward_config=base_config["walk_forward_config"],
                override_walk_forward_config=walk_forward_override,
            )
        config_override = base_config
    dispatch_runner_command(command_name=command_name, config_override=config_override)


if __name__ == "__main__":
    main()
