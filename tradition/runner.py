import argparse
from datetime import datetime

import pandas as pd

from tradition.config import build_tradition_config
from tradition.data_adapter import adapt_to_price_series
from tradition.data_fetcher import fetch_fund_data_with_cache
from tradition.data_loader import filter_single_fund, normalize_fund_data
from tradition.metrics import compute_return_metrics, save_equity_curve_plot
from tradition.strategies import generate_signals


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

    strategy_param_override = {}
    if args.ma_fast is not None:
        strategy_param_override.setdefault("ma_cross", {})["fast"] = int(args.ma_fast)
    if args.ma_slow is not None:
        strategy_param_override.setdefault("ma_cross", {})["slow"] = int(args.ma_slow)
    if args.momentum_window is not None:
        strategy_param_override.setdefault("momentum", {})["window"] = int(args.momentum_window)
    if len(strategy_param_override) > 0:
        override["strategy_param_dict"] = strategy_param_override
    return override


def build_arg_parser():
    # 第一阶段只开放单基金回测最需要的命令行参数，保持入口简单且可直接试验。
    parser = argparse.ArgumentParser(description="运行 tradition 时序择时入门版单基金或批量回测")
    parser.add_argument("--fund-code", dest="fund_code", help="目标基金代码，例如 007301")
    parser.add_argument("--fund-codes", dest="fund_codes", help="批量模式下使用的基金代码列表，逗号分隔")
    parser.add_argument(
        "--strategy-name",
        dest="strategy_name",
        choices=["buy_and_hold", "ma_cross", "momentum", "multi_factor_score"],
        help="策略名称",
    )
    parser.add_argument("--batch-run", action="store_true", help="按基金池批量运行当前策略并输出汇总表")
    parser.add_argument("--force-refresh", action="store_true", help="忽略当天缓存并重新拉取 AkShare 数据")
    parser.add_argument("--init-cash", dest="init_cash", type=float, help="初始资金")
    parser.add_argument("--fees", dest="fees", type=float, help="手续费率")
    parser.add_argument("--ma-fast", dest="ma_fast", type=int, help="ma_cross 策略短均线窗口")
    parser.add_argument("--ma-slow", dest="ma_slow", type=int, help="ma_cross 策略长均线窗口")
    parser.add_argument("--momentum-window", dest="momentum_window", type=int, help="momentum 策略动量窗口")
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


def extract_trade_count(portfolio):
    # 统一把 vectorbt 的交易统计转成标量，避免不同返回类型污染结果结构。
    trade_count = portfolio.trades.count()
    if hasattr(trade_count, "item"):
        return int(trade_count.item())
    return int(trade_count)


def build_summary_record(result):
    # 汇总表固定展开核心指标字段，便于跨基金横向比较。
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
    # 汇总表按 Sharpe 和年化收益降序排序，优先展示风险收益比更好的基金。
    if len(result_list) == 0:
        raise ValueError("result_list 为空，无法构建汇总表。")
    summary_df = pd.DataFrame([build_summary_record(result) for result in result_list])
    summary_df = summary_df.sort_values(["sharpe", "annual_return"], ascending=False).reset_index(drop=True)
    return summary_df


def save_summary_table(summary_df, output_dir, strategy_name):
    # 批量模式下把汇总结果落成 CSV，便于后续继续分析与比较。
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    output_path = output_dir / f"summary_{strategy_name}_{date_str}.csv"
    summary_df.to_csv(output_path, index=False)
    return output_path


def print_summary_table(summary_df, summary_path):
    # 终端只打印最关键列，避免批量模式输出过长难以阅读。
    print("批量回测汇总:")
    printable_df = summary_df[["fund_code", "strategy_name", "annual_return", "sharpe", "max_drawdown", "trade_count"]].copy()
    print(printable_df.to_string(index=False))
    print("汇总输出:", summary_path)


def resolve_fund_code_list(config):
    # 批量模式优先使用显式传入的基金列表，否则退回配置中的全部基金代码。
    fund_code_list = config.get("fund_code_list")
    if fund_code_list is not None:
        return [str(code).zfill(6) for code in fund_code_list]
    return sorted([str(code).zfill(6) for code in config["code_dict"].keys()])


def run_single_fund_strategy_from_data(normalized_data, config, fund_code, print_summary=True):
    # 复用同一份标准化数据执行单基金回测，避免批量模式对同一天缓存重复读取与解析。
    fund_code = str(fund_code).zfill(6)
    fund_df = filter_single_fund(normalized_data, fund_code=fund_code)
    price_series, data_mode = adapt_to_price_series(fund_df=fund_df)

    strategy_name = str(config["default_strategy_name"]).lower()
    entries, exits, strategy_params = generate_signals(
        price_series=price_series,
        strategy_name=strategy_name,
        strategy_params=config["strategy_param_dict"].get(strategy_name),
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
    equity_curve = portfolio.value()
    metric_dict = compute_return_metrics(equity_curve=equity_curve)

    date_str = datetime.today().strftime("%Y-%m-%d")
    output_path = config["output_dir"] / f"{fund_code}_{strategy_name}_{date_str}.png"
    save_equity_curve_plot(
        equity_curve=equity_curve,
        output_path=output_path,
        title=f"{fund_code} {strategy_name}",
    )

    result = {
        "fund_code": fund_code,
        "strategy_name": strategy_name,
        "strategy_params": strategy_params,
        "stats": metric_dict,
        "equity_curve": equity_curve,
        "output_path": output_path,
        "data_mode": data_mode,
        "sample_start": price_series.index.min(),
        "sample_end": price_series.index.max(),
        "trade_count": extract_trade_count(portfolio=portfolio),
    }
    if print_summary:
        print_run_summary(result=result)
    return result


def run_single_fund_strategy(config_override=None):
    # 单基金模式在数据准备后复用通用执行函数，保持与批量模式同一回测口径。
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
    summary_path = save_summary_table(
        summary_df=summary_df,
        output_dir=config["output_dir"],
        strategy_name=str(config["default_strategy_name"]).lower(),
    )
    print_summary_table(summary_df=summary_df, summary_path=summary_path)
    return {
        "result_list": result_list,
        "summary_df": summary_df,
        "summary_path": summary_path,
    }


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
    # 第一阶段支持单基金和批量模式，命令行只用于覆盖少量默认配置。
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cli_override = build_cli_override(args)
    config_override = None
    if len(cli_override) > 0:
        strategy_param_dict = cli_override.pop("strategy_param_dict", None)
        base_config = build_tradition_config(config_override=cli_override)
        if strategy_param_dict is not None:
            base_config["strategy_param_dict"] = merge_strategy_params(
                default_param_dict=base_config["strategy_param_dict"],
                override_param_dict=strategy_param_dict,
            )
        config_override = base_config
    if bool((config_override or {}).get("batch_run", False)):
        run_multi_fund_strategy(config_override=config_override)
        return
    run_single_fund_strategy(config_override=config_override)


if __name__ == "__main__":
    main()
