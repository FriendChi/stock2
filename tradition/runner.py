import argparse
from datetime import datetime

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
    if args.strategy_name is not None:
        override["default_strategy_name"] = str(args.strategy_name).lower()
    if args.force_refresh:
        override["force_refresh"] = True
    if args.init_cash is not None:
        override["init_cash"] = float(args.init_cash)
    if args.fees is not None:
        override["fees"] = float(args.fees)

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
    parser = argparse.ArgumentParser(description="运行 tradition 时序择时入门版单基金回测")
    parser.add_argument("--fund-code", dest="fund_code", help="目标基金代码，例如 007301")
    parser.add_argument(
        "--strategy-name",
        dest="strategy_name",
        choices=["buy_and_hold", "ma_cross", "momentum"],
        help="策略名称",
    )
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


def run_single_fund_strategy(config_override=None):
    # 统一入口收敛配置、拉数、信号回测和结果输出，便于后续新增策略时复用同一主链。
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
    print_run_summary(result=result)
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
    # 第一阶段固定走单基金时序策略回测，命令行只用于覆盖少量默认配置。
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
    run_single_fund_strategy(config_override=config_override)


if __name__ == "__main__":
    main()
