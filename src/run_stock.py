import argparse

from compute_return import add_return_column
from config import code_dict
from feature_engineering import build_aligned_price_df
from feature_engineering import build_feature_dict
from fetch_fund_data import fetch_or_load_fund_data
from backtest import run_backtest


def main(mode="train"):
    # 先准备基础净值数据并保留原有检查输出
    data = fetch_or_load_fund_data(code_dict=code_dict, data_dir="data")
    data = add_return_column(data)
    print(type(data))
    print(data.head())
    print(data.dtypes)
    print(data[data["code"] == "007301"].head())

    # 构建基金池宽表并对齐到公共有效区间
    fund_list = [
        "012832",
        "007883",
        "012700",
        "012043",
        "009180",
        "012323",
        "007339",
        "014345",
        "001593",
        "009052",
        "008702",
    ]
    price_df_aligned = build_aligned_price_df(data=data, fund_list=fund_list)

    # 按时间顺序切分训练和验证区间，默认前80%训练，后20%验证
    split_idx = int(len(price_df_aligned) * 0.8)
    if split_idx <= 0 or split_idx >= len(price_df_aligned):
        raise ValueError(
            "样本量不足，无法按前80%训练、后20%验证切分。"
            f" 当前样本数={len(price_df_aligned)}"
        )

    # 根据运行模式选择对应数据区间，避免训练和验证共用全量样本
    if mode == "train":
        price_df_selected = price_df_aligned.iloc[:split_idx].copy()
    elif mode == "valid":
        price_df_selected = price_df_aligned.iloc[split_idx:].copy()
    else:
        raise ValueError(f"不支持的mode: {mode}，仅支持 train 或 valid。")

    print("当前模式:", mode)
    print("切分位置:", split_idx)
    print("当前样本区间:", price_df_selected.index.min(), price_df_selected.index.max())
    print("当前样本形状:", price_df_selected.shape)

    # 特征仅基于当前模式对应的数据区间计算，避免验证阶段读取训练后段之外的数据
    feature_dict = build_feature_dict(price_df_selected)
    print(feature_dict.keys())
    print(feature_dict["momentum_20"].head())

    # 选择策略并执行回测，使用对齐后的价格表避免变量不一致
    strategy_name = "ma_cross"
    strategy_params = {"fast": 5, "slow": 20, "t_plus_one": True}
    pf, entries, exits = run_backtest(
        price_df=price_df_selected,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        init_cash=10000,
        fees=0.0,
    )

    # 输出策略统计并绘制净值曲线
    print("当前策略:", strategy_name)
    print("策略参数:", strategy_params)
    print(pf.stats())
    print(pf.stats(group_by=False))
    pf.value().plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 用命令行参数切换训练或验证模式，默认保持训练模式
    parser.add_argument(
        "--mode",
        choices=["train", "valid"],
        default="train",
        help="运行模式：train 使用前80%数据，valid 使用后20%数据。",
    )
    args = parser.parse_args()
    main(mode=args.mode)
