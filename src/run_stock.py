from compute_return import add_return_column
from config import code_dict
from feature_engineering import build_aligned_price_df
from feature_engineering import build_feature_dict
from fetch_fund_data import fetch_or_load_fund_data
from backtest import run_backtest


def main():
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

    # 计算特征并展示关键项，保持与 notebook 新增部分一致
    feature_dict = build_feature_dict(price_df_aligned)
    print(feature_dict.keys())
    print(feature_dict["momentum_20"].head())

    # 选择策略并执行回测，使用对齐后的价格表避免变量不一致
    strategy_name = "ma_cross"
    strategy_params = {"fast": 5, "slow": 20, "t_plus_one": True}
    pf, entries, exits = run_backtest(
        price_df=price_df_aligned,
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
    main()
