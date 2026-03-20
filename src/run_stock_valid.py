from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from compute_return import add_return_column
from config import code_dict
from feature_engineering import DataLayer
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
    price_df_aligned = DataLayer.build_aligned_price_df(data=data, fund_list=fund_list)

    # 验证段向前借用上下文窗口，缓解特征在验证起点的冷启动缺失
    split_result = DataLayer.split_train_valid(
        price_df=price_df_aligned,
        train_ratio=0.8,
        valid_context_window=120,
    )
    split_idx = split_result["split_idx"]
    price_df_selected = split_result["valid_df"]
    price_df_selected_with_context = split_result["valid_df_with_context"]
    print("当前模式:", "valid")
    print("切分位置:", split_idx)
    print("当前样本区间:", price_df_selected.index.min(), price_df_selected.index.max())
    print("当前样本形状:", price_df_selected.shape)
    print("验证上下文区间:", price_df_selected_with_context.index.min(), price_df_selected_with_context.index.max())
    print("验证上下文形状:", price_df_selected_with_context.shape)

    # 特征在验证模式下使用带上下文样本计算，避免窗口类特征在起始段缺失
    feature_dict = DataLayer.build_feature_dict(price_df_selected_with_context)
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
    # 保留绘图结果用于后续保存与弹窗展示
    ax = pf.value().plot()
    # 统一保存到数据目录，文件名包含策略名和日期
    output_dir = Path("/home/chi/stock2/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_path = output_dir / f"{strategy_name}_{date_str}_valid.png"
    ax.figure.savefig(output_path, dpi=150, bbox_inches="tight")
    print("图像已保存:", output_path)
    # 保留有图形界面时的弹窗显示能力
    plt.show()
    # 关闭图对象，避免批量运行时句柄累积
    plt.close(ax.figure)


if __name__ == "__main__":
    # 固定验证入口：无需再通过命令行参数指定模式
    main()
