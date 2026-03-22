from pathlib import Path

import pandas as pd

from compute_return import add_return_column
from config import code_dict
from feature_engineering import DataLayer
from fetch_fund_data import fetch_or_load_fund_data
from strategies import NeuralRankStrategy


def main():
    # 固定数据目录到项目根data，避免在src目录运行时误触发重复下载
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    # 训练优先复用本地最新缓存，避免无网络环境触发下载失败
    local_csv_list = sorted(data_dir.glob("fund_nav_*.csv"))
    if len(local_csv_list) > 0:
        latest_csv = local_csv_list[-1]
        print("使用本地缓存数据:", latest_csv)
        # 与数据拉取模块保持一致的类型与排序规范
        data = pd.read_csv(latest_csv, dtype={"code": str})
        data["date"] = pd.to_datetime(data["date"]).dt.date
        data["code"] = data["code"].astype(str).str.zfill(6)
        data = data.sort_values(["date", "code"]).reset_index(drop=True)
    else:
        # 无缓存时回退到原下载逻辑，保持原有能力不退化
        data = fetch_or_load_fund_data(code_dict=code_dict, data_dir=str(data_dir))
    data = add_return_column(data)

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

    # 训练参数集中在脚本内，便于复现实验并保持入口稳定
    rank_n = 7
    train_ratio = 0.8
    hidden_dim = 16
    epochs = 200
    lr = 0.01
    l2 = 0.0
    seed = 42

    # 先构建多特征与未来n日收益排名标签，再按时间切训练/验证
    labeled_df = DataLayer.build_labeled_table(
        price_df=price_df_aligned,
        n=rank_n,
        dropna=True,
    )
    split_result = DataLayer.split_train_valid(
        price_df=labeled_df,
        train_ratio=train_ratio,
        valid_context_window=0,
    )
    train_df = split_result["train_df"]
    valid_df = split_result["valid_df"]

    # 训练前打印去基金代码后的特征名，便于核对模型输入语义
    rank_suffix = f"_future_{rank_n}d_rank"
    # 标签列用于从labeled表中剔除监督目标，剩余即特征列
    rank_cols = [col for col in labeled_df.columns if str(col).endswith(rank_suffix)]
    # 保持与训练样本构建一致：非标签列全部视为特征列
    feature_cols = [str(col) for col in labeled_df.columns if col not in rank_cols]
    # 基金代码来自标签列前缀，和训练样本构建规则一致
    fund_codes = sorted([str(col).replace(rank_suffix, "") for col in rank_cols])
    # 去掉“基金代码_”前缀，仅保留特征后缀并去重
    feature_name_set = set()
    for col in feature_cols:
        for code in fund_codes:
            prefix = f"{code}_"
            if col.startswith(prefix):
                feature_name_set.add(col[len(prefix) :])
                break
    feature_names = sorted(feature_name_set)
    print("基金数量:", len(fund_codes))
    print("总特征数:", len(feature_names))
    print("去基金编号后的特征名:")
    print(feature_names)

    # 将数据层切分结果交给神经网络策略，执行排名监督训练
    artifact, metrics = NeuralRankStrategy.fit_from_labeled_split(
        train_df=train_df,
        valid_df=valid_df,
        rank_n=rank_n,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        l2=l2,
        seed=seed,
    )

    # 输出关键训练信息，便于确认训练是否成功与样本规模是否合理
    print("当前模式:", "train")
    print("训练目标:", f"future_{rank_n}d_rank")
    print("labeled_df形状:", labeled_df.shape)
    print("train_df形状:", train_df.shape)
    print("valid_df形状:", valid_df.shape)
    print("训练样本数:", metrics["train_samples"])
    print("验证样本数:", metrics["valid_samples"])
    print("训练集RankIC:", metrics["train_rank_ic"])
    print("验证集RankIC:", metrics["valid_rank_ic"])
    print("模型参数形状w1:", artifact["model"]["w1"].shape)
    print("模型参数形状w2:", artifact["model"]["w2"].shape)



if __name__ == "__main__":
    # 固定训练入口：无需再通过命令行参数指定模式
    main()
