from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd


def fetch_or_load_fund_data(code_dict, data_dir="data"):
    # 统一按当天日期组织文件名，保证重复执行可复用同一份数据
    today = datetime.today().strftime("%Y-%m-%d")
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    csv_path = data_path / f"fund_nav_{today}.csv"

    # 若当天文件已存在，则直接读取并继续后续流程
    if csv_path.exists():
        print("Today's data already exists, skip download:")
        print(csv_path)
        # 读取时固定为字符串，避免被自动推断成整数后丢失前导0
        data = pd.read_csv(csv_path, dtype={"code": str})
    else:
        print("Start downloading fund data...")
        data_list = []

        # 按基金代码批量下载净值数据，并统一字段后聚合
        for code, name in code_dict.items():
            print("downloading:", name)
            df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
            df = df.rename(columns={"净值日期": "date", "单位净值": "nav"})
            df["fund"] = name
            df["code"] = code
            data_list.append(df[["date", "code", "fund", "nav"]])

        data = pd.concat(data_list)
        data.to_csv(csv_path, index=False)
        print("Saved to:", csv_path)

    # 无论下载或读取分支，都统一数据类型和排序规则
    data["date"] = pd.to_datetime(data["date"]).dt.date
    # 统一基金代码格式，保证与字符串基金池过滤一致
    data["code"] = data["code"].astype(str).str.zfill(6)
    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    return data
