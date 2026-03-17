from compute_return import add_return_column
from config import code_dict
from fetch_fund_data import fetch_or_load_fund_data


def main():
    # 串联数据准备与收益率计算流程，保持与 notebook 原有输出一致
    data = fetch_or_load_fund_data(code_dict=code_dict, data_dir="data")
    data = add_return_column(data)

    print(type(data))
    print(data.head())
    print(data.dtypes)
    print(data[data["code"] == "007301"].head())


if __name__ == "__main__":
    main()
