import pandas as pd


def split_time_series_by_ratio(price_series, split_config):
    # 三段式切分严格按时间顺序进行，确保训练、验证、测试各自职责清晰。
    if not isinstance(split_config, dict):
        raise ValueError("split_config 必须为dict。")

    series = pd.Series(price_series, copy=True).dropna()
    if len(series) == 0:
        raise ValueError("price_series 为空，无法切分数据集。")

    train_ratio = float(split_config["train_ratio"])
    valid_ratio = float(split_config["valid_ratio"])
    test_ratio = float(split_config["test_ratio"])
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("train_ratio、valid_ratio、test_ratio 之和必须为 1.0。")

    min_segment_size = int(split_config["min_segment_size"])
    total_size = len(series)
    train_end = int(total_size * train_ratio)
    valid_end = int(total_size * (train_ratio + valid_ratio))

    train_series = series.iloc[:train_end].copy()
    valid_series = series.iloc[train_end:valid_end].copy()
    test_series = series.iloc[valid_end:].copy()

    # 每个区间都必须覆盖最小样本量，否则 Top-K 选参与最终测试都不可靠。
    if len(train_series) < min_segment_size or len(valid_series) < min_segment_size or len(test_series) < min_segment_size:
        raise ValueError(
            f"切分后样本不足，要求每段至少 {min_segment_size} 条，当前 train={len(train_series)}, valid={len(valid_series)}, test={len(test_series)}"
        )

    return {
        "train": train_series,
        "valid": valid_series,
        "test": test_series,
    }
