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


def build_walk_forward_fold_list(price_series, walk_forward_config, split_config):
    # walk-forward 按总窗口滚动，并在每折内部复用统一的比例切分逻辑。
    if not isinstance(walk_forward_config, dict):
        raise ValueError("walk_forward_config 必须为dict。")
    if not isinstance(split_config, dict):
        raise ValueError("split_config 必须为dict。")

    series = pd.Series(price_series, copy=True).dropna()
    if len(series) == 0:
        raise ValueError("price_series 为空，无法生成 walk-forward 折。")

    window_size = int(walk_forward_config["window_size"])
    step_size = int(walk_forward_config["step_size"])
    min_fold_count = int(walk_forward_config["min_fold_count"])
    if window_size <= 0 or step_size <= 0:
        raise ValueError("walk-forward 窗口长度必须为正整数。")

    fold_list = []
    total_size = len(series)
    fold_start = 0
    fold_id = 1
    while True:
        window_end = fold_start + window_size
        if window_end > total_size:
            break

        window_series = series.iloc[fold_start:window_end].copy()
        split_dict = split_time_series_by_ratio(price_series=window_series, split_config=split_config)
        train_series = split_dict["train"]
        valid_series = split_dict["valid"]
        test_series = split_dict["test"]
        fold_list.append(
            {
                "fold_id": fold_id,
                "train": train_series,
                "valid": valid_series,
                "test": test_series,
                "train_start": train_series.index.min(),
                "train_end": train_series.index.max(),
                "valid_start": valid_series.index.min(),
                "valid_end": valid_series.index.max(),
                "test_start": test_series.index.min(),
                "test_end": test_series.index.max(),
            }
        )
        fold_start += step_size
        fold_id += 1

    if len(fold_list) < min_fold_count:
        raise ValueError(f"walk-forward 可用折数不足，要求至少 {min_fold_count} 折，当前仅 {len(fold_list)} 折。")
    return fold_list
