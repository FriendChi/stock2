import numpy as np
import pandas as pd


class BuyAndHoldStrategy:
    @classmethod
    def generate_signals(cls, price_df, **params):
        # 第一日全量买入，后续不再产生入场与离场信号
        entries = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
        exits = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
        if len(price_df) > 0:
            entries.iloc[0, :] = True
        return entries.astype(bool), exits.astype(bool)


class MACrossStrategy:
    @classmethod
    def generate_signals(cls, price_df, fast=5, slow=20, t_plus_one=False, **params):
        # 计算快慢均线并生成原始入场与离场条件
        fast_ma = price_df.rolling(fast).mean()
        slow_ma = price_df.rolling(slow).mean()
        entry_raw = fast_ma > slow_ma
        exit_raw = fast_ma < slow_ma

        # 用shift的fill_value避免fillna在object列上的隐式降类型告警
        prev_entry = entry_raw.shift(1, fill_value=False).astype(bool)
        # exit信号同样使用fill_value保持布尔语义稳定
        prev_exit = exit_raw.shift(1, fill_value=False).astype(bool)
        entries = entry_raw & (~prev_entry)
        exits = exit_raw & (~prev_exit)

        if t_plus_one:
            # T+1时序平移后直接填False，避免后续fillna告警
            entries = entries.shift(1, fill_value=False).astype(bool)
            # exits与entries保持同一处理策略
            exits = exits.shift(1, fill_value=False).astype(bool)
        return entries.astype(bool), exits.astype(bool)


class MomentumStrategy:
    @classmethod
    def generate_signals(cls, price_df, window=20, t_plus_one=False, **params):
        # 基于窗口收益率构造动量入场与离场条件
        momentum = price_df.pct_change(window)
        entry_raw = momentum > 0
        exit_raw = momentum <= 0

        # 用shift的fill_value避免fillna在object列上的隐式降类型告警
        prev_entry = entry_raw.shift(1, fill_value=False).astype(bool)
        # exit信号同样使用fill_value保持布尔语义稳定
        prev_exit = exit_raw.shift(1, fill_value=False).astype(bool)
        entries = entry_raw & (~prev_entry)
        exits = exit_raw & (~prev_exit)

        if t_plus_one:
            # T+1时序平移后直接填False，避免后续fillna告警
            entries = entries.shift(1, fill_value=False).astype(bool)
            # exits与entries保持同一处理策略
            exits = exits.shift(1, fill_value=False).astype(bool)
        return entries.astype(bool), exits.astype(bool)


class NeuralRankStrategy:
    @classmethod
    def _build_samples(cls, returns_df, lookback):
        # 构建样本：用过去lookback日收益率预测下一日收益率
        x_list = []
        y_list = []
        pred_row_list = []
        pred_col_list = []
        pred_pos_list = []
        n_rows = len(returns_df.index)
        if n_rows <= lookback + 1:
            return None, None, None, None, None

        for pred_pos in range(lookback, n_rows):
            feat_end = pred_pos - 1
            feat_start = feat_end - lookback + 1
            if feat_start < 0:
                continue
            for col in returns_df.columns:
                feat = returns_df.iloc[feat_start : feat_end + 1][col].values.astype(float)
                target = float(returns_df.iloc[pred_pos][col])
                if np.isnan(feat).any() or np.isnan(target):
                    continue
                x_list.append(feat)
                y_list.append(target)
                pred_row_list.append(returns_df.index[pred_pos])
                pred_col_list.append(col)
                pred_pos_list.append(pred_pos)

        if len(x_list) == 0:
            return None, None, None, None, None
        x = np.asarray(x_list, dtype=float)
        y = np.asarray(y_list, dtype=float).reshape(-1, 1)
        pred_pos = np.asarray(pred_pos_list, dtype=int)
        return x, y, pred_row_list, pred_col_list, pred_pos

    @classmethod
    def train_model(
        cls,
        x_train,
        y_train,
        hidden_dim=16,
        epochs=200,
        lr=0.01,
        l2=0.0,
        seed=42,
    ):
        # 训练一个单隐层MLP回归器，用于预测下一日收益率
        rng = np.random.default_rng(seed)
        input_dim = x_train.shape[1]
        w1 = rng.normal(0, 0.1, size=(input_dim, hidden_dim))
        b1 = np.zeros((1, hidden_dim))
        w2 = rng.normal(0, 0.1, size=(hidden_dim, 1))
        b2 = np.zeros((1, 1))

        for _ in range(epochs):
            z1 = x_train @ w1 + b1
            a1 = np.maximum(0.0, z1)
            y_pred = a1 @ w2 + b2

            err = y_pred - y_train
            d_y_pred = 2.0 * err / x_train.shape[0]

            d_w2 = a1.T @ d_y_pred + l2 * w2
            d_b2 = np.sum(d_y_pred, axis=0, keepdims=True)

            d_a1 = d_y_pred @ w2.T
            d_z1 = d_a1 * (z1 > 0).astype(float)
            d_w1 = x_train.T @ d_z1 + l2 * w1
            d_b1 = np.sum(d_z1, axis=0, keepdims=True)

            w1 -= lr * d_w1
            b1 -= lr * d_b1
            w2 -= lr * d_w2
            b2 -= lr * d_b2

        return {
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
        }

    @classmethod
    def predict_scores(cls, model, x):
        # 使用训练好的MLP参数输出收益率预测值
        z1 = x @ model["w1"] + model["b1"]
        a1 = np.maximum(0.0, z1)
        y_pred = a1 @ model["w2"] + model["b2"]
        return y_pred.reshape(-1)

    @classmethod
    def _build_rank_samples_from_labeled_table(cls, labeled_df, rank_n):
        # 标签后缀固定为_future_{n}d_rank，和数据层命名保持一致
        rank_suffix = f"_future_{int(rank_n)}d_rank"
        # 先定位全部标签列，若为空直接报错避免静默训练空样本
        rank_cols = [col for col in labeled_df.columns if str(col).endswith(rank_suffix)]
        if len(rank_cols) == 0:
            raise ValueError(f"未找到排名标签列，期望后缀={rank_suffix}")

        # 从标签列名提取基金代码，后续按基金代码回收对应特征列
        fund_codes = [str(col).replace(rank_suffix, "") for col in rank_cols]
        # 收集特征后缀，确保所有基金使用相同语义的特征集合
        feat_suffix_set = set()
        for col in labeled_df.columns:
            col_str = str(col)
            # 标签列不参与特征集合构建
            if col_str in rank_cols:
                continue
            # 仅收集以“基金代码_”开头的宽表特征列
            for code in fund_codes:
                prefix = f"{code}_"
                if col_str.startswith(prefix):
                    feat_suffix_set.add(col_str[len(prefix) :])
                    break
        # 固定排序，保证不同运行次序下特征列顺序稳定
        feat_suffix_list = sorted(list(feat_suffix_set))
        if len(feat_suffix_list) == 0:
            raise ValueError("未找到可用特征列，无法构建排名训练样本。")

        # 逐基金构建对应特征列名，确保每个基金拿到同构特征向量
        fund_feature_cols = {}
        for code in fund_codes:
            fund_cols = [f"{code}_{suffix}" for suffix in feat_suffix_list]
            # 若某基金缺少任一特征列，提前失败避免训练阶段列错位
            missing_cols = [c for c in fund_cols if c not in labeled_df.columns]
            if len(missing_cols) > 0:
                raise ValueError(f"基金{code}缺少特征列，示例缺失={missing_cols[:3]}")
            fund_feature_cols[code] = fund_cols

        # 逐日逐基金展开样本：X为该基金特征，y为该基金未来n日排名标签
        x_list = []
        y_rank_list = []
        date_list = []
        code_list = []
        for row_idx in labeled_df.index:
            row_data = labeled_df.loc[row_idx]
            for code in fund_codes:
                label_col = f"{code}{rank_suffix}"
                # 排名标签缺失时跳过该样本，保持训练样本干净
                rank_value = row_data[label_col]
                if pd.isna(rank_value):
                    continue
                feat_values = row_data[fund_feature_cols[code]].values.astype(float)
                # 任一特征缺失则跳过，避免向模型输入NaN
                if np.isnan(feat_values).any():
                    continue
                x_list.append(feat_values)
                y_rank_list.append(float(rank_value))
                date_list.append(row_idx)
                code_list.append(code)

        # 全部为空时返回None，由上层统一处理可读错误
        if len(x_list) == 0:
            return None, None, None, None
        x = np.asarray(x_list, dtype=float)
        y_rank = np.asarray(y_rank_list, dtype=float).reshape(-1, 1)
        return x, y_rank, date_list, code_list

    @classmethod
    def train_rank_model(
        cls,
        x_train,
        y_rank_train,
        hidden_dim=16,
        epochs=200,
        lr=0.01,
        l2=0.0,
        seed=42,
    ):
        # 将“名次越小越好”转换为“分数越大越好”，用于监督回归训练
        y_rank_safe = np.maximum(y_rank_train.astype(float), 1.0)
        y_score = 1.0 / y_rank_safe
        # 复用现有MLP训练逻辑，最小改动接入排名监督标签
        model = cls.train_model(
            x_train=x_train,
            y_train=y_score,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=lr,
            l2=l2,
            seed=seed,
        )
        return model

    @classmethod
    def _compute_rank_ic(cls, date_list, pred_score, y_rank_true):
        # 无样本时直接返回NaN，避免空分组导致异常
        if len(date_list) == 0:
            return np.nan

        # 按日期聚合预测值与真实排名，计算逐日秩相关后取均值
        grouped = {}
        for i, day in enumerate(date_list):
            grouped.setdefault(day, {"pred": [], "rank": []})
            grouped[day]["pred"].append(float(pred_score[i]))
            grouped[day]["rank"].append(float(y_rank_true[i]))

        # 逐日将“预测分数降序名次”与“真实排名升序名次”做相关
        daily_ic = []
        for day in grouped:
            pred_values = np.asarray(grouped[day]["pred"], dtype=float)
            rank_values = np.asarray(grouped[day]["rank"], dtype=float)
            # 至少两个基金样本才有秩相关意义
            if len(pred_values) < 2:
                continue
            pred_rank = pd.Series(pred_values).rank(ascending=False, method="average").values
            true_rank = pd.Series(rank_values).rank(ascending=True, method="average").values
            # 常量序列会导致相关系数不可定义，直接跳过该日
            if np.std(pred_rank) < 1e-12 or np.std(true_rank) < 1e-12:
                continue
            corr = np.corrcoef(pred_rank, true_rank)[0, 1]
            if np.isfinite(corr):
                daily_ic.append(float(corr))

        # 若无有效日期，返回NaN提示评估样本不足
        if len(daily_ic) == 0:
            return np.nan
        return float(np.mean(daily_ic))

    @classmethod
    def fit_from_labeled_split(
        cls,
        train_df,
        valid_df,
        rank_n=7,
        hidden_dim=16,
        epochs=200,
        lr=0.01,
        l2=0.0,
        seed=42,
    ):
        # 从数据层产物中提取训练样本，标签为未来n日收益排名
        x_train, y_rank_train, train_dates, _ = cls._build_rank_samples_from_labeled_table(
            labeled_df=train_df,
            rank_n=rank_n,
        )
        if x_train is None:
            raise ValueError("训练集无法构建有效排名样本。")

        # 验证集同样展开为样本，用于训练后快速校验有效性
        x_valid, y_rank_valid, valid_dates, _ = cls._build_rank_samples_from_labeled_table(
            labeled_df=valid_df,
            rank_n=rank_n,
        )
        if x_valid is None:
            raise ValueError("验证集无法构建有效排名样本。")

        # 仅使用训练集统计量做标准化，防止验证信息泄露到训练过程
        x_mean = np.mean(x_train, axis=0, keepdims=True)
        x_std = np.std(x_train, axis=0, keepdims=True)
        x_std = np.where(x_std < 1e-8, 1.0, x_std)
        x_train_std = (x_train - x_mean) / x_std
        x_valid_std = (x_valid - x_mean) / x_std

        # 使用排名标签监督训练模型，输出可用于排序的预测分数
        model = cls.train_rank_model(
            x_train=x_train_std,
            y_rank_train=y_rank_train,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=lr,
            l2=l2,
            seed=seed,
        )
        train_pred = cls.predict_scores(model=model, x=x_train_std)
        valid_pred = cls.predict_scores(model=model, x=x_valid_std)

        # 汇总训练元信息，便于脚本侧打印与后续模型落地
        metrics = {
            "train_samples": int(x_train.shape[0]),
            "valid_samples": int(x_valid.shape[0]),
            "feature_dim": int(x_train.shape[1]),
            "train_rank_ic": cls._compute_rank_ic(
                date_list=train_dates,
                pred_score=train_pred,
                y_rank_true=y_rank_train.reshape(-1),
            ),
            "valid_rank_ic": cls._compute_rank_ic(
                date_list=valid_dates,
                pred_score=valid_pred,
                y_rank_true=y_rank_valid.reshape(-1),
            ),
        }
        # 返回模型与标准化统计量，支持训练后直接用于推理
        artifact = {
            "model": model,
            "x_mean": x_mean,
            "x_std": x_std,
            "rank_n": int(rank_n),
        }
        return artifact, metrics

    @classmethod
    def generate_signals(
        cls,
        price_df,
        lookback=20,
        hidden_dim=16,
        epochs=200,
        lr=0.01,
        l2=0.0,
        train_ratio=0.8,
        top_k=3,
        min_pred_return=0.0,
        t_plus_one=False,
        seed=42,
        **params,
    ):
        # 基于收益率序列构建样本并训练神经网络预测各基金下一日收益率
        entries = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
        exits = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
        if len(price_df.index) <= lookback + 1 or len(price_df.columns) == 0:
            return entries.astype(bool), exits.astype(bool)

        returns_df = price_df.pct_change()
        x_all, y_all, pred_row_list, pred_col_list, pred_pos = cls._build_samples(
            returns_df=returns_df,
            lookback=lookback,
        )
        if x_all is None:
            return entries.astype(bool), exits.astype(bool)

        # 按时间位置切训练样本，避免训练集使用未来时点标签
        train_end_pos = int(len(price_df.index) * train_ratio)
        train_mask = pred_pos < train_end_pos
        if np.sum(train_mask) < 10:
            return entries.astype(bool), exits.astype(bool)

        x_train = x_all[train_mask]
        y_train = y_all[train_mask]

        # 训练集特征标准化并复用于全量样本，保证训练与推理尺度一致
        x_mean = np.mean(x_train, axis=0, keepdims=True)
        x_std = np.std(x_train, axis=0, keepdims=True)
        x_std = np.where(x_std < 1e-8, 1.0, x_std)
        x_train_std = (x_train - x_mean) / x_std
        x_all_std = (x_all - x_mean) / x_std

        model = cls.train_model(
            x_train=x_train_std,
            y_train=y_train,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=lr,
            l2=l2,
            seed=seed,
        )
        pred_all = cls.predict_scores(model=model, x=x_all_std)

        # 将预测值回填到按日期和基金对齐的分数矩阵，便于横截面排名
        score_df = pd.DataFrame(np.nan, index=price_df.index, columns=price_df.columns)
        for i in range(len(pred_all)):
            row_idx = pred_row_list[i]
            col_name = pred_col_list[i]
            score_df.at[row_idx, col_name] = pred_all[i]

        # 每个交易日按预测收益率排名选取前top_k基金作为持仓候选
        hold_df = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
        for row_idx in score_df.index:
            row_score = score_df.loc[row_idx]
            row_score = row_score.dropna()
            if len(row_score) == 0:
                continue
            top_n = int(max(0, min(top_k, len(row_score))))
            if top_n == 0:
                continue
            selected = row_score.nlargest(top_n)
            selected = selected[selected > min_pred_return]
            if len(selected.index) > 0:
                hold_df.loc[row_idx, selected.index] = True

        # 从持仓状态序列生成入场和离场信号，保持与其他策略一致的布尔语义
        prev_hold = hold_df.shift(1, fill_value=False).astype(bool)
        entries = hold_df & (~prev_hold)
        exits = (~hold_df) & prev_hold

        if t_plus_one:
            # T+1模式下整体平移信号，避免未来信息同日成交
            entries = entries.shift(1, fill_value=False).astype(bool)
            exits = exits.shift(1, fill_value=False).astype(bool)
        return entries.astype(bool), exits.astype(bool)


strategy_dict = {
    "buy_and_hold": BuyAndHoldStrategy.generate_signals,
    "ma_cross": MACrossStrategy.generate_signals,
    "momentum": MomentumStrategy.generate_signals,
    "nn_rank": NeuralRankStrategy.generate_signals,
}
