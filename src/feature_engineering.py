import pandas as pd


class DataLayer:


    @classmethod
    def build_aligned_price_df(cls, data, fund_list):
        # 先过滤基金池，再透视成多基金宽表并按日期排序
        filtered = data[data["code"].isin(fund_list)].copy()
        # 过滤后为空时，直接给出可读错误，避免后续nan切片异常
        if filtered.empty:
            raise ValueError(
                "过滤后无数据：请检查code类型或fund_list取值。"
                f" data中示例code={data['code'].astype(str).head(5).tolist()}"
            )
        price_df = filtered.pivot(index="date", columns="code", values="nav").sort_index()
        price_df = price_df.dropna(how="all")
        # 透视后为空也提前失败，避免进入公共区间计算
        if price_df.empty:
            raise ValueError("透视后无有效净值数据，无法构建宽表。")

        # 对齐到所有基金都有数据的公共时间区间
        start_dates = price_df.apply(lambda s: s.first_valid_index())
        end_dates = price_df.apply(lambda s: s.last_valid_index())
        common_start = start_dates.max()
        common_end = end_dates.min()
        print("各基金公共起点:", common_start)
        print("各基金公共终点:", common_end)
        # 公共区间不可用时，显式报错并给出基金列信息
        if pd.isna(common_start) or pd.isna(common_end):
            raise ValueError(
                "基金公共区间为空，无法对齐。"
                f" 当前基金列={price_df.columns.astype(str).tolist()}"
            )

        price_df_aligned = price_df.loc[common_start:common_end].copy()
        price_df_aligned = price_df_aligned.dropna(how="any")
        # 对齐后为空时直接提示，便于定位是区间过窄还是缺失过多
        if price_df_aligned.empty:
            raise ValueError("对齐并去缺失后无数据，请调整基金池或时间范围。")
        print(price_df_aligned.head())
        print(price_df_aligned.shape)
        return price_df_aligned

    @classmethod
    def build_feature_dict(cls, price_df):
        feature_dict = {}

        # 基础收益类特征
        ret_1 = price_df.pct_change(1)
        ret_5 = price_df.pct_change(5)
        ret_10 = price_df.pct_change(10)
        ret_20 = price_df.pct_change(20)
        ret_60 = price_df.pct_change(60)
        feature_dict["ret_1"] = ret_1
        feature_dict["ret_5"] = ret_5
        feature_dict["ret_10"] = ret_10
        feature_dict["ret_20"] = ret_20
        feature_dict["ret_60"] = ret_60

        # 动量类特征
        momentum_5 = price_df / price_df.shift(5) - 1
        momentum_20 = price_df / price_df.shift(20) - 1
        momentum_60 = price_df / price_df.shift(60) - 1
        momentum_120 = price_df / price_df.shift(120) - 1
        feature_dict["momentum_5"] = momentum_5
        feature_dict["momentum_20"] = momentum_20
        feature_dict["momentum_60"] = momentum_60
        feature_dict["momentum_120"] = momentum_120

        # 波动率类特征
        daily_ret = price_df.pct_change()
        vol_5 = daily_ret.rolling(5).std()
        vol_10 = daily_ret.rolling(10).std()
        vol_20 = daily_ret.rolling(20).std()
        vol_60 = daily_ret.rolling(60).std()
        feature_dict["vol_5"] = vol_5
        feature_dict["vol_10"] = vol_10
        feature_dict["vol_20"] = vol_20
        feature_dict["vol_60"] = vol_60

        # 均线/趋势类特征
        ma_5 = price_df.rolling(5).mean()
        ma_10 = price_df.rolling(10).mean()
        ma_20 = price_df.rolling(20).mean()
        ma_60 = price_df.rolling(60).mean()
        ma_120 = price_df.rolling(120).mean()
        feature_dict["ma_ratio_5"] = price_df / ma_5 - 1
        feature_dict["ma_ratio_10"] = price_df / ma_10 - 1
        feature_dict["ma_ratio_20"] = price_df / ma_20 - 1
        feature_dict["ma_ratio_60"] = price_df / ma_60 - 1
        feature_dict["ma_ratio_120"] = price_df / ma_120 - 1
        feature_dict["ma_cross_5_20"] = ma_5 / ma_20 - 1
        feature_dict["ma_cross_10_60"] = ma_10 / ma_60 - 1
        feature_dict["ma_cross_20_120"] = ma_20 / ma_120 - 1

        # 回撤/位置类特征
        rolling_max_20 = price_df.rolling(20).max()
        rolling_max_60 = price_df.rolling(60).max()
        rolling_max_120 = price_df.rolling(120).max()
        drawdown_20 = price_df / rolling_max_20 - 1
        drawdown_60 = price_df / rolling_max_60 - 1
        drawdown_120 = price_df / rolling_max_120 - 1
        feature_dict["drawdown_20"] = drawdown_20
        feature_dict["drawdown_60"] = drawdown_60
        feature_dict["drawdown_120"] = drawdown_120

        rolling_min_20 = price_df.rolling(20).min()
        rolling_min_60 = price_df.rolling(60).min()
        feature_dict["position_20"] = (price_df - rolling_min_20) / (rolling_max_20 - rolling_min_20)
        feature_dict["position_60"] = (price_df - rolling_min_60) / (rolling_max_60 - rolling_min_60)
        return feature_dict

    @classmethod
    def add_future_return_rank_feature(cls, price_df, n=7):
        # 校验未来收益窗口参数，避免非正整数导致shift语义错误
        n = int(n)
        if n <= 0:
            raise ValueError(f"n必须为正整数，当前n={n}")

        # 计算未来n日收益率，并在每个交易日做基金横截面排名
        future_ret = price_df.shift(-n) / price_df - 1
        future_rank = future_ret.rank(axis=1, ascending=False, method="average")

        # 为新增特征列增加后缀，和原净值列并存输出
        rank_col_map = {col: f"{col}_future_{n}d_rank" for col in future_rank.columns}
        future_rank = future_rank.rename(columns=rank_col_map)
        output_df = pd.concat([price_df.copy(), future_rank], axis=1)
        return output_df

    @classmethod
    def build_feature_table(cls, price_df):
        # 将特征字典展平为单表结构，便于与标签按时间拼接
        feature_dict = cls.build_feature_dict(price_df)
        feature_table_list = []
        for feat_name, feat_df in feature_dict.items():
            # 对每个特征按基金列加后缀，避免不同特征列名冲突
            renamed = feat_df.rename(columns={col: f"{col}_{feat_name}" for col in feat_df.columns})
            feature_table_list.append(renamed)
        if len(feature_table_list) == 0:
            return pd.DataFrame(index=price_df.index)
        feature_table = pd.concat(feature_table_list, axis=1)
        return feature_table

    @classmethod
    def build_labeled_table(cls, price_df, n=7, dropna=True):
        # 构建特征表并叠加未来n日收益率排名标签列
        feature_table = cls.build_feature_table(price_df)
        rank_augmented_df = cls.add_future_return_rank_feature(price_df=price_df, n=n)
        rank_cols = [col for col in rank_augmented_df.columns if str(col).endswith(f"_future_{int(n)}d_rank")]
        rank_table = rank_augmented_df[rank_cols].copy()
        labeled_table = pd.concat([feature_table, rank_table], axis=1)
        # 按你的方案在拆分前统一去除包含缺失值的时间点
        if dropna:
            labeled_table = labeled_table.dropna(how="any")
        return labeled_table

    @classmethod
    def run_pipeline(
        cls,
        data,
        fund_list,
        scheme_id,
        train_ratio=0.8,
        rank_n=7,
        dropna=True,
    ):
        # 统一入口先构建公共对齐价格表，再按方案分支处理
        price_df_aligned = cls.build_aligned_price_df(data=data, fund_list=fund_list)

        if int(scheme_id) == 1:
            # 方案1：对齐后直接按时间顺序切分训练与验证
            split_result = cls.split_train_valid(
                price_df=price_df_aligned,
                train_ratio=train_ratio,
                valid_context_window=0,
            )
            return {
                "scheme_id": 1,
                "price_df_aligned": price_df_aligned,
                "processed_df": price_df_aligned,
                "split_result": split_result,
            }

        if int(scheme_id) == 2:
            # 方案2：对齐后构建特征与未来收益排名标签，再去NaN后切分
            feature_dict = cls.build_feature_dict(price_df_aligned)
            processed_df = cls.build_labeled_table(
                price_df=price_df_aligned,
                n=rank_n,
                dropna=dropna,
            )
            split_result = cls.split_train_valid(
                price_df=processed_df,
                train_ratio=train_ratio,
                valid_context_window=0,
            )
            return {
                "scheme_id": 2,
                "price_df_aligned": price_df_aligned,
                "feature_dict": feature_dict,
                "processed_df": processed_df,
                "split_result": split_result,
            }

        raise ValueError(f"不支持的scheme_id: {scheme_id}，仅支持1或2。")

    @classmethod
    def split_train_valid(cls, price_df, train_ratio=0.8, valid_context_window=0):
        # 按时间顺序做训练与验证切分，默认前80%训练后20%验证
        split_idx = int(len(price_df) * train_ratio)
        if split_idx <= 0 or split_idx >= len(price_df):
            raise ValueError(
                "样本量不足，无法按给定比例切分训练与验证。"
                f" 当前样本数={len(price_df)}"
            )

        train_df = price_df.iloc[:split_idx].copy()
        valid_df = price_df.iloc[split_idx:].copy()

        # 验证段可向前借用历史上下文，仅用于特征构建而不改变评估区间边界
        valid_context_window = max(0, int(valid_context_window))
        valid_start_idx_with_context = max(0, split_idx - valid_context_window)
        valid_df_with_context = price_df.iloc[valid_start_idx_with_context:].copy()

        return {
            "split_idx": split_idx,
            "train_df": train_df,
            "valid_df": valid_df,
            "valid_df_with_context": valid_df_with_context,
        }

    @classmethod
    def split_train_valid_test(
        cls,
        price_df,
        train_ratio=0.8,
        valid_ratio=0.1,
        test_ratio=0.1,
        valid_context_window=0,
        test_context_window=0,
    ):
        # 按时间顺序切分训练/验证/测试，默认比例如80%/10%/10%
        ratio_sum = float(train_ratio + valid_ratio + test_ratio)
        if abs(ratio_sum - 1.0) > 1e-8:
            raise ValueError(
                "train_ratio + valid_ratio + test_ratio 必须等于1.0。"
                f" 当前总和={ratio_sum}"
            )

        n_rows = len(price_df)
        train_end_idx = int(n_rows * train_ratio)
        valid_end_idx = int(n_rows * (train_ratio + valid_ratio))
        if train_end_idx <= 0 or valid_end_idx <= train_end_idx or valid_end_idx >= n_rows:
            raise ValueError(
                "样本量不足，无法按给定比例切分训练/验证/测试。"
                f" 当前样本数={n_rows}"
            )

        train_df = price_df.iloc[:train_end_idx].copy()
        valid_df = price_df.iloc[train_end_idx:valid_end_idx].copy()
        test_df = price_df.iloc[valid_end_idx:].copy()

        # 验证与测试可向前借用上下文，仅用于构建特征窗口
        valid_context_window = max(0, int(valid_context_window))
        valid_start_idx_with_context = max(0, train_end_idx - valid_context_window)
        valid_df_with_context = price_df.iloc[valid_start_idx_with_context:valid_end_idx].copy()

        test_context_window = max(0, int(test_context_window))
        test_start_idx_with_context = max(0, valid_end_idx - test_context_window)
        test_df_with_context = price_df.iloc[test_start_idx_with_context:].copy()

        return {
            "train_end_idx": train_end_idx,
            "valid_end_idx": valid_end_idx,
            "train_df": train_df,
            "valid_df": valid_df,
            "test_df": test_df,
            "valid_df_with_context": valid_df_with_context,
            "test_df_with_context": test_df_with_context,
        }


def build_aligned_price_df(data, fund_list):
    # 保留原函数入口，兼容现有调用方
    return DataLayer.build_aligned_price_df(data=data, fund_list=fund_list)


def build_feature_dict(price_df):
    # 保留原函数入口，兼容现有调用方
    return DataLayer.build_feature_dict(price_df=price_df)
