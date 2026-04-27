import pandas as pd
import pytest

from tradition import factor_analysis
def test_build_forward_return_series_uses_future_simple_return():
    price_series = pd.Series([1.0, 1.1, 1.21, 1.331, 1.4641, 1.61051], index=pd.date_range("2024-01-01", periods=6, freq="D"))
    forward_return_series = factor_analysis.build_forward_return_series(price_series=price_series, forward_window=5)
    assert abs(float(forward_return_series.iloc[0]) - 0.61051) < 1e-12
    assert pd.isna(forward_return_series.iloc[-1])

def test_build_factor_candidate_list_expands_multi_param_combinations():
    candidate_list = factor_analysis.build_factor_candidate_list(
        candidate_factor_name_list=["momentum", "ma_slope"],
        strategy_params={
            "factor_param_dict": {
                "momentum": {"window": 20},
                "ma_slope": {"window": 20, "lookback": 5},
            }
        },
    )
    momentum_candidate_list = [item for item in candidate_list if item["factor_name"] == "momentum"]
    ma_slope_candidate_list = [item for item in candidate_list if item["factor_name"] == "ma_slope"]
    assert len(momentum_candidate_list) == len(range(10, 121, 5))
    assert len(ma_slope_candidate_list) == len(range(10, 61, 5)) * len(range(2, 21, 1))
    assert ma_slope_candidate_list[0]["candidate_label"].startswith("ma_slope(")

def test_build_metric_summary_returns_zero_icir_for_constant_metric_series():
    summary = factor_analysis.build_metric_summary([0.1, 0.1, 0.1])
    assert abs(summary["mean"] - 0.1) < 1e-12
    assert abs(summary["std"] - 0.0) < 1e-12
    assert abs(summary["icir"] - 0.0) < 1e-12

def test_build_spearman_metric_summary_supports_exp_weighted():
    summary = factor_analysis.build_spearman_metric_summary(
        [0.1, 0.2, 0.5],
        ic_aggregation_config={"mode": "exp_weighted", "half_life": 1.0},
    )
    assert summary["count"] == 3
    assert summary["mean"] > 0.2
    assert summary["mean"] < 0.5
    assert summary["icir"] > 0.0

def test_build_positive_ic_ratio_uses_positive_valid_fold_ratio():
    assert abs(factor_analysis.build_positive_ic_ratio([0.1, -0.2, 0.0, 0.3, float("nan")]) - (2 / 4)) < 1e-12
    assert abs(factor_analysis.build_positive_ic_ratio([float("nan")]) - 0.0) < 1e-12

def test_build_trimmed_ic_mean_and_flip_count_follow_stability_rules():
    assert abs(factor_analysis.build_trimmed_ic_mean([1, 2, 3, 100, -50, 4, 5, 6, 7, 8]) - 4.5) < 1e-12
    assert factor_analysis.build_ic_flip_count([0.1, 0.2, -0.1, 0.0, -0.3, 0.4, float("nan")]) == 2
    assert factor_analysis.build_ic_flip_count([0.0, float("nan")]) == 0

def test_build_tail_reject_mask_rejects_union_of_worst_five_percent():
    summary_df = pd.DataFrame(
        [
            {
                "candidate_label": f"factor_{idx}",
                "train_valid_ic_mean_gap": 0.001 * idx,
                "valid_ic_flip_count": idx,
                "valid_trimmed_ic_gap": 0.002 * idx,
            }
            for idx in range(20)
        ]
    )
    tail_reject_flag_df = factor_analysis.build_tail_reject_mask(summary_df=summary_df, reject_ratio=0.05)
    assert bool(tail_reject_flag_df.loc[19, "abs_train_valid_ic_mean_gap_tail_rejected"]) is True
    assert bool(tail_reject_flag_df.loc[19, "valid_ic_flip_count_tail_rejected"]) is True
    assert bool(tail_reject_flag_df.loc[19, "abs_valid_trimmed_ic_gap_tail_rejected"]) is True
    assert bool(tail_reject_flag_df.loc[19, "stability_tail_rejected"]) is True
    assert int(tail_reject_flag_df["stability_tail_rejected"].sum()) == 1

def test_resolve_factor_group_name_raises_for_unknown_factor():
    with pytest.raises(ValueError, match="未定义因子"):
        factor_analysis.resolve_factor_group_name("unknown_factor")
