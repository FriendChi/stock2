import json
from pathlib import Path

import numpy as np
import pandas as pd

from tradition import factor_analysis
def test_run_strategy_advice_outputs_independent_json(monkeypatch, tmp_path):
    strategy_backtest_path = tmp_path / "strategy_backtest_007301_2026-04-09_abcdE1.json"
    factor_combination_path = tmp_path / "factor_combination_007301_2026-04-09_abcdD0.json"
    sample_date_index = pd.date_range("2024-01-01", periods=140, freq="D")
    strategy_backtest_payload = {
        "path_code": "abcdE1",
        "input_ref": {
            "factor_combination_path": str(factor_combination_path),
            "fund_code": "007301",
        },
        "strategy_backtest_output": {
            "fund_code": "007301",
            "preprocess_path": str(tmp_path / "unused_checked.csv"),
            "factor_combination_path": str(factor_combination_path),
            "candidate_label_list": [
                "007301__daily_growth_rate__momentum(window=10)__zscore",
                "512480__price__zscore",
            ],
            "candidate_weight_dict": {
                "007301__daily_growth_rate__momentum(window=10)__zscore": 0.7,
                "512480__price__zscore": 0.3,
            },
            "best_strategy_test_summary": {
                "position_function_name": "sigmoid",
                "position_function_params": {"center": 0.2, "slope": 2.0},
                "ema_span": 2,
                "trade_gate": 0.05,
            },
        },
    }
    strategy_backtest_path.write_text(json.dumps(strategy_backtest_payload, ensure_ascii=False), encoding="utf-8")
    factor_combination_payload = {
        "path_code": "abcdD0",
        "factor_combination_output": {
            "fund_code": "007301",
            "target_nav_column": "007301__cumulative_nav",
            "preprocess_path": str(tmp_path / "unused_checked.csv"),
            "factor_candidate_record_dict": {
                "007301__daily_growth_rate__momentum(window=10)__zscore": {
                    "candidate_label": "007301__daily_growth_rate__momentum(window=10)__zscore",
                },
                "512480__price__zscore": {
                    "candidate_label": "512480__price__zscore",
                },
            },
            "best_combination_selection_summary": {
                "candidate_label_list": [
                    "007301__daily_growth_rate__momentum(window=10)__zscore",
                    "512480__price__zscore",
                ],
                "candidate_weight_dict": {
                    "007301__daily_growth_rate__momentum(window=10)__zscore": 0.7,
                    "512480__price__zscore": 0.3,
                },
            },
        },
    }
    factor_combination_path.write_text(json.dumps(factor_combination_payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(
        factor_analysis.backtest,
        "build_tradition_config",
        lambda config_override=None: {
            "output_dir": tmp_path,
            "data_dir": tmp_path,
            "strategy_backtest_path": str(strategy_backtest_path),
            "default_fund_code": "007301",
            "linked_code_dict": {"007301": ["512480", "000510"]},
            "code_type_dict": {"007301": "fund", "512480": "fund", "000510": "index"},
            "force_refresh": False,
            "strategy_param_dict": {"multi_factor_score": {"score_window": 60}},
        },
    )
    monkeypatch.setattr(
        factor_analysis.backtest,
        "_fetch_feature_df_with_cache",
        lambda ak_module, code, code_type, cache_dir, force_refresh=False: (
            pd.DataFrame(
                {
                        "date": sample_date_index,
                        "price": np.linspace(1.0, 1.6, len(sample_date_index)),
                        "daily_growth_rate": np.linspace(0.1, 0.5, len(sample_date_index)),
                        "cumulative_nav": np.linspace(1.0, 1.6, len(sample_date_index)),
                    }
                )
                if str(code) == "007301"
                else (
                    pd.DataFrame(
                        {
                            "date": sample_date_index,
                            "price": np.linspace(2.0, 2.8, len(sample_date_index)),
                            "daily_growth_rate": np.linspace(0.05, 0.25, len(sample_date_index)),
                        }
                    )
                    if str(code) == "512480"
                    else pd.DataFrame(
                        {
                            "date": sample_date_index,
                            "price": np.linspace(3.0, 3.7, len(sample_date_index)),
                            "volume": np.linspace(100, 240, len(sample_date_index)),
                        }
                    )
                )
            ),
        )
    monkeypatch.setattr(
        factor_analysis.backtest,
        "_resolve_remote_feature_latest_date_dict",
        lambda code_type_dict: {
            "007301": pd.Timestamp("2024-05-19"),
            "512480": pd.Timestamp("2024-05-19"),
            "000510": pd.Timestamp("2024-05-19"),
        },
    )

    result = factor_analysis.run_strategy_advice(config_override={})

    assert result["fund_code"] == "007301"
    assert result["latest_date"] == "2024-05-19"
    assert result["previous_date"] == "2024-05-18"
    assert result["action"] in {"加仓", "减仓", "持有"}
    assert result["trade_gate"] == 0.05
    assert len(result["top_positive_contributors"]) > 0
    assert "007301__daily_growth_rate" in result["required_source_column_list"]
    assert "512480__price" in result["required_source_column_list"]
    assert Path(result["latest_wide_feature_path"]).exists()
    assert Path(result["latest_wide_feature_path"]).parent == tmp_path
    assert result["used_cached_wide_feature_snapshot"] is False
    assert result["latest_wide_feature_snapshot_date"] == "2024-05-19"
    assert result["remote_feature_latest_date_dict"]["007301"] == "2024-05-19"
    output_path = result["summary_path"]
    assert output_path.name.startswith("strategy_advice_007301_")
    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_payload["input_ref"]["fund_code"] == "007301"
    assert Path(output_payload["strategy_advice_output"]["latest_wide_feature_path"]).exists()
    assert output_payload["strategy_advice_output"]["used_cached_wide_feature_snapshot"] is False
    assert output_payload["strategy_advice_output"]["latest_date"] == "2024-05-19"
    assert output_payload["strategy_advice_output"]["previous_date"] == "2024-05-18"
    assert output_payload["strategy_advice_output"]["candidate_weight_dict"]["007301__daily_growth_rate__momentum(window=10)__zscore"] == 0.7

def test_run_strategy_advice_reuses_fresh_wide_feature_snapshot(monkeypatch, tmp_path):
    strategy_backtest_path = tmp_path / "strategy_backtest_007301_2026-04-09_abcdE1.json"
    factor_combination_path = tmp_path / "factor_combination_007301_2026-04-09_abcdD0.json"
    sample_date_index = pd.date_range("2024-01-01", periods=140, freq="D")
    snapshot_path = tmp_path / "strategy_advice_latest_wide_feature_007301.csv"
    snapshot_df = pd.DataFrame(
        {
            "date": sample_date_index.strftime("%Y-%m-%d"),
            "007301__daily_growth_rate": np.linspace(0.1, 0.5, len(sample_date_index)),
            "512480__price": np.linspace(2.0, 2.8, len(sample_date_index)),
            "000510__price": np.linspace(3.0, 3.7, len(sample_date_index)),
        }
    )
    snapshot_df.to_csv(snapshot_path, index=False)
    strategy_backtest_payload = {
        "path_code": "abcdE1",
        "input_ref": {
            "factor_combination_path": str(factor_combination_path),
            "fund_code": "007301",
        },
        "strategy_backtest_output": {
            "fund_code": "007301",
            "preprocess_path": str(tmp_path / "unused_checked.csv"),
            "factor_combination_path": str(factor_combination_path),
            "candidate_label_list": [
                "007301__daily_growth_rate__momentum(window=10)__zscore",
                "512480__price__zscore",
            ],
            "candidate_weight_dict": {
                "007301__daily_growth_rate__momentum(window=10)__zscore": 0.7,
                "512480__price__zscore": 0.3,
            },
            "best_strategy_test_summary": {
                "position_function_name": "sigmoid",
                "position_function_params": {"center": 0.2, "slope": 2.0},
                "ema_span": 2,
                "trade_gate": 0.05,
            },
        },
    }
    strategy_backtest_path.write_text(json.dumps(strategy_backtest_payload, ensure_ascii=False), encoding="utf-8")
    factor_combination_payload = {
        "path_code": "abcdD0",
        "factor_combination_output": {
            "fund_code": "007301",
            "target_nav_column": "007301__cumulative_nav",
            "preprocess_path": str(tmp_path / "unused_checked.csv"),
            "factor_candidate_record_dict": {
                "007301__daily_growth_rate__momentum(window=10)__zscore": {
                    "candidate_label": "007301__daily_growth_rate__momentum(window=10)__zscore",
                },
                "512480__price__zscore": {
                    "candidate_label": "512480__price__zscore",
                },
            },
            "best_combination_selection_summary": {
                "candidate_label_list": [
                    "007301__daily_growth_rate__momentum(window=10)__zscore",
                    "512480__price__zscore",
                ],
                "candidate_weight_dict": {
                    "007301__daily_growth_rate__momentum(window=10)__zscore": 0.7,
                    "512480__price__zscore": 0.3,
                },
            },
        },
    }
    factor_combination_path.write_text(json.dumps(factor_combination_payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(
        factor_analysis.backtest,
        "build_tradition_config",
        lambda config_override=None: {
            "output_dir": tmp_path,
            "data_dir": tmp_path,
            "strategy_backtest_path": str(strategy_backtest_path),
            "default_fund_code": "007301",
            "linked_code_dict": {"007301": ["512480", "000510"]},
            "code_type_dict": {"007301": "fund", "512480": "fund", "000510": "index"},
            "force_refresh": False,
            "strategy_param_dict": {"multi_factor_score": {"score_window": 60}},
        },
    )
    monkeypatch.setattr(
        factor_analysis.backtest,
        "_fetch_feature_df_with_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("命中宽表快照时不应再次回源更新")),
    )
    monkeypatch.setattr(
        factor_analysis.backtest,
        "_resolve_remote_feature_latest_date_dict",
        lambda code_type_dict: {
            "007301": pd.Timestamp("2024-05-19"),
            "512480": pd.Timestamp("2024-05-19"),
            "000510": pd.Timestamp("2024-05-19"),
        },
    )

    result = factor_analysis.run_strategy_advice(config_override={})

    assert result["used_cached_wide_feature_snapshot"] is True
    assert result["latest_wide_feature_snapshot_date"] == "2024-05-19"
