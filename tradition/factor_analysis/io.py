import json
from datetime import datetime
from pathlib import Path


def save_factor_selection_table(factor_selection_output, output_dir, fund_code):
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    output_path = output_dir / f"factor_selection_{str(fund_code).zfill(6)}_{date_str}.json"
    payload = {
        "factor_selection_output": factor_selection_output,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


def print_factor_selection_summary(result):
    print("因子筛选结果:")
    print("基金代码:", result["fund_code"])
    print("输入因子族:", ",".join(result["factor_group_list"]))
    print("候选参数化因子数量:", len(result["candidate_factor_list"]))
    print("训练集 Spearman IC 阈值:", result["threshold_config"]["train_min_spearman_ic"])
    print("训练集 Spearman ICIR 阈值:", result["threshold_config"]["train_min_spearman_icir"])
    print("训练通过参数化因子数量:", int(result["summary_df"]["train_passed"].astype(bool).sum()))
    print("验证通过参数化因子数量:", int(result["summary_df"]["valid_passed"].astype(bool).sum()))
    print("筛选后参数化因子列表:", result["selected_candidate_label_list"])
    if len(result["selected_summary_df"]) > 0:
        printable_df = result["selected_summary_df"][
            [
                "final_rank",
                "candidate_label",
                "factor_group",
                "train_spearman_positive_ic_ratio",
                "valid_spearman_ic_mean",
                "valid_spearman_icir",
                "valid_pearson_ic_mean",
                "valid_pearson_icir",
            ]
        ].copy()
        print(printable_df.to_string(index=False))
    print("汇总输出:", result["summary_path"])


def load_factor_selection_input(factor_selection_path):
    factor_selection_path = Path(factor_selection_path)
    if not factor_selection_path.exists():
        raise FileNotFoundError(f"factor_select 结果文件不存在: {factor_selection_path}")
    with factor_selection_path.open("r", encoding="utf-8") as input_file:
        factor_selection_input = json.load(input_file)
    if not isinstance(factor_selection_input, dict):
        raise ValueError("factor_select 结果文件必须是顶层字典。")
    factor_selection_output = factor_selection_input.get("factor_selection_output")
    if not isinstance(factor_selection_output, dict):
        raise ValueError("factor_select 结果文件缺少 factor_selection_output 子字典。")
    record_dict = factor_selection_output.get("record_dict")
    if not isinstance(record_dict, dict):
        raise ValueError("factor_select 结果文件缺少 record_dict 子字典。")
    return factor_selection_input, factor_selection_path


def resolve_fund_code_from_factor_selection_path(factor_selection_path):
    factor_selection_path = Path(factor_selection_path)
    path_stem_part_list = factor_selection_path.stem.split("_")
    if len(path_stem_part_list) >= 3 and str(path_stem_part_list[0]) == "factor" and str(path_stem_part_list[1]) == "selection":
        candidate_code = str(path_stem_part_list[2]).strip()
        if len(candidate_code) == 6 and candidate_code.isdigit():
            return candidate_code
    raise ValueError(f"无法从 factor_select 结果文件名解析基金代码: {factor_selection_path.name}")


def resolve_fund_code_from_factor_selection_input(factor_selection_input, factor_selection_path):
    factor_selection_output = dict(factor_selection_input.get("factor_selection_output", {}))
    candidate_code = str(factor_selection_output.get("fund_code", "")).strip()
    if len(candidate_code) > 0:
        candidate_code = candidate_code.zfill(6)
        if len(candidate_code) == 6 and candidate_code.isdigit():
            return candidate_code
    return resolve_fund_code_from_factor_selection_path(factor_selection_path)


def save_single_factor_stability_analysis_output(factor_selection_input, stability_analysis_output, output_dir, fund_code):
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    output_path = output_dir / f"single_factor_stability_{str(fund_code).zfill(6)}_{date_str}.json"
    factor_selection_output = dict(factor_selection_input.get("factor_selection_output", {}))
    payload = {
        "input_ref": {
            "factor_selection_path": str(output_dir / f"factor_selection_{str(fund_code).zfill(6)}_{date_str}.json"),
            "fund_code": str(factor_selection_output.get("fund_code", str(fund_code).zfill(6))),
        },
        "stability_analysis_output": stability_analysis_output,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


def load_stability_analysis_input(stability_analysis_path):
    stability_analysis_path = Path(stability_analysis_path)
    if not stability_analysis_path.exists():
        raise FileNotFoundError(f"稳定性分析结果文件不存在: {stability_analysis_path}")
    with stability_analysis_path.open("r", encoding="utf-8") as input_file:
        stability_analysis_input = json.load(input_file)
    if not isinstance(stability_analysis_input, dict):
        raise ValueError("稳定性分析结果文件必须是顶层字典。")
    stability_analysis_output = stability_analysis_input.get("stability_analysis_output")
    if not isinstance(stability_analysis_output, dict):
        raise ValueError("稳定性分析结果文件缺少 stability_analysis_output 子字典。")
    record_dict = stability_analysis_output.get("record_dict")
    if not isinstance(record_dict, dict):
        raise ValueError("稳定性分析结果文件缺少 record_dict 子字典。")
    return stability_analysis_input, stability_analysis_path


def resolve_fund_code_from_stability_analysis_input(stability_analysis_input, stability_analysis_path):
    stability_analysis_output = dict(stability_analysis_input.get("stability_analysis_output", {}))
    candidate_code = str(stability_analysis_output.get("fund_code", "")).strip()
    if len(candidate_code) > 0:
        candidate_code = candidate_code.zfill(6)
        if len(candidate_code) == 6 and candidate_code.isdigit():
            return candidate_code
    return resolve_fund_code_from_factor_selection_input(
        factor_selection_input=stability_analysis_input,
        factor_selection_path=stability_analysis_path,
    )


def save_factor_combination_output(dedup_selection_input, factor_combination_output, output_dir, fund_code):
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    output_path = output_dir / f"factor_combination_{str(fund_code).zfill(6)}_{date_str}.json"
    payload = {
        "input_ref": {
            "dedup_selection_path": str(factor_combination_output.get("dedup_selection_path", "")),
            "fund_code": str(factor_combination_output.get("fund_code", str(fund_code).zfill(6))),
        },
        "factor_combination_output": factor_combination_output,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


def print_factor_combination_summary(result):
    print("因子组合结果:")
    print("基金代码:", result["fund_code"])
    print("输入去冗余文件:", result["dedup_selection_path"])
    print("输入因子组合:", result["input_candidate_label_list"])
    print("组合对比优胜方法:", result["combination_compare_output"]["selected_method"])
    print("权重微调后组合:", result["best_combination_selection_summary"]["candidate_label_list"])
    print("权重微调后方法:", result["best_combination_selection_summary"]["selected_method"])
    print("汇总输出:", result["summary_path"])


def load_factor_combination_input(factor_combination_path):
    factor_combination_path = Path(factor_combination_path)
    if not factor_combination_path.exists():
        raise FileNotFoundError(f"因子组合结果文件不存在: {factor_combination_path}")
    with factor_combination_path.open("r", encoding="utf-8") as input_file:
        factor_combination_input = json.load(input_file)
    if not isinstance(factor_combination_input, dict):
        raise ValueError("因子组合结果文件必须是顶层字典。")
    factor_combination_output = factor_combination_input.get("factor_combination_output")
    if not isinstance(factor_combination_output, dict):
        raise ValueError("因子组合结果文件缺少 factor_combination_output 子字典。")
    if not isinstance(factor_combination_output.get("best_combination_selection_summary"), dict):
        raise ValueError("因子组合结果文件缺少 factor_combination_output.best_combination_selection_summary。")
    if not isinstance(factor_combination_output.get("factor_candidate_record_dict"), dict):
        raise ValueError("因子组合结果文件缺少 factor_combination_output.factor_candidate_record_dict。")
    return factor_combination_input, factor_combination_path


def resolve_fund_code_from_factor_combination_input(factor_combination_input, factor_combination_path):
    factor_combination_output = dict(factor_combination_input.get("factor_combination_output", {}))
    candidate_code = str(factor_combination_output.get("fund_code", "")).strip()
    if len(candidate_code) > 0:
        candidate_code = candidate_code.zfill(6)
        if len(candidate_code) == 6 and candidate_code.isdigit():
            return candidate_code
    return resolve_fund_code_from_dedup_selection_input(
        dedup_selection_input=factor_combination_input,
        dedup_selection_path=factor_combination_path,
    )


def save_strategy_backtest_output(factor_combination_input, strategy_backtest_output, output_dir, fund_code):
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    output_path = output_dir / f"strategy_backtest_{str(fund_code).zfill(6)}_{date_str}.json"
    payload = {
        "input_ref": {
            "factor_combination_path": str(strategy_backtest_output.get("factor_combination_path", "")),
            "fund_code": str(strategy_backtest_output.get("fund_code", str(fund_code).zfill(6))),
        },
        "strategy_backtest_output": strategy_backtest_output,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


def print_strategy_backtest_summary(result):
    print("策略回测结果:")
    print("基金代码:", result["fund_code"])
    print("输入因子组合文件:", result["factor_combination_path"])
    print("最终仓位函数:", result["best_strategy_test_summary"]["position_function_name"])
    print("最终组合因子:", result["best_strategy_test_summary"]["candidate_label_list"])
    print("最终 test Sharpe:", result["best_strategy_test_summary"]["test_result"]["stats"]["sharpe"])
    print("图像输出:", result["plot_path"])
    print("汇总输出:", result["summary_path"])


def save_single_factor_dedup_selection_output(stability_analysis_input, dedup_selection_output, output_dir, fund_code):
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    output_path = output_dir / f"single_factor_dedup_{str(fund_code).zfill(6)}_{date_str}.json"
    stability_analysis_output = dict(stability_analysis_input.get("stability_analysis_output", {}))
    payload = {
        "input_ref": {
            "stability_analysis_path": str(dedup_selection_output.get("stability_analysis_path", "")),
            "fund_code": str(stability_analysis_output.get("fund_code", str(fund_code).zfill(6))),
        },
        "dedup_selection_output": dedup_selection_output,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


def load_dedup_selection_input(dedup_selection_path):
    dedup_selection_path = Path(dedup_selection_path)
    if not dedup_selection_path.exists():
        raise FileNotFoundError(f"去冗余结果文件不存在: {dedup_selection_path}")
    with dedup_selection_path.open("r", encoding="utf-8") as input_file:
        dedup_selection_input = json.load(input_file)
    if not isinstance(dedup_selection_input, dict):
        raise ValueError("去冗余结果文件必须是顶层字典。")
    dedup_selection_output = dedup_selection_input.get("dedup_selection_output")
    if not isinstance(dedup_selection_output, dict):
        raise ValueError("去冗余结果文件缺少 dedup_selection_output 子字典。")
    if not isinstance(dedup_selection_output.get("record_dict"), dict):
        raise ValueError("去冗余结果文件缺少 dedup_selection_output.record_dict。")
    if not isinstance(dedup_selection_output.get("best_final_selection_summary"), dict):
        raise ValueError("去冗余结果文件缺少 dedup_selection_output.best_final_selection_summary。")
    return dedup_selection_input, dedup_selection_path


def resolve_fund_code_from_dedup_selection_input(dedup_selection_input, dedup_selection_path):
    dedup_selection_output = dict(dedup_selection_input.get("dedup_selection_output", {}))
    candidate_code = str(dedup_selection_output.get("fund_code", "")).strip()
    if len(candidate_code) > 0:
        candidate_code = candidate_code.zfill(6)
        if len(candidate_code) == 6 and candidate_code.isdigit():
            return candidate_code
    return resolve_fund_code_from_stability_analysis_input(
        stability_analysis_input=dedup_selection_input,
        stability_analysis_path=dedup_selection_path,
    )


def print_single_factor_dedup_selection_summary(result):
    print("单因子去冗余与正向选择结果:")
    print("基金代码:", result["fund_code"])
    print("输入稳定性文件:", result["stability_analysis_path"])
    print("输入稳定性保留候选数量:", len(result["selected_stability_candidate_list"]))
    print("相关性去冗余后候选数量:", len(result["corr_selected_summary_df"]))
    print("最终组合来源:", result["final_selected_source"])
    print("最终正向选择组合:", result["best_final_selection_summary"]["candidate_label_list"])
    print("汇总输出:", result["summary_path"])


def print_single_factor_stability_analysis_summary(result):
    print("单因子稳定性分析结果:")
    print("基金代码:", result["fund_code"])
    print("输入筛选文件:", result["factor_selection_path"])
    print("输入最终候选数量:", len(result["selected_factor_input_list"]))
    print("稳定性分析候选数量:", len(result["summary_df"]))
    print("稳定性尾部剔除后保留数量:", len(result["selected_summary_df"]))
    print("稳定性排序候选列表:", result["selected_candidate_label_list"])
    if len(result["selected_summary_df"]) > 0:
        printable_df = result["selected_summary_df"][
            [
                "stability_rank",
                "candidate_label",
                "factor_group",
                "train_spearman_ic_mean",
                "valid_spearman_ic_mean",
                "train_valid_ic_mean_gap",
                "valid_trimmed_ic_gap",
                "valid_ic_flip_count",
            ]
        ].copy()
        print(printable_df.to_string(index=False))
    print("汇总输出:", result["summary_path"])
