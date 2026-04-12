import json
from datetime import datetime
from pathlib import Path

from tradition.config import build_tradition_config

from .io import allocate_stage_csv_json_output_path
from .temp import (
    TARGET_CODE_TYPE_DICT,
    _build_checked_factor_table,
    _check_and_fill_wide_feature_table,
    build_code_feature_table,
)


def _resolve_feature_preprocess_code_type_dict(config):
    # 现阶段流程0先复用 temp 中已经验证过的数据类型映射，避免在配置层再引入半成品接口。
    config = dict(config)
    primary_code = str(config["default_fund_code"]).zfill(6)
    linked_code_list = [str(code).zfill(6) for code in list(dict(config.get("linked_code_dict", {})).get(primary_code, []))]
    resolved_code_type_dict = {primary_code: "fund"}
    for linked_code in linked_code_list:
        if linked_code not in TARGET_CODE_TYPE_DICT:
            raise ValueError(f"流程0缺少 linked code 的类型映射: {linked_code}")
        resolved_code_type_dict[linked_code] = str(TARGET_CODE_TYPE_DICT[linked_code])
    return resolved_code_type_dict


def _build_feature_preprocess_metadata(
    checked_feature_df,
    csv_path,
    path_code,
    fund_code,
    code_type_dict,
    source_column_list,
    candidate_factor_list,
    code_report_list,
    dropped_factor_report_list,
):
    # 元信息 JSON 显式声明目标列和特征列，避免流程1再从命名规则反推接口。
    checked_feature_df = checked_feature_df.copy()
    raw_feature_column_list = list(source_column_list)
    raw_feature_zscore_column_list = [f"{column}__zscore" for column in raw_feature_column_list if f"{column}__zscore" in checked_feature_df.columns]
    factor_feature_column_list = [
        column
        for column in checked_feature_df.columns
        if column not in {"date", *raw_feature_column_list, *raw_feature_zscore_column_list}
    ]
    target_price_column = f"{str(fund_code).zfill(6)}__price"
    target_nav_column = f"{str(fund_code).zfill(6)}__cumulative_nav"
    if target_nav_column not in checked_feature_df.columns:
        target_nav_column = target_price_column
    feature_column_list = list(raw_feature_zscore_column_list) + list(factor_feature_column_list)
    return {
        "fund_code": str(fund_code).zfill(6),
        "primary_code": str(fund_code).zfill(6),
        "analysis_date": datetime.today().strftime("%Y-%m-%d"),
        "data_mode": "feature_matrix",
        "path_code": str(path_code),
        "csv_path": str(Path(csv_path).resolve()),
        "target_price_column": target_price_column,
        "target_nav_column": target_nav_column,
        "feature_column_list": feature_column_list,
        "raw_feature_column_list": raw_feature_column_list,
        "raw_feature_zscore_column_list": raw_feature_zscore_column_list,
        "factor_feature_column_list": factor_feature_column_list,
        "linked_code_type_dict": {str(code).zfill(6): str(code_type) for code, code_type in dict(code_type_dict).items()},
        "candidate_factor_count": int(len(candidate_factor_list)),
        "row_count": int(len(checked_feature_df)),
        "column_count": int(len(checked_feature_df.columns)),
        "quality_summary": {
            "code_report_list": list(code_report_list),
            "dropped_factor_count": int(len(dropped_factor_report_list)),
        },
        "dropped_feature_list": [
            {
                "source_column": str(record["source_column"]),
                "candidate_label": str(record["candidate_label"]),
                "output_column": f"{str(record['source_column'])}__{str(record['candidate_label'])}__zscore",
                "raw_nan_ratio": float(record["raw_nan_ratio"]),
                "normalized_zero_ratio": float(record["normalized_zero_ratio"]),
                "drop_reason": "threshold_exceeded",
            }
            for record in list(dropped_factor_report_list)
        ],
    }


def run_feature_preprocess_single_fund(config_override=None):
    config = build_tradition_config(config_override=config_override)
    fund_code = str(config["default_fund_code"]).zfill(6)
    code_type_dict = _resolve_feature_preprocess_code_type_dict(config=config)
    raw_output_path, metadata_output_path, path_code = allocate_stage_csv_json_output_path(
        output_dir=config["output_dir"],
        output_prefix="feature_preprocess",
        fund_code=fund_code,
    )
    _, resolved_raw_output_path = build_code_feature_table(
        code_type_dict=code_type_dict,
        output_path=raw_output_path,
        primary_code=fund_code,
        force_refresh=bool(config["force_refresh"]),
    )
    checked_feature_df, checked_output_path, code_report_list = _check_and_fill_wide_feature_table(
        raw_output_path=resolved_raw_output_path,
        code_type_dict=code_type_dict,
        primary_code=fund_code,
    )
    checked_feature_df, checked_output_path, source_column_list, candidate_factor_list, dropped_factor_report_list = _build_checked_factor_table(
        checked_output_path=checked_output_path,
        strategy_params=config["strategy_param_dict"]["multi_factor_score"],
    )
    metadata_output = _build_feature_preprocess_metadata(
        checked_feature_df=checked_feature_df,
        csv_path=checked_output_path,
        path_code=path_code,
        fund_code=fund_code,
        code_type_dict=code_type_dict,
        source_column_list=source_column_list,
        candidate_factor_list=candidate_factor_list,
        code_report_list=code_report_list,
        dropped_factor_report_list=dropped_factor_report_list,
    )
    payload = {
        "path_code": str(path_code),
        "feature_preprocess_output": metadata_output,
    }
    with Path(metadata_output_path).open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    result = {
        "fund_code": fund_code,
        "path_code": path_code,
        "raw_output_path": Path(resolved_raw_output_path),
        "summary_path": Path(checked_output_path),
        "metadata_path": Path(metadata_output_path),
        "record_count": int(len(checked_feature_df)),
        "column_count": int(len(checked_feature_df.columns)),
        "dropped_factor_count": int(len(dropped_factor_report_list)),
    }
    print("特征预处理结果:")
    print("基金代码:", result["fund_code"])
    print("path_code:", result["path_code"])
    print("记录数:", result["record_count"])
    print("总列数:", result["column_count"])
    print("删除因子数:", result["dropped_factor_count"])
    print("原始输出:", result["raw_output_path"])
    print("特征输出:", result["summary_path"])
    print("元信息输出:", result["metadata_path"])
    return result
