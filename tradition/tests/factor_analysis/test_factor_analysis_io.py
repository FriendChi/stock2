import json
from pathlib import Path
import pytest
from tradition.factor_analysis import io

def test_path_code_validation_and_extraction():
    # 逻辑块：验证 path_code 格式合法性
    assert io.is_valid_path_code("aB1234") is True
    assert io.is_valid_path_code("123") is False
    assert io.is_valid_path_code("aB12345") is False
    
    # 逻辑块：从文件名中提取 path_code
    path = Path("factor_selection_000001_2024-01-01_aB1234.json")
    assert io.extract_path_code_from_path(path) == "aB1234"
    assert io.extract_path_code_from_path("invalid_file.json") is None

def test_resolve_input_path_code():
    # 逻辑块：从 payload 字典中解析 path_code
    payload = {"path_code": "xY9876"}
    assert io.resolve_input_path_code(payload) == "xY9876"
    
    # 逻辑块：当 payload 无效时从路径解析
    path = "factor_selection_000001_2024-01-01_aB1234.json"
    assert io.resolve_input_path_code({}, input_path=path) == "aB1234"
    
    # 逻辑块：兜底返回默认 code
    assert io.resolve_input_path_code({}) == io.DEFAULT_PATH_CODE

def test_allocate_stage_output_path(tmp_path):
    # 逻辑块：测试阶段输出路径的分配与编码演进
    # 设定初始 code 为 000000，第1阶段（index=1）分配后应保留第0位，修改第1位，后续清零
    inherited_code = "A00000" 
    output_path, path_code = io.allocate_stage_output_path(
        output_dir=tmp_path,
        output_prefix="test_prefix",
        fund_code="000001",
        stage_index=1,
        inherited_path_code=inherited_code
    )
    
    assert path_code.startswith("A")
    assert path_code[2:] == "0000"
    assert output_path.parent == tmp_path
    assert str(path_code) in output_path.name

def test_allocate_stage_csv_json_output_path(tmp_path):
    # 逻辑块：测试双文件（CSV/JSON）同步分配逻辑
    csv_path, json_path, path_code = io.allocate_stage_csv_json_output_path(
        output_dir=tmp_path,
        output_prefix="stage0",
        fund_code="000001"
    )
    
    assert csv_path.suffix == ".csv"
    assert json_path.suffix == ".json"
    assert csv_path.stem == json_path.stem
    assert path_code[1:] == "00000" # 流程0只改首位
