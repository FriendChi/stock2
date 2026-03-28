from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path


DEFAULT_CODE_DICT = {
    "007301": "半导体",
    "012832": "新能源",
    "007883": "医药",
    "012700": "证券",
    "012043": "酒",
    "009180": "消费",
    "012323": "医疗",
    "007339": "沪深300",
    "014345": "中证500",
    "001593": "创业",
    "009052": "中证红利",
    "008702": "黄金",
}

DEFAULT_STRATEGY_PARAM_DICT = {
    "buy_and_hold": {},
    "ma_cross": {
        "fast": 5,
        "slow": 20,
    },
    "momentum": {
        "window": 20,
    },
}


@dataclass
class TraditionConfig:
    # 固定项目根目录推导规则，避免在不同工作目录下运行时写错缓存和输出路径。
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    code_dict: dict = field(default_factory=lambda: deepcopy(DEFAULT_CODE_DICT))
    strategy_param_dict: dict = field(default_factory=lambda: deepcopy(DEFAULT_STRATEGY_PARAM_DICT))
    default_fund_code: str = "007301"
    default_strategy_name: str = "buy_and_hold"
    init_cash: float = 10000.0
    fees: float = 0.001
    force_refresh: bool = False
    cache_prefix: str = "tradition_fund"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "tradition" / "outputs"


def build_tradition_config(config_override=None):
    # 对外统一返回字典配置，减少 runner 和测试里对 dataclass 细节的耦合。
    config = TraditionConfig()
    config_dict = asdict(config)
    config_dict["project_root"] = config.project_root
    config_dict["data_dir"] = config.data_dir
    config_dict["output_dir"] = config.output_dir
    if config_override is not None:
        if not isinstance(config_override, dict):
            raise ValueError("config_override 必须为dict。")
        config_dict.update(config_override)
    return config_dict
