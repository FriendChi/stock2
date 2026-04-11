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

DEFAULT_LINKED_CODE_DICT = {
    "007301": ["512480", "000510"],
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
    "multi_factor_score": {
        "enabled_factor_list": [
            "momentum",
            "ma_trend_state",
            "ma_slope",
            "trend_r2",
            "trend_tvalue",
            "price_position",
            "breakout_strength",
            "donchian_breakout",
            "trend_residual",
            "volatility",
            "drawdown",
            "risk_adjusted_momentum",
            "sharpe_like_trend",
        ],
        "factor_param_dict": {
            "momentum": {"window": 40},
            "ma_trend_state": {"window": 60},
            "ma_slope": {"window": 20, "lookback": 5},
            "trend_r2": {"window": 20},
            "trend_tvalue": {"window": 20},
            "price_position": {"window": 60},
            "breakout_strength": {"window": 60},
            "donchian_breakout": {"window": 20},
            "trend_residual": {"window": 20},
            "volatility": {"window": 20},
            "drawdown": {"window": 60},
            "risk_adjusted_momentum": {"window": 20},
            "sharpe_like_trend": {"window": 20},
        },
        "search_strategy_param_name_list": [
            "entry_threshold",
            "exit_threshold",
        ],
        "score_window": 60,
        "factor_weight_dict": {
            "momentum": 0.12,
            "ma_trend_state": 0.10,
            "ma_slope": 0.08,
            "trend_r2": 0.08,
            "trend_tvalue": 0.08,
            "price_position": 0.08,
            "breakout_strength": 0.08,
            "donchian_breakout": 0.07,
            "trend_residual": 0.07,
            "volatility": 0.08,
            "drawdown": 0.08,
            "risk_adjusted_momentum": 0.10,
            "sharpe_like_trend": 0.06,
        },
        "entry_threshold": 0.2,
        "exit_threshold": 0.0,
    },
}

DEFAULT_DATA_SPLIT_DICT = {
    "train_ratio": 0.6,
    "valid_ratio": 0.2,
    "test_ratio": 0.2,
    "min_segment_size": 60,
}

DEFAULT_OPTIMIZATION_CONFIG = {
    "default_target_strategy_name": "multi_factor_score",
    "n_trials": 30,
    "top_k": 5,
    "study_direction": "maximize",
    "study_name_prefix": "tradition_optuna",
    "target_metric": "sharpe",
    "penalty_weight": 0.2,
}

DEFAULT_WALK_FORWARD_CONFIG = {
    "enabled": False,
    "window_size": 700,
    "step_size": 60,
    "min_fold_count": 1,
}

DEFAULT_RF_CONFIG = {
    "enabled": True,
    "curve_name": "中债国债收益率曲线",
    "tenor": "1年",
    "cache_prefix": "tradition_rf_cn_bond_yield",
}

DEFAULT_IC_AGGREGATION_MODE = "classic"
DEFAULT_IC_EXP_WEIGHT_HALF_LIFE = 3.0


@dataclass
class TraditionConfig:
    # 固定项目根目录推导规则，避免在不同工作目录下运行时写错缓存和输出路径。
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    code_dict: dict = field(default_factory=lambda: deepcopy(DEFAULT_CODE_DICT))
    linked_code_dict: dict = field(default_factory=lambda: deepcopy(DEFAULT_LINKED_CODE_DICT))
    strategy_param_dict: dict = field(default_factory=lambda: deepcopy(DEFAULT_STRATEGY_PARAM_DICT))
    data_split_dict: dict = field(default_factory=lambda: deepcopy(DEFAULT_DATA_SPLIT_DICT))
    optimization_config: dict = field(default_factory=lambda: deepcopy(DEFAULT_OPTIMIZATION_CONFIG))
    walk_forward_config: dict = field(default_factory=lambda: deepcopy(DEFAULT_WALK_FORWARD_CONFIG))
    rf_config: dict = field(default_factory=lambda: deepcopy(DEFAULT_RF_CONFIG))
    default_fund_code: str = "007301"
    default_strategy_name: str = "buy_and_hold"
    init_cash: float = 10000.0
    fees: float = 0.001
    force_refresh: bool = False
    cache_prefix: str = "tradition_fund"
    ic_aggregation_mode: str = DEFAULT_IC_AGGREGATION_MODE
    ic_exp_weight_half_life: float = DEFAULT_IC_EXP_WEIGHT_HALF_LIFE

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


def resolve_effective_code_dict(config):
    # 数据抓取代码集合允许按主基金代码做联动扩展，保证主流程可同时拿到辅助序列。
    if not isinstance(config, dict):
        raise ValueError("config 必须为dict。")
    base_code_dict = {
        str(code).zfill(6): str(name)
        for code, name in dict(config.get("code_dict", {})).items()
    }
    primary_code = str(config.get("default_fund_code", "")).zfill(6)
    if primary_code and primary_code not in base_code_dict:
        base_code_dict[primary_code] = primary_code
    linked_code_dict = dict(config.get("linked_code_dict", {}))
    linked_code_list = list(linked_code_dict.get(primary_code, []))
    for linked_code in linked_code_list:
        normalized_code = str(linked_code).zfill(6)
        if normalized_code not in base_code_dict:
            base_code_dict[normalized_code] = normalized_code
    return base_code_dict
