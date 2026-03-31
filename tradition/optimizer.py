from datetime import datetime

import pandas as pd

from tradition.factor_engine import build_factor_search_param_spec_dict
from tradition.strategies import get_strategy_params


def load_optuna_module():
    # 延迟导入 Optuna，避免普通回测路径对优化依赖产生强耦合。
    import optuna

    return optuna


def build_strategy_search_param_spec_dict(strategy_name):
    # 策略级搜索参数单独管理，和因子参数区分职责，避免被误塞进因子池。
    strategy_name = str(strategy_name).lower()
    if strategy_name == "multi_factor_score":
        return {
            "entry_threshold": (-0.2, 1.0, 0.1),
            "exit_threshold": (-0.8, 0.2, 0.1),
        }
    return {}


def suggest_strategy_params(trial, strategy_name, base_params):
    # 搜索空间和策略名绑定在优化模块内，runner 只消费产出的参数字典。
    strategy_name = str(strategy_name).lower()
    params = dict(base_params)

    if strategy_name == "ma_cross":
        fast = trial.suggest_int("fast", 3, 30)
        slow = trial.suggest_int("slow", fast + 5, 120)
        params.update({"fast": fast, "slow": slow})
        return params

    if strategy_name == "momentum":
        params["window"] = trial.suggest_int("window", 5, 120)
        return params

    if strategy_name == "multi_factor_score":
        search_factor_param_name_dict = {
            str(factor_name): [str(param_name) for param_name in param_name_list]
            for factor_name, param_name_list in dict(params.get("search_factor_param_name_dict", {})).items()
        }
        search_strategy_param_name_list = [
            str(param_name) for param_name in params.get("search_strategy_param_name_list", [])
        ]
        strategy_search_param_spec_dict = build_strategy_search_param_spec_dict(strategy_name=strategy_name)
        factor_search_param_spec_dict = build_factor_search_param_spec_dict(strategy_params=params)
        factor_param_dict = {
            str(factor_name): dict(param_dict) for factor_name, param_dict in dict(params["factor_param_dict"]).items()
        }
        for factor_name, param_name_list in search_factor_param_name_dict.items():
            if factor_name not in factor_search_param_spec_dict:
                raise ValueError(f"search_factor_param_name_dict 中存在未定义搜索范围的因子: {factor_name}")
            for param_name in param_name_list:
                if param_name not in factor_search_param_spec_dict[factor_name]:
                    raise ValueError(f"{factor_name} 未定义可搜索参数: {param_name}")
                low, high, step = factor_search_param_spec_dict[factor_name][param_name]
                trial_param_name = f"{factor_name}__{param_name}"
                factor_param_dict.setdefault(factor_name, {})
                factor_param_dict[factor_name][param_name] = trial.suggest_int(
                    trial_param_name,
                    int(low),
                    int(high),
                    step=1 if step is None else int(step),
                )
        params["factor_param_dict"] = factor_param_dict

        for param_name in search_strategy_param_name_list:
            if param_name not in strategy_search_param_spec_dict:
                raise ValueError(f"search_strategy_param_name_list 中存在未定义策略参数: {param_name}")

        if "entry_threshold" in search_strategy_param_name_list:
            low, high, step = strategy_search_param_spec_dict["entry_threshold"]
            params["entry_threshold"] = trial.suggest_float("entry_threshold", low, high, step=step)
        entry_threshold = float(params["entry_threshold"])

        if "exit_threshold" in search_strategy_param_name_list:
            low, high, step = strategy_search_param_spec_dict["exit_threshold"]
            exit_threshold = trial.suggest_float("exit_threshold", low, min(high, entry_threshold), step=step)
            params["exit_threshold"] = min(exit_threshold, entry_threshold)

        momentum_short_window = int(params["factor_param_dict"]["momentum_short"]["window"])
        momentum_mid_window = int(params["factor_param_dict"]["momentum_mid"]["window"])
        momentum_long_window = int(params["factor_param_dict"]["momentum_long"]["window"])
        params["factor_param_dict"]["momentum_mid"]["window"] = max(momentum_short_window, momentum_mid_window)
        params["factor_param_dict"]["momentum_long"]["window"] = max(
            int(params["factor_param_dict"]["momentum_mid"]["window"]),
            momentum_long_window,
        )
        return params

    raise ValueError(f"当前未定义可优化的 strategy_name: {strategy_name}")


def compute_objective_value(metric_dict, optimization_config):
    # 目标函数默认以 Sharpe 为主，对深回撤做惩罚，避免搜索过度偏向高波动参数。
    target_metric = str(optimization_config["target_metric"])
    if target_metric not in metric_dict:
        raise ValueError(f"metric_dict 中不存在目标指标: {target_metric}")
    penalty_weight = float(optimization_config["penalty_weight"])
    max_drawdown = float(metric_dict["max_drawdown"])
    drawdown_penalty = abs(min(0.0, max_drawdown))
    return float(metric_dict[target_metric]) - penalty_weight * drawdown_penalty


def build_trials_dataframe(study):
    # 统一把 trial 结果转成表结构，便于后续落盘和分析。
    trial_df = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
    if isinstance(trial_df, pd.DataFrame):
        return trial_df.reset_index(drop=True)
    return pd.DataFrame()



def resolve_top_k_count(optimization_config):
    # Top-K 候选数量按 n_trials 的 20% 动态放大，并保留最小 5 个候选的下限。
    n_trials = int(optimization_config["n_trials"])
    return max(5, int(n_trials * 0.2))

def build_top_k_params_list(study, strategy_name, base_params, optimization_config):
    # 只保留训练集 objective 最优的前 K 组参数，后续由验证集决定最终采用哪一组。
    resolved_base_params = get_strategy_params(strategy_name=strategy_name, strategy_params=base_params)
    top_k = resolve_top_k_count(optimization_config)
    completed_trial_list = [trial for trial in study.trials if trial.value is not None]
    sorted_trial_list = sorted(completed_trial_list, key=lambda trial: float(trial.value), reverse=True)

    top_k_params_list = []
    for trial in sorted_trial_list[:top_k]:
        candidate_params = suggest_strategy_params(
            trial=trial,
            strategy_name=strategy_name,
            base_params=resolved_base_params,
        )
        top_k_params_list.append(
            {
                "trial_number": int(trial.number),
                "train_objective": float(trial.value),
                "params": candidate_params,
                "source": "top_k",
            }
        )
    return top_k_params_list


def build_improving_best_params_list(study, strategy_name, base_params):
    # 按 trial 发生顺序记录每次刷新训练集历史最优值的参数，保留优化轨迹中的关键候选。
    resolved_base_params = get_strategy_params(strategy_name=strategy_name, strategy_params=base_params)
    completed_trial_list = [trial for trial in study.trials if trial.value is not None]
    sorted_trial_list = sorted(completed_trial_list, key=lambda trial: int(trial.number))

    improving_best_params_list = []
    best_value = None
    for trial in sorted_trial_list:
        current_value = float(trial.value)
        if best_value is None or current_value > best_value:
            best_value = current_value
            candidate_params = suggest_strategy_params(
                trial=trial,
                strategy_name=strategy_name,
                base_params=resolved_base_params,
            )
            improving_best_params_list.append(
                {
                    "trial_number": int(trial.number),
                    "train_objective": current_value,
                    "params": candidate_params,
                    "source": "improving_best",
                }
            )
    return improving_best_params_list


def optimize_strategy_params(strategy_name, base_params, evaluate_params_fn, optimization_config, study_name=None):
    # 优化执行层只负责编排 Study 生命周期，不关心数据切分和回测细节。
    optuna = load_optuna_module()
    strategy_name = str(strategy_name).lower()
    resolved_base_params = get_strategy_params(strategy_name=strategy_name, strategy_params=base_params)

    if study_name is None:
        prefix = str(optimization_config["study_name_prefix"])
        date_str = datetime.today().strftime("%Y%m%d")
        study_name = f"{prefix}_{strategy_name}_{date_str}"

    def objective(trial):
        # 每个 trial 先生成参数，再通过外部注入的评估函数执行训练集回测。
        params = suggest_strategy_params(
            trial=trial,
            strategy_name=strategy_name,
            base_params=resolved_base_params,
        )
        result = evaluate_params_fn(params)
        metric_dict = dict(result["stats"])
        objective_value = compute_objective_value(
            metric_dict=metric_dict,
            optimization_config=optimization_config,
        )
        trial.set_user_attr("trade_count", int(result["trade_count"]))
        trial.set_user_attr("annual_return", float(metric_dict["annual_return"]))
        trial.set_user_attr("sharpe", float(metric_dict["sharpe"]))
        trial.set_user_attr("max_drawdown", float(metric_dict["max_drawdown"]))
        return objective_value

    study = optuna.create_study(
        direction=str(optimization_config["study_direction"]),
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=study_name,
    )
    study.optimize(objective, n_trials=int(optimization_config["n_trials"]))
    return {
        "study": study,
        "study_name": study_name,
        "best_params": suggest_strategy_params(
            trial=study.best_trial if hasattr(study, "best_trial") else next(
                trial for trial in study.trials if trial.params == study.best_params
            ),
            strategy_name=strategy_name,
            base_params=resolved_base_params,
        ),
        "best_value": float(study.best_value),
        "top_k_params_list": build_top_k_params_list(
            study=study,
            strategy_name=strategy_name,
            base_params=base_params,
            optimization_config=optimization_config,
        ),
        "improving_best_params_list": build_improving_best_params_list(
            study=study,
            strategy_name=strategy_name,
            base_params=base_params,
        ),
        "trial_df": build_trials_dataframe(study=study),
    }
