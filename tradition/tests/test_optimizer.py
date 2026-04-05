import pandas as pd

from tradition import optimizer


class FakeTrial:
    def __init__(self, params, number=0, value=None):
        self.params = dict(params)
        self.user_attrs = {}
        self.number = int(number)
        self.value = value

    def suggest_int(self, name, low, high, step=1):
        return int(self.params.get(name, low))

    def suggest_float(self, name, low, high, step=None):
        return float(self.params.get(name, low))

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


class FakeStudy:
    def __init__(self, trials):
        self._trials = trials
        self.trials = trials
        self.best_params = {}
        self.best_value = None

    def optimize(self, objective, n_trials):
        values = []
        for trial in self._trials[:n_trials]:
            value = objective(trial)
            trial.value = value
            values.append((value, trial))
        self.best_value, best_trial = max(values, key=lambda item: item[0])
        self.best_params = dict(best_trial.params)

    def trials_dataframe(self, attrs=None):
        return pd.DataFrame(
            [
                {
                    "number": index,
                    "value": trial.value,
                    "params": trial.params,
                    "state": "COMPLETE",
                    "user_attrs": trial.user_attrs,
                }
                for index, trial in enumerate(self._trials)
            ]
        )


class FakeSampler:
    def __init__(self, seed=None):
        self.seed = seed


class FakeSamplers:
    TPESampler = FakeSampler


class FakeOptuna:
    samplers = FakeSamplers

    @staticmethod
    def create_study(direction, sampler, study_name):
        assert direction == "maximize"
        assert study_name.startswith("tradition_optuna")
        return FakeStudy(
            [
                FakeTrial({
                    "momentum__window": 20,
                    "ma_trend_state__window": 60,
                    "trend_r2__window": 20,
                    "entry_threshold": 0.2,
                    "exit_threshold": 0.0,
                }, number=0),
                FakeTrial({
                    "momentum__window": 30,
                    "ma_trend_state__window": 75,
                    "trend_r2__window": 25,
                    "entry_threshold": 0.3,
                    "exit_threshold": -0.1,
                }, number=1),
            ]
        )


def build_multi_factor_params():
    return {
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
        "search_strategy_param_name_list": ["entry_threshold", "exit_threshold"],
        "factor_param_dict": {
            "momentum": {"window": 10},
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
        "score_window": 60,
        "factor_weight_dict": {
            "momentum": 0.2,
            "ma_trend_state": 0.1,
            "ma_slope": 0.1,
            "trend_r2": 0.1,
            "trend_tvalue": 0.05,
            "price_position": 0.1,
            "breakout_strength": 0.1,
            "donchian_breakout": 0.05,
            "trend_residual": 0.05,
            "volatility": 0.1,
            "drawdown": 0.1,
            "risk_adjusted_momentum": 0.1,
            "sharpe_like_trend": 0.05,
        },
        "entry_threshold": 0.1,
        "exit_threshold": 0.0,
    }


def test_suggest_strategy_params_for_multi_factor_score():
    params = optimizer.suggest_strategy_params(
        trial=FakeTrial({
            "momentum__window": 20,
            "ma_trend_state__window": 75,
            "trend_r2__window": 25,
            "price_position__window": 70,
            "breakout_strength__window": 85,
            "volatility__window": 25,
            "drawdown__window": 65,
            "risk_adjusted_momentum__window": 30,
            "entry_threshold": 0.2,
            "exit_threshold": -0.1,
        }),
        strategy_name="multi_factor_score",
        base_params=build_multi_factor_params(),
    )
    assert params["factor_param_dict"]["momentum"]["window"] == 20
    assert params["factor_param_dict"]["ma_trend_state"]["window"] == 75
    assert params["factor_param_dict"]["trend_r2"]["window"] == 25
    assert params["factor_param_dict"]["price_position"]["window"] == 70
    assert params["factor_param_dict"]["breakout_strength"]["window"] == 85
    assert params["factor_param_dict"]["volatility"]["window"] == 25
    assert params["factor_param_dict"]["drawdown"]["window"] == 65
    assert params["factor_param_dict"]["risk_adjusted_momentum"]["window"] == 30
    assert params["entry_threshold"] == 0.2
    assert params["exit_threshold"] == -0.1


def test_suggest_strategy_params_caps_exit_threshold_by_entry_threshold():
    params = optimizer.suggest_strategy_params(
        trial=FakeTrial({
            "momentum__window": 20,
            "entry_threshold": 0.1,
            "exit_threshold": 0.2,
        }),
        strategy_name="multi_factor_score",
        base_params={
            "enabled_factor_list": ["momentum"],
            "search_strategy_param_name_list": ["entry_threshold", "exit_threshold"],
            "factor_param_dict": {
                "momentum": {"window": 10},
            },
            "score_window": 60,
            "factor_weight_dict": {"momentum": 1.0},
            "entry_threshold": 0.1,
            "exit_threshold": 0.0,
        },
    )
    assert params["entry_threshold"] == 0.1
    assert params["exit_threshold"] <= params["entry_threshold"]


def test_suggest_strategy_params_raises_for_unknown_strategy_search_param():
    try:
        optimizer.suggest_strategy_params(
            trial=FakeTrial({"entry_threshold": 0.1}),
            strategy_name="multi_factor_score",
            base_params={
                "enabled_factor_list": ["momentum"],
                "search_strategy_param_name_list": ["unknown_threshold"],
                "factor_param_dict": {
                    "momentum": {"window": 10},
                },
                "score_window": 60,
                "factor_weight_dict": {"momentum": 1.0},
                "entry_threshold": 0.1,
                "exit_threshold": 0.0,
            },
        )
        raise AssertionError("预期应抛出 ValueError")
    except ValueError as exc:
        assert "search_strategy_param_name_list 中存在未定义策略参数" in str(exc)


def test_compute_objective_value_penalizes_drawdown():
    value = optimizer.compute_objective_value(
        metric_dict={"sharpe": 1.0, "max_drawdown": -0.3},
        optimization_config={"target_metric": "sharpe", "penalty_weight": 0.2},
    )
    assert round(value, 4) == 0.94


def test_resolve_top_k_count_uses_trial_ratio_with_minimum():
    assert optimizer.resolve_top_k_count({"n_trials": 3}) == 5
    assert optimizer.resolve_top_k_count({"n_trials": 20}) == 5
    assert optimizer.resolve_top_k_count({"n_trials": 50}) == 10


def test_build_top_k_params_list_sorts_by_train_objective():
    study = FakeStudy(
        [
            FakeTrial({"window": 10}, number=0, value=0.2),
            FakeTrial({"window": 20}, number=1, value=0.5),
            FakeTrial({"window": 30}, number=2, value=0.3),
        ]
    )
    top_k = optimizer.build_top_k_params_list(
        study=study,
        strategy_name="momentum",
        base_params={"window": 20},
        optimization_config={"n_trials": 50},
    )
    assert [item["trial_number"] for item in top_k] == [1, 2, 0]


def test_build_improving_best_params_list_keeps_only_new_bests():
    study = FakeStudy(
        [
            FakeTrial({"window": 10}, number=0, value=0.2),
            FakeTrial({"window": 20}, number=1, value=0.5),
            FakeTrial({"window": 30}, number=2, value=0.4),
            FakeTrial({"window": 40}, number=3, value=0.8),
        ]
    )
    improving = optimizer.build_improving_best_params_list(
        study=study,
        strategy_name="momentum",
        base_params={"window": 20},
    )
    assert [item["trial_number"] for item in improving] == [0, 1, 3]


def test_optimize_strategy_params_returns_best_result(monkeypatch):
    monkeypatch.setattr(optimizer, "load_optuna_module", lambda: FakeOptuna)

    def evaluate_params_fn(params):
        sharpe = 1.0 if params["factor_param_dict"]["momentum"]["window"] == 30 else 0.8
        return {
            "stats": {
                "annual_return": 0.1,
                "sharpe": sharpe,
                "max_drawdown": -0.2,
            },
            "trade_count": 3,
        }

    result = optimizer.optimize_strategy_params(
        strategy_name="multi_factor_score",
        base_params=build_multi_factor_params(),
        evaluate_params_fn=evaluate_params_fn,
        optimization_config={
            "n_trials": 2,
            "top_k": 2,
            "study_direction": "maximize",
            "study_name_prefix": "tradition_optuna",
            "target_metric": "sharpe",
            "penalty_weight": 0.2,
        },
    )

    assert result["best_params"]["factor_param_dict"]["momentum"]["window"] == 30
    assert result["best_value"] > 0.9
    assert len(result["top_k_params_list"]) == 2
    assert len(result["improving_best_params_list"]) == 2
    assert not result["trial_df"].empty
