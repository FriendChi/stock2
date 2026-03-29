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
                    "momentum_window_short": 20,
                    "momentum_window_long": 60,
                    "entry_threshold": 0.2,
                    "exit_threshold": 0.0,
                }, number=0),
                FakeTrial({
                    "momentum_window_short": 30,
                    "momentum_window_long": 90,
                    "entry_threshold": 0.3,
                    "exit_threshold": -0.1,
                }, number=1),
            ]
        )


def test_suggest_strategy_params_for_multi_factor_score():
    params = optimizer.suggest_strategy_params(
        trial=FakeTrial({
            "momentum_window_short": 20,
            "momentum_window_long": 80,
            "entry_threshold": 0.2,
            "exit_threshold": -0.1,
        }),
        strategy_name="multi_factor_score",
        base_params={
            "momentum_window_short": 10,
            "momentum_window_long": 60,
            "volatility_window": 20,
            "drawdown_window": 60,
            "score_window": 60,
            "factor_weight_dict": {"momentum_20": 0.3},
            "entry_threshold": 0.1,
            "exit_threshold": 0.0,
        },
    )
    assert params["momentum_window_short"] == 20
    assert params["momentum_window_long"] == 80
    assert params["entry_threshold"] == 0.2
    assert params["exit_threshold"] == -0.1


def test_suggest_strategy_params_caps_exit_threshold_by_entry_threshold():
    params = optimizer.suggest_strategy_params(
        trial=FakeTrial({
            "momentum_window_short": 20,
            "momentum_window_long": 80,
            "entry_threshold": 0.1,
            "exit_threshold": 0.2,
        }),
        strategy_name="multi_factor_score",
        base_params={
            "momentum_window_short": 10,
            "momentum_window_long": 60,
            "volatility_window": 20,
            "drawdown_window": 60,
            "score_window": 60,
            "factor_weight_dict": {"momentum_20": 0.3},
            "entry_threshold": 0.1,
            "exit_threshold": 0.0,
        },
    )
    assert params["entry_threshold"] == 0.1
    assert params["exit_threshold"] <= params["entry_threshold"]


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
        sharpe = 1.0 if params["momentum_window_short"] == 30 else 0.8
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
        base_params={
            "momentum_window_short": 20,
            "momentum_window_long": 60,
            "volatility_window": 20,
            "drawdown_window": 60,
            "score_window": 60,
            "factor_weight_dict": {
                "momentum_20": 0.30,
                "momentum_60": 0.30,
                "volatility_20": 0.20,
                "drawdown_60": 0.20,
            },
            "entry_threshold": 0.2,
            "exit_threshold": 0.0,
        },
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

    assert result["best_params"]["momentum_window_short"] == 30
    assert result["best_value"] > 0.9
    assert len(result["top_k_params_list"]) == 2
    assert len(result["improving_best_params_list"]) == 2
    assert not result["trial_df"].empty
