from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import scripts.run_strategy_optimization as strategy_opt


def test_run_strategy_optimization_keeps_low_win_rate_candidate_when_asymmetry_is_strong(monkeypatch):
    saved_metrics: list[dict] = []
    captured_optimizer = {}

    class FakeDB:
        def __init__(self, db_path: str):
            self.db_path = db_path

        def get_distinct_tickers(self, limit=None):
            return ["AAPL"]

        def get_forecast_regression_summary(self, start_date=None, end_date=None, model_type=None):
            if model_type == "SAMOSSA":
                return {"samossa": {"rmse": 1.0}}
            return {"ensemble": {"rmse": 0.95}}

        def save_strategy_config(self, regime=None, params=None, metrics=None, score=None):
            saved_metrics.append(dict(metrics or {}))
            return 1

    class FakeOptimizer:
        def __init__(self, search_space, objectives, constraints=None, random_state=None):
            self.search_space = search_space
            self.objectives = objectives
            self.constraints = constraints or {}
            captured_optimizer["constraints"] = self.constraints
            captured_optimizer["objectives"] = self.objectives

        def run(self, n_candidates, evaluation_fn, regime=None):
            candidate = strategy_opt.StrategyCandidate(params={"alpha": 0.5}, regime=regime)
            metrics = evaluation_fn(candidate)
            return [SimpleNamespace(candidate=candidate, metrics=metrics, score=1.0)]

    def fake_simulate_candidate(**kwargs):
        return {
            "total_profit": 25.0,
            "total_return": 0.25,
            "alpha": 0.03,
            "information_ratio": 1.4,
            "beta": 0.92,
            "tracking_error": 0.08,
            "r_squared": 0.64,
            "profit_factor": 2.1,
            "win_rate": 0.20,
            "total_trades": 6,
            "max_drawdown": 0.08,
            "omega_ratio": 2.40,
            "payoff_asymmetry": 1.70,
            "payoff_asymmetry_support_ok": True,
            "payoff_asymmetry_effective": 1.50,
            "expected_shortfall": -0.012,
            "cvar_95": -0.020,
            "fractional_kelly_fat_tail": 0.05,
            "omega_cliff_drop_ratio": 0.15,
            "omega_cliff_ok": True,
            "omega_right_tail_ok": True,
            "omega_ci_lower": 1.20,
            "omega_ci_upper": 3.10,
            "omega_ci_width": 1.90,
            "expected_shortfall_raw": -0.012,
            "expected_shortfall_to_edge": 2.5,
            "es_to_edge_bounded": True,
            "barbell_path_risk_ok": True,
            "regime_realism_ok": True,
            "anti_barbell_ok": True,
            "anti_barbell_reason": "all anti-barbell checks passed",
            "anti_barbell_evidence": {
                "anti_barbell_ok": True,
                "anti_barbell_reason": "all anti-barbell checks passed",
            },
            "rmse_ratio_vs_baseline": 0.95,
            "rmse_within_threshold": 1.0,
        }

    monkeypatch.setattr(strategy_opt, "DatabaseManager", FakeDB)
    monkeypatch.setattr(strategy_opt, "StrategyOptimizer", FakeOptimizer)
    monkeypatch.setattr(strategy_opt, "simulate_candidate", fake_simulate_candidate)

    strategy_opt.main.callback(
        config_path="config/strategy_optimization_config.yml",
        db_path="data/portfolio_maximizer.db",
        n_candidates=1,
        regime="default",
        verbose=False,
    )
    assert saved_metrics, "Expected optimizer results to be persisted"
    assert saved_metrics[0]["win_rate"] == 0.20
    assert saved_metrics[0]["total_return"] == 0.25
    assert saved_metrics[0]["alpha"] == 0.03
    assert saved_metrics[0]["information_ratio"] == 1.4
    assert saved_metrics[0]["profit_factor"] == 2.1
    assert saved_metrics[0]["omega_ratio"] is not None
    assert saved_metrics[0]["payoff_asymmetry"] is not None
    assert saved_metrics[0]["payoff_asymmetry_effective"] is not None
    assert saved_metrics[0]["anti_barbell_ok"] is True
    assert saved_metrics[0]["barbell_path_risk_ok"] is True
    assert saved_metrics[0]["regime_realism_ok"] is True
    assert saved_metrics[0]["rmse_ratio_vs_baseline"] == 0.95
    assert saved_metrics[0]["rmse_within_threshold"] == 1.0
    assert "payoff_asymmetry" not in captured_optimizer["objectives"]
    assert "payoff_asymmetry_effective" in captured_optimizer["objectives"]
    assert "alpha" in captured_optimizer["objectives"]
    assert "information_ratio" in captured_optimizer["objectives"]
    assert captured_optimizer["constraints"]["min"]["omega_ratio"] == pytest.approx(1.0)
    assert captured_optimizer["constraints"]["min"]["alpha"] == pytest.approx(0.0)
    assert captured_optimizer["constraints"]["min"]["information_ratio"] == pytest.approx(0.0)
    assert captured_optimizer["constraints"]["min"]["payoff_asymmetry_effective"] == pytest.approx(1.10)
    assert captured_optimizer["constraints"]["min"]["omega_monotonicity_ok"] is True
    assert captured_optimizer["constraints"]["min"]["omega_cliff_ok"] is True
    assert captured_optimizer["constraints"]["min"]["omega_right_tail_ok"] is True
    assert captured_optimizer["constraints"]["min"]["es_to_edge_bounded"] is True
    assert captured_optimizer["constraints"]["min"]["barbell_path_risk_ok"] is True
    assert captured_optimizer["constraints"]["min"]["regime_realism_ok"] is True
    assert captured_optimizer["constraints"]["min"]["anti_barbell_ok"] is True
    assert captured_optimizer["constraints"]["min"]["total_return"] == pytest.approx(0.0)
    assert captured_optimizer["constraints"]["max"]["max_drawdown"] == pytest.approx(0.30)
    assert captured_optimizer["constraints"]["max"]["rmse_ratio_vs_baseline"] == pytest.approx(1.10)


def test_run_strategy_optimization_fails_closed_when_regression_summary_missing(monkeypatch):
    class FakeDB:
        def __init__(self, db_path: str):
            self.db_path = db_path

        def get_distinct_tickers(self, limit=None):
            return ["AAPL"]

        def get_forecast_regression_summary(self, start_date=None, end_date=None, model_type=None):
            return {}

        def save_strategy_config(self, regime=None, params=None, metrics=None, score=None):
            return 1

    class FakeOptimizer:
        def __init__(self, search_space, objectives, constraints=None, random_state=None):
            self.search_space = search_space

        def run(self, n_candidates, evaluation_fn, regime=None):
            candidate = strategy_opt.StrategyCandidate(params={"alpha": 0.5}, regime=regime)
            evaluation_fn(candidate)
            return []

    def fake_simulate_candidate(**kwargs):
        return {
            "total_profit": 10.0,
            "total_return": 0.10,
            "alpha": 0.01,
            "information_ratio": 0.8,
            "beta": 1.0,
            "tracking_error": 0.12,
            "r_squared": 0.50,
            "profit_factor": 1.5,
            "win_rate": 0.40,
            "total_trades": 4,
            "max_drawdown": 0.05,
            "omega_ratio": 1.20,
            "payoff_asymmetry": 1.05,
            "payoff_asymmetry_effective": 1.02,
            "expected_shortfall": -0.02,
            "cvar_95": -0.03,
            "fractional_kelly_fat_tail": 0.03,
            "barbell_path_risk_ok": True,
            "regime_realism_ok": True,
            "anti_barbell_ok": False,
            "anti_barbell_reason": "path_risk",
            "rmse_ratio_vs_baseline": 1.20,
            "rmse_within_threshold": 0.0,
        }

    monkeypatch.setattr(strategy_opt, "DatabaseManager", FakeDB)
    monkeypatch.setattr(strategy_opt, "StrategyOptimizer", FakeOptimizer)
    monkeypatch.setattr(strategy_opt, "simulate_candidate", fake_simulate_candidate)

    with pytest.raises(Exception) as excinfo:
        strategy_opt.main.callback(
            config_path="config/strategy_optimization_config.yml",
            db_path="data/portfolio_maximizer.db",
            n_candidates=1,
            regime="default",
            verbose=False,
        )

    assert "requires ensemble and baseline RMSE" in str(excinfo.value)


def test_strategy_optimization_config_has_no_win_rate_constraint():
    """Barbell policy: win_rate must NOT be a hard constraint.
    Low win rate is valid when payoff asymmetry is large (p*avg_win > (1-p)*avg_loss).
    """
    cfg_path = Path("config/strategy_optimization_config.yml")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    constraints = cfg.get("strategy_optimization", {}).get("constraints", {})
    min_constraints = constraints.get("min", {})
    assert "win_rate" not in min_constraints, (
        "win_rate must not be a hard min constraint — barbell asymmetry policy permits "
        "low win rate when avg_win/avg_loss is large. Remove 'win_rate' from constraints.min."
    )


def test_strategy_optimization_config_barbell_objectives_present():
    """Barbell policy: payoff asymmetry and tail metrics must be in optimizer objectives."""
    cfg_path = Path("config/strategy_optimization_config.yml")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    root = cfg.get("strategy_optimization", {})
    objectives = root.get("objectives", {})
    search_space = root.get("search_space", {})
    assert "omega_ratio" in objectives, "omega_ratio must be an optimizer objective (barbell asymmetry)"
    assert "payoff_asymmetry_effective" in objectives, "payoff_asymmetry_effective must be an optimizer objective (support-aware asymmetry engine)"
    assert "expected_shortfall" in objectives, "expected_shortfall must be an optimizer objective (downside bound)"
    assert objectives.get("expected_shortfall", 0) < 0, "expected_shortfall weight must be negative (penalize tail loss)"
    assert "alpha" in objectives, "alpha must be an optimizer objective (benchmark-relative excess return)"
    assert "information_ratio" in objectives, "information_ratio must be an optimizer objective (benchmark-relative quality)"
    for key in (
        "ensemble_weight_sarimax",
        "ensemble_weight_garch",
        "ensemble_weight_samossa",
        "ensemble_weight_mssa_rl",
    ):
        assert key in search_space, f"{key} must be in the candidate search space for ensemble routing"
