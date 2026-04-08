from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_strategy_optimization as strategy_opt


def test_run_strategy_optimization_keeps_low_win_rate_candidate_when_asymmetry_is_strong(monkeypatch):
    saved_metrics: list[dict] = []

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

        def run(self, n_candidates, evaluation_fn, regime=None):
            candidate = strategy_opt.StrategyCandidate(params={"alpha": 0.5}, regime=regime)
            metrics = evaluation_fn(candidate)
            return [SimpleNamespace(candidate=candidate, metrics=metrics, score=1.0)]

    def fake_backtest_candidate(**kwargs):
        returns = pd.Series([0.04, -0.01, 0.05, -0.015, 0.03, -0.01, 0.06, -0.02, 0.04, -0.01, 0.05, -0.015])
        return SimpleNamespace(
            total_profit=25.0,
            total_return=0.25,
            profit_factor=2.1,
            win_rate=0.20,
            total_trades=6,
            max_drawdown=0.08,
            strategy_returns=returns,
        )

    monkeypatch.setattr(strategy_opt, "DatabaseManager", FakeDB)
    monkeypatch.setattr(strategy_opt, "StrategyOptimizer", FakeOptimizer)
    monkeypatch.setattr(strategy_opt, "backtest_candidate", fake_backtest_candidate)

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
    assert saved_metrics[0]["profit_factor"] == 2.1
    assert saved_metrics[0]["omega_ratio"] is not None


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

    def fake_backtest_candidate(**kwargs):
        returns = pd.Series([0.02] * 12)
        return SimpleNamespace(
            total_profit=10.0,
            total_return=0.10,
            profit_factor=1.5,
            win_rate=0.40,
            total_trades=4,
            max_drawdown=0.05,
            strategy_returns=returns,
        )

    monkeypatch.setattr(strategy_opt, "DatabaseManager", FakeDB)
    monkeypatch.setattr(strategy_opt, "StrategyOptimizer", FakeOptimizer)
    monkeypatch.setattr(strategy_opt, "backtest_candidate", fake_backtest_candidate)

    with pytest.raises(Exception) as excinfo:
        strategy_opt.main.callback(
            config_path="config/strategy_optimization_config.yml",
            db_path="data/portfolio_maximizer.db",
            n_candidates=1,
            regime="default",
            verbose=False,
        )

    assert "requires ensemble and baseline RMSE" in str(excinfo.value)
