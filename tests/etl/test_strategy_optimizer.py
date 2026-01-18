import math

from etl.strategy_optimizer import StrategyOptimizer, StrategyCandidate
from etl.database_manager import DatabaseManager


def test_sample_candidate_respects_search_space_bounds():
    search_space = {
        "weight": {"type": "continuous", "bounds": [0.0, 1.0]},
        "kelly_cap": {"type": "integer", "bounds": [1, 5]},
        "style": {"type": "categorical", "choices": ["market", "limit_bias"]},
    }
    optimizer = StrategyOptimizer(
        search_space=search_space,
        objectives={},
        random_state=42,
    )

    candidate = optimizer.sample_candidate(regime="test_regime")

    assert 0.0 <= candidate.params["weight"] <= 1.0
    assert 1 <= candidate.params["kelly_cap"] <= 5
    assert candidate.params["style"] in ["market", "limit_bias"]
    assert candidate.regime == "test_regime"


def test_run_applies_constraints_and_sorts_by_score():
    search_space = {
        "alpha": {"type": "continuous", "bounds": [0.0, 1.0]},
    }
    objectives = {"total_return": 1.0, "max_drawdown": -1.0}
    constraints = {"min": {"win_rate": 0.5}, "max": {"max_drawdown": 0.2}}

    optimizer = StrategyOptimizer(
        search_space=search_space,
        objectives=objectives,
        constraints=constraints,
        random_state=0,
    )

    def evaluation_fn(candidate: StrategyCandidate):
        # Higher alpha gives higher total_return; other metrics fixed.
        w = float(candidate.params["alpha"])
        return {
            "total_return": w,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
        }

    evaluations = optimizer.run(
        n_candidates=10,
        evaluation_fn=evaluation_fn,
        regime="r1",
    )

    assert evaluations, "Expected at least one evaluation"
    # All evaluations must satisfy constraints
    assert all(e.metrics["win_rate"] >= 0.5 for e in evaluations)
    assert all(e.metrics["max_drawdown"] <= 0.2 for e in evaluations)
    # Scores must be sorted descending
    scores = [e.score for e in evaluations]
    assert scores == sorted(scores, reverse=True)


def test_constraints_can_filter_all_candidates():
    search_space = {"alpha": {"type": "continuous", "bounds": [0.0, 1.0]}}
    constraints = {"min": {"win_rate": 0.9}}  # Intentionally strict
    optimizer = StrategyOptimizer(
        search_space=search_space,
        objectives={"total_return": 1.0},
        constraints=constraints,
        random_state=1,
    )

    def evaluation_fn(candidate: StrategyCandidate):
        return {
            "total_return": 0.5,
            "win_rate": 0.5,
        }

    evaluations = optimizer.run(
        n_candidates=5,
        evaluation_fn=evaluation_fn,
        regime=None,
    )

    assert evaluations == []


def test_constraints_bypass_when_no_trades():
    search_space = {"alpha": {"type": "continuous", "bounds": [0.0, 1.0]}}
    constraints = {"min": {"win_rate": 0.9}}
    optimizer = StrategyOptimizer(
        search_space=search_space,
        objectives={"total_return": 1.0},
        constraints=constraints,
        random_state=1,
    )

    def evaluation_fn(candidate: StrategyCandidate):
        return {
            "total_return": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,  # triggers constraint bypass
        }

    evaluations = optimizer.run(
        n_candidates=3,
        evaluation_fn=evaluation_fn,
        regime=None,
    )

    assert evaluations, "Zero-trade candidates should bypass constraints"


def test_strategy_config_persistence_roundtrip(tmp_path):
    db_file = tmp_path / "test_strategy.db"
    db = DatabaseManager(db_path=str(db_file))

    params = {"alpha": 0.3, "style": "market"}
    metrics = {"total_return": 10.0, "win_rate": 0.6}

    row_id = db.save_strategy_config(
        regime="test_regime",
        params=params,
        metrics=metrics,
        score=1.23,
    )
    assert row_id > 0

    best = db.get_best_strategy_config("test_regime")
    assert best is not None
    assert best["regime"] == "test_regime"
    assert best["params"]["alpha"] == params["alpha"]
    assert best["params"]["style"] == params["style"]
    assert best["metrics"]["total_return"] == metrics["total_return"]


def test_equity_curve_and_backtester_sanity(tmp_path):
    from backtesting.candidate_backtester import backtest_candidate
    import pandas as pd

    db_file = tmp_path / "test_strategy.db"
    db = DatabaseManager(db_path=str(db_file))

    # Seed minimal OHLCV data
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    sample = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 5,
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
            "volume": [1000, 1000, 1000, 1000, 1000],
            "date": dates.strftime("%Y-%m-%d"),
        }
    )
    sample.to_sql("ohlcv_data", db.conn, if_exists="append", index=False)

    result = backtest_candidate(
        db_manager=db,
        tickers=["AAPL"],
        start="2024-01-01",
        end="2024-01-10",
        candidate_params={"diversification_penalty": 0.0, "sizing_kelly_fraction_cap": 0.1},
        guardrails={"min_expected_return": 0.0},
    )

    assert result.total_trades >= 0
    assert result.profit_factor >= 0
