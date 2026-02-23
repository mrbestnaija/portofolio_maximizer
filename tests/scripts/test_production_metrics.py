"""Tests for production metrics API shape and integrity-enforcer filtering."""
import os
import pytest
from pathlib import Path
from etl.database_manager import DatabaseManager

# Canonical DB location
_DEFAULT_DB = str(Path(__file__).parent.parent.parent / "data" / "portfolio_maximizer.db")
_DB_PATH = os.environ.get("PORTFOLIO_DB_PATH", _DEFAULT_DB)
_DB_EXISTS = os.path.exists(_DB_PATH)


def test_performance_summary_returns_expected_keys():
    """get_performance_summary() returns a dict with the required metric keys."""
    with DatabaseManager() as db:
        data = db.get_performance_summary()
    assert isinstance(data, dict), "Should return a dict"
    required_keys = {"total_trades", "total_profit", "win_rate", "profit_factor"}
    missing = required_keys - data.keys()
    assert not missing, f"Missing keys in summary: {missing}"


def test_performance_summary_types():
    """get_performance_summary() numeric fields are numeric (or None)."""
    with DatabaseManager() as db:
        data = db.get_performance_summary()
    assert isinstance(data["total_trades"], int), "total_trades should be int"
    if data["total_profit"] is not None:
        assert isinstance(data["total_profit"], (int, float)), "total_profit should be numeric"


@pytest.mark.skipif(not _DB_EXISTS, reason=f"DB not found at {_DB_PATH}")
def test_integrity_enforcer_canonical_metrics():
    """PnLIntegrityEnforcer.get_canonical_metrics() returns production-only metrics."""
    from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer

    with PnLIntegrityEnforcer(_DB_PATH) as enforcer:
        metrics = enforcer.get_canonical_metrics()
    assert hasattr(metrics, "total_realized_pnl"), "Metrics should have total_realized_pnl"
    assert hasattr(metrics, "win_rate"), "Metrics should have win_rate"
    assert hasattr(metrics, "total_round_trips"), "Metrics should have total_round_trips"
    if metrics.total_round_trips > 0:
        assert 0.0 <= metrics.win_rate <= 1.0, f"win_rate out of range: {metrics.win_rate}"
