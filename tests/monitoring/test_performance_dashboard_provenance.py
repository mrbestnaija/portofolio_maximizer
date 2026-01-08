from __future__ import annotations

from datetime import datetime

from etl.database_manager import DatabaseManager
from monitoring.performance_dashboard import PerformanceDashboard


def test_performance_dashboard_flags_synthetic_not_proof(tmp_path):
    db_path = tmp_path / "dashboard.db"
    db = DatabaseManager(str(db_path))
    db.save_trade_execution(
        ticker="AAPL",
        trade_date=datetime.utcnow(),
        action="BUY",
        shares=1.0,
        price=100.0,
        total_value=100.0,
        commission=0.0,
        data_source="synthetic",
        execution_mode="synthetic",
        synthetic_dataset_id="syn_dashboard",
        synthetic_generator_version="v1",
        run_id="run_dash",
        realized_pnl=1.0,
        realized_pnl_pct=0.01,
    )
    db.close()

    dashboard = PerformanceDashboard(db_path=str(db_path))
    snapshot = dashboard.generate_live_metrics(lookback_days=30)

    assert snapshot.metrics["data_origin"] == "synthetic"
    assert snapshot.metrics["profitability_proof"] is False
    assert snapshot.alerts and "Synthetic data present" in snapshot.alerts[0]
    assert snapshot.provenance.get("origin") == "synthetic"

