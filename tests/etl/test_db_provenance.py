from __future__ import annotations

from datetime import datetime

from etl.database_manager import DatabaseManager


def test_record_run_provenance_sets_profitability_proof_false():
    db = DatabaseManager(":memory:")
    db.record_run_provenance(
        run_id="run_1",
        execution_mode="synthetic",
        data_source="synthetic",
        synthetic_dataset_id="syn_test",
        synthetic_generator_version="v1",
    )
    assert db.get_metadata("profitability_proof") == "false"
    assert db.get_metadata("profitability_proof_reason") == "synthetic_data"
    db.close()


def test_data_provenance_marks_synthetic_when_trades_tagged(tmp_path):
    db_path = tmp_path / "synthetic.db"
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
        synthetic_dataset_id="syn_demo",
        synthetic_generator_version="v1",
        run_id="run_demo",
        realized_pnl=1.0,
        realized_pnl_pct=0.01,
    )
    summary = db.get_data_provenance_summary()
    assert summary["origin"] == "synthetic"
    assert summary["trade_sources"]["synthetic"] >= 1
    assert "syn_demo" in summary["synthetic_dataset_ids"]
    db.close()

