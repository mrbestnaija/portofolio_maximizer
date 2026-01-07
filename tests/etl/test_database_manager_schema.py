"""Schema migration and validation tests for DatabaseManager."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

from etl.database_manager import DatabaseManager


def _create_legacy_llm_risks_schema(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_risks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            assessment_date DATE NOT NULL,
            risk_level TEXT CHECK(risk_level IN ('low', 'medium', 'high')),
            risk_score INTEGER CHECK(risk_score BETWEEN 0 AND 100),
            portfolio_weight REAL,
            concerns TEXT,
            recommendation TEXT,
            model_name TEXT NOT NULL,
            var_95 REAL,
            max_drawdown REAL,
            volatility REAL,
            latency_seconds REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, assessment_date, model_name)
        )
        """
    )
    conn.commit()
    conn.close()


def test_migrates_risk_level_constraint_to_include_extreme(tmp_path: Path):
    """Existing databases without 'extreme' risk level are upgraded automatically."""
    db_file = tmp_path / "legacy_pm.db"
    _create_legacy_llm_risks_schema(db_file)

    manager = DatabaseManager(str(db_file))
    try:
        risk_record: Dict[str, float] = {
            'risk_level': 'extreme',
            'risk_score': 95,
            'portfolio_weight': 0.25,
            'concerns': ['volatility spike'],
            'recommendation': 'Reduce exposure',
        }
        row_id = manager.save_llm_risk(
            ticker='TEST',
            date='2025-10-30',
            risk=risk_record,
            model_name='qwen:14b-chat-q4_K_M',
            latency=1.2,
        )
        assert row_id != -1

        cursor = manager.conn.cursor()
        cursor.execute("SELECT risk_level FROM llm_risks WHERE ticker = 'TEST'")
        stored = cursor.fetchone()
        assert stored is not None
        assert stored[0] == 'extreme'
    finally:
        manager.conn.close()


def test_save_signal_validation_records_audit_trail(tmp_path: Path):
    db_file = tmp_path / "validation_pm.db"
    manager = DatabaseManager(str(db_file))
    try:
        signal_id = manager.save_llm_signal(
            ticker='AAPL',
            date='2025-10-30',
            signal={'action': 'BUY', 'confidence': 0.8, 'reasoning': 'Test case', 'entry_price': 150.0},
            model_name='deepseek-coder:6.7b-instruct-q4_K_M',
            latency=0.5,
            validation_status='validated'
        )
        assert signal_id != -1

        validation_payload = {
            'validator_version': 'v1',
            'confidence_score': 0.82,
            'recommendation': 'BUY',
            'warnings': ['synthetic warning'],
            'quality_metrics': {'basic_validation': 1.0},
        }
        validation_id = manager.save_signal_validation(signal_id, validation_payload)
        assert validation_id != -1

        cursor = manager.conn.cursor()
        cursor.execute("SELECT recommendation, warnings FROM llm_signal_validations WHERE signal_id = ?", (signal_id,))
        stored = cursor.fetchone()
        assert stored is not None
        assert stored[0] == 'BUY'
        assert 'synthetic warning' in stored[1]
    finally:
        manager.conn.close()


def test_recent_signals_backtest_helpers(tmp_path: Path):
    """DatabaseManager exposes helper utilities for signal analytics."""
    db_file = tmp_path / "signal_pm.db"
    manager = DatabaseManager(str(db_file))
    try:
        # Seed historical signals across several days
        base_date = datetime(2025, 1, 1)
        for offset in range(5):
            signal_date = (base_date + timedelta(days=offset)).strftime("%Y-%m-%d")
            manager.save_llm_signal(
                ticker='MSFT',
                date=signal_date,
                signal={
                    'action': 'BUY' if offset % 2 == 0 else 'SELL',
                    'confidence': 0.65,
                    'reasoning': f'synthetic #{offset}',
                    'entry_price': 100 + offset,
                    'signal_timestamp': (base_date + timedelta(days=offset)).isoformat(),
                },
                model_name='deepseek-coder:6.7b-instruct-q4_K_M',
                latency=0.4,
                validation_status='validated',
            )

        reference_ts = base_date + timedelta(days=4)
        recent = manager.fetch_recent_signals(
            'MSFT',
            reference_timestamp=reference_ts,
            lookback_days=3,
            limit=10,
        )
        assert len(recent) >= 3
        assert all(row['ticker'] == 'MSFT' for row in recent)

        latest_signal_id = recent[-1]['id']
        manager.update_signal_performance(
            latest_signal_id,
            {
                'actual_return': 0.05,
                'annual_return': 0.12,
                'sharpe_ratio': 1.1,
                'information_ratio': 0.3,
                'hit_rate': 0.6,
                'profit_factor': 1.4,
            },
        )

        cursor = manager.conn.cursor()
        cursor.execute(
            "SELECT actual_return, backtest_annual_return, backtest_hit_rate "
            "FROM llm_signals WHERE id = ?",
            (latest_signal_id,),
        )
        stored_metrics = cursor.fetchone()
        assert stored_metrics is not None
        assert stored_metrics[0] == 0.05
        assert stored_metrics[1] == 0.12
        assert stored_metrics[2] == 0.6

        report = SimpleNamespace(
            trades_analyzed=5,
            hit_rate=0.6,
            profit_factor=1.4,
            sharpe_ratio=1.1,
            annual_return=0.12,
            information_ratio=0.3,
            information_coefficient=0.2,
            p_value=0.04,
            statistically_significant=True,
            bootstrap_intervals={'sharpe_ratio_ci': (0.9, 1.2)},
            statistical_summary={'p_value': 0.04, 'significant': True},
            autocorrelation={'ljung_box_p': 0.7},
        )
        manager.save_signal_backtest_summary('MSFT', lookback_days=30, report=report)

        cursor.execute(
            "SELECT hit_rate, profit_factor, sharpe_ratio "
            "FROM llm_signal_backtests WHERE ticker = ?",
            ('MSFT',),
        )
        stored_summary = cursor.fetchone()
        assert stored_summary is not None
        assert stored_summary[0] == 0.6
        assert stored_summary[1] == 1.4
        assert stored_summary[2] == 1.1
    finally:
        manager.conn.close()


def test_forecast_regression_metrics_column(tmp_path: Path):
    db_path = tmp_path / "forecast_metrics.db"
    manager = DatabaseManager(str(db_path))
    try:
        forecast_payload = {
            "model_type": "SARIMAX",
            "forecast_horizon": 1,
            "forecast_value": 150.25,
            "model_order": {"order": (1, 1, 1)},
            "aic": 100.5,
            "bic": 110.2,
            "diagnostics": {"ljung_box_pvalue": 0.9},
            "regression_metrics": {"rmse": 0.5, "smape": 0.1, "tracking_error": 0.2},
        }
        row_id = manager.save_forecast("AAPL", "2025-11-08", forecast_payload)
        assert row_id != -1

        cursor = manager.conn.cursor()
        cursor.execute(
            "SELECT regression_metrics FROM time_series_forecasts WHERE id = ?",
            (row_id,),
        )
        stored = cursor.fetchone()
        assert stored is not None
        assert '"rmse": 0.5' in stored[0]
    finally:
        manager.close()


def test_get_forecasts_and_update_regression_metrics(tmp_path: Path) -> None:
    db_path = tmp_path / "forecast_query.db"
    manager = DatabaseManager(str(db_path))
    try:
        combined_id = manager.save_forecast(
            "AAPL",
            "2025-11-09",
            {
                "model_type": "COMBINED",
                "forecast_horizon": 2,
                "forecast_value": 101.0,
                "model_order": {},
            },
        )
        sarimax_id = manager.save_forecast(
            "AAPL",
            "2025-11-09",
            {
                "model_type": "SARIMAX",
                "forecast_horizon": 2,
                "forecast_value": 100.0,
                "model_order": {},
            },
        )
        assert combined_id != -1
        assert sarimax_id != -1

        combined_rows = manager.get_forecasts("AAPL", model_types=["COMBINED"], limit=10)
        assert len(combined_rows) == 1
        assert combined_rows[0]["id"] == combined_id

        updated = manager.update_forecast_regression_metrics(combined_id, {"rmse": 0.123, "smape": 0.5})
        assert updated is True

        refreshed = manager.get_forecasts("AAPL", model_types=["COMBINED"], limit=10)
        assert len(refreshed) == 1
        raw_metrics = refreshed[0].get("regression_metrics")
        assert raw_metrics
        assert '"rmse": 0.123' in raw_metrics
    finally:
        manager.close()


def test_performance_summary_filters_by_run_id(tmp_path: Path):
    db_path = tmp_path / "perf_summary.db"
    manager = DatabaseManager(str(db_path))
    try:
        manager.save_trade_execution(
            ticker="AAPL",
            trade_date="2026-01-01",
            action="SELL",
            shares=1,
            price=100.0,
            total_value=100.0,
            commission=0.0,
            run_id="run1",
            realized_pnl=10.0,
            realized_pnl_pct=0.10,
        )
        manager.save_trade_execution(
            ticker="AAPL",
            trade_date="2026-01-01",
            action="SELL",
            shares=1,
            price=99.0,
            total_value=99.0,
            commission=0.0,
            run_id="run1",
            realized_pnl=-5.0,
            realized_pnl_pct=-0.05,
        )
        manager.save_trade_execution(
            ticker="MSFT",
            trade_date="2026-01-01",
            action="SELL",
            shares=1,
            price=200.0,
            total_value=200.0,
            commission=0.0,
            run_id="run2",
            realized_pnl=20.0,
            realized_pnl_pct=0.10,
        )

        run1 = manager.get_performance_summary(run_id="run1")
        assert run1["total_trades"] == 2
        assert run1["win_rate"] == 0.5
        assert run1["profit_factor"] == 2.0

        run2 = manager.get_performance_summary(run_id="run2")
        assert run2["total_trades"] == 1
        assert run2["win_rate"] == 1.0
        assert run2["profit_factor"] == float("inf")

        overall = manager.get_performance_summary()
        assert overall["total_trades"] == 3
    finally:
        manager.close()
