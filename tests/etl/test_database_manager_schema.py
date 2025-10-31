"""Schema migration and validation tests for DatabaseManager."""

import sqlite3
from pathlib import Path
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
