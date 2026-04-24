import json
import sqlite3
from pathlib import Path

import pandas as pd

from scripts import accumulate_classifier_labels as acc


def _seed_trade_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE trade_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                realized_pnl REAL,
                exit_reason TEXT,
                holding_period_days INTEGER,
                effective_horizon INTEGER,
                ticker_status_snapshot TEXT,
                ts_signal_id TEXT,
                is_close INTEGER,
                is_diagnostic INTEGER DEFAULT 0,
                is_synthetic INTEGER DEFAULT 0,
                is_contaminated INTEGER DEFAULT 0
            );
            """
        )
        conn.execute(
            """
            INSERT INTO trade_executions
            (ticker, realized_pnl, exit_reason, holding_period_days, effective_horizon,
             ticker_status_snapshot, ts_signal_id, is_close, is_diagnostic, is_synthetic, is_contaminated)
            VALUES
            ('AAPL', 25.0, 'TAKE_PROFIT', 3, 8, 'HEALTHY', 'ts_AAPL_1', 1, 0, 0, 0)
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_accumulate_prefers_db_ticker_status_snapshot(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "labels.db"
    jsonl_path = tmp_path / "quant_validation.jsonl"
    dataset_path = tmp_path / "directional_dataset.parquet"
    _seed_trade_db(db_path)
    jsonl_path.write_text(
        json.dumps(
            {
                "signal_id": "ts_AAPL_1",
                "ticker": "AAPL",
                "action": "BUY",
                "timestamp": "2026-04-21T10:00:00Z",
                "classifier_features": {
                    "snr": 2.4,
                    "ensemble_pred_return": 0.012,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        acc,
        "compute_eligibility",
        lambda db_path=None: {"tickers": {"AAPL": {"status": "LAB_ONLY"}}},
    )

    result = acc.accumulate(
        jsonl_path=jsonl_path,
        dataset_path=dataset_path,
        db_path=db_path,
    )

    assert result["n_matched"] == 1
    df = pd.read_parquet(dataset_path)
    assert df.loc[0, "ticker_status_snapshot"] == "HEALTHY"
    assert df.loc[0, "y_take_profit"] == 1
