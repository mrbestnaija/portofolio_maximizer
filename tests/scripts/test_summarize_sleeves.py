from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from click.testing import CliRunner


def _write_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE trade_executions (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                trade_date TEXT,
                asset_class TEXT,
                instrument_type TEXT,
                realized_pnl REAL,
                is_close INTEGER DEFAULT 1,
                is_diagnostic INTEGER DEFAULT 0,
                is_synthetic INTEGER DEFAULT 0
            )
            """
        )
        as_of = date(2026, 4, 22)
        old_date = (as_of - timedelta(days=400)).isoformat()
        recent_dates = [
            (as_of - timedelta(days=3)).isoformat(),
            (as_of - timedelta(days=2)).isoformat(),
            (as_of - timedelta(days=1)).isoformat(),
        ]
        rows = [
            (1, "NVDA", old_date, "equity", "stock", -100.0, 1, 0, 0),
            (2, "NVDA", old_date, "equity", "stock", -100.0, 1, 0, 0),
            (3, "NVDA", recent_dates[0], "equity", "stock", 75.0, 1, 0, 0),
            (4, "NVDA", recent_dates[1], "equity", "stock", 80.0, 1, 0, 0),
            (5, "NVDA", recent_dates[2], "equity", "stock", 90.0, 1, 0, 0),
            (6, "BTC-USD", recent_dates[2], "crypto", "crypto", 500.0, 1, 0, 1),
        ]
        conn.executemany(
            """
            INSERT INTO trade_executions
                (id, ticker, trade_date, asset_class, instrument_type, realized_pnl,
                 is_close, is_diagnostic, is_synthetic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.execute(
            """
            CREATE VIEW production_closed_trades AS
            SELECT *
            FROM trade_executions
            WHERE is_close = 1
              AND COALESCE(is_diagnostic, 0) = 0
              AND COALESCE(is_synthetic, 0) = 0
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_summarize_sleeves_uses_canonical_rolling_window(tmp_path: Path, monkeypatch) -> None:
    from risk.barbell_policy import BarbellConfig
    from scripts import summarize_sleeves as mod

    db_path = tmp_path / "portfolio.db"
    out_path = tmp_path / "summary.json"
    _write_db(db_path)

    monkeypatch.setattr(
        mod.BarbellConfig,
        "from_yaml",
        classmethod(
            lambda cls: BarbellConfig(
                enable_barbell_allocation=True,
                enable_barbell_validation=True,
                enable_antifragility_tests=False,
                safe_min=0.0,
                safe_max=1.0,
                risk_max=1.0,
                safe_symbols=["CASH"],
                core_symbols=["NVDA"],
                speculative_symbols=["AAPL"],
                core_max=1.0,
                core_max_per=1.0,
                spec_max=1.0,
                spec_max_per=1.0,
                risk_symbols=["NVDA", "AAPL"],
            )
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        mod.main,
        [
            "--db-path",
            str(db_path),
            "--lookback-days",
            "30",
            "--as-of-date",
            "2026-04-22",
            "--min-trades",
            "2",
            "--output",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["source_view"] == "production_closed_trades"
    assert payload["window"]["state"] == "rolling_window"
    assert payload["window"]["lookback_days"] == 30
    assert payload["window"]["source_view"] == "production_closed_trades"
    assert payload["sleeves"][0]["ticker"] == "NVDA"
    assert payload["sleeves"][0]["trades"] == 3
    assert payload["sleeves"][0]["total_profit"] > 0
    assert "BTC-USD" not in {row["ticker"] for row in payload["sleeves"]}
