"""
Tests for scripts/compute_ticker_eligibility.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _make_row(ticker: str, n: int, wr: float, pf: float) -> dict:
    return {"ticker": ticker, "n_trades": n, "win_rate": wr, "profit_factor": pf}


def _make_db(tmp_path: Path, rows: list[dict]) -> Path:
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            realized_pnl REAL,
            is_close INTEGER DEFAULT 1,
            is_diagnostic INTEGER DEFAULT 0,
            is_synthetic INTEGER DEFAULT 0
        )
        """
    )
    for row in rows:
        wins = int(row["n_trades"] * row["win_rate"] + 0.5)
        losses = row["n_trades"] - wins
        win_pnl = 100.0 if row["profit_factor"] > 1.0 else 50.0
        loss_pnl = -100.0 / row["profit_factor"] if row["profit_factor"] > 0.001 else -100.0
        for _ in range(wins):
            conn.execute(
                "INSERT INTO trade_executions(ticker, realized_pnl, is_close) VALUES (?, ?, 1)",
                (row["ticker"], win_pnl),
            )
        for _ in range(losses):
            conn.execute(
                "INSERT INTO trade_executions(ticker, realized_pnl, is_close) VALUES (?, ?, 1)",
                (row["ticker"], loss_pnl),
            )
    conn.execute(
        """
        CREATE VIEW production_closed_trades AS
        SELECT * FROM trade_executions
        WHERE is_close = 1
          AND COALESCE(is_diagnostic, 0) = 0
          AND COALESCE(is_synthetic, 0) = 0
        """
    )
    conn.commit()
    conn.close()
    return db


class TestClassification:
    def test_healthy_all_criteria_met(self):
        from scripts.compute_ticker_eligibility import classify_ticker

        assert classify_ticker(_make_row("NVDA", 25, 0.60, 2.0), set()) == "HEALTHY"

    def test_weak_low_wr_sufficient_trades(self):
        from scripts.compute_ticker_eligibility import classify_ticker

        assert classify_ticker(_make_row("AAPL", 7, 0.14, 0.50), set()) == "WEAK"

    def test_lab_only_explicit_flag(self):
        from scripts.compute_ticker_eligibility import classify_ticker

        assert classify_ticker(_make_row("AAPL", 25, 0.60, 2.0), {"AAPL"}) == "LAB_ONLY"

    def test_borderline_path_collapses_to_lab_only(self):
        from scripts.compute_ticker_eligibility import classify_ticker_details

        status, reasons = classify_ticker_details(_make_row("MSFT", 20, 0.40, 1.50), set())
        assert status == "LAB_ONLY"
        assert reasons


class TestComputation:
    def test_result_schema_and_reasons(self, tmp_path):
        from scripts.compute_ticker_eligibility import compute_eligibility

        db = _make_db(
            tmp_path,
            [
                {"ticker": "NVDA", "n_trades": 25, "win_rate": 0.60, "profit_factor": 2.5},
                {"ticker": "AAPL", "n_trades": 7, "win_rate": 0.14, "profit_factor": 0.5},
            ],
        )
        result = compute_eligibility(db_path=db)
        assert set(result.keys()) >= {
            "generated_utc",
            "db_path",
            "n_tickers",
            "tickers",
            "summary",
            "routing_note",
            "thresholds",
            "source_thresholds",
        }
        assert "BORDERLINE" not in result["summary"]
        assert "source_hashes" in result["thresholds"]
        assert all("reasons" in info and info["reasons"] for info in result["tickers"].values())

    def test_threshold_source_lock(self):
        from scripts.compute_ticker_eligibility import (
            HEALTHY_MIN_PROFIT_FACTOR,
            HEALTHY_MIN_TRADES,
            HEALTHY_MIN_WIN_RATE,
        )
        from scripts.robustness_thresholds import (
            R3_MIN_PROFIT_FACTOR,
            R3_MIN_TRADES,
            R3_MIN_WIN_RATE,
        )

        assert HEALTHY_MIN_WIN_RATE == R3_MIN_WIN_RATE
        assert HEALTHY_MIN_PROFIT_FACTOR == R3_MIN_PROFIT_FACTOR
        assert HEALTHY_MIN_TRADES == R3_MIN_TRADES

    def test_missing_db_returns_empty(self, tmp_path):
        from scripts.compute_ticker_eligibility import compute_eligibility

        result = compute_eligibility(db_path=tmp_path / "missing.db")
        assert result["n_tickers"] == 0
        assert result["tickers"] == {}
        assert "db_missing" in result["errors"]
        assert "eligibility_query_error" in result["warnings"]


class TestCLI:
    def test_json_output_schema(self, tmp_path, capsys):
        from scripts.compute_ticker_eligibility import main

        db = _make_db(
            tmp_path,
            [{"ticker": "NVDA", "n_trades": 25, "win_rate": 0.60, "profit_factor": 2.5}],
        )
        out = tmp_path / "elig.json"
        rc = main(["--db", str(db), "--output", str(out), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["summary"]["HEALTHY"] == 1
        assert "BORDERLINE" not in payload["summary"]
