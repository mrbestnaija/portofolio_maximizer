"""
Tests for scripts/compute_ticker_eligibility.py
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path


def _make_row(
    ticker: str,
    n: int,
    wr: float,
    pf: float,
    *,
    omega: float = 0.0,
    payoff: float = 0.0,
    tp_freq: float = 0.0,
) -> dict:
    return {
        "ticker": ticker,
        "n_trades": n,
        "win_rate": wr,
        "profit_factor": pf,
        "omega_ratio": omega,
        "payoff_asymmetry_effective": payoff,
        "take_profit_frequency": tp_freq,
    }


def _make_db(tmp_path: Path, rows: list[dict]) -> Path:
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            trade_date TEXT,
            realized_pnl REAL,
            entry_price REAL,
            close_size REAL,
            shares REAL,
            exit_reason TEXT,
            holding_period_days INTEGER,
            effective_horizon INTEGER,
            is_close INTEGER DEFAULT 1,
            is_diagnostic INTEGER DEFAULT 0,
            is_synthetic INTEGER DEFAULT 0
        )
        """
    )
    for row in rows:
        n_trades = int(row["n_trades"])
        trade_date = str(row.get("trade_date") or date.today().isoformat())
        take_profit_count = int(row.get("take_profit_count", max(1, round(n_trades * row.get("take_profit_frequency", 0.0)))))
        stop_loss_count = int(row.get("stop_loss_count", max(0, n_trades - take_profit_count)))
        time_exit_count = int(row.get("time_exit_count", 0))
        entry_price = float(row.get("entry_price", 100.0))
        tp_pnl = float(row.get("take_profit_pnl", 100.0))
        stop_pnl = float(row.get("stop_loss_pnl", -20.0))
        time_pnl = float(row.get("time_exit_pnl", 0.0))
        holding_period_tp = int(row.get("take_profit_holding_period", 2))
        holding_period_stop = int(row.get("stop_loss_holding_period", 4))
        holding_period_time = int(row.get("time_exit_holding_period", 10))
        for _ in range(take_profit_count):
            conn.execute(
                """
                INSERT INTO trade_executions
                    (ticker, trade_date, realized_pnl, entry_price, close_size, shares, exit_reason,
                     holding_period_days, effective_horizon, is_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (row["ticker"], trade_date, tp_pnl, entry_price, 1.0, 1.0, "TAKE_PROFIT", holding_period_tp, 10),
            )
        for _ in range(stop_loss_count):
            conn.execute(
                """
                INSERT INTO trade_executions
                    (ticker, trade_date, realized_pnl, entry_price, close_size, shares, exit_reason,
                     holding_period_days, effective_horizon, is_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (row["ticker"], trade_date, stop_pnl, entry_price, 1.0, 1.0, "STOP_LOSS", holding_period_stop, 10),
            )
        for _ in range(time_exit_count):
            conn.execute(
                """
                INSERT INTO trade_executions
                    (ticker, trade_date, realized_pnl, entry_price, close_size, shares, exit_reason,
                     holding_period_days, effective_horizon, is_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (row["ticker"], trade_date, time_pnl, entry_price, 1.0, 1.0, "TIME_EXIT", holding_period_time, 10),
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

        assert (
            classify_ticker(
                _make_row("NVDA", 25, 0.60, 2.0, omega=1.2, payoff=2.5, tp_freq=0.20),
                set(),
            )
            == "HEALTHY"
        )

    def test_weak_low_wr_sufficient_trades(self):
        from scripts.compute_ticker_eligibility import classify_ticker

        assert (
            classify_ticker(
                _make_row("AAPL", 7, 0.14, 0.50, omega=1.1, payoff=1.1, tp_freq=0.06),
                set(),
            )
            == "WEAK"
        )

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
                {
                    "ticker": "NVDA",
                    "n_trades": 25,
                    "win_rate": 0.60,
                    "profit_factor": 2.5,
                    "take_profit_count": 20,
                    "stop_loss_count": 5,
                    "take_profit_pnl": 120.0,
                    "stop_loss_pnl": -20.0,
                    "omega_ratio": 1.2,
                    "payoff_asymmetry_effective": 2.5,
                    "take_profit_frequency": 0.8,
                },
                {
                    "ticker": "AAPL",
                    "n_trades": 7,
                    "win_rate": 0.14,
                    "profit_factor": 0.5,
                    "take_profit_count": 1,
                    "stop_loss_count": 6,
                    "take_profit_pnl": 40.0,
                    "stop_loss_pnl": -20.0,
                    "omega_ratio": 0.9,
                    "payoff_asymmetry_effective": 1.1,
                    "take_profit_frequency": 0.14,
                },
            ],
        )
        result = compute_eligibility(db_path=db)
        assert set(result.keys()) >= {
            "generated_utc",
            "db_path",
            "window",
            "source_view",
            "n_tickers",
            "tickers",
            "summary",
            "routing_note",
            "thresholds",
            "source_thresholds",
        }
        assert result["summary"]["HEALTHY"] == 1
        assert result["summary"]["WEAK"] == 1
        assert result["window"]["state"] == "rolling_window"
        assert result["source_view"] == "production_closed_trades"
        assert result["tickers"]["NVDA"]["beats_ngn_hurdle"] is True
        assert result["tickers"]["NVDA"]["take_profit_frequency"] >= 0.05
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

    def test_rolling_window_re_admits_recent_recovery(self, tmp_path):
        from scripts.compute_ticker_eligibility import compute_eligibility

        as_of = date(2026, 4, 22).isoformat()
        recent_start = date.fromisoformat(as_of) - timedelta(days=9)
        old_date = (date.fromisoformat(as_of) - timedelta(days=400)).isoformat()
        recent_rows = []
        for offset in range(20):
            recent_rows.append(
                {
                    "ticker": "AAPL",
                    "n_trades": 1,
                    "win_rate": 1.0,
                    "profit_factor": 99.0,
                    "take_profit_count": 1,
                    "stop_loss_count": 0,
                    "take_profit_pnl": 120.0,
                    "stop_loss_pnl": -20.0,
                    "omega_ratio": 1.8,
                    "payoff_asymmetry_effective": 2.4,
                    "take_profit_frequency": 1.0,
                    "trade_date": (recent_start + timedelta(days=offset % 10)).isoformat(),
                }
            )
        for offset in range(5):
            recent_rows.append(
                {
                    "ticker": "AAPL",
                    "n_trades": 1,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "take_profit_count": 0,
                    "stop_loss_count": 1,
                    "take_profit_pnl": 120.0,
                    "stop_loss_pnl": -20.0,
                    "omega_ratio": 0.4,
                    "payoff_asymmetry_effective": 0.7,
                    "take_profit_frequency": 0.0,
                    "trade_date": (recent_start + timedelta(days=offset % 10)).isoformat(),
                }
            )
        old_rows = [
            {
                "ticker": "AAPL",
                "n_trades": 1,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "take_profit_count": 0,
                "stop_loss_count": 1,
                "take_profit_pnl": 80.0,
                "stop_loss_pnl": -100.0,
                "omega_ratio": 0.5,
                "payoff_asymmetry_effective": 0.6,
                "take_profit_frequency": 0.0,
                "trade_date": old_date,
            }
            for _ in range(20)
        ]
        db = _make_db(tmp_path, old_rows + recent_rows)

        rolling = compute_eligibility(db_path=db, lookback_days=30, as_of_date=as_of)
        lifetime = compute_eligibility(db_path=db, lookback_days=800, as_of_date=as_of)

        assert rolling["window"]["state"] == "rolling_window"
        assert rolling["window"]["source_view"] == "production_closed_trades"
        assert rolling["tickers"]["AAPL"]["status"] == "HEALTHY"
        assert lifetime["tickers"]["AAPL"]["status"] in {"WEAK", "LAB_ONLY"}
        assert rolling["tickers"]["AAPL"]["n_trades"] < lifetime["tickers"]["AAPL"]["n_trades"]


class TestCLI:
    def test_json_output_schema(self, tmp_path, capsys):
        from scripts.compute_ticker_eligibility import main

        db = _make_db(
            tmp_path,
            [
                {
                    "ticker": "NVDA",
                    "n_trades": 25,
                    "win_rate": 0.60,
                    "profit_factor": 2.5,
                    "take_profit_count": 20,
                    "stop_loss_count": 5,
                    "take_profit_pnl": 120.0,
                    "stop_loss_pnl": -20.0,
                    "omega_ratio": 1.2,
                    "payoff_asymmetry_effective": 2.5,
                    "take_profit_frequency": 0.8,
                }
            ],
        )
        out = tmp_path / "elig.json"
        rc = main(["--db", str(db), "--output", str(out), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["summary"]["HEALTHY"] == 1
        assert payload["summary"]["WEAK"] == 0
        assert payload["tickers"]["NVDA"]["take_profit_frequency"] >= 0.05
