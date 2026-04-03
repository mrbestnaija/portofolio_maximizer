"""
Tests for scripts/compute_context_quality.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _make_db(
    tmp_path: Path,
    trades: list[dict],
    *,
    confidence_cols: tuple[str, ...] = ("base_confidence",),
    add_regime_col: bool = True,
    add_tsf_table: bool = True,
    include_tsf_join_col: bool = True,
    create_view: bool = True,
) -> Path:
    db = tmp_path / "ctx.db"
    conn = sqlite3.connect(str(db))
    conf_defs = ", ".join(f"{name} REAL" for name in confidence_cols)
    conf_insert_cols = ", ".join(confidence_cols)
    conn.execute(
        f"""
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            realized_pnl REAL,
            is_close INTEGER DEFAULT 1,
            is_diagnostic INTEGER DEFAULT 0,
            is_synthetic INTEGER DEFAULT 0,
            is_contaminated INTEGER DEFAULT 0,
            ts_signal_id TEXT
            {"," if conf_defs else ""}
            {conf_defs}
        )
        """
    )
    if add_tsf_table:
        tsf_join_col = "ts_signal_id TEXT," if include_tsf_join_col else ""
        if add_regime_col:
            conn.execute(
                f"""
                CREATE TABLE time_series_forecasts (
                    id INTEGER PRIMARY KEY,
                    {tsf_join_col}
                    detected_regime TEXT
                )
                """
            )
        else:
            conn.execute(
                f"""
                CREATE TABLE time_series_forecasts (
                    id INTEGER PRIMARY KEY,
                    {tsf_join_col.rstrip(',')}
                )
                """
            )

    for idx, trade in enumerate(trades, start=1):
        ts_id = trade.get("ts_signal_id", f"ts_{trade['ticker']}_{idx}")
        values = [
            trade["ticker"],
            trade["pnl"],
            trade.get("is_diagnostic", 0),
            trade.get("is_synthetic", 0),
            trade.get("is_contaminated", 0),
            ts_id,
        ]
        if confidence_cols:
            for name in confidence_cols:
                values.append(trade.get(name))
            placeholders = ", ".join(["?"] * (6 + len(confidence_cols)))
            conn.execute(
                f"""
                INSERT INTO trade_executions(
                    ticker, realized_pnl, is_diagnostic, is_synthetic, is_contaminated, ts_signal_id, {conf_insert_cols}
                ) VALUES ({placeholders})
                """,
                values,
            )
        else:
            conn.execute(
                """
                INSERT INTO trade_executions(
                    ticker, realized_pnl, is_diagnostic, is_synthetic, is_contaminated, ts_signal_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        if add_tsf_table and add_regime_col and trade.get("regime"):
            if include_tsf_join_col:
                conn.execute(
                    "INSERT INTO time_series_forecasts(ts_signal_id, detected_regime) VALUES (?, ?)",
                    (ts_id, trade["regime"]),
                )
            else:
                conn.execute(
                    "INSERT INTO time_series_forecasts(detected_regime) VALUES (?)",
                    (trade["regime"],),
                )

    if create_view:
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


class TestContextQuality:
    def test_missing_confidence_calibrated_column_still_loads(self, tmp_path):
        from scripts.compute_context_quality import compute_context_quality

        db = _make_db(
            tmp_path,
            [
                {"ticker": "X", "pnl": 100.0, "base_confidence": 0.62, "regime": "HIGH_VOL_TRENDING"},
                {"ticker": "X", "pnl": -50.0, "base_confidence": 0.57, "regime": "HIGH_VOL_TRENDING"},
            ],
            confidence_cols=("base_confidence",),
        )
        result = compute_context_quality(db_path=db)
        assert result["n_total_trades"] == 2
        assert result["regime_quality"]["HIGH_VOL_TRENDING"]["n"] == 2
        assert "0.60-0.70" in result["confidence_bin_quality"]

    def test_missing_effective_confidence_also_works(self, tmp_path):
        from scripts.compute_context_quality import compute_context_quality

        db = _make_db(
            tmp_path,
            [{"ticker": "X", "pnl": 50.0, "base_confidence": 0.60, "confidence_calibrated": 0.61}],
            confidence_cols=("base_confidence", "confidence_calibrated"),
        )
        result = compute_context_quality(db_path=db)
        assert result["n_total_trades"] == 1
        assert result["confidence_bin_quality"]

    def test_all_confidence_columns_absent_keeps_rows(self, tmp_path):
        from scripts.compute_context_quality import compute_context_quality

        db = _make_db(tmp_path, [{"ticker": "X", "pnl": 50.0, "regime": "TRENDING"}], confidence_cols=())
        result = compute_context_quality(db_path=db)
        assert result["n_total_trades"] == 1
        assert result["n_trades_no_confidence"] == 1
        assert result["n_confidence_out_of_range"] == 0
        assert result["confidence_bin_quality"] == {}
        assert result["partial_data"] is True

    def test_out_of_range_confidence_is_counted_and_excluded_from_bins(self, tmp_path):
        from scripts.compute_context_quality import compute_context_quality

        db = _make_db(
            tmp_path,
            [{"ticker": "X", "pnl": 50.0, "base_confidence": 1.2, "regime": "TRENDING"}],
        )
        result = compute_context_quality(db_path=db)
        assert result["n_total_trades"] == 1
        assert result["n_confidence_out_of_range"] == 1
        assert result["confidence_bin_quality"] == {}
        assert "confidence_out_of_range" in result["warnings"]

    def test_missing_detected_regime_column_uses_unknown(self, tmp_path):
        from scripts.compute_context_quality import _UNKNOWN_REGIME, compute_context_quality

        db = _make_db(
            tmp_path,
            [{"ticker": "X", "pnl": 50.0, "base_confidence": 0.60}],
            add_regime_col=False,
        )
        result = compute_context_quality(db_path=db)
        assert _UNKNOWN_REGIME in result["regime_quality"]
        assert result["partial_data"] is True

    def test_missing_forecast_join_key_uses_unknown(self, tmp_path):
        from scripts.compute_context_quality import _UNKNOWN_REGIME, compute_context_quality

        db = _make_db(
            tmp_path,
            [{"ticker": "X", "pnl": 10.0, "base_confidence": 0.61, "regime": "TRENDING"}],
            include_tsf_join_col=False,
        )
        result = compute_context_quality(db_path=db)
        assert _UNKNOWN_REGIME in result["regime_quality"]
        assert "forecast_ts_signal_id_missing" in result["warnings"]

    def test_missing_db_returns_empty(self, tmp_path):
        from scripts.compute_context_quality import compute_context_quality

        result = compute_context_quality(db_path=tmp_path / "missing.db")
        assert result["n_total_trades"] == 0
        assert result["regime_quality"] == {}

    def test_result_schema_has_partial_metadata(self, tmp_path):
        from scripts.compute_context_quality import compute_context_quality

        db = _make_db(
            tmp_path,
            [{"ticker": "X", "pnl": 50.0, "base_confidence": None, "regime": "TRENDING"}],
        )
        result = compute_context_quality(db_path=db)
        assert set(result.keys()) >= {
            "generated_utc",
            "db_path",
            "n_total_trades",
            "n_trades_no_confidence",
            "regime_quality",
            "confidence_bin_quality",
            "ticker_regime_quality",
            "schema_used",
            "thresholds",
            "partial_data",
            "warnings",
        }

    def test_fallback_excludes_null_and_contaminated_rows(self, tmp_path):
        from scripts.compute_context_quality import compute_context_quality

        db = _make_db(
            tmp_path,
            [
                {"ticker": "X", "pnl": 50.0, "base_confidence": 0.60, "regime": "TRENDING", "is_diagnostic": None},
                {"ticker": "Y", "pnl": 25.0, "base_confidence": 0.62, "regime": "TRENDING", "is_contaminated": 1},
                {"ticker": "Z", "pnl": 75.0, "base_confidence": 0.64, "regime": "TRENDING"},
            ],
            create_view=False,
        )
        result = compute_context_quality(db_path=db)
        assert result["n_total_trades"] == 1


class TestCLI:
    def test_json_output_to_stdout(self, tmp_path, capsys):
        from scripts.compute_context_quality import main

        db = _make_db(tmp_path, [{"ticker": "X", "pnl": 100.0, "base_confidence": 0.60}])
        out = tmp_path / "cq.json"
        rc = main(["--db", str(db), "--output", str(out), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "regime_quality" in payload
        assert "confidence_bin_quality" in payload
