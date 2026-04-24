from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts import family_calibration_writer as mod


def _write_audit(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE trade_executions (
                id INTEGER PRIMARY KEY,
                ts_signal_id TEXT,
                ticker TEXT,
                realized_pnl REAL,
                holding_period_days REAL,
                trade_date TEXT,
                exit_reason TEXT,
                is_close INTEGER DEFAULT 0,
                is_synthetic INTEGER DEFAULT 0,
                is_diagnostic INTEGER DEFAULT 0,
                execution_mode TEXT,
                bar_timestamp TEXT,
                bar_high REAL,
                bar_low REAL,
                bar_close REAL
            )
            """
        )
        conn.execute(
            """
            CREATE VIEW production_closed_trades AS
            SELECT *
            FROM trade_executions
            WHERE is_close = 1
              AND COALESCE(is_synthetic, 0) = 0
              AND COALESCE(is_diagnostic, 0) = 0
            """
        )
        conn.execute(
            "CREATE TABLE ohlcv_data(date TEXT, ticker TEXT, adj_close REAL, close REAL)"
        )
        for idx, close in enumerate(range(100, 115), start=1):
            conn.execute(
                "INSERT INTO ohlcv_data(date, ticker, adj_close, close) VALUES (?, 'AAPL', ?, ?)",
                (f"2026-04-{idx:02d}", float(close), float(close)),
            )
        conn.executemany(
            """
            INSERT INTO trade_executions (
                id, ts_signal_id, ticker, realized_pnl, holding_period_days, trade_date,
                exit_reason, is_close, is_synthetic, is_diagnostic, execution_mode,
                bar_timestamp, bar_high, bar_low, bar_close
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    1,
                    "ts_garch_closed",
                    "AAPL",
                    12.5,
                    2.0,
                    "2026-04-05",
                    "TAKE_PROFIT",
                    1,
                    0,
                    0,
                    "live",
                    "2026-04-05T15:30:00Z",
                    106.0,
                    102.0,
                    105.0,
                ),
                (
                    2,
                    "ts_sarimax_closed",
                    "AAPL",
                    -4.0,
                    3.0,
                    "2026-04-06",
                    "STOP_LOSS",
                    1,
                    0,
                    0,
                    "live",
                    "2026-04-06T15:30:00Z",
                    107.0,
                    101.0,
                    103.0,
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_build_family_calibration_row_records_attribution_regimes_and_shadow_metrics(tmp_path: Path) -> None:
    db_path = tmp_path / "pmx.db"
    _build_db(db_path)

    audit_dir = tmp_path / "logs" / "forecast_audits" / "production"
    _write_audit(
        audit_dir / "forecast_audit_ts_garch_closed.json",
        {
            "dataset": {"end": "2026-04-05T00:00:00Z", "ticker": "AAPL", "detected_regime": "HIGH_VOL_TRENDING", "forecast_horizon": 2},
            "signal_context": {
                "ts_signal_id": "ts_garch_closed",
                "model_type": "GARCH",
                "snr": 0.9,
                "snr_gate_blocked": True,
                "execution_policy_blocked": False,
                "hold_reason": "SNR",
                "execution_mode": "live",
                "signal_timestamp": "2026-04-05T15:00:00Z",
                "bar_timestamp": "2026-04-05T14:00:00Z",
                "bar_high": 106.0,
                "bar_low": 102.0,
                "bar_close": 105.0,
            },
        },
    )
    _write_audit(
        audit_dir / "forecast_audit_ts_sarimax_closed.json",
        {
            "dataset": {"end": "2026-04-06T00:00:00Z", "ticker": "AAPL", "detected_regime": "LIQUID_RANGEBOUND", "forecast_horizon": 2},
            "signal_context": {
                "ts_signal_id": "ts_sarimax_closed",
                "model_type": "SARIMAX",
                "snr": 1.6,
                "snr_gate_blocked": False,
                "execution_policy_blocked": True,
                "hold_reason": "EVIDENCE",
                "execution_mode": "live",
                "signal_timestamp": "2026-04-06T15:00:00Z",
                "bar_timestamp": "2026-04-06T14:00:00Z",
                "bar_high": 107.0,
                "bar_low": 101.0,
                "bar_close": 103.0,
            },
        },
    )
    _write_audit(
        audit_dir / "forecast_audit_ts_sarimax_closed.json",
        {
            "dataset": {"end": "2026-04-06T00:00:00Z", "ticker": "AAPL", "detected_regime": "LIQUID_RANGEBOUND", "forecast_horizon": 2},
            "signal_context": {
                "ts_signal_id": "ts_sarimax_closed",
                "model_type": "SARIMAX",
                "snr": 1.6,
                "snr_gate_blocked": False,
                "execution_policy_blocked": True,
                "hold_reason": "EVIDENCE",
                "execution_mode": "live",
                "signal_timestamp": "2026-04-06T15:00:00Z",
                "bar_timestamp": "2026-04-06T14:00:00Z",
                "bar_high": 107.0,
                "bar_low": 101.0,
                "bar_close": 103.0,
            },
        },
    )
    _write_audit(
        audit_dir / "forecast_audit_ts_shadow.json",
        {
            "dataset": {"end": "2026-04-07T00:00:00Z", "ticker": "AAPL", "detected_regime": "CRISIS", "forecast_horizon": 2},
            "signal_context": {
                "ts_signal_id": "ts_shadow",
                "model_type": "GARCH",
                "snr": 0.75,
                "snr_gate_blocked": False,
                "execution_policy_blocked": False,
                "execution_mode": "synthetic",
                "signal_timestamp": "2026-04-07T15:00:00Z",
                "bar_timestamp": "2026-04-07T14:00:00Z",
                "bar_high": 110.0,
                "bar_low": 100.0,
                "bar_close": 104.0,
                "bar_position_proxy": 0.5,
            },
        },
    )

    row = mod.build_family_calibration_row(
        db_path=db_path,
        audit_dir=audit_dir,
        window_cycles=20,
        window_start_utc="2026-04-01T00:00:00Z",
        window_end_utc="2026-04-11T00:00:00Z",
    )

    assert row["schema_version"] == 1
    assert row["regime_distribution"] == {
        "CRISIS": 1,
        "HIGH_VOL_TRENDING": 1,
        "LIQUID_RANGEBOUND": 1,
    }
    assert row["analysis_gate_passed"] is True
    assert row["attribution_available"] is True
    assert row["closed_trades_by_model_family"]["GARCH"]["count"] == 1
    assert row["closed_trades_by_model_family"]["SARIMAX"]["count"] == 1
    assert row["shadow_trade_metrics"]["count"] == 1
    assert row["shadow_trade_metrics"]["spread_proxy_note"] == "range-based, upward-biased"
    assert row["shadow_trade_metrics"]["bar_range_fraction"]["p50"] == pytest.approx(round((110.0 - 100.0) / 104.0, 6))
    assert row["shadow_trade_metrics"]["bar_position_proxy"]["p50"] == pytest.approx(0.5)
    assert row["family_stats"]["GARCH"]["blocked_by_snr"] == 1
    assert row["family_stats"]["SARIMAX"]["blocked_by_evidence"] == 1


def test_load_family_calibration_rows_skips_malformed_rows(tmp_path: Path) -> None:
    path = tmp_path / "family_calibration.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"schema_version": 1, "window_cycles": 20}),
                "{not-json",
                json.dumps({"schema_version": 1, "window_cycles": 21}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = mod.load_family_calibration_rows(path)
    assert len(rows) == 2
    assert rows[0]["window_cycles"] == 20
    assert rows[1]["window_cycles"] == 21


def test_family_calibration_cli_appends_a_row(tmp_path: Path) -> None:
    db_path = tmp_path / "pmx.db"
    _build_db(db_path)
    audit_dir = tmp_path / "logs" / "forecast_audits" / "production"
    _write_audit(
        audit_dir / "forecast_audit_ts_garch_closed.json",
        {
            "dataset": {"end": "2026-04-05T00:00:00Z", "ticker": "AAPL", "detected_regime": "HIGH_VOL_TRENDING", "forecast_horizon": 2},
            "signal_context": {
                "ts_signal_id": "ts_garch_closed",
                "model_type": "GARCH",
                "snr": 0.9,
                "snr_gate_blocked": True,
                "execution_policy_blocked": False,
                "hold_reason": "SNR",
                "execution_mode": "live",
                "signal_timestamp": "2026-04-05T15:00:00Z",
                "bar_timestamp": "2026-04-05T14:00:00Z",
                "bar_high": 106.0,
                "bar_low": 102.0,
                "bar_close": 105.0,
            },
        },
    )
    output = tmp_path / "family_calibration.jsonl"

    assert mod.main([
        "--db",
        str(db_path),
        "--audit-dir",
        str(audit_dir),
        "--output",
        str(output),
        "--window-cycles",
        "20",
        "--window-start-utc",
        "2026-04-01T00:00:00Z",
        "--window-end-utc",
        "2026-04-11T00:00:00Z",
    ]) == 0
    rows = mod.load_family_calibration_rows(output)
    assert len(rows) == 1
    assert rows[0]["schema_version"] == 1
