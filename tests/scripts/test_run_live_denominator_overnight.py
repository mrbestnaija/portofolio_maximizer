from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from scripts.run_live_denominator_overnight import (
    _forecast_audit_sanitization_cmd,
    _seconds_until_next_weekday,
    extract_fresh_trade_signals,
)


def test_extract_fresh_trade_signals_filters_trade_rows_and_counts_matches(tmp_path: Path) -> None:
    summary_path = tmp_path / "latest_summary.json"
    db_path = tmp_path / "portfolio_maximizer.db"

    summary_payload = {
        "generated_utc": "2026-03-06T22:00:45+00:00",
        "dataset_windows": [
            {
                "file": "forecast_audit_20260305_101010.json",
                "context_type": "TRADE",
                "outcome_status": "MATCHED",
                "outcome_reason": "ONE_TO_ONE_MATCH",
                "counts_toward_linkage_denominator": True,
                "ts_signal_id": "ts_old",
            },
            {
                "file": "forecast_audit_20260306_111111.json",
                "context_type": "TRADE",
                "outcome_status": "MATCHED",
                "outcome_reason": "ONE_TO_ONE_MATCH",
                "counts_toward_linkage_denominator": True,
                "ts_signal_id": "ts_match",
            },
            {
                "file": "forecast_audit_20260306_222222.json",
                "context_type": "TRADE",
                "outcome_status": "NON_TRADE_CONTEXT",
                "outcome_reason": "MISSING_TICKER",
                "counts_toward_linkage_denominator": False,
                "ts_signal_id": "ts_missing",
            },
            {
                "file": "forecast_audit_20260306_333333.json",
                "context_type": "RESEARCH",
                "outcome_status": "MATCHED",
                "outcome_reason": "ONE_TO_ONE_MATCH",
                "counts_toward_linkage_denominator": True,
                "ts_signal_id": "ts_ignore",
            },
        ],
    }
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE production_closed_trades (ts_signal_id TEXT)")
        conn.execute("INSERT INTO production_closed_trades(ts_signal_id) VALUES ('ts_match')")
        conn.execute("INSERT INTO production_closed_trades(ts_signal_id) VALUES ('ts_ignore')")
        conn.commit()
    finally:
        conn.close()

    signals = extract_fresh_trade_signals(summary_path, db_path)

    assert signals["latest_day"] == "20260306"
    assert signals["fresh_trade_context_rows_raw"] == 2
    assert signals["fresh_trade_rows"] == 1
    assert signals["fresh_trade_exclusions"] == {
        "non_trade_context": 0,
        "invalid_context": 0,
        "missing_execution_metadata": 0,
    }
    assert signals["fresh_trade_diagnostics"] == {"non_trade_context_rows": 1}
    assert signals["fresh_linkage_included"] == 1
    assert signals["fresh_production_valid_rows"] == 1
    assert signals["fresh_production_valid_matched"] == 1


def test_extract_fresh_trade_signals_counts_missing_execution_metadata_reason(tmp_path: Path) -> None:
    summary_path = tmp_path / "latest_summary.json"
    db_path = tmp_path / "portfolio_maximizer.db"
    summary_path.write_text(
        json.dumps(
            {
                "generated_utc": "2026-03-06T22:00:45+00:00",
                "dataset_windows": [
                    {
                        "file": "forecast_audit_20260306_111111.json",
                        "context_type": "TRADE",
                        "outcome_status": "INVALID_CONTEXT",
                        "outcome_reason": "MISSING_EXECUTION_METADATA",
                        "counts_toward_linkage_denominator": False,
                        "ts_signal_id": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sqlite3.connect(str(db_path)).close()

    signals = extract_fresh_trade_signals(summary_path, db_path)

    assert signals["fresh_trade_context_rows_raw"] == 1
    assert signals["fresh_trade_rows"] == 1
    assert signals["fresh_trade_exclusions"] == {
        "non_trade_context": 0,
        "invalid_context": 1,
        "missing_execution_metadata": 1,
    }
    assert signals["fresh_trade_diagnostics"] == {"non_trade_context_rows": 0}
    assert signals["fresh_linkage_included"] == 0
    assert signals["fresh_production_valid_rows"] == 0
    assert signals["fresh_production_valid_matched"] == 0


def test_seconds_until_next_weekday_skips_weekends() -> None:
    saturday = datetime.fromisoformat("2026-03-07T05:25:29+00:00")
    monday = datetime.fromisoformat("2026-03-09T05:25:29+00:00")

    assert _seconds_until_next_weekday(saturday) == 172800
    assert _seconds_until_next_weekday(monday) == 0


def test_forecast_audit_sanitization_command_targets_production_root(tmp_path: Path) -> None:
    cmd = _forecast_audit_sanitization_cmd(tmp_path / "logs" / "forecast_audits")

    audit_dir = Path(cmd[cmd.index("--audit-dir") + 1])
    eval_dir = Path(cmd[cmd.index("--eval-audit-dir") + 1])
    quarantine_dir = Path(cmd[cmd.index("--quarantine-dir") + 1])

    assert audit_dir.name == "production"
    assert audit_dir.parent.name == "forecast_audits"
    assert eval_dir.name == "production_eval"
    assert quarantine_dir.name == "quarantine"
    assert cmd[-1] == "--apply"
