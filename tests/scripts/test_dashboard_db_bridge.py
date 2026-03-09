from __future__ import annotations

import datetime
import pytest
import sqlite3

from scripts import dashboard_db_bridge as mod


def test_build_dashboard_payload_from_sqlite(tmp_path) -> None:
    db_path = tmp_path / "pmx.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE ohlcv_data(date TEXT, ticker TEXT, close REAL)")
        conn.execute(
            "CREATE TABLE trading_signals(id INTEGER PRIMARY KEY, ticker TEXT, action TEXT, confidence REAL, expected_return REAL, source TEXT, signal_timestamp TEXT, created_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE trade_executions(id INTEGER PRIMARY KEY, ticker TEXT, action TEXT, shares REAL, price REAL, trade_date TEXT, created_at TEXT, realized_pnl REAL, realized_pnl_pct REAL, mid_slippage_bps REAL, run_id TEXT)"
        )
        conn.execute(
            "CREATE TABLE portfolio_positions(id INTEGER PRIMARY KEY, ticker TEXT, position_date TEXT, shares REAL, average_cost REAL)"
        )
        conn.execute(
            "CREATE TABLE data_quality_snapshots(id INTEGER PRIMARY KEY, ticker TEXT, quality_score REAL, missing_pct REAL, coverage REAL, outlier_frac REAL, source TEXT, created_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE performance_metrics(id INTEGER PRIMARY KEY, total_return REAL, total_return_pct REAL, win_rate REAL, profit_factor REAL, num_trades INTEGER, created_at TEXT)"
        )

        conn.execute("INSERT INTO ohlcv_data(date,ticker,close) VALUES ('2026-01-01','AAPL',100.0)")
        conn.execute("INSERT INTO ohlcv_data(date,ticker,close) VALUES ('2026-01-02','AAPL',101.0)")
        conn.execute(
            "INSERT INTO trading_signals(ticker,action,confidence,expected_return,source,signal_timestamp,created_at) VALUES (?,?,?,?,?,?,?)",
            ("AAPL", "BUY", 0.8, 0.01, "TIME_SERIES", "2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO trade_executions(ticker,action,shares,price,trade_date,created_at,realized_pnl,realized_pnl_pct,mid_slippage_bps,run_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("AAPL", "SELL", 10, 101.0, "2026-01-02", "2026-01-02T00:00:01Z", 5.0, 0.005, 10.0, "RID"),
        )
        conn.execute(
            "INSERT INTO trade_executions(ticker,action,shares,price,trade_date,created_at,realized_pnl,realized_pnl_pct,mid_slippage_bps,run_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("AAPL", "BUY", 10, 100.0, "2026-01-01", "2026-01-01T00:00:01Z", None, None, 10.0, "RID"),
        )
        recent_date = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        conn.execute(
            "INSERT INTO portfolio_positions(ticker,position_date,shares,average_cost) VALUES (?,?,?,?)",
            ("AAPL", recent_date, 10, 101.0),
        )
        conn.execute(
            "INSERT INTO data_quality_snapshots(ticker,quality_score,missing_pct,coverage,outlier_frac,source,created_at) VALUES (?,?,?,?,?,?,?)",
            ("AAPL", 0.9, 0.0, 1.0, 0.0, "yfinance", "2026-01-02T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO performance_metrics(total_return,total_return_pct,win_rate,profit_factor,num_trades,created_at) VALUES (?,?,?,?,?,?)",
            (5.0, 0.0002, 1.0, 2.0, 1, "2026-01-02T00:00:00Z"),
        )
        conn.commit()
    finally:
        conn.close()

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        payload = mod.build_dashboard_payload(
            conn=conn,
            tickers=["AAPL"],
            lookback_days=10,
            max_signals=10,
            max_trades=10,
            db_path=db_path,
            read_path=db_path,
        )
    finally:
        conn.close()

    assert payload["meta"]["run_id"]
    assert payload["meta"]["payload_schema_version"] == 2
    assert payload["meta"]["payload_digest"]
    assert payload["meta"]["storage"]["db_path"] == str(db_path)
    assert payload["meta"]["tickers"] == ["AAPL"]
    assert payload["meta"]["ticker_buckets"]["AAPL"] in {"safe", "core", "speculative", "other"}
    assert payload["pnl"]["absolute"] == 5.0
    assert payload["trade_count"] == 1
    assert payload["signals"] and payload["signals"][0]["ticker"] == "AAPL"
    assert payload["trade_events"] and payload["trade_events"][0]["ticker"] == "AAPL"
    assert payload["price_series"]["AAPL"] and payload["price_series"]["AAPL"][-1]["close"] == 101.0
    assert payload["positions"]["AAPL"]["shares"] == 10
    assert {e["event_type"] for e in payload["trade_events"]} == {"ENTRY", "EXIT_PROFIT"}


def test_positions_fallback_uses_average_cost(tmp_path) -> None:
    db_path = tmp_path / "pmx_avg_cost.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE ohlcv_data(date TEXT, ticker TEXT, close REAL)")
        conn.execute(
            "CREATE TABLE trading_signals(id INTEGER PRIMARY KEY, ticker TEXT, action TEXT, confidence REAL, expected_return REAL, source TEXT, signal_timestamp TEXT, created_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE trade_executions(id INTEGER PRIMARY KEY, ticker TEXT, action TEXT, shares REAL, price REAL, trade_date TEXT, created_at TEXT, realized_pnl REAL, realized_pnl_pct REAL, mid_slippage_bps REAL, run_id TEXT)"
        )
        conn.execute(
            "CREATE TABLE portfolio_positions(id INTEGER PRIMARY KEY, ticker TEXT, position_date TEXT, shares REAL, average_cost REAL)"
        )
        conn.execute(
            "CREATE TABLE data_quality_snapshots(id INTEGER PRIMARY KEY, ticker TEXT, quality_score REAL, missing_pct REAL, coverage REAL, outlier_frac REAL, source TEXT, created_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE performance_metrics(id INTEGER PRIMARY KEY, total_return REAL, total_return_pct REAL, win_rate REAL, profit_factor REAL, num_trades INTEGER, created_at TEXT)"
        )

        conn.execute("INSERT INTO ohlcv_data(date,ticker,close) VALUES ('2026-01-01','AAPL',12.0)")
        conn.execute(
            "INSERT INTO trade_executions(ticker,action,shares,price,trade_date,created_at,realized_pnl,realized_pnl_pct,mid_slippage_bps,run_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("AAPL", "BUY", 10, 10.0, "2026-01-01", "2026-01-01T00:00:01Z", None, None, 10.0, "RID"),
        )
        conn.execute(
            "INSERT INTO trade_executions(ticker,action,shares,price,trade_date,created_at,realized_pnl,realized_pnl_pct,mid_slippage_bps,run_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("AAPL", "SELL", 5, 12.0, "2026-01-02", "2026-01-02T00:00:01Z", 10.0, 0.1, 10.0, "RID"),
        )
        conn.execute(
            "INSERT INTO performance_metrics(total_return,total_return_pct,win_rate,profit_factor,num_trades,created_at) VALUES (?,?,?,?,?,?)",
            (5.0, 0.0002, 1.0, 2.0, 1, "2026-01-02T00:00:00Z"),
        )
        conn.commit()
    finally:
        conn.close()

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        payload = mod.build_dashboard_payload(
            conn=conn,
            tickers=["AAPL"],
            lookback_days=10,
            max_signals=10,
            max_trades=10,
        )
    finally:
        conn.close()

    pos = payload["positions"]["AAPL"]
    assert pos["shares"] == 5
    assert pos["entry_price"] == 10.0


def test_robustness_sidecars_merge_without_breaking_payload(tmp_path, monkeypatch) -> None:
    elig = tmp_path / "elig.json"
    ctx = tmp_path / "ctx.json"
    perf = tmp_path / "metrics.json"
    elig.write_text('{"summary":{"HEALTHY":1,"WEAK":1,"LAB_ONLY":0},"tickers":{"GS":{"status":"WEAK"}},"warnings":[]}', encoding="utf-8")
    ctx.write_text('{"n_total_trades":4,"n_trades_no_confidence":1,"partial_data":true,"regime_quality":{"A":{}}, "confidence_bin_quality":{"0.60-0.65":{}}}', encoding="utf-8")
    perf.write_text('{"status":"WARN","warnings":["sufficiency_not_green"],"sufficiency":{"status":"INSUFFICIENT"},"chart_paths":{"a":"b"},"coverage_ratio":0.2}', encoding="utf-8")

    monkeypatch.setattr(mod, "DEFAULT_ELIGIBILITY_PATH", elig)
    monkeypatch.setattr(mod, "DEFAULT_CONTEXT_QUALITY_PATH", ctx)
    monkeypatch.setattr(mod, "DEFAULT_PERFORMANCE_METRICS_PATH", perf)

    robustness = mod._robustness_payload()
    assert robustness["status"] == "WARN"
    assert robustness["eligibility_summary"]["WEAK"] == 1
    assert robustness["weak_tickers"] == ["GS"]
    assert robustness["context_quality_summary"]["partial_data"] is True
    assert robustness["sufficiency"]["status"] == "INSUFFICIENT"


def test_robustness_warns_when_chart_paths_missing(tmp_path, monkeypatch) -> None:
    elig = tmp_path / "elig.json"
    ctx = tmp_path / "ctx.json"
    perf = tmp_path / "metrics.json"
    elig.write_text('{"summary":{"HEALTHY":1,"WEAK":0,"LAB_ONLY":0},"tickers":{"GS":{"status":"HEALTHY"}},"warnings":[]}', encoding="utf-8")
    ctx.write_text('{"n_total_trades":4,"n_trades_no_confidence":0,"partial_data":false,"regime_quality":{"A":{}}, "confidence_bin_quality":{"0.60-0.65":{}}}', encoding="utf-8")
    perf.write_text(
        '{"status":"PASS","warnings":[],"sufficiency":{"status":"SUFFICIENT"},"chart_paths":{"per_ticker_wr_pf":"visualizations/performance/missing_chart.png"}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "DEFAULT_ELIGIBILITY_PATH", elig)
    monkeypatch.setattr(mod, "DEFAULT_CONTEXT_QUALITY_PATH", ctx)
    monkeypatch.setattr(mod, "DEFAULT_PERFORMANCE_METRICS_PATH", perf)

    robustness = mod._robustness_payload()
    assert robustness["status"] == "WARN"
    assert "chart_missing:per_ticker_wr_pf" in robustness["warnings"]


def test_robustness_payload_marks_missing_without_sidecars(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "DEFAULT_ELIGIBILITY_PATH", tmp_path / "missing_elig.json")
    monkeypatch.setattr(mod, "DEFAULT_CONTEXT_QUALITY_PATH", tmp_path / "missing_ctx.json")
    monkeypatch.setattr(mod, "DEFAULT_PERFORMANCE_METRICS_PATH", tmp_path / "missing_perf.json")

    robustness = mod._robustness_payload()
    assert robustness["status"] == "MISSING"
    assert robustness["warnings"]


def test_robustness_marks_stale_when_sidecar_age_exceeds_policy(tmp_path, monkeypatch) -> None:
    elig = tmp_path / "elig.json"
    ctx = tmp_path / "ctx.json"
    perf = tmp_path / "metrics.json"
    old_ts = "2026-01-01T00:00:00Z"
    elig.write_text(
        '{"generated_utc":"%s","summary":{"HEALTHY":1,"WEAK":0,"LAB_ONLY":0},"tickers":{},"warnings":[]}'
        % old_ts,
        encoding="utf-8",
    )
    ctx.write_text(
        '{"generated_utc":"%s","n_total_trades":4,"n_trades_no_confidence":0,"partial_data":false,"regime_quality":{"A":{}}, "confidence_bin_quality":{"0.60-0.65":{}}}'
        % old_ts,
        encoding="utf-8",
    )
    perf.write_text(
        '{"generated_utc":"%s","status":"PASS","warnings":[],"sufficiency":{"status":"SUFFICIENT"},"chart_paths":{}}'
        % old_ts,
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "DEFAULT_ELIGIBILITY_PATH", elig)
    monkeypatch.setattr(mod, "DEFAULT_CONTEXT_QUALITY_PATH", ctx)
    monkeypatch.setattr(mod, "DEFAULT_PERFORMANCE_METRICS_PATH", perf)
    monkeypatch.setenv("PMX_SIDECAR_MAX_AGE_MINUTES", "120")

    robustness = mod._robustness_payload()
    assert robustness["status"] == "STALE"
    assert robustness["freshness_status"] == "STALE"
    assert robustness["freshness_reason"] == "STALE_SIDECAR"


def test_robustness_optional_forecast_summary_stale_is_warn_only(tmp_path, monkeypatch) -> None:
    elig = tmp_path / "elig.json"
    ctx = tmp_path / "ctx.json"
    perf = tmp_path / "metrics.json"
    forecast = tmp_path / "forecast_summary.json"

    elig.write_text(
        '{"generated_utc":"2099-01-01T00:00:00Z","summary":{"HEALTHY":1,"WEAK":0,"LAB_ONLY":0},"tickers":{},"warnings":[]}',
        encoding="utf-8",
    )
    ctx.write_text(
        '{"generated_utc":"2099-01-01T00:00:00Z","n_total_trades":4,"n_trades_no_confidence":0,"partial_data":false,"regime_quality":{"A":{}}, "confidence_bin_quality":{"0.60-0.65":{}}}',
        encoding="utf-8",
    )
    perf.write_text(
        '{"generated_utc":"2099-01-01T00:00:00Z","status":"PASS","warnings":[],"sufficiency":{"status":"SUFFICIENT"},"chart_paths":{}}',
        encoding="utf-8",
    )
    forecast.write_text(
        '{"generated_utc":"2020-01-01T00:00:00Z","telemetry_contract":{"schema_version":3}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "DEFAULT_ELIGIBILITY_PATH", elig)
    monkeypatch.setattr(mod, "DEFAULT_CONTEXT_QUALITY_PATH", ctx)
    monkeypatch.setattr(mod, "DEFAULT_PERFORMANCE_METRICS_PATH", perf)
    monkeypatch.setattr(mod, "DEFAULT_FORECAST_SUMMARY_PATH", forecast)
    monkeypatch.setenv("PMX_SIDECAR_MAX_AGE_MINUTES", "120")

    robustness = mod._robustness_payload()
    assert robustness["status"] == "WARN"
    assert "stale_sidecar:forecast_summary" in robustness["warnings"]


def test_live_denominator_payload_merges_watcher_sidecar(tmp_path, monkeypatch) -> None:
    watcher = tmp_path / "live_denominator_latest.json"
    watcher.write_text(
        """
        {
          "run_meta": {
            "run_id": "RID",
            "tickers": ["AAPL", "MSFT"],
            "cycles": 30,
            "sleep_seconds": 86400,
            "progress_linkage_threshold": 2
          },
          "cycles": [
            {
              "latest_day": "20260306",
              "fresh_trade_rows": 1,
              "fresh_trade_context_rows_raw": 4,
              "fresh_trade_diagnostics": {"non_trade_context_rows": 3},
              "fresh_trade_exclusions": {"invalid_context": 0, "missing_execution_metadata": 0},
              "fresh_linkage_included": 1,
              "fresh_production_valid_matched": 0,
              "progress_triggered": false
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "DEFAULT_LIVE_DENOMINATOR_PATH", watcher)

    payload = mod._live_denominator_payload()

    assert payload["status"] == "WAITING"
    assert payload["cycles_completed"] == 1
    assert payload["current"]["fresh_trade_rows"] == 1
    assert payload["run_meta"]["tickers"] == ["AAPL", "MSFT"]


def test_quant_validation_payload_summarizes_status(tmp_path, monkeypatch) -> None:
    quant_log = tmp_path / "quant_validation.jsonl"
    quant_log.write_text(
        '\n'.join(
            [
                '{"status":"PASS","expected_profit":1.0,"timestamp":"2026-03-07T00:00:00Z"}',
                '{"status":"FAIL","expected_profit":-1.0,"timestamp":"2026-03-07T00:05:00Z"}',
            ]
        ),
        encoding="utf-8",
    )
    cfg = tmp_path / "forecaster_monitoring.yml"
    cfg.write_text(
        """
forecaster_monitoring:
  quant_validation:
    max_fail_fraction: 0.95
    max_negative_expected_profit_fraction: 0.50
    warn_fail_fraction: 0.25
    warn_negative_expected_profit_fraction: 0.25
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "DEFAULT_QUANT_VALIDATION_LOG_PATH", quant_log)
    monkeypatch.setattr(mod, "DEFAULT_MONITORING_CONFIG_PATH", cfg)

    payload = mod._quant_validation_payload()

    assert payload["status"] == "YELLOW"
    assert payload["total"] == 2
    assert payload["pass_count"] == 1
    assert payload["fail_count"] == 1
    assert payload["path"] == str(quant_log)


# ---------------------------------------------------------------------------
# P0 guardrail smoke tests: verify both connect helpers harden connections
# ---------------------------------------------------------------------------

def test_connect_ro_blocks_dangerous_pragma(tmp_path) -> None:
    """_connect_ro must apply guardrails so blocked PRAGMAs raise OperationalError."""
    db_path = tmp_path / "ro_test.db"
    # Seed a minimal DB so the file exists and can be opened read-only.
    seed = sqlite3.connect(str(db_path))
    seed.execute("CREATE TABLE t(x INTEGER)")
    seed.commit()
    seed.close()

    conn = mod._connect_ro(db_path)
    try:
        with pytest.raises(sqlite3.DatabaseError):  # authorizer denial raises DatabaseError
            conn.execute("PRAGMA journal_mode=DELETE")
    finally:
        conn.close()


def test_connect_rw_blocks_dangerous_pragma_after_setup(tmp_path) -> None:
    """_connect_rw must apply guardrails after WAL setup so journal_mode cannot be changed again."""
    db_path = tmp_path / "rw_test.db"
    conn = mod._connect_rw(db_path)
    try:
        with pytest.raises(sqlite3.DatabaseError):  # authorizer denial raises DatabaseError
            conn.execute("PRAGMA journal_mode=DELETE")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Agent-B dashboard truthfulness contract tests
# ---------------------------------------------------------------------------

def _make_minimal_db(db_path, *, with_exit_reason: bool = False) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE ohlcv_data(date TEXT, ticker TEXT, close REAL)")
    conn.execute(
        "CREATE TABLE trading_signals(id INTEGER PRIMARY KEY, ticker TEXT, action TEXT,"
        " confidence REAL, expected_return REAL, source TEXT, signal_timestamp TEXT, created_at TEXT)"
    )
    exit_col = ", exit_reason TEXT" if with_exit_reason else ""
    conn.execute(
        f"CREATE TABLE trade_executions(id INTEGER PRIMARY KEY, ticker TEXT, action TEXT,"
        f" shares REAL, price REAL, trade_date TEXT, created_at TEXT, realized_pnl REAL,"
        f" realized_pnl_pct REAL, mid_slippage_bps REAL, run_id TEXT, is_diagnostic INTEGER"
        f" DEFAULT 0, is_synthetic INTEGER DEFAULT 0{exit_col})"
    )
    conn.execute(
        "CREATE TABLE portfolio_positions(id INTEGER PRIMARY KEY, ticker TEXT,"
        " position_date TEXT, shares REAL, average_cost REAL)"
    )
    conn.execute(
        "CREATE TABLE data_quality_snapshots(id INTEGER PRIMARY KEY, ticker TEXT,"
        " quality_score REAL, missing_pct REAL, coverage REAL, outlier_frac REAL, source TEXT, created_at TEXT)"
    )
    conn.commit()
    conn.close()


def test_positions_stale_uses_filtered_execution_fallback(tmp_path, monkeypatch) -> None:
    """When portfolio_positions.position_date is older than max_age_days, the bridge
    must fall back to trade_executions and set positions_stale=True in the payload."""
    db_path = tmp_path / "stale_test.db"
    _make_minimal_db(db_path)
    conn = sqlite3.connect(db_path)
    stale_date = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
    conn.execute(
        "INSERT INTO portfolio_positions(ticker,position_date,shares,average_cost) VALUES (?,?,?,?)",
        ("MSFT", stale_date, 5, 300.0),
    )
    # Open BUY the fallback will see
    conn.execute(
        "INSERT INTO trade_executions(ticker,action,shares,price,trade_date,created_at,"
        " realized_pnl,realized_pnl_pct,mid_slippage_bps,run_id,is_diagnostic,is_synthetic)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("MSFT", "BUY", 3, 300.0, stale_date, stale_date + "T00:00:00Z",
         None, None, 0.0, "RID", 0, 0),
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    monkeypatch.setenv("PMX_POSITIONS_MAX_AGE_DAYS", "14")
    try:
        payload = mod.build_dashboard_payload(
            conn=conn, tickers=["MSFT"], lookback_days=10,
            max_signals=5, max_trades=5, db_path=db_path,
        )
    finally:
        conn.close()

    assert payload["positions_stale"] is True
    assert payload["positions_source"] in {"trade_executions_fallback_stale", "trade_executions_fallback"}
    assert any("stale" in c.lower() for c in payload["checks"])


def test_performance_unknown_when_metrics_missing(tmp_path) -> None:
    """When performance_metrics table has no rows (and PnLIntegrityEnforcer is
    unavailable), the bridge must emit None values and performance_unknown=True
    rather than zeroes that are indistinguishable from a real zero-PnL session."""
    db_path = tmp_path / "no_perf.db"
    _make_minimal_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE performance_metrics(id INTEGER PRIMARY KEY,"
        " total_return REAL, total_return_pct REAL, win_rate REAL,"
        " profit_factor REAL, num_trades INTEGER, created_at TEXT)"
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        payload = mod.build_dashboard_payload(
            conn=conn, tickers=["AAPL"], lookback_days=10,
            max_signals=5, max_trades=5, db_path=db_path,
        )
    finally:
        conn.close()

    assert payload["performance_unknown"] is True
    assert payload["pnl"]["absolute"] is None
    assert payload["win_rate"] is None
    assert payload["trade_count"] is None


def test_trade_events_include_exit_reason_when_available(tmp_path) -> None:
    """Trade events payload must include exit_reason from trade_executions when the
    column exists, rather than always emitting None."""
    db_path = tmp_path / "exit_reason_test.db"
    _make_minimal_db(db_path, with_exit_reason=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO trade_executions(ticker,action,shares,price,trade_date,created_at,"
        " realized_pnl,realized_pnl_pct,mid_slippage_bps,run_id,is_diagnostic,is_synthetic,exit_reason)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("NVDA", "SELL", 2, 150.0, "2026-03-01", "2026-03-01T10:00:00Z",
         30.0, 0.01, 5.0, "RID", 0, 0, "stop_loss"),
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        payload = mod.build_dashboard_payload(
            conn=conn, tickers=["NVDA"], lookback_days=30,
            max_signals=5, max_trades=10, latest_run_only=False, db_path=db_path,
        )
    finally:
        conn.close()

    events = payload["trade_events"]
    assert events, "Expected at least one trade event"
    exit_reasons = [e.get("exit_reason") for e in events]
    assert "stop_loss" in exit_reasons, (
        f"exit_reason 'stop_loss' not found in trade events. Got: {exit_reasons}"
    )


def test_provenance_origin_is_mixed_when_trade_sources_are_mixed(tmp_path) -> None:
    """data_origin must be 'mixed' when trade_sources contains both synthetic and
    non-synthetic entries, even if ohlcv_sources contains no non-synthetic rows.

    Regression for: origin stayed 'synthetic' because the mixed-origin check only
    inspected ohlcv_sources, ignoring trade_sources entirely.
    """
    db_path = tmp_path / "mixed_provenance.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trade_executions("
        "id INTEGER PRIMARY KEY, ticker TEXT, action TEXT, shares REAL, price REAL,"
        " trade_date TEXT, created_at TEXT, realized_pnl REAL, realized_pnl_pct REAL,"
        " mid_slippage_bps REAL, run_id TEXT, is_diagnostic INTEGER DEFAULT 0,"
        " is_synthetic INTEGER DEFAULT 0, data_source TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ohlcv_data(id INTEGER PRIMARY KEY, source TEXT)"
    )
    # Insert synthetic trade source
    conn.execute(
        "INSERT INTO trade_executions(ticker,action,shares,price,trade_date,created_at,"
        "realized_pnl,realized_pnl_pct,mid_slippage_bps,run_id,is_diagnostic,is_synthetic,data_source)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("AAPL", "BUY", 1, 100.0, "2026-01-01", "2026-01-01T00:00:00Z",
         None, None, None, "R1", 0, 0, "synthetic"),
    )
    # Insert non-synthetic (yfinance) trade source
    conn.execute(
        "INSERT INTO trade_executions(ticker,action,shares,price,trade_date,created_at,"
        "realized_pnl,realized_pnl_pct,mid_slippage_bps,run_id,is_diagnostic,is_synthetic,data_source)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("MSFT", "BUY", 1, 200.0, "2026-01-02", "2026-01-02T00:00:00Z",
         None, None, None, "R2", 0, 0, "yfinance"),
    )
    # ohlcv_data has only synthetic rows — verifies the fix checks trade_sources too
    conn.execute("INSERT INTO ohlcv_data(source) VALUES (?)", ("synthetic",))
    conn.commit()
    conn.close()

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        provenance = mod._provenance_summary(conn)
    finally:
        conn.close()

    assert provenance["origin"] == "mixed", (
        f"Expected origin='mixed' for mixed trade_sources, got '{provenance['origin']}'. "
        f"trade_sources={provenance['trade_sources']}, ohlcv_sources={provenance['ohlcv_sources']}"
    )
    assert "synthetic" in provenance["trade_sources"]
    assert "yfinance" in provenance["trade_sources"]


def test_advanced_metrics_unknown_when_performance_metrics_table_empty(tmp_path) -> None:
    """Regression for Defect 4: when PnLIntegrityEnforcer returns realized PnL but the
    performance_metrics table is empty, the payload must surface advanced_metrics_unknown=True
    so the UI does not collapse 'partial performance state' into 'all known'.

    Ensures performance_unknown=False (realized PnL IS available) while
    advanced_metrics_unknown=True (aggregate analytics are NOT available).
    """
    import unittest.mock as mock

    db_path = tmp_path / "partial_perf.db"
    _make_minimal_db(db_path)
    conn = sqlite3.connect(db_path)
    # Add performance_metrics table but leave it empty (no aggregation has run)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS performance_metrics("
        "id INTEGER PRIMARY KEY, total_return REAL, total_return_pct REAL,"
        " win_rate REAL, profit_factor REAL, num_trades INTEGER, created_at TEXT)"
    )
    conn.commit()
    conn.close()

    # Simulate PnLIntegrityEnforcer returning valid canonical metrics so the
    # integrity path succeeds (performance_unknown=False) while performance_metrics
    # table stays empty (advanced_metrics_unknown=True should be set).
    fake_metrics = mock.MagicMock()
    fake_metrics.total_realized_pnl = 500.0
    fake_metrics.win_rate = 0.6
    fake_metrics.profit_factor = 1.8
    fake_metrics.total_round_trips = 10
    fake_metrics.avg_win = 75.0
    fake_metrics.avg_loss = -30.0
    fake_metrics.largest_win = 200.0
    fake_metrics.largest_loss = -80.0
    fake_metrics.diagnostic_trades_excluded = 0
    fake_metrics.synthetic_trades_excluded = 0
    fake_metrics.opening_legs_with_pnl = 0

    fake_enforcer = mock.MagicMock()
    fake_enforcer.__enter__ = mock.Mock(return_value=fake_enforcer)
    fake_enforcer.__exit__ = mock.Mock(return_value=False)
    fake_enforcer.get_canonical_metrics.return_value = fake_metrics

    with mock.patch.object(mod, "_canonical_metrics_pnl_integrity", return_value={
        "pnl_abs": 500.0, "pnl_pct": 0.0, "win_rate": 0.6,
        "profit_factor": 1.8, "trade_count": 10,
        "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
        "avg_win": 75.0, "avg_loss": -30.0,
        "largest_win": 200.0, "largest_loss": -80.0,
        "diagnostic_excluded": 0, "synthetic_excluded": 0, "double_count_violations": 0,
    }):
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            payload = mod.build_dashboard_payload(
                conn=conn, tickers=["AAPL"], lookback_days=10,
                max_signals=5, max_trades=5, db_path=db_path,
            )
        finally:
            conn.close()

    # Realized PnL is known (from integrity enforcer)
    assert payload["performance_unknown"] is False, (
        "performance_unknown should be False when canonical PnL is available"
    )
    # But advanced analytics (Sharpe, drawdown, etc.) are NOT available
    assert payload["advanced_metrics_unknown"] is True, (
        "advanced_metrics_unknown should be True when performance_metrics table is empty "
        "even though canonical PnL metrics exist"
    )
    # Realized PnL data should still be present
    assert payload["pnl"]["absolute"] == 500.0
    assert payload["win_rate"] == 0.6
