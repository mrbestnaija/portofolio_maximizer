from __future__ import annotations

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
        conn.execute(
            "INSERT INTO portfolio_positions(ticker,position_date,shares,average_cost) VALUES (?,?,?,?)",
            ("AAPL", "2026-01-02", 10, 101.0),
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
        )
    finally:
        conn.close()

    assert payload["meta"]["run_id"]
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
