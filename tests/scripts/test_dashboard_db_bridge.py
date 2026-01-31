from __future__ import annotations

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
