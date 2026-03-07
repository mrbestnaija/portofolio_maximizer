from __future__ import annotations

from datetime import datetime, timezone
import pytest
import sqlite3

from scripts import dashboard_db_bridge as mod


def test_build_dashboard_payload_from_sqlite(tmp_path, monkeypatch) -> None:
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

    monkeypatch.setenv("PMX_POSITIONS_MAX_AGE_DAYS", "36500")
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


def test_empty_performance_metrics_reports_unknown(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "pmx_perf_unknown.db"
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
        conn.execute(
            "INSERT INTO portfolio_positions(ticker,position_date,shares,average_cost) VALUES (?,?,?,?)",
            ("AAPL", "2026-01-01", 5, 99.0),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(mod, "_canonical_metrics_pnl_integrity", lambda _: {})
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

    assert payload["performance_unknown"] is True
    assert payload["performance"]["performance_unknown"] is True
    assert payload["pnl"]["absolute"] is None
    assert payload["pnl"]["pct"] is None
    assert payload["win_rate"] is None
    assert payload["trade_count"] is None


def test_positions_stale_falls_back_and_filters_non_production_rows(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "pmx_positions_stale.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE ohlcv_data(date TEXT, ticker TEXT, close REAL)")
        conn.execute(
            "CREATE TABLE trading_signals(id INTEGER PRIMARY KEY, ticker TEXT, action TEXT, confidence REAL, expected_return REAL, source TEXT, signal_timestamp TEXT, created_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE trade_executions(id INTEGER PRIMARY KEY, ticker TEXT, action TEXT, shares REAL, price REAL, trade_date TEXT, created_at TEXT, realized_pnl REAL, realized_pnl_pct REAL, mid_slippage_bps REAL, run_id TEXT, is_diagnostic INTEGER DEFAULT 0, is_synthetic INTEGER DEFAULT 0)"
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
        conn.execute(
            "INSERT INTO portfolio_positions(ticker,position_date,shares,average_cost) VALUES (?,?,?,?)",
            ("AAPL", "2026-01-01", 999, 1.0),
        )
        conn.execute("INSERT INTO ohlcv_data(date,ticker,close) VALUES ('2026-01-02','AAPL',101.0)")
        conn.execute("INSERT INTO ohlcv_data(date,ticker,close) VALUES ('2026-01-02','TSLA',202.0)")
        conn.execute("INSERT INTO ohlcv_data(date,ticker,close) VALUES ('2026-01-02','MSFT',303.0)")
        conn.execute(
            "INSERT INTO trade_executions(ticker,action,shares,price,trade_date,created_at,run_id,is_diagnostic,is_synthetic) VALUES (?,?,?,?,?,?,?,?,?)",
            ("AAPL", "BUY", 10, 100.0, "2026-01-02", "2026-01-02T00:00:01Z", "RID", 0, 0),
        )
        conn.execute(
            "INSERT INTO trade_executions(ticker,action,shares,price,trade_date,created_at,run_id,is_diagnostic,is_synthetic) VALUES (?,?,?,?,?,?,?,?,?)",
            ("TSLA", "BUY", 10, 200.0, "2026-01-02", "2026-01-02T00:00:02Z", "RID", 1, 0),
        )
        conn.execute(
            "INSERT INTO trade_executions(ticker,action,shares,price,trade_date,created_at,run_id,is_diagnostic,is_synthetic) VALUES (?,?,?,?,?,?,?,?,?)",
            ("MSFT", "BUY", 10, 300.0, "2026-01-02", "2026-01-02T00:00:03Z", "RID", 0, 1),
        )
        conn.execute(
            "INSERT INTO performance_metrics(total_return,total_return_pct,win_rate,profit_factor,num_trades,created_at) VALUES (?,?,?,?,?,?)",
            (1.0, 0.001, 0.5, 1.1, 1, "2026-01-02T00:00:00Z"),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("PMX_POSITIONS_MAX_AGE_DAYS", "14")
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        payload = mod.build_dashboard_payload(
            conn=conn,
            tickers=["AAPL", "TSLA", "MSFT"],
            lookback_days=10,
            max_signals=10,
            max_trades=20,
            db_path=db_path,
            read_path=db_path,
        )
    finally:
        conn.close()

    assert payload["positions_stale"] is True
    assert payload["positions_source"] == "trade_executions_fallback"
    assert "AAPL" in payload["positions"]
    assert "TSLA" not in payload["positions"]
    assert "MSFT" not in payload["positions"]


def test_positions_fresh_snapshot_remains_authoritative(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "pmx_positions_fresh.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE ohlcv_data(date TEXT, ticker TEXT, close REAL)")
        conn.execute(
            "CREATE TABLE trading_signals(id INTEGER PRIMARY KEY, ticker TEXT, action TEXT, confidence REAL, expected_return REAL, source TEXT, signal_timestamp TEXT, created_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE trade_executions(id INTEGER PRIMARY KEY, ticker TEXT, action TEXT, shares REAL, price REAL, trade_date TEXT, created_at TEXT, realized_pnl REAL, realized_pnl_pct REAL, mid_slippage_bps REAL, run_id TEXT, is_diagnostic INTEGER DEFAULT 0, is_synthetic INTEGER DEFAULT 0)"
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
        today = datetime.now(timezone.utc).date().isoformat()
        conn.execute(
            "INSERT INTO portfolio_positions(ticker,position_date,shares,average_cost) VALUES (?,?,?,?)",
            ("AAPL", today, 7, 123.0),
        )
        conn.execute("INSERT INTO ohlcv_data(date,ticker,close) VALUES ('2026-01-02','AAPL',124.0)")
        conn.execute(
            "INSERT INTO trade_executions(ticker,action,shares,price,trade_date,created_at,run_id,is_diagnostic,is_synthetic) VALUES (?,?,?,?,?,?,?,?,?)",
            ("AAPL", "BUY", 100, 1.0, "2026-01-02", "2026-01-02T00:00:01Z", "RID", 0, 0),
        )
        conn.execute(
            "INSERT INTO performance_metrics(total_return,total_return_pct,win_rate,profit_factor,num_trades,created_at) VALUES (?,?,?,?,?,?)",
            (1.0, 0.001, 0.5, 1.1, 1, "2026-01-02T00:00:00Z"),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("PMX_POSITIONS_MAX_AGE_DAYS", "14")
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        payload = mod.build_dashboard_payload(
            conn=conn,
            tickers=["AAPL"],
            lookback_days=10,
            max_signals=10,
            max_trades=20,
            db_path=db_path,
            read_path=db_path,
        )
    finally:
        conn.close()

    assert payload["positions_stale"] is False
    assert payload["positions_source"] == "portfolio_positions"
    assert payload["positions"]["AAPL"]["shares"] == 7


def test_trade_events_include_exit_reason(tmp_path) -> None:
    db_path = tmp_path / "pmx_exit_reason.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE ohlcv_data(date TEXT, ticker TEXT, close REAL)")
        conn.execute(
            "CREATE TABLE trading_signals(id INTEGER PRIMARY KEY, ticker TEXT, action TEXT, confidence REAL, expected_return REAL, source TEXT, signal_timestamp TEXT, created_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE trade_executions(id INTEGER PRIMARY KEY, ticker TEXT, action TEXT, shares REAL, price REAL, trade_date TEXT, created_at TEXT, realized_pnl REAL, realized_pnl_pct REAL, mid_slippage_bps REAL, run_id TEXT, exit_reason TEXT)"
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
        conn.execute(
            "INSERT INTO trade_executions(ticker,action,shares,price,trade_date,created_at,realized_pnl,realized_pnl_pct,mid_slippage_bps,run_id,exit_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("AAPL", "SELL", 5, 95.0, "2026-01-02", "2026-01-02T00:00:02Z", -25.0, -0.05, 10.0, "RID", "STOP_LOSS"),
        )
        conn.execute(
            "INSERT INTO performance_metrics(total_return,total_return_pct,win_rate,profit_factor,num_trades,created_at) VALUES (?,?,?,?,?,?)",
            (-25.0, -0.01, 0.0, 0.0, 1, "2026-01-02T00:00:00Z"),
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
            max_trades=20,
            db_path=db_path,
            read_path=db_path,
        )
    finally:
        conn.close()

    assert payload["trade_events"], "Expected trade events in payload"
    assert payload["trade_events"][-1]["exit_reason"] == "STOP_LOSS"


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
