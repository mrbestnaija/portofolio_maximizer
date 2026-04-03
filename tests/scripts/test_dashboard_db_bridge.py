from __future__ import annotations

import json
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
    monkeypatch.setenv("PMX_SIDECAR_MAX_AGE_MINUTES", "999999")

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
    monkeypatch.setenv("PMX_SIDECAR_MAX_AGE_MINUTES", "999999")

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


def test_evidence_payload_reads_production_gate_and_audit_snapshot(tmp_path, monkeypatch) -> None:
    gate = tmp_path / "production_gate_latest.json"
    gate.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-15T18:22:13Z",
                "phase3_ready": False,
                "phase3_reason": "GATES_FAIL,ARTIFACT_STALE_OR_UNBOUND",
                "artifact_binding": {
                    "pass": False,
                    "reason_codes": ["NO_LIVE_CYCLE_TIMESTAMP"],
                },
                "production_profitability_gate": {
                    "status": "FAIL",
                    "gate_semantics_status": "INCONCLUSIVE_BLOCKED",
                },
                "profitability_proof": {
                    "status": "FAIL",
                    "evidence_progress": {
                        "remaining_closed_trades": 30,
                        "remaining_trading_days": 21,
                    },
                },
                "telemetry_contract": {
                    "status": "INCONCLUSIVE_BLOCKED",
                    "severity": "HIGH",
                    "generated_utc": "2026-03-15T18:22:13Z",
                },
            }
        ),
        encoding="utf-8",
    )

    audit_db = tmp_path / "dashboard_audit.db"
    conn = sqlite3.connect(audit_db)
    try:
        conn.execute(
            """
            CREATE TABLE dashboard_snapshots(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                run_id TEXT,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO dashboard_snapshots(created_at, run_id, payload_json) VALUES (?,?,?)",
            ("2026-03-15T18:21:51Z", "RID-1", "{}"),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(mod, "DEFAULT_PRODUCTION_GATE_PATH", gate)
    monkeypatch.setattr(mod, "DEFAULT_AUDIT_DB_PATH", audit_db)

    evidence = {
        "production_gate": mod._production_gate_summary(),
        "dashboard_audit": mod._dashboard_audit_summary(),
    }
    alerts = mod._operator_alerts(
        provenance={"origin": "mixed"},
        evidence=evidence,
        robustness={"status": "WARN"},
        signal_count=0,
        trade_count=0,
        price_series_count=0,
    )

    assert evidence["production_gate"]["status"] == "FAIL"
    assert evidence["production_gate"]["artifact_binding_pass"] is False
    assert evidence["production_gate"]["remaining_closed_trades"] == 30
    assert evidence["dashboard_audit"]["status"] == "OK"
    assert evidence["dashboard_audit"]["snapshot_count"] == 1
    assert any("Artifact binding failed" in alert for alert in alerts)
    assert any("Data origin is not fully live" in alert for alert in alerts)


def test_operator_alerts_surface_audit_db_error(tmp_path, monkeypatch) -> None:
    audit_db = tmp_path / "dashboard_audit.db"
    audit_db.write_text("not a sqlite database", encoding="utf-8")
    monkeypatch.setattr(mod, "DEFAULT_AUDIT_DB_PATH", audit_db)

    audit = mod._dashboard_audit_summary()
    alerts = mod._operator_alerts(
        provenance={"origin": "live"},
        evidence={"dashboard_audit": audit, "production_gate": {"status": "PASS", "phase3_ready": True}},
        robustness={"status": "OK"},
        signal_count=1,
        trade_count=1,
        price_series_count=1,
    )

    assert audit["status"] == "ERROR"
    assert any("cannot be queried cleanly" in alert for alert in alerts)
    assert all("no persisted snapshots yet" not in alert for alert in alerts)


def test_operator_alerts_surface_production_gate_refresh_failures() -> None:
    alerts = mod._operator_alerts(
        provenance={"origin": "live"},
        evidence={
            "production_gate": {"status": "PASS", "phase3_ready": True, "freshness_status": "FRESH"},
            "dashboard_audit": {"status": "OK"},
            "production_gate_refresh": {
                "enabled": True,
                "attempted": True,
                "ok": False,
                "reason": "timeout",
                "detail": "timed out after 120.0s",
            },
        },
        robustness={"status": "OK"},
        signal_count=1,
        trade_count=1,
        price_series_count=1,
    )

    assert any("auto-refresh failed" in alert for alert in alerts)


def test_maybe_refresh_production_gate_artifact_triggers_on_stale_artifact(tmp_path, monkeypatch) -> None:
    gate = tmp_path / "production_gate_latest.json"
    gate.write_text(
        json.dumps({"timestamp_utc": "2026-01-01T00:00:00Z", "production_profitability_gate": {"status": "PASS"}}),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    monkeypatch.setenv("PMX_PRODUCTION_GATE_MAX_AGE_MINUTES", "30")
    monkeypatch.setattr(
        mod,
        "_refresh_production_gate_artifact",
        lambda **kwargs: calls.append(dict(kwargs)) or {
            "enabled": True,
            "attempted": True,
            "ok": True,
            "status": "OK",
            "reason": "stale_artifact",
            "attempted_utc": "2026-04-02T07:15:00Z",
            "artifact_path": gate.as_posix(),
            "returncode": 1,
            "artifact_refreshed": True,
            "artifact_age_minutes": 0.1,
            "artifact_freshness_status": "FRESH",
            "artifact_freshness_reason": None,
            "generated_utc": "2026-04-02T07:15:00Z",
            "detail": None,
        },
    )

    status, last_attempt = mod._maybe_refresh_production_gate_artifact(
        db_path=tmp_path / "portfolio_maximizer.db",
        artifact_path=gate,
        timeout_seconds=120.0,
        min_interval_seconds=300.0,
        last_attempt_monotonic=None,
        force=False,
        python_bin="python",
        actor="test_actor",
    )

    assert calls
    assert status["ok"] is True
    assert status["reason"] == "stale_artifact"
    assert last_attempt is not None


def test_maybe_merge_with_existing_preserves_last_success_for_same_gate_artifact(tmp_path) -> None:
    dashboard_json = tmp_path / "dashboard_data.json"
    dashboard_json.write_text(
        json.dumps(
            {
                "evidence": {
                    "production_gate": {
                        "generated_utc": "2026-04-02T12:12:49Z",
                    },
                    "production_gate_refresh": {
                        "enabled": True,
                        "attempted": True,
                        "ok": True,
                        "status": "OK",
                        "reason": "artifact_refreshed",
                        "attempted_utc": "2026-04-02T12:12:50Z",
                        "generated_utc": "2026-04-02T12:12:49Z",
                        "actor": "dashboard_launch",
                        "last_success_utc": "2026-04-02T12:12:50Z",
                        "last_success_reason": "artifact_refreshed",
                        "last_success_generated_utc": "2026-04-02T12:12:49Z",
                        "last_success_actor": "dashboard_launch",
                        "last_success_status": "OK",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    merged = mod._maybe_merge_with_existing(
        dashboard_json,
        {
            "evidence": {
                "production_gate": {
                    "generated_utc": "2026-04-02T12:12:49Z",
                },
                "production_gate_refresh": {
                    "enabled": True,
                    "attempted": False,
                    "ok": None,
                    "status": "SKIPPED",
                    "reason": "fresh_artifact",
                    "generated_utc": "2026-04-02T12:12:49Z",
                    "actor": "dashboard_bridge",
                },
            }
        },
    )

    refresh = merged["evidence"]["production_gate_refresh"]
    assert refresh["status"] == "SKIPPED"
    assert refresh["actor"] == "dashboard_bridge"
    assert refresh["last_success_actor"] == "dashboard_launch"
    assert refresh["last_success_generated_utc"] == "2026-04-02T12:12:49Z"


def test_persist_snapshot_bootstraps_audit_db_schema(tmp_path) -> None:
    audit_db = tmp_path / "dashboard_audit.db"

    mod._persist_snapshot(
        audit_db,
        {
            "meta": {"run_id": "RID-BOOTSTRAP"},
            "signals": [],
            "trade_events": [],
        },
    )

    conn = sqlite3.connect(audit_db)
    try:
        row = conn.execute("SELECT COUNT(*) FROM dashboard_snapshots").fetchone()
    finally:
        conn.close()

    assert row is not None
    assert int(row[0]) == 1


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
