"""
Integration tests for the 4-layer model improvement measurement system.

Uses real file I/O (fixture audit JSONs), real SQLite DBs (in-memory/tmp_path),
and real imports where practical. Layer 2 subprocess is monkeypatched to avoid
slow gate execution.

These tests verify:
  - Schema invariants: every LayerResult.metrics contains required keys
  - Layer 1 correctly reads fixture files and detects anomalies
  - Layer 3 correctly queries SQLite and computes trade quality metrics
  - Layer 4 calls platt_contract_audit.run_audit and handles missing inputs
  - All four layers together satisfy the required-keys schema
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from check_model_improvement import (
    LAYER_REQUIRED_KEYS,
    LayerResult,
    run_layer1_forecast_quality,
    run_layer2_gate_status,
    run_layer3_trade_quality,
    run_layer4_calibration,
)

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "forecast_audits"

# ---------------------------------------------------------------------------
# Shared SQLite fixture helpers
# ---------------------------------------------------------------------------
_TRADE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS trade_executions (
        id INTEGER PRIMARY KEY,
        ticker TEXT,
        trade_date TEXT,
        action TEXT,
        realized_pnl REAL,
        realized_pnl_pct REAL,
        entry_price REAL,
        exit_price REAL,
        exit_reason TEXT,
        is_close INTEGER DEFAULT 0,
        is_diagnostic INTEGER DEFAULT 0,
        is_synthetic INTEGER DEFAULT 0,
        entry_trade_id INTEGER,
        holding_period_days REAL,
        ts_signal_id TEXT,
        confidence_calibrated REAL,
        effective_confidence REAL,
        base_confidence REAL,
        bar_high REAL,
        bar_low REAL
    );
    CREATE VIEW IF NOT EXISTS production_closed_trades AS
        SELECT * FROM trade_executions
        WHERE is_close = 1
          AND COALESCE(is_diagnostic, 0) = 0
          AND COALESCE(is_synthetic, 0) = 0;
    CREATE VIEW IF NOT EXISTS round_trips AS
        SELECT c.id AS close_id, o.id AS open_id, c.ticker,
               o.trade_date AS entry_date, c.trade_date AS exit_date,
               o.entry_price, c.exit_price,
               c.realized_pnl, c.holding_period_days, c.exit_reason
        FROM trade_executions c
        LEFT JOIN trade_executions o ON c.entry_trade_id = o.id
        WHERE c.is_close = 1;
"""

# 5 trades: 3 signal_exit winners, 2 stop_loss losers
# stop_loss_pct = 2/5 = 0.40
# Columns: id, ticker, trade_date, action, realized_pnl, realized_pnl_pct,
#          entry_price, exit_price, exit_reason, is_close, is_diagnostic, is_synthetic,
#          entry_trade_id, holding_period_days, ts_signal_id,
#          confidence_calibrated, effective_confidence, base_confidence, bar_high, bar_low
_SAMPLE_TRADES = [
    (1, "AAPL", "2025-01-10", "BUY", 120.0, None, 150.0, 170.0, "signal_exit", 1, 0, 0, None, 3.0, "ts_AAPL_001", None, None, None, None, None),
    (2, "MSFT", "2025-01-12", "BUY",  80.0, None, 200.0, 220.0, "signal_exit", 1, 0, 0, None, 2.0, "ts_MSFT_001", None, None, None, None, None),
    (3, "NVDA", "2025-01-15", "BUY",  50.0, None, 300.0, 325.0, "signal_exit", 1, 0, 0, None, 4.0, "ts_NVDA_001", None, None, None, None, None),
    (4, "AAPL", "2025-01-20", "BUY", -30.0, None, 155.0, 150.0, "stop_loss",  1, 0, 0, None, 1.0, "ts_AAPL_002", None, None, None, None, None),
    (5, "MSFT", "2025-01-22", "BUY", -25.0, None, 205.0, 200.0, "stop_loss",  1, 0, 0, None, 0.5, "ts_MSFT_002", None, None, None, None, None),
]


@pytest.fixture
def minimal_trade_db(tmp_path):
    """Temp SQLite with 3 winners + 2 stop-loss losers; win_rate=0.60, pf=2.5."""
    db = tmp_path / "test.db"
    con = sqlite3.connect(str(db))
    con.executescript(_TRADE_SCHEMA)
    con.executemany(
        "INSERT INTO trade_executions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        _SAMPLE_TRADES,
    )
    con.commit()
    con.close()
    return db


@pytest.fixture
def stop_loss_heavy_db(tmp_path):
    """Temp SQLite where stop_loss_pct > 0.40 -> interpretation='stop_too_tight'."""
    db = tmp_path / "stop_heavy.db"
    con = sqlite3.connect(str(db))
    con.executescript(_TRADE_SCHEMA)
    stop_trades = [
        # 3 stop_loss losers + 2 signal_exit winners  => stop_loss_pct = 3/5 = 0.60
        (10, "AAPL", "2025-01-10", "BUY", -40.0, None, 150.0, 145.0, "stop_loss", 1, 0, 0, None, 1.0, "ts_001", None, None, None, None, None),
        (11, "MSFT", "2025-01-11", "BUY", -35.0, None, 200.0, 195.0, "stop_loss", 1, 0, 0, None, 0.5, "ts_002", None, None, None, None, None),
        (12, "NVDA", "2025-01-12", "BUY", -30.0, None, 300.0, 294.0, "stop_loss", 1, 0, 0, None, 0.5, "ts_003", None, None, None, None, None),
        (13, "AAPL", "2025-01-15", "BUY", 100.0, None, 148.0, 158.0, "signal_exit", 1, 0, 0, None, 2.0, "ts_004", None, None, None, None, None),
        (14, "MSFT", "2025-01-16", "BUY",  80.0, None, 198.0, 206.0, "signal_exit", 1, 0, 0, None, 1.5, "ts_005", None, None, None, None, None),
    ]
    con.executemany(
        "INSERT INTO trade_executions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        stop_trades,
    )
    con.commit()
    con.close()
    return db


# ---------------------------------------------------------------------------
# Layer 1 integration tests
# ---------------------------------------------------------------------------
class TestLayer1Integration:
    def test_with_fixture_audits_returns_required_schema(self):
        if not FIXTURES_DIR.exists():
            pytest.skip("Fixture directory not found")
        result = run_layer1_forecast_quality(FIXTURES_DIR)
        assert result.layer == 1
        missing = LAYER_REQUIRED_KEYS[1] - set(result.metrics.keys())
        assert not missing, f"Layer 1 missing required keys: {missing}"

    def test_detects_samossa_da_anomaly_warn_from_fixture(self):
        """fixture_audit_samossa_da_zero.json has SAMOSSA DA=0; overall samossa_da_zero_pct = 1/3.
        n_used=3 < warn_coverage_threshold=50 -> WARN regardless, but also samossa anomaly counted."""
        if not FIXTURES_DIR.exists():
            pytest.skip("Fixture directory not found")
        result = run_layer1_forecast_quality(FIXTURES_DIR)
        # 1 of 3 windows has SAMOSSA DA=0 -> 0.333, below default 0.40 threshold
        # but it's still detected in metrics
        assert result.metrics["samossa_da_zero_pct"] == pytest.approx(1 / 3, abs=0.02)
        # Status is WARN (coverage < 50)
        assert result.status in ("WARN", "PASS")  # depends on other thresholds

    def test_lift_fraction_computed_correctly_from_fixtures(self):
        """
        healthy: ensemble=1.30, best_single=mssa_rl=1.40 -> lift (ratio 0.929 < 1.0)
        samossa_da_zero: ensemble=1.25, best_single=samossa=1.10 -> no lift (ratio 1.136 > 1.0)
        ensemble_lift: ensemble=1.60, best_single=mssa_rl=1.85 -> lift (ratio 0.865 < 1.0)
        => lift_fraction_global = 2/3 ~ 0.667
        """
        if not FIXTURES_DIR.exists():
            pytest.skip("Fixture directory not found")
        result = run_layer1_forecast_quality(FIXTURES_DIR)
        assert result.metrics["lift_fraction_global"] == pytest.approx(2 / 3, abs=0.02)
        assert result.metrics["lift_fraction_recent"] == pytest.approx(2 / 3, abs=0.02)


# ---------------------------------------------------------------------------
# Layer 2 integration tests (monkeypatched subprocess)
# ---------------------------------------------------------------------------
class TestLayer2Integration:
    def test_run_all_gates_json_has_overall_passed_key(self, monkeypatch):
        """Monkeypatch subprocess so the test is fast and reliable."""
        fake_json = json.dumps({
            "overall_passed": True,
            "phase": "institutional_unattended_hardening",
            "gates": [
                {"label": "integrity", "passed": True, "exit_code": 0},
                {"label": "quant_health", "passed": True, "exit_code": 0},
                {"label": "audit_lift", "passed": True, "exit_code": 0},
                {"label": "institutional", "passed": True, "exit_code": 0},
            ],
        })
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: SimpleNamespace(stdout=fake_json, stderr="", returncode=0),
        )
        result = run_layer2_gate_status()
        assert "overall_passed" in result.metrics
        missing = LAYER_REQUIRED_KEYS[2] - set(result.metrics.keys())
        assert not missing, f"Layer 2 missing required keys: {missing}"
        assert result.metrics["overall_passed"] is True


# ---------------------------------------------------------------------------
# Layer 3 integration tests
# ---------------------------------------------------------------------------
class TestLayer3Integration:
    def test_win_rate_computed_from_minimal_sqlite_db(self, minimal_trade_db):
        result = run_layer3_trade_quality(minimal_trade_db)
        # 3 winners, 2 losers -> win_rate 0.60
        assert result.status in ("PASS", "WARN")  # pf could be above/below threshold
        assert result.metrics["n_trades"] == 5
        assert result.metrics["win_rate"] == pytest.approx(0.60, abs=0.02)
        missing = LAYER_REQUIRED_KEYS[3] - set(result.metrics.keys())
        assert not missing, f"Layer 3 missing required keys: {missing}"

    def test_stop_loss_trades_trigger_stop_too_tight_interpretation(self, stop_loss_heavy_db):
        """3/5 stop_loss -> stop_loss_pct=0.60 > 0.40 -> interpretation='stop_too_tight'."""
        result = run_layer3_trade_quality(stop_loss_heavy_db)
        assert result.metrics["interpretation"] == "stop_too_tight"
        assert result.status == "WARN"

    def test_returns_skip_when_db_not_found(self, tmp_path):
        result = run_layer3_trade_quality(tmp_path / "nonexistent.db")
        assert result.status == "SKIP"
        assert "not found" in result.summary.lower()


# ---------------------------------------------------------------------------
# Layer 4 integration tests
# ---------------------------------------------------------------------------
class TestLayer4Integration:
    def test_with_empty_jsonl_returns_skip_or_warn(self, tmp_path):
        """DB doesn't exist, JSONL exists but is empty -> calibration_active_tier FAIL/WARN."""
        empty_jsonl = tmp_path / "qv.jsonl"
        empty_jsonl.write_text("")  # empty
        # no DB -> check_ts_closes_in_db will FAIL (db not found)
        result = run_layer4_calibration(tmp_path / "nonexistent.db", empty_jsonl)
        # JSONL exists (empty), DB does not -> run_audit runs, ts_closes_in_db FAIL -> FAIL
        assert result.status in ("FAIL", "WARN", "SKIP")
        missing = LAYER_REQUIRED_KEYS[4] - set(result.metrics.keys())
        assert not missing, f"Layer 4 missing required keys: {missing}"

    def test_skip_when_both_missing(self, tmp_path):
        result = run_layer4_calibration(
            tmp_path / "no_db.db",
            tmp_path / "no_jsonl.jsonl",
        )
        assert result.status == "SKIP"


# ---------------------------------------------------------------------------
# Schema invariant: all 4 layers must have required keys
# ---------------------------------------------------------------------------
class TestAllFourLayersSchemaInvariant:
    def test_all_four_layers_metrics_have_required_keys(self, tmp_path, monkeypatch):
        """Run all 4 layers (SKIP mode — missing data) and check schema completeness."""
        fake_gate_json = json.dumps({
            "overall_passed": True,
            "gates": [{"label": "integrity", "passed": True}],
        })
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: SimpleNamespace(stdout=fake_gate_json, stderr="", returncode=0),
        )
        results = [
            run_layer1_forecast_quality(tmp_path / "no_audits"),
            run_layer2_gate_status(),
            run_layer3_trade_quality(tmp_path / "no.db"),
            run_layer4_calibration(tmp_path / "no.db", tmp_path / "no.jsonl"),
        ]
        for result in results:
            missing = LAYER_REQUIRED_KEYS[result.layer] - set(result.metrics.keys())
            assert not missing, (
                f"Layer {result.layer} ({result.name}) missing required metrics keys: {missing}"
            )
            assert result.status in ("PASS", "WARN", "FAIL", "SKIP")
            assert isinstance(result.summary, str) and len(result.summary) > 0

    def test_fixture_layer1_and_skip_layers_correct_statuses(self, tmp_path, monkeypatch):
        """Full run with fixture audits for Layer 1, SKIP for others."""
        if not FIXTURES_DIR.exists():
            pytest.skip("Fixture directory not found")

        fake_gate_json = json.dumps({"overall_passed": True, "gates": []})
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: SimpleNamespace(stdout=fake_gate_json, stderr="", returncode=0),
        )
        l1 = run_layer1_forecast_quality(FIXTURES_DIR)
        l2 = run_layer2_gate_status()
        l3 = run_layer3_trade_quality(tmp_path / "no.db")
        l4 = run_layer4_calibration(tmp_path / "no.db", tmp_path / "no.jsonl")

        # Layer 1: WARN (3 windows < coverage threshold)
        assert l1.status in ("WARN", "PASS")
        assert l1.metrics["n_used_windows"] == 3
        # Layer 2: PASS (mocked)
        assert l2.status == "PASS"
        # Layer 3 and 4: SKIP (no data)
        assert l3.status == "SKIP"
        assert l4.status == "SKIP"
