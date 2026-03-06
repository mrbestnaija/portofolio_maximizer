"""Tests for scripts/capital_readiness_check.py (Phase 7.30).

All tests use monkeypatch to mock the dependency layer — no live DB or
adversarial runner required.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.capital_readiness_check import run_capital_readiness


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_finding(id_: str, severity: str, passed: bool) -> object:
    finding = MagicMock()
    finding.id = id_
    finding.severity = severity
    finding.passed = passed
    return finding


def _layer_result(
    layer: int,
    status: str = "PASS",
    metrics: dict | None = None,
    summary: str = "ok",
) -> object:
    from scripts.check_model_improvement import LayerResult
    return LayerResult(
        layer=layer,
        name=f"Layer{layer}",
        status=status,
        metrics=metrics or {},
        summary=summary,
    )


def _patch_all_passing(monkeypatch):
    """Monkeypatch all 5 rule checks to return 'all clear' defaults."""
    # R1: no adversarial confirmed
    import scripts.capital_readiness_check as mod
    monkeypatch.setattr(
        mod, "_check_r1_adversarial",
        lambda db, audit: (True, "", {"n_critical_high_confirmed": 0, "confirmed_ids": []}),
    )
    # R2: gate artifact fresh and passed
    monkeypatch.setattr(
        mod, "_check_r2_gate_artifact",
        lambda: (True, "", {"gate_overall_passed": True, "gate_age_hours": 2.0}),
    )
    # R3: good trade quality
    monkeypatch.setattr(
        mod, "_check_r3_trade_quality",
        lambda db: (True, "", {"n_trades": 50, "win_rate": 0.55, "profit_factor": 1.80}),
    )
    # R4: good calibration
    monkeypatch.setattr(
        mod, "_check_r4_calibration",
        lambda db, jsonl: (True, "", {"calibration_tier": "db_local", "brier_score": 0.18}),
    )
    # R5: CI positive (no warning)
    monkeypatch.setattr(
        mod, "_check_r5_lift_ci",
        lambda db, audit: ("", {"lift_ci_low": 0.05}),
    )
    # R6: no lifecycle integrity issues
    monkeypatch.setattr(
        mod, "_check_r6_lifecycle_integrity",
        lambda db: (
            True,
            "",
            {
                "close_before_entry_count": 0,
                "closed_missing_exit_reason_count": 0,
                "high_integrity_violation_count": 0,
            },
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCapitalReadinessR1:
    def test_fails_when_critical_adversarial_confirmed(self, monkeypatch, tmp_path):
        import scripts.capital_readiness_check as mod
        _patch_all_passing(monkeypatch)
        monkeypatch.setattr(
            mod, "_check_r1_adversarial",
            lambda db, audit: (
                False,
                "R1: 1 confirmed CRITICAL/HIGH finding(s): ['LEAK-01']",
                {"n_critical_high_confirmed": 1, "confirmed_ids": ["LEAK-01"]},
            ),
        )
        result = run_capital_readiness(tmp_path / "x.db", tmp_path, tmp_path / "q.jsonl")
        assert result["ready"] is False
        assert result["verdict"] in ("FAIL", "INSUFFICIENT_DATA")
        assert any("R1" in r for r in result["reasons"])

    def test_fails_when_high_adversarial_confirmed(self, monkeypatch, tmp_path):
        import scripts.capital_readiness_check as mod
        _patch_all_passing(monkeypatch)
        monkeypatch.setattr(
            mod, "_check_r1_adversarial",
            lambda db, audit: (
                False,
                "R1: 1 confirmed CRITICAL/HIGH finding(s): ['WIRE-01']",
                {"n_critical_high_confirmed": 1, "confirmed_ids": ["WIRE-01"]},
            ),
        )
        result = run_capital_readiness(tmp_path / "x.db", tmp_path, tmp_path / "q.jsonl")
        assert result["ready"] is False
        assert any("R1" in r for r in result["reasons"])


class TestCapitalReadinessR2:
    def test_fails_when_gate_artifact_missing(self, monkeypatch, tmp_path):
        import scripts.capital_readiness_check as mod
        _patch_all_passing(monkeypatch)
        monkeypatch.setattr(
            mod, "_check_r2_gate_artifact",
            lambda: (False, "R2: gate_status_latest.json not found", {"gate_overall_passed": None, "gate_age_hours": None}),
        )
        result = run_capital_readiness(tmp_path / "x.db", tmp_path, tmp_path / "q.jsonl")
        assert result["ready"] is False
        assert any("R2" in r for r in result["reasons"])

    def test_fails_when_gate_status_not_passed(self, monkeypatch, tmp_path):
        import scripts.capital_readiness_check as mod
        _patch_all_passing(monkeypatch)
        monkeypatch.setattr(
            mod, "_check_r2_gate_artifact",
            lambda: (False, "R2: gate artifact overall_passed=False", {"gate_overall_passed": False, "gate_age_hours": 1.0}),
        )
        result = run_capital_readiness(tmp_path / "x.db", tmp_path, tmp_path / "q.jsonl")
        assert result["ready"] is False

    def test_fails_when_gate_artifact_stale_26h(self, monkeypatch, tmp_path):
        import scripts.capital_readiness_check as mod
        _patch_all_passing(monkeypatch)
        monkeypatch.setattr(
            mod, "_check_r2_gate_artifact",
            lambda: (False, "R2: gate artifact is stale (27.0h >= 26.0h)", {"gate_overall_passed": True, "gate_age_hours": 27.0}),
        )
        result = run_capital_readiness(tmp_path / "x.db", tmp_path, tmp_path / "q.jsonl")
        assert result["ready"] is False


class TestCapitalReadinessR3:
    def test_fails_when_too_few_trades(self, monkeypatch, tmp_path):
        import scripts.capital_readiness_check as mod
        _patch_all_passing(monkeypatch)
        monkeypatch.setattr(
            mod, "_check_r3_trade_quality",
            lambda db: (False, "R3: only 12 trades (min 20)", {"n_trades": 12, "win_rate": 0.58, "profit_factor": 1.71}),
        )
        result = run_capital_readiness(tmp_path / "x.db", tmp_path, tmp_path / "q.jsonl")
        assert result["ready"] is False
        assert any("R3" in r for r in result["reasons"])


class TestCapitalReadinessR4:
    def test_fails_when_calibration_inactive(self, monkeypatch, tmp_path):
        import scripts.capital_readiness_check as mod
        _patch_all_passing(monkeypatch)
        monkeypatch.setattr(
            mod, "_check_r4_calibration",
            lambda db, jsonl: (False, "R4: calibration tier is 'inactive'", {"calibration_tier": "inactive", "brier_score": None}),
        )
        result = run_capital_readiness(tmp_path / "x.db", tmp_path, tmp_path / "q.jsonl")
        assert result["ready"] is False
        assert any("R4" in r for r in result["reasons"])


class TestCapitalReadinessR5:
    def test_warns_but_passes_when_lift_ci_spans_zero(self, monkeypatch, tmp_path):
        """R5 is advisory — CI spanning zero emits WARNING but does not prevent PASS."""
        import scripts.capital_readiness_check as mod
        _patch_all_passing(monkeypatch)
        monkeypatch.setattr(
            mod, "_check_r5_lift_ci",
            lambda db, audit: (
                "R5: lift CI [-0.003, 0.012] spans zero (win_fraction=48.0%) -- lift not statistically confirmed",
                {"lift_ci_low": -0.003},
            ),
        )
        result = run_capital_readiness(tmp_path / "x.db", tmp_path, tmp_path / "q.jsonl")
        assert result["ready"] is True, (
            f"R5 advisory should not block readiness; got verdict={result['verdict']}"
        )
        assert result["verdict"] == "PASS"
        assert any("R5" in w for w in result["warnings"]), "R5 warning must appear in warnings list"
        assert result["reasons"] == [], "R5 must not add to reasons list"


class TestCapitalReadinessPasses:
    def test_passes_when_all_conditions_met(self, monkeypatch, tmp_path):
        """All R1–R4 clear → ready=True, verdict=PASS."""
        import scripts.capital_readiness_check as mod
        _patch_all_passing(monkeypatch)
        result = run_capital_readiness(tmp_path / "x.db", tmp_path, tmp_path / "q.jsonl")
        assert result["ready"] is True
        assert result["verdict"] == "PASS"
        assert result["reasons"] == []

    def test_returns_insufficient_data_when_db_missing(self, monkeypatch, tmp_path):
        """Missing DB → R3/R4 both None → INSUFFICIENT_DATA verdict."""
        import scripts.capital_readiness_check as mod
        _patch_all_passing(monkeypatch)
        # Simulate R3 returning None (insufficient data)
        monkeypatch.setattr(
            mod, "_check_r3_trade_quality",
            lambda db: (None, "R3: db not found -- /nonexistent/db.db", {"n_trades": None, "win_rate": None, "profit_factor": None}),
        )
        monkeypatch.setattr(
            mod, "_check_r4_calibration",
            lambda db, jsonl: (None, "R4: calibration skipped -- db missing", {"calibration_tier": None, "brier_score": None}),
        )
        result = run_capital_readiness(tmp_path / "x.db", tmp_path, tmp_path / "q.jsonl")
        assert result["ready"] is False
        assert result["verdict"] == "INSUFFICIENT_DATA"

    def test_fails_when_r6_lifecycle_integrity_violations_present(self, monkeypatch, tmp_path):
        import scripts.capital_readiness_check as mod

        _patch_all_passing(monkeypatch)
        monkeypatch.setattr(
            mod,
            "_check_r6_lifecycle_integrity",
            lambda db: (
                False,
                "R6: HIGH lifecycle violation(s): close_before_entry=1, closed_missing_exit_reason=1",
                {
                    "close_before_entry_count": 1,
                    "closed_missing_exit_reason_count": 1,
                    "high_integrity_violation_count": 2,
                },
            ),
        )
        result = run_capital_readiness(tmp_path / "x.db", tmp_path, tmp_path / "q.jsonl")
        assert result["ready"] is False
        assert result["verdict"] == "FAIL"
        assert any(reason.startswith("R6:") for reason in result["reasons"])

    def test_r6_legacy_trades_excluded_from_close_before_entry(self, tmp_path):
        """Legacy trades (ts_signal_id LIKE 'legacy_%') are not flagged for close_before_entry
        even when bar_timestamp is inverted — those timestamps come from forecast windows."""
        import sqlite3
        from scripts.capital_readiness_check import _check_r6_lifecycle_integrity

        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        conn.executescript(
            """
            CREATE TABLE trade_executions (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                trade_date TEXT,
                action TEXT,
                realized_pnl REAL,
                is_close INTEGER DEFAULT 0,
                is_diagnostic INTEGER DEFAULT 0,
                is_synthetic INTEGER DEFAULT 0,
                bar_timestamp TEXT,
                exit_reason TEXT,
                entry_trade_id INTEGER,
                ts_signal_id TEXT
            );
            -- Legacy open leg (entry bar = July 25)
            INSERT INTO trade_executions VALUES
              (1,'MSFT','2026-02-10','BUY',NULL,0,0,0,'2025-07-25T00:00:00+00:00','TIME_EXIT',NULL,'legacy_2026-02-10_1');
            -- Legacy close leg (close bar = July 18, EARLIER than entry — legacy artifact)
            INSERT INTO trade_executions VALUES
              (2,'MSFT','2026-02-10','SELL',72.5,1,0,0,'2025-07-18T00:00:00+00:00','TIME_EXIT',1,'legacy_2026-02-10_2');
            CREATE VIEW production_closed_trades AS
              SELECT * FROM trade_executions
              WHERE is_close=1 AND COALESCE(is_diagnostic,0)=0 AND COALESCE(is_synthetic,0)=0;
            """
        )
        conn.close()

        ok, msg, metrics = _check_r6_lifecycle_integrity(db)
        assert ok is True, f"Legacy trade should not trigger R6: {msg}"
        assert metrics["close_before_entry_count"] == 0
        assert metrics["high_integrity_violation_count"] == 0
