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
