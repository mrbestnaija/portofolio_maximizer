"""
Code-contract tests for the Platt calibration implementation.

Purpose: Make implementation drift immediately visible as CI failures.

These tests assert WHAT the code does, not just that it runs. If any test
breaks, it means either:
  (a) The implementation changed without updating docs/agents -- fix the docs, or
  (b) The docs are correct and someone broke the implementation -- fix the code.

Covers:
  1. Classifier identity     -- LogisticRegression, never isotonic
  2. Fallback chain order    -- JSONL -> DB-local -> DB-global
  3. HOLD entries contract   -- HOLD signals structurally cannot reconcile
  4. Contract audit script   -- platt_contract_audit.py runs and self-validates
  5. Bootstrap outcome guard -- overnight refresh checks ts_* close count post-bootstrap
"""

from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path

import pytest


# ===========================================================================
# 1. Classifier identity
# ===========================================================================

class TestClassifierIdentity:
    """_calibrate_confidence must use LogisticRegression (classic Platt scaling)."""

    def _get_source(self) -> str:
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        return inspect.getsource(TimeSeriesSignalGenerator._calibrate_confidence)

    def test_uses_logistic_regression(self):
        """Implementation must use LogisticRegression -- matches docs and audit."""
        source = self._get_source()
        assert "LogisticRegression" in source, (
            "_calibrate_confidence no longer uses LogisticRegression.\n"
            "If you changed the classifier:\n"
            "  1. Update PHASE_7.14_GATE_RECALIBRATION.md (remove isotonic claim)\n"
            "  2. Update AGENTS.md Platt contract rules\n"
            "  3. Update this test to reflect the new classifier"
        )

    def test_does_not_use_isotonic(self):
        """Isotonic regression must never appear -- docs previously made this false claim."""
        source = self._get_source().lower()
        assert "isotonicregression" not in source, (
            "IsotonicRegression introduced in _calibrate_confidence.\n"
            "Phase 7.14 docs explicitly corrected an isotonic claim. "
            "If this change is intentional, update docs and this test."
        )
        assert "isotonic" not in source, (
            "String 'isotonic' (e.g., method='isotonic') found in _calibrate_confidence.\n"
            "If isotonic calibration is intentional, update docs and this test."
        )

    def test_jsonl_loader_method_exists(self):
        """_load_jsonl_outcome_pairs must exist as the JSONL tier entry point."""
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        assert hasattr(TimeSeriesSignalGenerator, "_load_jsonl_outcome_pairs"), (
            "_load_jsonl_outcome_pairs removed -- JSONL tier broken."
        )

    def test_db_loader_method_exists(self):
        """_load_realized_outcome_pairs must exist as the DB tier entry point."""
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        assert hasattr(TimeSeriesSignalGenerator, "_load_realized_outcome_pairs"), (
            "_load_realized_outcome_pairs removed -- DB fallback tier broken."
        )


# ===========================================================================
# 2. Fallback chain order
# ===========================================================================

class TestFallbackChainOrder:
    """Fallback priority must be JSONL -> DB-local -> DB-global."""

    def _get_source(self) -> str:
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        return inspect.getsource(TimeSeriesSignalGenerator._calibrate_confidence)

    def test_jsonl_before_db_local(self):
        source = self._get_source()
        jsonl_pos = source.find("_load_jsonl_outcome_pairs")
        db_local_pos = source.find("db_local")
        assert jsonl_pos >= 0, "_load_jsonl_outcome_pairs not found in source"
        assert db_local_pos >= 0, "'db_local' marker not found in source"
        assert jsonl_pos < db_local_pos, (
            "Fallback chain broken: db_local tier appears before JSONL check.\n"
            "Priority must be JSONL -> DB-local -> DB-global."
        )

    def test_db_local_before_db_global(self):
        source = self._get_source()
        db_local_pos = source.find("db_local")
        db_global_pos = source.find("db_global")
        assert db_local_pos >= 0, "'db_local' not found in source"
        assert db_global_pos >= 0, "'db_global' not found in source"
        assert db_local_pos < db_global_pos, (
            "Fallback chain broken: db_global tier appears before db_local."
        )

    def test_minimum_threshold_is_30(self):
        """Threshold of 30 pairs is documented -- must not silently change."""
        source = self._get_source()
        has_threshold = "< 30" in source or "n < 30" in source or "< PLATT_MIN_PAIRS" in source
        assert has_threshold, (
            "Minimum pair threshold '30' not found in _calibrate_confidence.\n"
            "If the threshold changed, update docs and this test."
        )


# ===========================================================================
# 3. HOLD entries are structurally unreconcilable
# ===========================================================================

class TestHoldEntriesContract:
    """HOLD signals cannot produce is_close=1 trades -- reconciler must not count them as pending."""

    def _make_db(self, path: Path, signal_id: str) -> None:
        conn = sqlite3.connect(str(path))
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE trade_executions ("
            "id INTEGER PRIMARY KEY, ts_signal_id TEXT, "
            "realized_pnl REAL, realized_pnl_pct REAL, is_close INTEGER, action TEXT)"
        )
        cur.execute(
            "INSERT INTO trade_executions VALUES (1, ?, 50.0, 0.05, 1, 'BUY')",
            (signal_id,),
        )
        conn.commit()
        conn.close()

    def test_hold_entry_never_receives_outcome(self, tmp_path):
        """HOLD-action entries must not get an outcome, even if DB has matching ts_signal_id."""
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"

        buy_sid = "ts_AAPL_20260101T120000Z_a1b2_0001"
        self._make_db(db, buy_sid)

        # Both share the same signal_id prefix -- HOLD should never match
        entries = [
            {"signal_id": buy_sid, "confidence": 0.75, "action": "BUY"},
            {"signal_id": "ts_AAPL_20260101T120001Z_a1b2_0002", "confidence": 0.60, "action": "HOLD"},
        ]
        log.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

        reconcile(db_path=db, log_path=log, dry_run=False)

        result = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
        hold_entry = next(e for e in result if e.get("action") == "HOLD")
        assert "outcome" not in hold_entry, (
            "HOLD-action entry received an outcome. HOLDs never produce is_close=1 trades.\n"
            "If reconciler logic changed, verify HOLD cannot accidentally match a BUY/SELL close."
        )

    def test_buy_entry_with_closed_trade_receives_outcome(self, tmp_path):
        """Sanity: BUY entry with matching closed trade must receive outcome."""
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"

        sid = "ts_MSFT_20260101T120000Z_c3d4_0001"
        self._make_db(db, sid)

        log.write_text(
            json.dumps({"signal_id": sid, "confidence": 0.80, "action": "BUY"}) + "\n",
            encoding="utf-8",
        )

        _, updated, _ = reconcile(db_path=db, log_path=log, dry_run=False)
        assert updated == 1, "BUY entry with matching closed trade must receive outcome"


# ===========================================================================
# 4. Contract audit script self-validation
# ===========================================================================

class TestPlattContractAuditScript:
    """platt_contract_audit.py must be importable and its current-code checks must pass."""

    def test_script_is_importable(self):
        import importlib
        mod = importlib.import_module("scripts.platt_contract_audit")
        assert hasattr(mod, "run_audit")
        assert hasattr(mod, "Finding")

    def test_classifier_check_passes_on_current_code(self):
        from scripts.platt_contract_audit import check_classifier_identity
        finding = check_classifier_identity()
        assert finding.status == "PASS", (
            f"Classifier identity check FAILED: {finding.detail}\n"
            "Implementation drifted from LogisticRegression or a forbidden classifier appeared."
        )

    @pytest.mark.parametrize("forbidden", ["IsotonicRegression", "CalibratedClassifierCV", "isotonic"])
    def test_classifier_check_blocks_forbidden_tokens_case_insensitive(self, monkeypatch, forbidden):
        import scripts.platt_contract_audit as mod

        fake_source = (
            "def _calibrate_confidence(self):\n"
            "    model = LogisticRegression()\n"
            f"    marker = '{forbidden}'\n"
            "    return model\n"
        )
        monkeypatch.setattr(mod.inspect, "getsource", lambda *_args, **_kwargs: fake_source)

        finding = mod.check_classifier_identity()
        assert finding.status == "FAIL"
        assert "Forbidden classifier token" in finding.detail

    def test_fallback_order_check_passes_on_current_code(self):
        from scripts.platt_contract_audit import check_fallback_chain_order
        finding = check_fallback_chain_order()
        assert finding.status == "PASS", (
            f"Fallback chain order check FAILED: {finding.detail}\n"
            "Tier order changed or marker strings renamed."
        )

    def test_run_audit_with_missing_paths_returns_findings_not_crash(self, tmp_path):
        from scripts.platt_contract_audit import run_audit
        findings = run_audit(
            db_path=tmp_path / "no_db.db",
            jsonl_path=tmp_path / "no.jsonl",
        )
        assert isinstance(findings, list)
        assert len(findings) >= 1
        assert all(f.status in {"PASS", "FAIL", "WARN", "SKIP"} for f in findings)

    def test_finding_as_dict_has_required_keys(self):
        from scripts.platt_contract_audit import Finding
        f = Finding("test_check", "PASS", "All good.")
        d = f.as_dict()
        assert set(d.keys()) == {"check", "status", "detail"}


# ===========================================================================
# 5. Bootstrap outcome guard exists in overnight refresh
# ===========================================================================

class TestBootstrapOutcomeGuard:
    """run_overnight_refresh.py must check ts_* close count post-bootstrap.

    Without this guard, a bootstrap that produces 0 closed trades (e.g., due to
    cycles-vs-bars mismatch where max_holding never fires at a fixed as-of-date)
    silently reports success. This test ensures the guard is present in code.
    """

    _REFRESH_PATH = Path("scripts/run_overnight_refresh.py")

    def _get_source(self) -> str:
        assert self._REFRESH_PATH.exists(), (
            "scripts/run_overnight_refresh.py not found -- cannot verify bootstrap guard."
        )
        return self._REFRESH_PATH.read_text(encoding="utf-8")

    def test_ts_close_count_check_exists(self):
        """Refresh script must contain a query or function that counts ts_* closed trades."""
        source = self._get_source()
        has_ts_query = (
            "ts_signal_id LIKE 'ts_%'" in source
            or "_count_ts_closes" in source
            or "ts_closes" in source
        )
        assert has_ts_query, (
            "Bootstrap outcome guard missing from run_overnight_refresh.py.\n"
            "The script must count ts_* is_close=1 trades before and after bootstrap.\n"
            "If delta == 0, increment errors -- silent success on zero output is not allowed."
        )

    def test_bootstrap_guard_increments_errors_on_failure(self):
        """Guard must use the errors counter so the summary reports failures correctly."""
        source = self._get_source()
        assert "errors +=" in source, (
            "run_overnight_refresh.py does not use 'errors +=' counter.\n"
            "Bootstrap guard must increment errors when delta ts_* closes == 0,\n"
            "so the final 'Errors: N' summary line reflects the bootstrap failure."
        )


class TestPlattJsonlPairCounting:
    """JSONL tier must count only actionable BUY/SELL outcomes with non-null outcome payload."""

    def test_active_tier_ignores_null_outcomes_and_hold_actions(self, tmp_path):
        from scripts.platt_contract_audit import check_calibration_active_tier

        log = tmp_path / "qv.jsonl"
        db = tmp_path / "missing.db"
        rows = []
        for i in range(20):
            rows.append({"signal_id": f"sig_null_{i}", "action": "BUY", "confidence": 0.7, "outcome": None})
        for i in range(20):
            rows.append(
                {
                    "signal_id": f"sig_hold_{i}",
                    "action": "HOLD",
                    "confidence": 0.7,
                    "outcome": {"win": bool(i % 2), "pnl_pct": 0.01},
                }
            )
        log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        finding = check_calibration_active_tier(db_path=db, jsonl_path=log)
        assert finding.status == "FAIL"
        assert "[NONE]" in finding.detail

    def test_hold_inflation_treats_outcome_null_as_pending(self, tmp_path):
        from scripts.platt_contract_audit import check_hold_inflation

        log = tmp_path / "qv.jsonl"
        rows = [{"signal_id": f"hold_{i}", "action": "HOLD", "confidence": 0.55, "outcome": None} for i in range(40)]
        rows.append({"signal_id": "buy_done", "action": "BUY", "confidence": 0.8, "outcome": {"win": True}})
        log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        finding = check_hold_inflation(log)
        assert finding.status == "WARN"
        assert "structurally unreconcilable" in finding.detail
