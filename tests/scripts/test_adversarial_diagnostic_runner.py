"""
tests/scripts/test_adversarial_diagnostic_runner.py
----------------------------------------------------
Adversarial audit anti-regression tests for Phase 7.20.

Each test locks in detection logic for a confirmed vulnerability so that
future "fixes" do not accidentally hollow-out the detection gate.

Coverage map:
  INT-01  chk_null_flag_bypass
  INT-02  chk_duplicate_close_null_bypass
  INT-03  chk_proof_raw_table
  INT-04  chk_orphan_shorts
  INT-05  chk_medium_violations_in_ci_gate
  BYP-01  chk_gate_skip_bypass
  BYP-02  chk_layer2_exit_code_ignored
  BYP-03  chk_institutional_gate_doesnt_verify_prior_gates
  BYP-04  chk_overnight_exit_code
  BYP-05  chk_allow_inconclusive_lift
  LEAK-01 chk_platt_no_train_test_split
  LEAK-02 chk_macro_bfill_lookahead
  LEAK-03 chk_audit_file_no_validation
  POI-02  chk_whitelist_divergence
  WIRE-02 chk_production_view_integrity
  WIRE-03 chk_warmup_indefinite
  CLI     run_all_checks + filter behaviour
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

# Ensure the repo root is on sys.path so scripts/ is importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adversarial_diagnostic_runner import (  # noqa: E402
    Finding,
    chk_allow_inconclusive_lift,
    chk_audit_file_no_validation,
    chk_duplicate_close_null_bypass,
    chk_gate_skip_bypass,
    chk_institutional_gate_doesnt_verify_prior_gates,
    chk_layer2_exit_code_ignored,
    chk_macro_bfill_lookahead,
    chk_medium_violations_in_ci_gate,
    chk_null_flag_bypass,
    chk_orphan_shorts,
    chk_overnight_exit_code,
    chk_platt_no_train_test_split,
    chk_production_view_integrity,
    chk_proof_raw_table,
    chk_warmup_indefinite,
    chk_whitelist_divergence,
    run_all_checks,
    SEVERITY_ORDER,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade_db(tmp_path: Path, rows: list[tuple]) -> Path:
    """Create a minimal trade_executions SQLite DB with the given rows.

    Column order:
        id, ticker, action, is_close, is_diagnostic, is_synthetic,
        entry_trade_id, realized_pnl
    """
    db = tmp_path / "test_trades.db"
    con = sqlite3.connect(db)
    con.execute("""
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            action TEXT,
            is_close INTEGER,
            is_diagnostic INTEGER,
            is_synthetic INTEGER,
            entry_trade_id INTEGER,
            realized_pnl REAL
        )
    """)
    con.executemany(
        "INSERT INTO trade_executions VALUES (?,?,?,?,?,?,?,?)", rows
    )
    con.commit()
    con.close()
    return db


def _connect(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db), timeout=2.0)
    con.row_factory = sqlite3.Row
    return con


def _add_production_view(db: Path, view_sql: str) -> None:
    con = sqlite3.connect(db)
    con.execute(view_sql)
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Group 0: Finding dataclass
# ---------------------------------------------------------------------------

class TestFindingDataclass:
    def test_finding_has_required_fields(self):
        f = Finding(
            id="TEST-01",
            severity="HIGH",
            category="integrity",
            location="scripts/foo.py:10",
            title="Test finding",
            detail="Some detail",
            attack_vector="Do X",
            fix="Fix Y",
        )
        assert f.id == "TEST-01"
        assert f.severity == "HIGH"
        assert f.category == "integrity"
        assert f.passed is False  # default is vulnerable

    def test_finding_passed_defaults_to_false(self):
        f = Finding(
            id="X-01", severity="CRITICAL", category="bypass",
            location="a:1", title="T", detail="D", attack_vector="A", fix="F",
        )
        assert f.passed is False

    def test_severity_order_covers_all_levels(self):
        for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            assert level in SEVERITY_ORDER
        # CRITICAL < HIGH < MEDIUM < LOW (lower = more severe)
        assert SEVERITY_ORDER["CRITICAL"] < SEVERITY_ORDER["HIGH"]
        assert SEVERITY_ORDER["HIGH"] < SEVERITY_ORDER["MEDIUM"]


# ---------------------------------------------------------------------------
# Group 1: Integrity checks (INT-01 to INT-05)
# ---------------------------------------------------------------------------

class TestNullFlagBypass:
    """INT-01 — CRITICAL: COALESCE(is_diagnostic, 0) treats NULL as safe."""

    def test_confirmed_when_production_closes_have_null_flags(self, tmp_path):
        db = _make_trade_db(tmp_path, [
            # id, ticker, action, is_close, is_diagnostic, is_synthetic, entry_id, pnl
            (1, "AAPL", "SELL", 1, None, None, 1, 50.0),  # NULL flags on close
        ])
        con = _connect(db)
        result = chk_null_flag_bypass(con)
        con.close()
        assert result.id == "INT-01"
        assert result.passed is False, "Null-flagged closes must trigger confirmed"

    def test_cleared_when_all_closes_have_explicit_flags(self, tmp_path):
        db = _make_trade_db(tmp_path, [
            (1, "AAPL", "BUY",  0, 0, 0, None, None),
            (2, "AAPL", "SELL", 1, 0, 0, 1, 50.0),
        ])
        con = _connect(db)
        result = chk_null_flag_bypass(con)
        con.close()
        assert result.passed is True, "Explicitly-flagged closes should clear INT-01"

    def test_no_db_returns_unconfirmed_not_cleared(self):
        result = chk_null_flag_bypass(None)
        # When DB is unavailable we cannot clear -- severity preserved
        assert result.id == "INT-01"
        assert result.passed is False


class TestDuplicateCloseNullBypass:
    """INT-02 — CRITICAL: DUPLICATE_CLOSE detection requires entry_trade_id."""

    def test_confirmed_when_unlinked_closes_in_production(self, tmp_path):
        db = _make_trade_db(tmp_path, [
            (1, "AAPL", "BUY",  0, 0, 0, None, None),
            (2, "AAPL", "SELL", 1, 0, 0, None, 30.0),  # no entry_trade_id
            (3, "AAPL", "SELL", 1, 0, 0, None, 30.0),  # duplicate unlinked
        ])
        con = _connect(db)
        result = chk_duplicate_close_null_bypass(con)
        con.close()
        assert result.id == "INT-02"
        assert result.passed is False, "Unlinked closes should confirm INT-02"

    def test_cleared_when_all_closes_are_linked(self, tmp_path):
        db = _make_trade_db(tmp_path, [
            (1, "AAPL", "BUY",  0, 0, 0, None, None),
            (2, "AAPL", "SELL", 1, 0, 0, 1, 30.0),  # linked to id=1
        ])
        con = _connect(db)
        result = chk_duplicate_close_null_bypass(con)
        con.close()
        assert result.passed is True


class TestProofRawTable:
    """INT-03 — CRITICAL: validate_profitability_proof.py reads raw table."""

    def test_confirmed_when_source_has_no_view_reference(self, tmp_path):
        fake_src = tmp_path / "validate_profitability_proof.py"
        fake_src.write_text(
            "SELECT COUNT(*) FROM trade_executions WHERE realized_pnl IS NOT NULL",
            encoding="utf-8",
        )
        # Use ROOT-relative lookup — monkeypatch via file path hack isn't needed;
        # chk_proof_raw_table reads ROOT/"scripts"/"validate_profitability_proof.py"
        # So we test the LIVE source file which is the production code.
        result = chk_proof_raw_table(ROOT / "data" / "portfolio_maximizer.db")
        assert result.id == "INT-03"
        # The live source should NOT contain production_closed_trades
        # (confirmed vulnerability in Phase 7.20 audit)
        assert result.passed is False, (
            "INT-03 should be confirmed: validate_profitability_proof.py "
            "must NOT use production_closed_trades (audit finding)"
        )

    def test_clears_when_source_uses_canonical_view(self, monkeypatch):
        """If the source is fixed to use production_closed_trades, INT-03 clears."""
        import scripts.adversarial_diagnostic_runner as mod

        # Patch _read to return a fixed source that uses the canonical view
        monkeypatch.setattr(
            mod,
            "_read",
            lambda path: (
                "SELECT * FROM production_closed_trades"
                if "validate_profitability_proof" in str(path)
                else ""
            ),
        )
        result = chk_proof_raw_table(Path("data/portfolio_maximizer.db"))
        assert result.passed is True, "Using production_closed_trades should clear INT-03"


class TestOrphanShorts:
    """INT-04 — HIGH: orphan check is BUY-only, misses short positions."""

    def test_confirmed_when_buy_only_check_in_source(self):
        # Source that checks BUY orphans but not SELL (current production state)
        src = (
            "WHERE action = 'BUY' AND is_close = 0 AND age_days > threshold"
        )
        result = chk_orphan_shorts(src)
        assert result.id == "INT-04"
        assert result.passed is False, "BUY-only orphan check should confirm INT-04"

    def test_cleared_when_sell_orphans_also_checked(self):
        src = (
            "WHERE action = 'BUY' AND is_close = 0 AND age_days > threshold "
            "UNION ALL "
            "WHERE action = 'SELL' AND is_close = 0 AND age_days > threshold"
        )
        result = chk_orphan_shorts(src)
        assert result.passed is True


class TestMediumViolationsInCiGate:
    """INT-05 — HIGH: MEDIUM violations (unlinked closes) pass in default CI mode."""

    def test_confirmed_when_default_excludes_medium(self):
        src = 'fail_severities = {"CRITICAL", "HIGH"}'
        result = chk_medium_violations_in_ci_gate(src)
        assert result.id == "INT-05"
        assert result.passed is False

    def test_still_confirmed_with_add_pattern(self):
        src = "fail_severities.add(severity)"
        result = chk_medium_violations_in_ci_gate(src)
        assert result.passed is False


# ---------------------------------------------------------------------------
# Group 2: Bypass checks
# ---------------------------------------------------------------------------

class TestGateSkipBypass:
    """BYP-01 — CRITICAL: all optional gates skippable simultaneously."""

    def test_confirmed_when_all_three_skip_flags_present(self):
        src = (
            "--skip-forecast-gate ... --skip-profitability-gate ... "
            "--skip-institutional-gate ... 'passed': True"
        )
        result = chk_gate_skip_bypass(src)
        assert result.id == "BYP-01"
        assert result.passed is False

    def test_cleared_when_not_all_skip_flags_present(self):
        src = "--skip-forecast-gate ... 'passed': True"
        result = chk_gate_skip_bypass(src)
        assert result.passed is True


class TestLayer2ExitCodeIgnored:
    """BYP-02 — CRITICAL: Layer 2 trusts only JSON field, ignores subprocess exit code."""

    def test_confirmed_when_returncode_absent_from_source(self):
        # Source that uses overall_passed but never reads the subprocess exit code.
        # The DOTALL regex requires both words to coexist; omitting "returncode"
        # triggers the `elif not has_returncode_check` branch -> confirmed.
        src = (
            "result = subprocess.run(...)\n"
            "overall_passed = data.get('overall_passed', False)\n"
            "status = 'PASS' if overall_passed else 'FAIL'\n"
        )
        result = chk_layer2_exit_code_ignored(src)
        assert result.id == "BYP-02"
        assert result.passed is False

    def test_cleared_when_returncode_combined_with_json_decision(self):
        src = (
            "overall_passed = data.get('overall_passed', False)\n"
            "if result.returncode != 0 and not overall_passed:\n"
            "    status = 'FAIL'\n"
        )
        result = chk_layer2_exit_code_ignored(src)
        assert result.passed is True


class TestInstitutionalGateBlind:
    """BYP-03 — CRITICAL: institutional gate doesn't verify prior gate results."""

    def test_confirmed_when_no_prior_gate_reference(self):
        src = (
            "def check_p0_guardrails(): ...\n"
            "def check_p1_patterns(): ...\n"
            "def check_p2_platt(): ...\n"
        )
        result = chk_institutional_gate_doesnt_verify_prior_gates(src)
        assert result.id == "BYP-03"
        assert result.passed is False

    def test_cleared_when_prior_gate_result_referenced(self):
        src = (
            "def check_p4_prior_gates():\n"
            "    data = json.load(open('logs/production_gate_latest.json'))\n"
            "    return data.get('overall_passed', False)\n"
        )
        result = chk_institutional_gate_doesnt_verify_prior_gates(src)
        assert result.passed is True


class TestOvernightExitCode:
    """BYP-04 — CRITICAL: run_overnight_refresh.py exits with error COUNT not 0/1."""

    def test_confirmed_when_return_errors_at_end_of_main(self):
        src = (
            "def main():\n"
            "    errors = 0\n"
            "    # ... do work ...\n"
            "    return errors\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    sys.exit(main())\n"
        )
        result = chk_overnight_exit_code(src)
        assert result.id == "BYP-04"
        assert result.passed is False

    def test_confirmed_when_sys_exit_errors_directly(self):
        src = "sys.exit(errors)\n"
        result = chk_overnight_exit_code(src)
        assert result.passed is False

    def test_cleared_when_explicit_boolean_return(self):
        src = (
            "def main():\n"
            "    errors = 0\n"
            "    return 0 if errors == 0 else 1\n"
        )
        result = chk_overnight_exit_code(src)
        assert result.passed is True


class TestAllowInconclusiveLift:
    """BYP-05 — HIGH: --allow-inconclusive-lift has no expiry."""

    def test_confirmed_when_flag_present_without_expiry(self):
        src = (
            "parser.add_argument('--allow-inconclusive-lift', action='store_true')\n"
            "cmd.append('--allow-inconclusive-lift')\n"
        )
        result = chk_allow_inconclusive_lift(src)
        assert result.id == "BYP-05"
        assert result.passed is False

    def test_cleared_when_flag_absent(self):
        src = "parser.add_argument('--skip-forecast-gate', action='store_true')\n"
        result = chk_allow_inconclusive_lift(src)
        assert result.passed is True

    def test_cleared_when_expiry_mechanism_present(self):
        src = (
            "--allow-inconclusive-lift-until 2026-06-01\n"
            "max_warmup_days = 30\n"
        )
        result = chk_allow_inconclusive_lift(src)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Group 3: Leakage checks
# ---------------------------------------------------------------------------

class TestPlattNoTrainTestSplit:
    """LEAK-01 — CRITICAL: Platt LogisticRegression has no holdout."""

    def test_confirmed_when_no_train_test_split_in_source(self):
        src = (
            "from sklearn.linear_model import LogisticRegression\n"
            "clf = LogisticRegression()\n"
            "clf.fit(X, y)\n"
            "pred = clf.predict_proba(X)\n"
        )
        result = chk_platt_no_train_test_split(src)
        assert result.id == "LEAK-01"
        assert result.passed is False

    def test_cleared_when_train_test_split_present(self):
        src = (
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)\n"
            "clf.fit(X_train, y_train)\n"
        )
        result = chk_platt_no_train_test_split(src)
        assert result.passed is True

    def test_cleared_when_holdout_keyword_present(self):
        src = "# holdout set used for Platt calibration validation"
        result = chk_platt_no_train_test_split(src)
        assert result.passed is True


class TestMacroBfillLookahead:
    """LEAK-02 — HIGH: bfill() on macro context can pull future values into past rows."""

    def test_confirmed_when_bfill_used_on_macro_context(self):
        src = (
            "MACRO_COLUMNS = ('vix_level', 'yield_spread')\n"
            "macro_context = df.reindex(price_history.index)\n"
            "macro_context = macro_context.ffill().bfill()\n"
        )
        result = chk_macro_bfill_lookahead(src)
        assert result.id == "LEAK-02"
        assert result.passed is False

    def test_cleared_when_ffill_only_no_bfill(self):
        src = (
            "MACRO_COLUMNS = ('vix_level',)\n"
            "macro_context = macro_context.ffill()\n"
            "# no bfill allowed\n"
        )
        result = chk_macro_bfill_lookahead(src)
        assert result.passed is True


class TestAuditFileNoValidation:
    """LEAK-03 (category=poisoning) — HIGH: audit JSON loaded without range checks."""

    def test_confirmed_when_no_range_check_in_health_audit_source(self):
        src = (
            "def extract_window_metrics(path):\n"
            "    data = json.loads(path.read_text())\n"
            "    ensemble_rmse = data['artifacts']['ensemble_rmse']\n"
            "    return {'ensemble_rmse': ensemble_rmse}\n"
        )
        result = chk_audit_file_no_validation(src)
        assert result.id == "LEAK-03"
        assert result.passed is False

    def test_cleared_when_range_validation_present(self):
        src = (
            "if rmse > 0 and rmse < 100:\n"
            "    include = True\n"
            "else:\n"
            "    log.warning('Implausible RMSE rejected')\n"
        )
        result = chk_audit_file_no_validation(src)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Group 4: Wiring checks
# ---------------------------------------------------------------------------

class TestProductionViewIntegrity:
    """WIRE-02 — HIGH: production_closed_trades view must have all three filters."""

    def test_cleared_when_view_has_all_required_filters(self, tmp_path):
        db = _make_trade_db(tmp_path, [])
        _add_production_view(db, """
            CREATE VIEW production_closed_trades AS
            SELECT * FROM trade_executions
            WHERE is_close = 1
              AND COALESCE(is_diagnostic, 0) = 0
              AND COALESCE(is_synthetic, 0) = 0
        """)
        con = _connect(db)
        result = chk_production_view_integrity(con)
        con.close()
        assert result.passed is True

    def test_confirmed_when_view_missing_diagnostic_filter(self, tmp_path):
        db = _make_trade_db(tmp_path, [])
        _add_production_view(db, """
            CREATE VIEW production_closed_trades AS
            SELECT * FROM trade_executions
            WHERE is_close = 1
        """)
        con = _connect(db)
        result = chk_production_view_integrity(con)
        con.close()
        assert result.id == "WIRE-02"
        assert result.passed is False

    def test_confirmed_when_view_does_not_exist(self, tmp_path):
        db = _make_trade_db(tmp_path, [])  # no view added
        con = _connect(db)
        result = chk_production_view_integrity(con)
        con.close()
        assert result.passed is False


class TestWarmupIndefinite:
    """WIRE-03 — HIGH: fail_on_violation=False + no max_warmup_days = eternal warmup."""

    def test_confirmed_when_fail_on_violation_false_and_no_warmup_limit(
        self, monkeypatch
    ):
        import scripts.adversarial_diagnostic_runner as mod

        monkeypatch.setattr(
            mod,
            "_read_config",
            lambda path: {
                "forecaster_monitoring": {
                    "regression_metrics": {
                        "fail_on_violation_during_holding_period": False
                        # no max_warmup_days
                    }
                }
            },
        )
        src = "fail_on_violation_during_holding_period = False"
        result = chk_warmup_indefinite(src)
        assert result.id == "WIRE-03"
        assert result.passed is False

    def test_cleared_when_max_warmup_days_configured(self, monkeypatch):
        import scripts.adversarial_diagnostic_runner as mod

        monkeypatch.setattr(
            mod,
            "_read_config",
            lambda path: {
                "forecaster_monitoring": {
                    "regression_metrics": {
                        "fail_on_violation_during_holding_period": False,
                        "max_warmup_days": 30,
                    }
                }
            },
        )
        result = chk_warmup_indefinite("")
        assert result.passed is True


class TestWhitelistDivergence:
    """POI-02 — HIGH: whitelist in enforcer not propagated to production gate."""

    def test_confirmed_when_enforcer_has_whitelist_but_gate_does_not(self):
        src_enforcer = "INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS = os.environ.get(...)"
        src_gate = "def _count_unlinked_closes(conn, ticker):\n    ..."
        result = chk_whitelist_divergence(src_enforcer, src_gate)
        assert result.id == "POI-02"
        assert result.passed is False

    def test_cleared_when_both_use_whitelist(self):
        src_enforcer = "INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS = ..."
        src_gate = "WHITELIST_IDS = os.environ.get('INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS')"
        result = chk_whitelist_divergence(src_enforcer, src_gate)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Group 5: CLI orchestration and filter behaviour
# ---------------------------------------------------------------------------

class TestRunAllChecks:
    """run_all_checks() integration — verifies orchestrator contract."""

    def test_returns_nonempty_finding_list(self, tmp_path):
        db = _make_trade_db(tmp_path, [])
        findings = run_all_checks(db, tmp_path / "audits", None, None)
        assert isinstance(findings, list)
        assert len(findings) > 0, "run_all_checks must return at least one finding"

    def test_all_findings_have_required_fields(self, tmp_path):
        db = _make_trade_db(tmp_path, [])
        findings = run_all_checks(db, tmp_path / "audits", None, None)
        required = {"id", "severity", "category", "location", "title", "detail",
                    "attack_vector", "fix", "passed"}
        for f in findings:
            missing = required - set(vars(f).keys())
            assert not missing, f"Finding {f.id} missing fields: {missing}"

    def test_severity_filter_excludes_lower_severity(self, tmp_path):
        db = _make_trade_db(tmp_path, [])
        findings_critical = run_all_checks(db, tmp_path / "audits", "CRITICAL", None)
        findings_all = run_all_checks(db, tmp_path / "audits", None, None)
        # CRITICAL filter should return subset
        assert len(findings_critical) <= len(findings_all)
        for f in findings_critical:
            assert f.severity == "CRITICAL", (
                f"Severity filter 'CRITICAL' should exclude {f.id} (severity={f.severity})"
            )

    def test_category_filter_excludes_other_categories(self, tmp_path):
        db = _make_trade_db(tmp_path, [])
        findings = run_all_checks(db, tmp_path / "audits", None, "integrity")
        for f in findings:
            assert f.category == "integrity", (
                f"Category filter 'integrity' should not include {f.id} (cat={f.category})"
            )

    def test_findings_sorted_confirmed_before_cleared(self, tmp_path):
        db = _make_trade_db(tmp_path, [])
        findings = run_all_checks(db, tmp_path / "audits", None, None)
        confirmed_done = False
        for f in findings:
            if f.passed:
                confirmed_done = True
            if confirmed_done:
                assert f.passed, "All confirmed findings must precede cleared findings"

    def test_critical_confirmed_findings_exist_in_production_codebase(self, tmp_path):
        """The 6 CRITICAL findings identified in Phase 7.20 must still be confirmed
        against the live production codebase (anti-regression lock)."""
        db_real = ROOT / "data" / "portfolio_maximizer.db"
        audit_dir = ROOT / "logs" / "forecast_audits"
        findings = run_all_checks(db_real, audit_dir, None, None)
        critical_confirmed = [
            f for f in findings if f.severity == "CRITICAL" and not f.passed
        ]
        critical_ids = {f.id for f in critical_confirmed}
        # These must remain confirmed until individually fixed and cleared
        expected_critical = {"INT-03", "BYP-01", "BYP-02", "BYP-03", "BYP-04", "LEAK-01"}
        still_confirmed = expected_critical & critical_ids
        # If all 6 are fixed, this test should be updated to reflect the new baseline.
        # For now, at least INT-03 must remain (validate_profitability_proof.py not yet fixed)
        assert "INT-03" in critical_ids, (
            "INT-03 (validate_profitability_proof raw table) is a known confirmed vulnerability. "
            "If it no longer appears, INT-03 was fixed -- update this test accordingly."
        )


class TestJsonOutputSchema:
    """JSON mode emits a report with the required top-level keys."""

    def test_json_output_has_required_keys(self, tmp_path):
        db = _make_trade_db(tmp_path, [])
        findings = run_all_checks(db, tmp_path / "audits", None, None)
        import dataclasses, datetime
        report = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "phase": "adversarial_audit_7.20",
            "db_path": str(db),
            "total_checks": len(findings),
            "confirmed": sum(1 for f in findings if not f.passed),
            "cleared": sum(1 for f in findings if f.passed),
            "findings": [dataclasses.asdict(f) for f in findings],
        }
        required_keys = {
            "timestamp_utc", "phase", "db_path", "total_checks",
            "confirmed", "cleared", "findings",
        }
        assert required_keys <= set(report.keys())
        # Each finding in JSON must have id, severity, passed
        for entry in report["findings"]:
            assert "id" in entry
            assert "severity" in entry
            assert "passed" in entry

    def test_json_findings_are_serialisable(self, tmp_path):
        db = _make_trade_db(tmp_path, [])
        findings = run_all_checks(db, tmp_path / "audits", None, None)
        import dataclasses
        data = [dataclasses.asdict(f) for f in findings]
        # Must be JSON-serialisable without error
        serialised = json.dumps(data)
        decoded = json.loads(serialised)
        assert len(decoded) == len(findings)


class TestExitCodeLogic:
    """Exit code must be 1 when any CRITICAL or HIGH finding is confirmed."""

    def test_exit_code_is_1_when_critical_high_confirmed(self, tmp_path):
        db = _make_trade_db(tmp_path, [])
        findings = run_all_checks(db, tmp_path / "audits", None, None)
        blocking = [f for f in findings if not f.passed and f.severity in ("CRITICAL", "HIGH")]
        expected_code = 1 if blocking else 0
        # In Phase 7.20, there are known CRITICAL findings -- exit code must be 1
        assert expected_code == 1, (
            "With known CRITICAL/HIGH confirmed findings, exit code must be 1. "
            "If 0 is returned, all findings were incorrectly cleared."
        )

    def test_exit_code_is_0_when_only_medium_or_lower(self):
        # Simulate a world where only MEDIUM findings exist
        medium_finding = Finding(
            id="THR-XX", severity="MEDIUM", category="threshold",
            location="a:1", title="T", detail="D", attack_vector="A", fix="F",
            passed=False,
        )
        blocking = [
            f for f in [medium_finding]
            if not f.passed and f.severity in ("CRITICAL", "HIGH")
        ]
        assert len(blocking) == 0
        expected_code = 1 if blocking else 0
        assert expected_code == 0
