"""
Unit tests for scripts/check_model_improvement.py.

All external dependencies (subprocess, imports, DB) are monkeypatched so these
tests run without any live data, database, or heavy computation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure scripts/ is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from check_model_improvement import (
    LAYER_REQUIRED_KEYS,
    LayerResult,
    compare_baseline,
    run_layer1_forecast_quality,
    run_layer2_gate_status,
    run_layer3_trade_quality,
    run_layer4_calibration,
    save_baseline,
)

# ---------------------------------------------------------------------------
# Fixtures dir (uses real fixture files for Layer 1 smoke tests)
# ---------------------------------------------------------------------------
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "forecast_audits"


# ---------------------------------------------------------------------------
# Helper to build a fake subprocess result
# ---------------------------------------------------------------------------
def _fake_proc(stdout: str, returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


# ---------------------------------------------------------------------------
# Layer 1 tests
# ---------------------------------------------------------------------------
class TestLayer1ForecastQuality:
    def test_returns_skip_when_audit_dir_empty(self, tmp_path):
        result = run_layer1_forecast_quality(tmp_path / "nonexistent")
        assert result.layer == 1
        assert result.status == "SKIP"
        assert result.metrics.get("n_total_files") == 0 or result.metrics.get("n_total_files") is None

    def test_returns_skip_when_no_json_files_in_dir(self, tmp_path):
        (tmp_path / "some_other_file.txt").write_text("hello")
        result = run_layer1_forecast_quality(tmp_path)
        assert result.status == "SKIP"

    def test_computes_lift_and_data_quality_metrics_from_fixtures(self):
        """Layer 1 with the 3 committed fixture files: healthy, samossa_da_zero, ensemble_lift."""
        if not FIXTURES_DIR.exists():
            pytest.skip("Fixture directory not found")
        result = run_layer1_forecast_quality(FIXTURES_DIR)
        # 3 files, all valid — no malformed, no missing
        assert result.metrics["n_total_files"] == 3
        assert result.metrics["n_used_windows"] == 3
        assert result.metrics["n_skipped_malformed"] == 0
        assert result.metrics["n_skipped_missing_metrics"] == 0
        # 2 of 3 fixtures have lift (healthy: 1.30<1.40, ensemble_lift: 1.60<1.85)
        # samossa_da_zero: ensemble 1.25 > best_single 1.10 -> no lift
        assert result.metrics["lift_fraction_global"] == pytest.approx(2 / 3, abs=0.02)
        # Only 3 windows < warn_coverage_threshold=50 -> WARN
        assert result.status == "WARN"

    def test_warns_when_samossa_da_zero_pct_exceeds_threshold(self, tmp_path):
        """All 3 fixture files replaced with copies having SAMOSSA DA=0.0."""
        # Write 3 audit files all with SAMOSSA DA=0
        for i in range(3):
            audit = {
                "dataset": {"ticker": "X", "start": f"2025-0{i+1}-01", "end": f"2025-0{i+1}-30", "length": 100},
                "summary": {"forecast_horizon": 5},
                "artifacts": {
                    "evaluation_metrics": {
                        "garch":    {"rmse": 1.5, "directional_accuracy": 0.5, "smape": 3.0},
                        "samossa":  {"rmse": 1.2, "directional_accuracy": 0.0, "smape": 2.5},
                        "mssa_rl":  {"rmse": 1.8, "directional_accuracy": 0.6, "smape": 4.0},
                        "ensemble": {"rmse": 1.3, "directional_accuracy": 0.3},
                    },
                    "ensemble_weights": {"garch": 0.3, "samossa": 0.4, "mssa_rl": 0.3},
                },
            }
            (tmp_path / f"forecast_audit_test_{i}.json").write_text(json.dumps(audit))
        result = run_layer1_forecast_quality(tmp_path, warn_da_zero_pct=0.40)
        # samossa_da_zero_pct = 1.0 (all 3 windows have SAMOSSA DA=0)
        assert result.metrics["samossa_da_zero_pct"] == pytest.approx(1.0)
        assert result.status == "WARN"

    def test_warns_when_n_used_windows_below_coverage_threshold(self):
        """3 fixture windows < default warn_coverage_threshold=50."""
        if not FIXTURES_DIR.exists():
            pytest.skip("Fixture directory not found")
        result = run_layer1_forecast_quality(FIXTURES_DIR, warn_coverage_threshold=50)
        assert result.metrics["n_used_windows"] == 3
        assert result.status == "WARN"
        assert "coverage_threshold" in result.summary or "n_used" in result.summary


# ---------------------------------------------------------------------------
# Layer 2 tests
# ---------------------------------------------------------------------------
class TestLayer2GateStatus:
    def test_pass_when_overall_passed_true(self, monkeypatch):
        fake_json = json.dumps({
            "overall_passed": True,
            "gates": [
                {"label": "integrity", "passed": True},
                {"label": "quant_health", "passed": True},
                {"label": "audit_lift", "passed": True},
                {"label": "institutional", "passed": True},
            ],
        })
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: _fake_proc(fake_json, 0))
        result = run_layer2_gate_status()
        assert result.status == "PASS"
        assert result.metrics["overall_passed"] is True
        assert result.metrics["n_gates_passed"] == 4
        assert result.metrics["n_gates_failed"] == 0

    def test_fail_when_overall_passed_false(self, monkeypatch):
        """Gate outcome must never be softened — FAIL is FAIL."""
        fake_json = json.dumps({
            "overall_passed": False,
            "gates": [
                {"label": "audit_lift", "passed": False},
                {"label": "integrity", "passed": True},
            ],
        })
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: _fake_proc(fake_json, 1))
        result = run_layer2_gate_status()
        assert result.status == "FAIL"
        assert result.metrics["overall_passed"] is False
        assert result.metrics["n_gates_failed"] == 1

    def test_skip_when_script_not_found(self, tmp_path):
        result = run_layer2_gate_status(root=tmp_path)
        assert result.status == "SKIP"
        assert "not found" in result.summary.lower() or "skip" in result.summary.lower()

    def test_fail_on_timeout(self, monkeypatch):
        import subprocess
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("x", 180)),
        )
        result = run_layer2_gate_status()
        assert result.status == "FAIL"
        assert "timeout" in result.summary.lower() or result.error is not None


# ---------------------------------------------------------------------------
# Layer 3 tests
# ---------------------------------------------------------------------------
class TestLayer3TradeQuality:
    def test_skip_when_db_not_found(self, tmp_path):
        result = run_layer3_trade_quality(tmp_path / "nonexistent.db")
        assert result.status == "SKIP"
        assert "not found" in result.summary.lower()

    def test_pass_with_healthy_metrics(self, tmp_path, monkeypatch):
        # Create an empty placeholder DB so db_path.exists() returns True
        fake_db = tmp_path / "test.db"
        fake_db.touch()

        # Mock PnLIntegrityEnforcer
        mock_canonical = MagicMock()
        mock_canonical.total_round_trips = 50
        mock_canonical.win_rate = 0.60
        mock_canonical.profit_factor = 2.0
        mock_canonical.total_realized_pnl = 1500.0
        mock_canonical.avg_win = 80.0
        mock_canonical.avg_loss = -40.0

        mock_enforcer = MagicMock()
        mock_enforcer.get_canonical_metrics.return_value = mock_canonical
        mock_enforcer.conn = MagicMock()

        mock_trades_df = MagicMock()
        mock_trades_df.__len__ = lambda _: 50

        mock_gap = {
            "total_trades": 50,
            "overall_win_rate": 0.60,
            "stop_loss_pct": 0.20,
            "time_exit_pct": 0.30,
            "signal_exit_pct": 0.50,
            "interpretation": "mix",
        }

        with patch("check_model_improvement.PnLIntegrityEnforcer", return_value=mock_enforcer), \
             patch("check_model_improvement.load_production_trades", return_value=mock_trades_df), \
             patch("check_model_improvement.diagnose_direction_gap", return_value=mock_gap):
            result = run_layer3_trade_quality(fake_db)

        assert result.status == "PASS"
        assert result.metrics["win_rate"] == pytest.approx(0.60)
        assert result.metrics["profit_factor"] == pytest.approx(2.0)
        assert result.metrics["n_trades"] == 50
        assert result.metrics["interpretation"] == "mix"

    def test_warn_when_win_rate_below_threshold(self, tmp_path, monkeypatch):
        fake_db = tmp_path / "test.db"
        fake_db.touch()

        mock_canonical = MagicMock()
        mock_canonical.total_round_trips = 30
        mock_canonical.win_rate = 0.38        # below warn threshold 0.45
        mock_canonical.profit_factor = 1.50
        mock_canonical.total_realized_pnl = 200.0
        mock_canonical.avg_win = 80.0
        mock_canonical.avg_loss = -40.0

        mock_enforcer = MagicMock()
        mock_enforcer.get_canonical_metrics.return_value = mock_canonical
        mock_enforcer.conn = MagicMock()

        mock_gap = {
            "total_trades": 30,
            "overall_win_rate": 0.38,
            "stop_loss_pct": 0.30,
            "time_exit_pct": 0.20,
            "signal_exit_pct": 0.50,
            "interpretation": "mix",
        }

        with patch("check_model_improvement.PnLIntegrityEnforcer", return_value=mock_enforcer), \
             patch("check_model_improvement.load_production_trades", return_value=MagicMock()), \
             patch("check_model_improvement.diagnose_direction_gap", return_value=mock_gap):
            result = run_layer3_trade_quality(fake_db, win_rate_warn=0.45)

        assert result.status == "WARN"
        assert "win_rate" in result.summary


# ---------------------------------------------------------------------------
# Layer 4 tests
# ---------------------------------------------------------------------------
class TestLayer4Calibration:
    def test_skip_when_both_missing(self, tmp_path):
        result = run_layer4_calibration(
            tmp_path / "nonexistent.db",
            tmp_path / "nonexistent.jsonl",
        )
        assert result.status == "SKIP"

    def test_pass_from_platt_audit(self, tmp_path, monkeypatch):
        """run_audit returns all PASS findings."""
        fake_db = tmp_path / "test.db"
        fake_db.touch()

        class FakeFinding:
            def __init__(self, check, status, detail):
                self.check = check
                self.status = status
                self.detail = detail

        fake_findings = [
            FakeFinding("classifier_identity", "PASS", "LogisticRegression confirmed."),
            FakeFinding("fallback_chain_order", "PASS", "Order confirmed."),
            FakeFinding("hold_inflation", "PASS", "HOLD 5.0% of pending."),
            FakeFinding("ts_closes_in_db", "PASS", "42 ts_* closed trades."),
            FakeFinding("calibration_active_tier", "PASS",
                        "[TIER_3_DB_GLOBAL] Active: 42 DB pairs."),
            FakeFinding("calibration_quality", "PASS",
                        "n=42, ECE=0.0321 (threshold 0.15), Brier=0.2100 (no-skill=0.25)."),
        ]

        with patch("check_model_improvement.run_audit", return_value=fake_findings):
            result = run_layer4_calibration(fake_db, tmp_path / "qv.jsonl")

        assert result.status == "PASS"
        assert result.metrics["calibration_active_tier"] == "db_local"
        assert result.metrics["brier_score"] == pytest.approx(0.2100, abs=0.001)
        assert result.metrics["ece"] == pytest.approx(0.0321, abs=0.001)

    def test_fail_when_inactive_tier(self, tmp_path, monkeypatch):
        fake_db = tmp_path / "test.db"
        fake_db.touch()

        class FakeFinding:
            def __init__(self, check, status, detail):
                self.check = check; self.status = status; self.detail = detail

        fake_findings = [
            FakeFinding("calibration_active_tier", "FAIL",
                        "[NONE] No active tier. JSONL=0, DB=5."),
            FakeFinding("calibration_quality", "SKIP", "Insufficient pairs."),
        ]

        with patch("check_model_improvement.run_audit", return_value=fake_findings):
            result = run_layer4_calibration(fake_db, tmp_path / "qv.jsonl")

        assert result.status == "FAIL"
        assert result.metrics["calibration_active_tier"] == "inactive"


# ---------------------------------------------------------------------------
# Baseline save/compare
# ---------------------------------------------------------------------------
class TestBaselineSaveCompare:
    def test_save_baseline_writes_json(self, tmp_path):
        results = [
            LayerResult(1, "Forecast Quality", "WARN",
                        {"lift_fraction_global": 0.03, "lift_fraction_recent": 0.05,
                         "samossa_da_zero_pct": 0.55, "n_used_windows": 3,
                         "n_skipped_malformed": 0, "n_skipped_missing_metrics": 0,
                         "n_total_files": 3},
                        "WARN | summary"),
        ]
        out = tmp_path / "baseline.json"
        save_baseline(results, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "timestamp_utc" in data
        assert data["results"][0]["layer"] == 1

    def test_compare_baseline_shows_delta(self, tmp_path):
        baseline_results = [
            LayerResult(1, "Forecast Quality", "WARN",
                        {"lift_fraction_global": 0.03, "lift_fraction_recent": 0.04,
                         "samossa_da_zero_pct": 0.55, "n_used_windows": 10,
                         "n_skipped_malformed": 0, "n_skipped_missing_metrics": 0,
                         "n_total_files": 10},
                        "WARN"),
        ]
        baseline_path = tmp_path / "baseline.json"
        save_baseline(baseline_results, baseline_path)

        current_results = [
            LayerResult(1, "Forecast Quality", "PASS",
                        {"lift_fraction_global": 0.12, "lift_fraction_recent": 0.10,
                         "samossa_da_zero_pct": 0.20, "n_used_windows": 15,
                         "n_skipped_malformed": 0, "n_skipped_missing_metrics": 0,
                         "n_total_files": 15},
                        "PASS"),
        ]
        comparison = compare_baseline(current_results, baseline_path)
        assert "Forecast Quality" in comparison
        lift_delta = comparison["Forecast Quality"]["lift_fraction_global"]["delta"]
        assert lift_delta == pytest.approx(0.12 - 0.03, abs=1e-5)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------
class TestCLIExitCodes:
    def test_cli_exit_0_when_all_layers_pass_or_warn_or_skip(self, tmp_path, monkeypatch):
        """All layers SKIP/WARN (no data) → exit 0."""
        from check_model_improvement import main

        # Layer 2 uses subprocess to call run_all_gates.py; monkeypatch it to PASS
        # so the test is self-contained regardless of real gate state.
        fake_gate_json = json.dumps({
            "overall_passed": True,
            "gates": [{"label": "integrity", "passed": True}],
        })
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: _fake_proc(fake_gate_json, 0))

        # Layer 1 SKIP (nonexistent dir); Layer 2 PASS (mocked); Layer 3/4 SKIP (missing DB)
        exit_code = main([
            "--audit-dir", str(tmp_path / "nonexistent_audits"),
            "--db", str(tmp_path / "nonexistent.db"),
            "--jsonl-path", str(tmp_path / "nonexistent.jsonl"),
        ])
        assert exit_code == 0

    def test_cli_exit_1_when_any_layer_fails(self, tmp_path, monkeypatch):
        """Monkeypatch run_layer2 to FAIL → exit 1."""
        from check_model_improvement import main

        fail_result = LayerResult(2, "Gate Status", "FAIL", {"overall_passed": False,
                                                               "n_gates_passed": 0,
                                                               "n_gates_failed": 4}, "FAIL")
        monkeypatch.setattr("check_model_improvement.run_layer2_gate_status",
                            lambda *a, **kw: fail_result)

        exit_code = main([
            "--layer", "2",
            "--audit-dir", str(tmp_path),
            "--db", str(tmp_path / "nonexistent.db"),
        ])
        assert exit_code == 1
