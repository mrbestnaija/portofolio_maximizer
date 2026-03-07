"""Integration tests for Phase 7.25 lift significance CI (Layer 1 schema + behaviour).

Tests verify that:
- Layer 1 metrics dict includes all 5 new CI keys (schema contract).
- When CI spans zero with >= 20 windows, Layer 1 emits a WARN.
- When lift is consistently positive, the CI confirms it.
- The bootstrap CI is numerically stable under extreme RMSE inputs.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.check_model_improvement import LAYER_REQUIRED_KEYS, run_layer1_forecast_quality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_audit(tmp_path: Path, window_id: str, ens_rmse: float, best_rmse: float) -> None:
    """Write a minimal forecast_audit JSON file with the required structure.

    extract_window_metrics() reads artifacts.evaluation_metrics, not top-level.
    All 3 MODELS (garch, samossa, mssa_rl) must be present with rmse entries.
    Each window must have a unique dataset.start so _window_fingerprint gives distinct keys
    (otherwise all windows with None fields collapse to the same fingerprint → deduplicated).
    """
    data = {
        "window_id": window_id,
        "window_end": f"2025-01-15",
        "dataset": {
            "ticker": "TEST",
            "start": f"start-{window_id}",   # guaranteed unique per window_id
            "end": f"end-{window_id}",
            "length": 60,
        },
        "artifacts": {
            "evaluation_metrics": {
                "ensemble": {
                    "rmse": ens_rmse,
                    "directional_accuracy": 0.55,
                },
                "garch": {
                    "rmse": best_rmse,
                    "directional_accuracy": 0.55,
                },
                "samossa": {
                    "rmse": best_rmse + 0.01,
                    "directional_accuracy": 0.55,
                },
                "mssa_rl": {
                    "rmse": best_rmse + 0.02,
                    "directional_accuracy": 0.55,
                },
            },
        },
    }
    (tmp_path / f"forecast_audit_{window_id}.json").write_text(
        json.dumps(data), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLayer1LiftSignificanceSchema:
    """Layer 1 must include all 5 CI keys in its metrics dict."""

    def test_layer1_includes_ci_metrics_in_required_schema(self):
        """LAYER_REQUIRED_KEYS[1] must declare all 5 Phase-7.25 CI keys."""
        required = LAYER_REQUIRED_KEYS[1]
        for key in ("lift_mean", "lift_ci_low", "lift_ci_high", "lift_win_fraction",
                    "lift_ci_insufficient_data"):
            assert key in required, f"LAYER_REQUIRED_KEYS[1] missing '{key}'"

    def test_layer1_ci_keys_present_in_returned_metrics(self, tmp_path):
        """run_layer1_forecast_quality must return all 5 CI keys in the metrics dict."""
        for i in range(10):
            _write_audit(tmp_path, f"w{i:02d}", ens_rmse=1.5, best_rmse=2.0)

        result = run_layer1_forecast_quality(audit_dir=tmp_path)
        for key in ("lift_mean", "lift_ci_low", "lift_ci_high", "lift_win_fraction",
                    "lift_ci_insufficient_data"):
            assert key in result.metrics, f"metrics dict missing '{key}' (status={result.status})"


class TestLayer1LiftCIBehaviour:
    """Layer 1 status/warn behaviour driven by CI results."""

    def test_layer1_warns_when_ci_spans_zero_with_sufficient_data(self, tmp_path):
        """With balanced wins/losses and n>=20, Layer 1 should emit a CI-span-zero WARN.

        Uses warn_coverage_threshold=5 to avoid the separate n_used coverage WARN firing
        first and blocking the CI check (which only fires inside 'if status == PASS').
        """
        # 10 windows where ensemble wins (delta +0.1), 10 where it loses (delta -0.1)
        for i in range(10):
            _write_audit(tmp_path, f"win{i:02d}", ens_rmse=1.9, best_rmse=2.0)
        for i in range(10):
            _write_audit(tmp_path, f"los{i:02d}", ens_rmse=2.0, best_rmse=1.9)

        # warn_coverage_threshold=5 prevents coverage WARN from firing (n_used=20 > 5)
        # so the CI check inside 'if status == PASS' is reachable.
        result = run_layer1_forecast_quality(
            audit_dir=tmp_path,
            warn_coverage_threshold=5,
            warn_lift_threshold=0.01,  # lift=0.5 >> 0.01 → no lift WARN
        )
        # CI spans zero AND n>=20 → must be WARN (never silent PASS)
        assert result.status in ("WARN", "FAIL"), (
            f"Expected WARN or FAIL for balanced lift data, got {result.status}\n"
            f"summary: {result.summary}"
        )
        # The CI WARN reason should appear in the summary
        assert "ci" in result.summary.lower() or "CI" in result.summary, (
            f"WARN message should mention CI span-zero, got summary: {result.summary}"
        )

    def test_layer1_ci_above_zero_with_strong_lift_fixtures(self, tmp_path):
        """With 30 windows where ensemble always wins by a large margin, CI_low > 0."""
        for i in range(30):
            _write_audit(tmp_path, f"w{i:02d}", ens_rmse=1.0, best_rmse=3.0)

        result = run_layer1_forecast_quality(audit_dir=tmp_path)
        ci_low = result.metrics.get("lift_ci_low")
        assert ci_low is not None, "lift_ci_low missing from metrics"
        assert not math.isnan(ci_low), "lift_ci_low is NaN"
        assert ci_low > 0.0, f"lift_ci_low={ci_low:.4f} should be > 0 for strong lift"
        insufficient = result.metrics.get("lift_ci_insufficient_data")
        assert insufficient is False

    def test_layer1_fails_when_ci_definitively_negative(self, tmp_path):
        """Both CI bounds < 0 (ensemble definitively worse) → Layer 1 FAIL, not just WARN.

        Uses 25 windows where ensemble always loses (ens_rmse=2.0, best=1.0).
        delta = best - ensemble = -1.0 per window.  CI: both bounds << 0.
        """
        for i in range(25):
            _write_audit(tmp_path, f"w{i:02d}", ens_rmse=2.0, best_rmse=1.0)

        result = run_layer1_forecast_quality(
            audit_dir=tmp_path,
            warn_coverage_threshold=5,   # prevent coverage WARN from masking CI signal
            warn_lift_threshold=0.01,
        )
        assert result.status == "FAIL", (
            f"Expected FAIL when CI is definitively negative (both bounds < 0), "
            f"got {result.status}\nsummary: {result.summary}"
        )
        assert "definitively" in result.summary.lower(), (
            f"FAIL message should mention 'definitively', got: {result.summary}"
        )
        ci_high = result.metrics.get("lift_ci_high")
        assert ci_high is not None and ci_high < 0.0, (
            f"lift_ci_high should be < 0, got {ci_high}"
        )


class TestLiftSignificanceNumericalStabilityIntegration:
    """Bootstrap CI must be stable under extreme RMSE inputs (no NaN/inf)."""

    def test_lift_significance_stable_under_extreme_rmse_inputs(self, tmp_path):
        """Extreme RMSE values (1e-8 and 1e8) must not cause NaN/inf in CI output."""
        for i in range(10):
            _write_audit(tmp_path, f"tiny{i:02d}", ens_rmse=1e-8, best_rmse=2e-8)
        for i in range(10):
            _write_audit(tmp_path, f"huge{i:02d}", ens_rmse=1e7, best_rmse=1e8)

        result = run_layer1_forecast_quality(audit_dir=tmp_path)
        for key in ("lift_mean", "lift_ci_low", "lift_ci_high", "lift_win_fraction"):
            val = result.metrics.get(key)
            assert val is not None, f"'{key}' missing from metrics"
            if isinstance(val, float):
                assert math.isfinite(val), f"'{key}'={val} is non-finite under extreme inputs"
