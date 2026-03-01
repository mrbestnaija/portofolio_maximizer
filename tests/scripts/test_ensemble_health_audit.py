"""Tests for scripts/ensemble_health_audit.py"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.ensemble_health_audit import (
    MODELS,
    compute_adaptive_weights,
    compute_per_model_summary,
    compute_shapley_attribution,
    extract_window_metrics,
    generate_markdown_report,
    load_audit_windows,
    main,
    update_config_adaptive_weights,
)


def _make_audit(
    garch_rmse: float = 2.0,
    samossa_rmse: float = 3.0,
    mssa_rl_rmse: float = 2.5,
    garch_da: float = 0.55,
    samossa_da: float = 0.0,
    mssa_rl_da: float = 0.55,
    ensemble_rmse: float = 2.2,
    ensemble_da: float = 0.55,
    start: str = "2025-01-01",
    end: str = "2025-06-01",
    length: int = 100,
    horizon: int = 10,
    regime: str | None = "HIGH_VOL_TRENDING",
    garch_smape: float = 0.02,
    samossa_smape: float = 0.03,
    mssa_rl_smape: float = 0.025,
) -> dict:
    runs = []
    if regime:
        runs.append({"model": "regime", "metadata": {"regime": regime}})
    return {
        "_path": f"forecast_audit_{start.replace('-','')}_{end.replace('-','')}.json",
        "_mtime": 1.0,
        "dataset": {"start": start, "end": end, "length": length, "ticker": "AAPL"},
        "summary": {"forecast_horizon": horizon},
        "artifacts": {
            "evaluation_metrics": {
                "garch": {
                    "rmse": garch_rmse,
                    "smape": garch_smape,
                    "directional_accuracy": garch_da,
                    "n_observations": 30,
                },
                "samossa": {
                    "rmse": samossa_rmse,
                    "smape": samossa_smape,
                    "directional_accuracy": samossa_da,
                    "n_observations": 30,
                },
                "mssa_rl": {
                    "rmse": mssa_rl_rmse,
                    "smape": mssa_rl_smape,
                    "directional_accuracy": mssa_rl_da,
                    "n_observations": 30,
                },
                "ensemble": {
                    "rmse": ensemble_rmse,
                    "smape": 0.022,
                    "directional_accuracy": ensemble_da,
                    "n_observations": 30,
                },
            },
            "ensemble_weights": {"garch": 0.4, "samossa": 0.4, "mssa_rl": 0.2},
        },
        "runs": runs,
    }


def _write_audit_file(directory: Path, name: str, **kwargs) -> Path:
    data = _make_audit(**kwargs)
    p = directory / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


class TestExtractWindowMetrics:
    def test_returns_expected_keys(self):
        audit = _make_audit()
        w = extract_window_metrics(audit)
        assert w is not None
        assert "window_id" in w
        assert "model_metrics" in w
        assert "best_single_model" in w
        assert "rmse_ratio" in w
        assert "ensemble_weights" in w
        assert "regime" in w

    def test_handles_missing_model_rmse(self):
        audit = _make_audit()
        del audit["artifacts"]["evaluation_metrics"]["garch"]["rmse"]
        w = extract_window_metrics(audit)
        assert w is None

    def test_handles_missing_evaluation_metrics(self):
        audit = _make_audit()
        audit["artifacts"]["evaluation_metrics"] = {}
        w = extract_window_metrics(audit)
        assert w is None

    def test_handles_missing_ensemble_rmse(self):
        audit = _make_audit()
        del audit["artifacts"]["evaluation_metrics"]["ensemble"]["rmse"]
        w = extract_window_metrics(audit)
        assert w is None

    def test_best_single_deterministic_by_rmse_then_smape(self):
        # garch=1.0 rmse, smape=0.01 → should win
        audit = _make_audit(
            garch_rmse=1.0, samossa_rmse=2.0, mssa_rl_rmse=2.0,
            garch_smape=0.01, samossa_smape=0.02, mssa_rl_smape=0.02,
        )
        w = extract_window_metrics(audit)
        assert w["best_single_model"] == "garch"

    def test_best_single_tie_broken_by_smape(self):
        # garch and samossa same RMSE; samossa has lower sMAPE → samossa wins
        audit = _make_audit(
            garch_rmse=2.0, samossa_rmse=2.0, mssa_rl_rmse=3.0,
            garch_smape=0.05, samossa_smape=0.02, mssa_rl_smape=0.05,
        )
        w = extract_window_metrics(audit)
        assert w["best_single_model"] == "samossa"

    def test_regime_extracted_from_runs(self):
        audit = _make_audit(regime="CRISIS")
        w = extract_window_metrics(audit)
        assert w["regime"] == "CRISIS"

    def test_regime_none_when_no_regime_run(self):
        audit = _make_audit(regime=None)
        w = extract_window_metrics(audit)
        assert w["regime"] is None


class TestPerModelSummary:
    def test_counts_best_single_correctly(self):
        windows = [
            extract_window_metrics(_make_audit(garch_rmse=1.0, samossa_rmse=2.0, mssa_rl_rmse=3.0)),
            extract_window_metrics(_make_audit(garch_rmse=1.0, samossa_rmse=2.0, mssa_rl_rmse=3.0)),
            extract_window_metrics(_make_audit(garch_rmse=5.0, samossa_rmse=2.0, mssa_rl_rmse=0.5)),
        ]
        summary = compute_per_model_summary(windows)
        assert summary["garch"]["times_best_single"] == 2
        assert summary["mssa_rl"]["times_best_single"] == 1
        assert summary["samossa"]["times_best_single"] == 0

    def test_counts_da_zero_windows(self):
        windows = [
            extract_window_metrics(_make_audit(samossa_da=0.0)),
            extract_window_metrics(_make_audit(samossa_da=0.0)),
            extract_window_metrics(_make_audit(samossa_da=0.55)),
        ]
        summary = compute_per_model_summary(windows)
        assert summary["samossa"]["da_zero_windows"] == 2

    def test_mean_rmse_computed(self):
        windows = [
            extract_window_metrics(_make_audit(garch_rmse=2.0)),
            extract_window_metrics(_make_audit(garch_rmse=4.0)),
        ]
        summary = compute_per_model_summary(windows)
        assert abs(summary["garch"]["mean_rmse"] - 3.0) < 1e-6


class TestAdaptiveWeights:
    def _windows_with_rmse(self, garch=2.0, samossa=3.0, mssa_rl=2.5):
        return [
            extract_window_metrics(_make_audit(garch_rmse=garch, samossa_rmse=samossa, mssa_rl_rmse=mssa_rl))
        ]

    def test_normalized_to_sum_1(self):
        windows = [extract_window_metrics(_make_audit()) for _ in range(5)]
        candidates, params = compute_adaptive_weights(windows)
        primary = candidates[0]
        total = sum(primary.values())
        assert abs(total - 1.0) < 1e-4, f"Weights should sum to 1.0, got {total}"

    def test_hard_zero_when_rmse_exceeds_1_2x_median(self):
        # SAMOSSA RMSE = 10x others → hard zero
        windows = [
            extract_window_metrics(_make_audit(garch_rmse=2.0, samossa_rmse=20.0, mssa_rl_rmse=2.0))
            for _ in range(5)
        ]
        candidates, params = compute_adaptive_weights(windows)
        primary = candidates[0]
        # SAMOSSA should be hard-zeroed (not present or weight=0)
        assert primary.get("samossa", 0.0) < 1e-6

    def test_da_penalty_caps_weight(self):
        # SAMOSSA DA=0 → weight capped at 0.10
        windows = [
            extract_window_metrics(
                _make_audit(
                    garch_rmse=2.0, samossa_rmse=1.5, mssa_rl_rmse=2.5,
                    samossa_da=0.0,  # DA=0 → penalty
                    garch_da=0.55, mssa_rl_da=0.55,
                )
            )
            for _ in range(5)
        ]
        candidates, params = compute_adaptive_weights(windows, da_floor=0.10, da_cap_weight=0.10)
        primary = candidates[0]
        assert primary.get("samossa", 0.0) <= 0.101, (
            f"DA-penalized samossa should be <= 0.10, got {primary.get('samossa')}"
        )

    def test_all_da_zero_fallback(self):
        # All models DA=0 → degraded_da_fallback=True, weights by RMSE only
        windows = [
            extract_window_metrics(
                _make_audit(garch_da=0.0, samossa_da=0.0, mssa_rl_da=0.0)
            )
            for _ in range(5)
        ]
        candidates, params = compute_adaptive_weights(windows, da_floor=0.10)
        assert params["degraded_da_fallback"] is True
        # Weights still sum to 1
        primary = candidates[0]
        assert abs(sum(primary.values()) - 1.0) < 1e-6

    def test_diversity_guard_clamps_top_weight_at_0_90(self):
        # One model much better → exp decay gives it >0.90 before guard
        windows = [
            extract_window_metrics(
                _make_audit(garch_rmse=0.1, samossa_rmse=100.0, mssa_rl_rmse=100.0)
            )
            for _ in range(5)
        ]
        candidates, params = compute_adaptive_weights(windows)
        primary = candidates[0]
        assert max(primary.values()) <= 0.901, (
            f"Diversity guard should clamp top weight to 0.90, got {max(primary.values())}"
        )

    def test_da_zero_samossa_gets_capped_weight(self):
        windows = [
            extract_window_metrics(
                _make_audit(samossa_da=0.0, garch_da=0.55, mssa_rl_da=0.55)
            )
            for _ in range(10)
        ]
        candidates, _ = compute_adaptive_weights(windows, da_floor=0.10, da_cap_weight=0.10)
        primary = candidates[0]
        assert primary.get("samossa", 0.0) <= 0.101

    def test_three_candidates_returned(self):
        windows = [extract_window_metrics(_make_audit()) for _ in range(5)]
        candidates, _ = compute_adaptive_weights(windows)
        assert len(candidates) == 3

    def test_empty_windows_returns_empty(self):
        candidates, params = compute_adaptive_weights([])
        assert candidates == []
        assert params == {}


class TestLoadAuditWindows:
    def test_deduplicates_by_default(self, tmp_path):
        # Two files with same fingerprint
        _write_audit_file(tmp_path, "forecast_audit_20260101_000000.json", start="2025-01-01", end="2025-06-01")
        time.sleep(0.05)
        _write_audit_file(tmp_path, "forecast_audit_20260102_000000.json", start="2025-01-01", end="2025-06-01")
        windows = load_audit_windows(tmp_path, dedupe=True)
        assert len(windows) == 1

    def test_no_dedup_returns_all(self, tmp_path):
        _write_audit_file(tmp_path, "forecast_audit_20260101_000000.json", start="2025-01-01", end="2025-06-01")
        _write_audit_file(tmp_path, "forecast_audit_20260102_000000.json", start="2025-01-01", end="2025-06-01")
        windows = load_audit_windows(tmp_path, dedupe=False)
        assert len(windows) == 2


class TestMarkdownReport:
    def _make_windows_and_summary(self):
        windows = [extract_window_metrics(_make_audit()) for _ in range(3)]
        summary = compute_per_model_summary(windows)
        return windows, summary

    def test_contains_model_names(self):
        windows, summary = self._make_windows_and_summary()
        candidates, params = compute_adaptive_weights(windows)
        shapley = {m: 0.1 for m in MODELS}
        report = generate_markdown_report(windows, summary, shapley, candidates, params, 0)
        for m in MODELS:
            assert m in report

    def test_contains_lift_fraction(self):
        windows, summary = self._make_windows_and_summary()
        candidates, params = compute_adaptive_weights(windows)
        shapley = {m: 0.0 for m in MODELS}
        report = generate_markdown_report(windows, summary, shapley, candidates, params, 4)
        assert "4 duplicates" in report

    def test_da_zero_callout_when_samossa_anomaly(self):
        # Build windows where samossa DA=0 in all 6+ windows
        windows = [
            extract_window_metrics(_make_audit(samossa_da=0.0)) for _ in range(7)
        ]
        summary = compute_per_model_summary(windows)
        candidates, params = compute_adaptive_weights(windows)
        shapley = {m: 0.0 for m in MODELS}
        report = generate_markdown_report(windows, summary, shapley, candidates, params, 0)
        assert "SAMOSSA DA=0 anomaly" in report


class TestUpdateConfig:
    def test_writes_adaptive_section(self, tmp_path):
        config_path = tmp_path / "forecasting_config.yml"
        config_path.write_text(
            yaml.dump({"ensemble": {"enabled": True, "candidate_weights": [{"garch": 1.0}]}}),
            encoding="utf-8",
        )
        candidates = [{"garch": 0.5, "mssa_rl": 0.5}]
        params = {"recent_n": 20, "lambda_decay": 1.0, "da_floor": 0.10, "da_cap_weight": 0.10}
        update_config_adaptive_weights(config_path, candidates, params)
        updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "adaptive_candidate_weights" in updated
        assert "weights" in updated["adaptive_candidate_weights"]

    def test_does_not_overwrite_static_weights(self, tmp_path):
        static = [{"garch": 1.0}, {"samossa": 1.0}]
        config_path = tmp_path / "forecasting_config.yml"
        config_path.write_text(
            yaml.dump({"ensemble": {"candidate_weights": static}}), encoding="utf-8"
        )
        update_config_adaptive_weights(
            config_path,
            [{"mssa_rl": 1.0}],
            {"recent_n": 20, "lambda_decay": 1.0, "da_floor": 0.10, "da_cap_weight": 0.10},
        )
        updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        # Static candidate_weights must still be there
        assert updated["ensemble"]["candidate_weights"] == static


class TestCLI:
    def test_exits_0_with_valid_audits(self, tmp_path):
        for i in range(3):
            _write_audit_file(
                tmp_path,
                f"forecast_audit_2026010{i}_000000.json",
                start=f"2025-0{i+1}-01",
                end=f"2025-0{i+2}-01",
            )
        rc = main(["--audit-dir", str(tmp_path)])
        assert rc == 0

    def test_exits_1_when_no_valid_audits(self, tmp_path):
        rc = main(["--audit-dir", str(tmp_path)])
        assert rc == 1

    def test_e2e_writes_markdown_and_config_from_fake_audits(self, tmp_path):
        """End-to-end smoke: fake audit JSONs -> markdown + config written."""
        audit_dir = tmp_path / "audits"
        audit_dir.mkdir()
        health_dir = tmp_path / "health"
        config_path = tmp_path / "forecasting_config.yml"
        config_path.write_text(
            yaml.dump({"ensemble": {"enabled": True, "candidate_weights": [{"garch": 1.0}]}}),
            encoding="utf-8",
        )
        # Write 5 distinct audit files
        for i in range(5):
            _write_audit_file(
                audit_dir,
                f"forecast_audit_2026010{i}_000000.json",
                start=f"2025-0{i+1}-01",
                end=f"2025-0{i+2}-01",
            )
        import scripts.ensemble_health_audit as mod

        orig_health_dir = mod.ENSEMBLE_HEALTH_DIR
        mod.ENSEMBLE_HEALTH_DIR = health_dir
        try:
            rc = main(
                [
                    "--audit-dir", str(audit_dir),
                    "--write-report",
                    "--write-config",
                    "--config-path", str(config_path),
                ]
            )
        finally:
            mod.ENSEMBLE_HEALTH_DIR = orig_health_dir

        assert rc == 0
        md_files = list(health_dir.glob("ensemble_health_*.md"))
        assert len(md_files) == 1, "Markdown report should be written"
        assert md_files[0].stat().st_size > 100

        updated_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "adaptive_candidate_weights" in updated_cfg, "Config should have adaptive weights"
        assert "weights" in updated_cfg["adaptive_candidate_weights"]


# ---------------------------------------------------------------------------
# Property-based fuzz tests for compute_adaptive_weights
# ---------------------------------------------------------------------------

from hypothesis import given, settings, assume
from hypothesis import strategies as st


def _make_window_row(
    garch_rmse: float = 2.0,
    garch_da: float = 0.55,
    samossa_rmse: float = 3.0,
    samossa_da: float = 0.50,
    mssa_rl_rmse: float = 2.5,
    mssa_rl_da: float = 0.55,
) -> dict:
    """Minimal window dict compatible with extract_window_metrics output."""
    return {
        "window_id": "w001",
        "ticker": "TEST",
        "regime": None,
        "window_start": "2025-01-01",
        "window_end": "2025-06-01",
        "n_obs": 100,
        "horizon": 10,
        "model_metrics": {
            "garch": {"rmse": garch_rmse, "da": garch_da, "smape": 0.05},
            "samossa": {"rmse": samossa_rmse, "da": samossa_da, "smape": 0.07},
            "mssa_rl": {"rmse": mssa_rl_rmse, "da": mssa_rl_da, "smape": 0.06},
        },
        "ensemble_rmse": 2.0,
        "ensemble_da": 0.53,
        "ensemble_weights": {"garch": 0.33, "samossa": 0.33, "mssa_rl": 0.34},
        "best_single_model": "garch",
        "rmse_ratio": 1.0,
    }


@st.composite
def _random_windows(draw):
    """Generate 1-25 random window dicts with plausible RMSE and DA values."""
    n = draw(st.integers(min_value=1, max_value=25))
    rows = []
    for _ in range(n):
        rows.append(
            _make_window_row(
                garch_rmse=draw(st.floats(0.5, 10.0, allow_nan=False, allow_infinity=False)),
                garch_da=draw(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)),
                samossa_rmse=draw(st.floats(0.5, 10.0, allow_nan=False, allow_infinity=False)),
                samossa_da=draw(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)),
                mssa_rl_rmse=draw(st.floats(0.5, 10.0, allow_nan=False, allow_infinity=False)),
                mssa_rl_da=draw(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)),
            )
        )
    return rows


class TestComputeAdaptiveWeightsProperties:
    """Hypothesis property-based tests for compute_adaptive_weights().

    Encodes the CONTRACT from the function docstring:
    - Every candidate sums to 1.0 ± 1e-4.
    - No negative weights.
    - When all DA < da_floor, degraded_da_fallback=True.
    """

    @given(windows=_random_windows(), da_floor=st.floats(0.05, 0.40))
    @settings(max_examples=150, deadline=None)
    def test_candidates_always_normalized(self, windows, da_floor):
        """Every returned candidate must sum to 1.0 (within 1e-4)."""
        candidates, _ = compute_adaptive_weights(
            windows, recent_n=len(windows), da_floor=da_floor, da_cap_weight=0.10
        )
        for i, cand in enumerate(candidates):
            total = sum(cand.values())
            assert abs(total - 1.0) < 1e-4, (
                f"candidate[{i}] sums to {total:.6f} (expected 1.0). "
                f"n_windows={len(windows)}, da_floor={da_floor}"
            )

    @given(windows=_random_windows(), da_floor=st.floats(0.05, 0.40))
    @settings(max_examples=150, deadline=None)
    def test_no_negative_weights(self, windows, da_floor):
        """No candidate should have negative weights."""
        candidates, _ = compute_adaptive_weights(
            windows, recent_n=len(windows), da_floor=da_floor, da_cap_weight=0.10
        )
        for i, cand in enumerate(candidates):
            for m, w in cand.items():
                assert w >= 0.0, f"candidate[{i}]['{m}'] = {w} < 0"

    @given(windows=_random_windows())
    @settings(max_examples=150, deadline=None)
    def test_all_da_zero_triggers_fallback(self, windows):
        """When every model has DA=0 (< da_floor), degraded_da_fallback must be True."""
        # Force all DAs to 0 so the all-DA-zero fallback activates
        zero_da_windows = []
        for w in windows:
            row = dict(w)
            row["model_metrics"] = {
                m: dict(v, da=0.0) for m, v in w["model_metrics"].items()
            }
            zero_da_windows.append(row)
        _, params = compute_adaptive_weights(
            zero_da_windows, recent_n=len(zero_da_windows), da_floor=0.10
        )
        assert params["degraded_da_fallback"] is True, (
            "All models with DA=0 should trigger degraded_da_fallback"
        )

    @given(windows=_random_windows())
    @settings(max_examples=150, deadline=None)
    def test_primary_candidate_respects_diversity_guard(self, windows):
        """Primary candidate's top model weight must be ≤ 0.90."""
        candidates, params = compute_adaptive_weights(windows, recent_n=len(windows))
        primary = candidates[0]
        max_weight = max(primary.values()) if primary else 0.0
        assert max_weight <= 0.90 + 1e-6, (
            f"Primary candidate top weight={max_weight:.4f} violates diversity guard (≤0.90). "
            f"diversity_clamped={params['diversity_clamped']}"
        )

    @given(
        windows=_random_windows(),
        da_floor=st.floats(0.05, 0.45),
        da_cap_weight=st.floats(0.10, 0.40),
    )
    @settings(max_examples=250, deadline=None)
    def test_penalized_models_respect_cap(self, windows, da_floor, da_cap_weight):
        """Every DA-penalized model must have weight ≤ da_cap_weight in the primary candidate.

        This guards against the redistribution-to-penalized bug where below-cap
        penalized models receive freed budget from above-cap penalized models and
        grow past da_cap_weight.
        """
        from collections import defaultdict

        candidates, params = compute_adaptive_weights(
            windows, recent_n=len(windows), da_floor=da_floor, da_cap_weight=da_cap_weight
        )
        if params.get("degraded_da_fallback"):
            return  # DA penalty was skipped (all DA < floor) — no cap to enforce

        # Recompute mean DA per model to know which are penalized
        da_sums: dict = defaultdict(list)
        for w in windows:
            for m, metrics in w.get("model_metrics", {}).items():
                if m in MODELS:
                    da_sums[m].append(float(metrics.get("da", 0.0)))
        mean_da = {
            m: (sum(da_sums[m]) / len(da_sums[m]) if da_sums[m] else 0.0)
            for m in MODELS
        }
        penalized = {m for m in MODELS if mean_da.get(m, 0.0) < da_floor}

        primary = candidates[0]
        for m in penalized:
            w_val = primary.get(m, 0.0)
            assert w_val <= da_cap_weight + 1e-6, (
                f"Penalized model '{m}' weight={w_val:.6f} > da_cap_weight={da_cap_weight:.4f}. "
                f"mean_da[{m}]={mean_da.get(m, 0):.4f} < da_floor={da_floor:.4f}. "
                f"n_windows={len(windows)}"
            )
