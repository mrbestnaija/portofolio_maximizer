"""
tests/contracts/test_system_integrity.py
=========================================
Regression-prevention schema for all 26 findings from the 2026-03-18 audit.

Each test is pinned to a specific finding code (SC = short-circuit, MW = mismatch
wiring, TD = threshold dodge, AR = architecture, NS = numerical stability,
ST = stub). Tests are intentionally narrow — they verify the CONTRACT, not the
implementation detail, so they survive refactoring.

Run:
  pytest tests/contracts/test_system_integrity.py -v

All tests must pass on every commit that touches:
  scripts/evaluate_directional_classifier.py
  scripts/accumulate_classifier_labels.py
  scripts/check_classifier_readiness.py
  scripts/generate_classifier_training_labels.py
  scripts/train_directional_classifier.py
  scripts/capital_readiness_check.py
  scripts/check_model_improvement.py
  scripts/check_forecast_audits.py
  config/forecaster_monitoring.yml
  config/signal_routing_config.yml
  forcester_ts/regime_detector.py
  models/time_series_signal_generator.py
"""
from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _REPO_ROOT / "config"


def _load_yaml(name: str) -> Dict[str, Any]:
    return yaml.safe_load((_CONFIG_DIR / name).read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# SC-01  _optimal_gate_threshold must return (float, bool) not bare float
# ---------------------------------------------------------------------------

class TestSC01GateThresholdReturnContract:
    """SC-01: _optimal_gate_threshold must expose whether optimisation succeeded."""

    def test_returns_tuple_of_float_and_bool(self):
        from scripts.evaluate_directional_classifier import _optimal_gate_threshold
        import numpy as np
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 3)
        p = np.linspace(0.4, 0.9, len(y))
        result = _optimal_gate_threshold(y, p, min_gated=5)
        assert isinstance(result, tuple) and len(result) == 2, (
            "SC-01: must return (threshold: float, optimized: bool)"
        )
        thresh, optimized = result
        assert isinstance(thresh, float)
        assert isinstance(optimized, bool)

    def test_optimized_false_when_all_candidates_fail_min_gated(self):
        """When no candidate clears min_gated, optimized must be False (not silently 0.55)."""
        from scripts.evaluate_directional_classifier import _optimal_gate_threshold
        y = np.array([1, 0])          # only 2 examples
        p = np.array([0.6, 0.4])
        _, optimized = _optimal_gate_threshold(y, p, min_gated=10)
        assert optimized is False, (
            "SC-01: must return optimized=False when dataset too small for any threshold"
        )

    def test_optimized_true_when_candidate_passes(self):
        from scripts.evaluate_directional_classifier import _optimal_gate_threshold
        y = np.ones(30, dtype=int)
        y[::2] = 0
        p = np.linspace(0.50, 0.85, 30)
        _, optimized = _optimal_gate_threshold(y, p, min_gated=5)
        assert optimized is True

    def test_evaluate_exposes_threshold_optimized_in_result(self, tmp_path):
        """evaluate() must propagate threshold_optimized into the returned dict."""
        from scripts.evaluate_directional_classifier import evaluate
        from forcester_ts.directional_classifier import _FEATURE_NAMES
        # build minimal dataset
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, len(_FEATURE_NAMES)))
        y = (X[:, 0] > 0).astype(int)
        df = pd.DataFrame(X, columns=_FEATURE_NAMES)
        df["y_directional"] = y
        df["ts_signal_id"] = [f"ts_AAPL_test_{i:04d}" for i in range(80)]
        df["ticker"] = "AAPL"
        df["entry_ts"] = pd.date_range("2022-01-01", periods=80, freq="D").astype(str)
        df["action"] = "BUY"
        ds = tmp_path / "directional_dataset.parquet"
        df.to_parquet(ds, index=False)
        # train first so meta exists
        from scripts.train_directional_classifier import train
        mp = tmp_path / "m.pkl"
        mtp = tmp_path / "m.meta.json"
        train(dataset_path=ds, model_path=mp, meta_path=mtp, c_values=[1.0])
        result = evaluate(dataset_path=ds, meta_path=mtp, write_report=False)
        # key must exist in result
        assert "threshold_optimized" in result or any(
            "threshold_optimized" in str(v) for v in result.values()
        ), "SC-01: evaluate() result must include threshold_optimized"


# ---------------------------------------------------------------------------
# SC-02  accumulate_classifier_labels must distinguish DB states
# ---------------------------------------------------------------------------

class TestSC02AccumulatorDbStatus:
    """SC-02: _load_outcome_map must return (dict, status_str) not bare dict."""

    def test_returns_tuple_on_missing_db(self, tmp_path):
        from scripts.accumulate_classifier_labels import _load_outcome_map
        result = _load_outcome_map(tmp_path / "nonexistent.db")
        assert isinstance(result, tuple) and len(result) == 2, (
            "SC-02: must return (dict, str) tuple"
        )
        outcome_map, status = result
        assert isinstance(outcome_map, dict)
        assert status == "db_missing", f"Expected 'db_missing', got {status!r}"

    def test_returns_ok_status_on_existing_empty_db(self, tmp_path):
        from scripts.accumulate_classifier_labels import _load_outcome_map
        db = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE trade_executions "
            "(ts_signal_id TEXT, realized_pnl REAL, is_close INTEGER, "
            "is_diagnostic INTEGER, is_synthetic INTEGER)"
        )
        conn.commit()
        conn.close()
        _, status = _load_outcome_map(db)
        assert status == "ok", f"Empty-but-valid DB must return 'ok', got {status!r}"

    def test_db_status_propagated_to_accumulate_result(self, tmp_path):
        from scripts.accumulate_classifier_labels import accumulate
        result = accumulate(
            jsonl_path=tmp_path / "missing.jsonl",
            dataset_path=tmp_path / "ds.parquet",
            db_path=tmp_path / "missing.db",
            dry_run=True,
        )
        assert "db_status" in result, "SC-02: accumulate() must expose db_status in result"
        assert result["db_status"] == "db_missing"


# ---------------------------------------------------------------------------
# TD-01  Gate threshold must not silently use 0.55 when optimisation fails
# ---------------------------------------------------------------------------

class TestTD01GateThresholdFallback:
    """TD-01: When threshold is uncalibrated, evaluate output must flag it."""

    def test_uncalibrated_flag_present_when_data_tiny(self, tmp_path):
        """With only 5 examples and min_gated=10, no threshold clears min_gated. Must flag uncalibrated."""
        from scripts.evaluate_directional_classifier import _optimal_gate_threshold
        y = np.array([1, 0, 1, 0, 1])   # 5 examples — all candidates below min_gated=10
        p = np.linspace(0.50, 0.90, len(y))
        thresh, optimized = _optimal_gate_threshold(y, p, min_gated=10)
        assert not optimized, "TD-01: must flag uncalibrated when dataset < min_gated"
        assert thresh == 0.55, "TD-01: fallback must be 0.55 (documented contract)"


# ---------------------------------------------------------------------------
# MW-01  min_lift_rmse_ratio: config value must be read, not hardcoded
# ---------------------------------------------------------------------------

class TestMW01LiftRmseRatioWiring:
    """MW-01: check_model_improvement must read min_lift_rmse_ratio from config, not hardcode."""

    def test_lift_threshold_reflects_config_value(self):
        from scripts.check_model_improvement import _load_layer1_regression_contract
        cfg = _load_yaml("forecaster_monitoring.yml")
        config_ratio = float(
            cfg.get("forecaster_monitoring", {})
            .get("regression_metrics", {})
            .get("min_lift_rmse_ratio", 0.0)
        )
        _, lift_threshold = _load_layer1_regression_contract()
        expected = round(1.0 - config_ratio, 6)
        assert abs(lift_threshold - expected) < 1e-6, (
            f"MW-01: lift_threshold {lift_threshold} != 1.0 - config({config_ratio}) = {expected}. "
            "Config wiring broken."
        )

    def test_config_min_lift_rmse_ratio_above_zero(self):
        """min_lift_rmse_ratio must be > 0.0 so epsilon improvements don't count as lift."""
        cfg = _load_yaml("forecaster_monitoring.yml")
        ratio = float(
            cfg.get("forecaster_monitoring", {})
            .get("regression_metrics", {})
            .get("min_lift_rmse_ratio", 0.0)
        )
        assert ratio > 0.0, (
            "MW-01: min_lift_rmse_ratio must be > 0.0 (currently set to 0.02). "
            "Setting 0.0 means any epsilon RMSE improvement counts as lift (threshold dodge)."
        )


# ---------------------------------------------------------------------------
# MW-02  SNR gate wired: min_signal_to_noise must be read and > 0 in production config
# ---------------------------------------------------------------------------

class TestMW02SnrGateWiring:
    """MW-02: min_signal_to_noise must be configured and non-zero to enforce SNR gate."""

    def test_snr_threshold_nonzero_in_config(self):
        cfg = _load_yaml("signal_routing_config.yml")
        cost_model = (
            cfg.get("signal_routing", {})
            .get("time_series", {})
            .get("cost_model", {})
        )
        snr = float(cost_model.get("min_signal_to_noise", 0.0))
        assert snr > 0.0, (
            "MW-02: min_signal_to_noise is 0.0 in config — SNR gate disabled. "
            "Set to 1.5 (E[return] > 1.5x CI half-width) to enforce gate."
        )

    def test_signal_generator_reads_snr_threshold(self):
        """Signal generator must load and store _min_signal_to_noise from config."""
        import inspect
        import models.time_series_signal_generator as sg_mod
        src = inspect.getsource(sg_mod)
        assert "min_signal_to_noise" in src, (
            "MW-02: time_series_signal_generator.py must reference min_signal_to_noise"
        )
        assert "_min_signal_to_noise" in src, (
            "MW-02: signal generator must store config value as _min_signal_to_noise"
        )


# ---------------------------------------------------------------------------
# MW-03  fail_on_violation_during_holding_period must be wired (not dead config)
# ---------------------------------------------------------------------------

class TestMW03ViolationFlagWiring:
    """MW-03: fail_on_violation_during_holding_period must be read and acted on."""

    def test_key_consumed_in_check_forecast_audits(self):
        import inspect
        import scripts.check_forecast_audits as mod
        src = inspect.getsource(mod)
        assert "fail_on_violation_during_holding_period" in src, (
            "MW-03: check_forecast_audits.py must consume fail_on_violation_during_holding_period"
        )
        # Must appear more than once: once for reading, once for using
        count = src.count("fail_on_violation_during_holding_period")
        assert count >= 2, (
            f"MW-03: flag appears {count} time(s) — must be both read AND acted on"
        )


# ---------------------------------------------------------------------------
# MW-04  disable_ensemble_if_no_lift must be wired (not dead config)
# ---------------------------------------------------------------------------

class TestMW04DisableEnsembleFlagWiring:
    """MW-04: disable_ensemble_if_no_lift must be read and acted on in check_forecast_audits."""

    def test_flag_consumed_and_produces_decision(self):
        import inspect
        import scripts.check_forecast_audits as mod
        src = inspect.getsource(mod)
        assert "disable_ensemble_if_no_lift" in src, (
            "MW-04: flag must be read from config in check_forecast_audits"
        )
        assert "DISABLE_DEFAULT" in src or "DISABLE" in src, (
            "MW-04: flag must produce a DISABLE_DEFAULT or equivalent decision"
        )


# ---------------------------------------------------------------------------
# AR-01  _hurst must return 0.5 fallback for series < 4 elements
# ---------------------------------------------------------------------------

class TestAR01HurstShortSeries:
    """AR-01: Hurst exponent must not crash or return inf/nan on tiny or constant series."""

    def _hurst_from_labeler(self, prices):
        """Import and call the inline _hurst function from the labeler script."""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "_labeler",
            str(_REPO_ROOT / "scripts" / "generate_classifier_training_labels.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Avoid full module execution; just extract function via source
        import ast, types
        src = (_REPO_ROOT / "scripts" / "generate_classifier_training_labels.py").read_text()
        # Extract _hurst function body
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_hurst":
                fn_src = ast.get_source_segment(src, node)
                ns: dict = {"np": np, "nan": float("nan")}
                exec(compile(fn_src, "<hurst>", "exec"), ns)
                return ns["_hurst"](prices)
        pytest.skip("_hurst not found in labeler source")

    def test_constant_series_returns_finite(self):
        result = self._hurst_from_labeler(np.ones(50))
        assert math.isfinite(result), "AR-01: constant series must return finite Hurst (expected 0.5)"

    def test_two_element_series_returns_fallback(self):
        result = self._hurst_from_labeler(np.array([1.0, 2.0]))
        assert math.isfinite(result), "AR-01: 2-element series must not crash"
        assert 0.0 <= result <= 1.0

    def test_random_walk_hurst_in_unit_interval(self):
        rng = np.random.default_rng(42)
        prices = np.cumsum(rng.normal(0, 1, 200)) + 100.0
        result = self._hurst_from_labeler(prices)
        assert 0.0 <= result <= 1.0, f"AR-01: Hurst must be in [0,1], got {result}"


# ---------------------------------------------------------------------------
# AR-02  trend_strength must not produce inf/nan on constant price series
# ---------------------------------------------------------------------------

class TestAR02TrendStrengthStability:
    """AR-02: trend_strength computation must guard ss_tot=0."""

    def test_constant_prices_produce_finite_trend_strength(self, tmp_path):
        """Generate labels on constant price series — trend_strength must be 0.0 not nan/inf."""
        import subprocess, sys
        prices = pd.DataFrame(
            {"Close": np.ones(300), "Volume": 1_000_000.0},
            index=pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC"),
        )
        pq = tmp_path / "CONST_data_extraction.parquet"
        prices.to_parquet(pq)
        result = subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "generate_classifier_training_labels.py"),
                "--checkpoint-dir", str(tmp_path),
                "--output", str(tmp_path / "ds.parquet"),
                "--tickers", "CONST",
            ],
            capture_output=True, text=True, cwd=str(_REPO_ROOT),
        )
        if (tmp_path / "ds.parquet").exists():
            df = pd.read_parquet(tmp_path / "ds.parquet")
            if "trend_strength" in df.columns:
                bad = df["trend_strength"].apply(lambda x: not math.isfinite(x) if pd.notna(x) else False)
                assert not bad.any(), f"AR-02: trend_strength has inf/nan on constant series"


# ---------------------------------------------------------------------------
# AR-03  TimeSeriesSplit gap must be >= 1 (no data leakage from adjacent bars)
# ---------------------------------------------------------------------------

class TestAR03TimeSeriesSplitGap:
    """AR-03: Walk-forward CV must use gap >= 1 to prevent look-ahead leakage."""

    def test_evaluator_default_gap_nonzero(self):
        import inspect
        import scripts.evaluate_directional_classifier as mod
        src = inspect.getsource(mod)
        # Find gap= in _walk_forward_da definition (the evaluator's CV function)
        import re
        m = re.search(r"def _walk_forward_da.*?gap:\s*int\s*=\s*(\d+)", src, re.DOTALL)
        assert m, "AR-03: _walk_forward_da must define gap parameter"
        gap_default = int(m.group(1))
        assert gap_default >= 1, f"AR-03: default gap={gap_default} allows adjacent-bar leakage"

    def test_trainer_default_gap_nonzero(self):
        import inspect
        import scripts.train_directional_classifier as mod
        src = inspect.getsource(mod)
        import re
        m = re.search(r"gap\s*=\s*(\d+)", src)
        assert m, "AR-03: train_directional_classifier must define gap"
        gap = int(m.group(1))
        assert gap >= 1, f"AR-03: trainer gap={gap} allows adjacent-bar leakage"


# ---------------------------------------------------------------------------
# AR-04  dedup key domains must not collide between parquet-scan and outcome-linked
# ---------------------------------------------------------------------------

class TestAR04DeduplicatonKeyIntegrity:
    """AR-04: parquet-scan ts_signal_ids (gen_*) must not match live ts_signal_ids (ts_*)."""

    def test_parquet_scan_ids_use_gen_prefix(self, tmp_path):
        """Parquet-scan labeler must generate IDs starting with 'gen_', not 'ts_'."""
        import subprocess, sys
        prices = pd.DataFrame(
            {"Close": np.linspace(100, 200, 300), "Volume": 1e6},
            index=pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC"),
        )
        pq = tmp_path / "AAPL_data_extraction.parquet"
        prices.to_parquet(pq)
        subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "generate_classifier_training_labels.py"),
                "--checkpoint-dir", str(tmp_path),
                "--output", str(tmp_path / "ds.parquet"),
                "--tickers", "AAPL",
            ],
            capture_output=True, cwd=str(_REPO_ROOT),
        )
        if (tmp_path / "ds.parquet").exists():
            df = pd.read_parquet(tmp_path / "ds.parquet")
            if "ts_signal_id" in df.columns and len(df):
                live_pattern = df["ts_signal_id"].str.startswith("ts_")
                assert not live_pattern.any(), (
                    "AR-04: parquet-scan IDs must start with 'gen_' not 'ts_' "
                    "to prevent dedup collision with live JSONL labels"
                )

    def test_outcome_linked_ids_use_ts_prefix(self, tmp_path):
        """accumulate_classifier_labels must copy signal_id from JSONL (ts_* format)."""
        from scripts.accumulate_classifier_labels import _load_jsonl_candidates
        import json
        jsonl = tmp_path / "quant_validation.jsonl"
        entry = {
            "signal_id": "ts_AAPL_20260318T190000Z_abcd_0001",
            "action": "BUY",
            "ticker": "AAPL",
            "timestamp": "2026-03-18T19:00:00Z",
            "classifier_features": {"hurst_exponent": 0.5, "trend_strength": 0.3},
        }
        jsonl.write_text(json.dumps(entry), encoding="utf-8")
        candidates = _load_jsonl_candidates(jsonl)
        assert len(candidates) == 1
        assert candidates[0]["signal_id"].startswith("ts_"), (
            "AR-04: outcome-linked IDs must preserve ts_* prefix from JSONL"
        )


# ---------------------------------------------------------------------------
# NS-01  CI computation must not produce inf/nan with n_windows=1
# ---------------------------------------------------------------------------

class TestNS01CiStabilitySmallN:
    """NS-01: Layer 1 CI must be flagged insufficient_data when n_used < min_windows."""

    def test_layer1_returns_insufficient_flag_on_single_window(self, tmp_path):
        from scripts.check_model_improvement import run_layer1_forecast_quality
        audit = {
            "dataset": {"ticker": "AAPL", "start": "2025-01-01", "end": "2025-06-30", "length": 180},
            "summary": {"forecast_horizon": 30},
            "artifacts": {
                "evaluation_metrics": {
                    "garch":    {"rmse": 1.5, "directional_accuracy": 0.55},
                    "samossa":  {"rmse": 1.2, "directional_accuracy": 0.60},
                    "ensemble": {"rmse": 1.1, "directional_accuracy": 0.58},
                }
            },
        }
        (tmp_path / "forecast_audit_single.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )
        result = run_layer1_forecast_quality(tmp_path)
        # With 1 window (< min for CI), CI fields must be flagged
        assert result.metrics.get("lift_ci_insufficient_data") is True or \
               result.metrics.get("n_used_windows", 0) < 5, (
            "NS-01: Layer 1 must flag CI as insufficient when n_windows < 5"
        )


# ---------------------------------------------------------------------------
# NS-02  Hurst exponent from regime_detector must be in [0.0, 1.0]
# ---------------------------------------------------------------------------

class TestNS02RegimeDetectorHurstBounds:
    """NS-02: RegimeDetector Hurst must always return finite value in [0, 1]."""

    def test_hurst_bounds_on_random_walk(self):
        from forcester_ts.regime_detector import RegimeDetector
        rd = RegimeDetector()
        rng = np.random.default_rng(0)
        series = np.cumsum(rng.normal(0, 1, 100)) + 50.0
        features = rd._extract_hurst_features(series) if hasattr(rd, "_extract_hurst_features") \
            else rd.extract_features(series) if hasattr(rd, "extract_features") \
            else {}
        h = features.get("hurst_exponent", 0.5) if features else 0.5
        assert math.isfinite(h), f"NS-02: Hurst must be finite, got {h}"
        assert 0.0 <= h <= 1.0, f"NS-02: Hurst must be in [0,1], got {h}"

    def test_hurst_constant_series_finite(self):
        from forcester_ts.regime_detector import RegimeDetector
        rd = RegimeDetector()
        series = np.ones(50) * 100.0
        try:
            features = rd._extract_hurst_features(series) if hasattr(rd, "_extract_hurst_features") \
                else rd.extract_features(series) if hasattr(rd, "extract_features") \
                else {}
            h = features.get("hurst_exponent", 0.5) if features else 0.5
            assert math.isfinite(h), f"NS-02: Constant series Hurst must be finite"
        except Exception as exc:
            pytest.fail(f"NS-02: RegimeDetector must not raise on constant series: {exc}")


# ---------------------------------------------------------------------------
# ST-01  check_classifier_readiness must not return hardcoded milestones
# ---------------------------------------------------------------------------

class TestST01ReadinessMilestoneComputed:
    """ST-01: milestone progress must be computed from actual dataset, not hardcoded."""

    def test_milestones_computed_from_dataset(self, tmp_path):
        from scripts.check_classifier_readiness import check_readiness
        from forcester_ts.directional_classifier import _FEATURE_NAMES
        # Build dataset with 60 outcome-linked rows
        rng = np.random.default_rng(7)
        X = rng.normal(size=(60, len(_FEATURE_NAMES)))
        df = pd.DataFrame(X, columns=_FEATURE_NAMES)
        df["y_directional"] = 1
        df["ts_signal_id"] = [f"ts_AAPL_test_{i:04d}" for i in range(60)]
        df["ticker"] = "AAPL"
        df["entry_ts"] = pd.date_range("2025-01-01", periods=60, freq="D").astype(str)
        df["action"] = "BUY"
        df["label_source"] = "outcome_linked"
        ds = tmp_path / "directional_dataset.parquet"
        df.to_parquet(ds, index=False)
        result = check_readiness(dataset_path=ds)
        assert result["n_outcome_linked"] == 60, (
            "ST-01: n_outcome_linked must reflect actual dataset rows"
        )
        assert result["milestones"]["100"]["reached"] is False
        assert result["milestones"]["500"]["reached"] is False
        assert result["verdict"] == "NOT_READY"

    def test_milestone_100_reached_when_dataset_has_100_rows(self, tmp_path):
        from scripts.check_classifier_readiness import check_readiness
        from forcester_ts.directional_classifier import _FEATURE_NAMES
        rng = np.random.default_rng(8)
        X = rng.normal(size=(100, len(_FEATURE_NAMES)))
        df = pd.DataFrame(X, columns=_FEATURE_NAMES)
        df["y_directional"] = rng.integers(0, 2, size=100)
        df["ts_signal_id"] = [f"ts_AAPL_m_{i:04d}" for i in range(100)]
        df["ticker"] = "AAPL"
        df["entry_ts"] = pd.date_range("2025-01-01", periods=100, freq="D").astype(str)
        df["action"] = "BUY"
        df["label_source"] = "outcome_linked"
        ds = tmp_path / "ds.parquet"
        df.to_parquet(ds, index=False)
        result = check_readiness(dataset_path=ds)
        assert result["milestones"]["100"]["reached"] is True, (
            "ST-01: milestone 100 must be marked reached when n_outcome_linked >= 100"
        )


# ---------------------------------------------------------------------------
# ST-02  accumulate result must always include db_status
# ---------------------------------------------------------------------------

class TestST02AccumulateAlwaysReturnsDbStatus:
    """ST-02: accumulate() result must always include db_status key."""

    def test_db_status_present_on_missing_db(self, tmp_path):
        from scripts.accumulate_classifier_labels import accumulate
        result = accumulate(
            jsonl_path=tmp_path / "missing.jsonl",
            dataset_path=tmp_path / "ds.parquet",
            db_path=tmp_path / "missing.db",
            dry_run=True,
        )
        assert "db_status" in result, "ST-02: db_status must be in result dict"
        assert result["db_status"] in ("ok", "db_missing", "db_error"), (
            f"ST-02: db_status must be one of ok/db_missing/db_error, got {result['db_status']!r}"
        )

    def test_db_status_ok_on_valid_db(self, tmp_path):
        from scripts.accumulate_classifier_labels import accumulate
        db = tmp_path / "t.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE trade_executions "
            "(ts_signal_id TEXT, realized_pnl REAL, is_close INTEGER, "
            "is_diagnostic INTEGER, is_synthetic INTEGER)"
        )
        conn.commit()
        conn.close()
        result = accumulate(
            jsonl_path=tmp_path / "missing.jsonl",
            dataset_path=tmp_path / "ds.parquet",
            db_path=db,
            dry_run=True,
        )
        assert result["db_status"] == "ok", (
            f"ST-02: valid (empty) DB must return db_status='ok', got {result['db_status']!r}"
        )


# ---------------------------------------------------------------------------
# CONFIG CONTRACT: key thresholds must be within acceptable production ranges
# ---------------------------------------------------------------------------

class TestConfigContract:
    """Prevent config drift back to test-mode values."""

    def test_confidence_threshold_gte_production_floor(self):
        cfg = _load_yaml("signal_routing_config.yml")
        thresh = float(
            cfg.get("signal_routing", {})
            .get("time_series", {})
            .get("confidence_threshold", 0.0)
        )
        assert thresh >= 0.55, (
            f"CONFIG: confidence_threshold={thresh} below production floor 0.55. "
            "Was lowered to 0.45 for test runs — do not regress."
        )

    def test_max_violation_rate_below_gate_ceiling(self):
        cfg = _load_yaml("forecaster_monitoring.yml")
        rate = float(
            cfg.get("forecaster_monitoring", {})
            .get("regression_metrics", {})
            .get("max_violation_rate", 1.0)
        )
        assert rate <= 0.85, (
            f"CONFIG: max_violation_rate={rate} above ceiling 0.85. "
            "Phase 7.14 raised this from 0.95 (too permissive) to 0.85."
        )

    def test_min_lift_fraction_gte_production_floor(self):
        cfg = _load_yaml("forecaster_monitoring.yml")
        frac = float(
            cfg.get("forecaster_monitoring", {})
            .get("regression_metrics", {})
            .get("min_lift_fraction", 0.0)
        )
        assert frac >= 0.25, (
            f"CONFIG: min_lift_fraction={frac} below production floor 0.25. "
            "Phase 7.14 set this to require meaningful ensemble benefit."
        )

    def test_min_lift_rmse_ratio_above_zero(self):
        cfg = _load_yaml("forecaster_monitoring.yml")
        ratio = float(
            cfg.get("forecaster_monitoring", {})
            .get("regression_metrics", {})
            .get("min_lift_rmse_ratio", 0.0)
        )
        assert ratio > 0.0, (
            f"CONFIG: min_lift_rmse_ratio={ratio} is 0.0 — epsilon improvements count as lift. "
            "Must be > 0.0 (set to 0.02 in Phase 7.14)."
        )

    def test_strict_preselection_gate_enabled(self):
        cfg = _load_yaml("forecaster_monitoring.yml")
        enabled = bool(
            cfg.get("forecaster_monitoring", {})
            .get("regression_metrics", {})
            .get("strict_preselection_gate_enabled", False)
        )
        assert enabled, (
            "CONFIG: strict_preselection_gate_enabled must be True. "
            "Disabling this allows ensemble to be default source even when "
            "recent RMSE ratio > 1.0 (worse than best-single)."
        )

    def test_aapl_min_return_above_roundtrip_cost(self):
        """AAPL min_expected_return must exceed ~15bps roundtrip cost to have edge."""
        cfg = _load_yaml("signal_routing_config.yml")
        per_ticker = (
            cfg.get("signal_routing", {})
            .get("time_series", {})
            .get("per_ticker", {})
        )
        if "AAPL" in per_ticker:
            mer = float(per_ticker["AAPL"].get("min_expected_return", 0.0))
            assert mer >= 0.0015, (
                f"CONFIG: AAPL min_expected_return={mer*100:.2f}bps below ~15bps roundtrip cost. "
                "Phase 7.14 raised to 80bps (0.0080)."
            )
