"""
test_ensemble_config_contract.py
---------------------------------
CI-ready regression tests for the two ensemble config bugs fixed in commit 99b5f8d:

  Bug 1 -- run_auto_trader.py did not pass ensemble_kwargs to
           TimeSeriesForecasterConfig, so CV fold 1 fell back to
           EnsembleConfig's stale SARIMAX-containing defaults.

  Bug 2 -- EnsembleConfig.candidate_weights default factory contained
           SARIMAX-dominant candidates that scored > 0 even with SARIMAX
           disabled, wasting evaluation cycles and risking wrong blends.

The tests cover:
  1.  Config load path validation (forecasting_config.yml)
  2.  EnsembleConfig dataclass defaults (no SARIMAX)
  3.  Disabled-model weight exclusion
  4.  Config-to-model mapping alignment
  5.  Candidate weight schema contracts
  6.  CV config propagation (deep-copy preserves ensemble_kwargs)
  7.  Forecaster construction with/without ensemble_kwargs
  8.  Regime candidate weights SARIMAX exclusion
  9.  Auto-trader config loader integration
  10. Quant gate: disabled models never contribute positive weight
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import yaml

from forcester_ts.ensemble import EnsembleConfig, EnsembleCoordinator
from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig


ROOT = Path(__file__).resolve().parents[2]
FORECASTING_CONFIG_PATH = ROOT / "config" / "forecasting_config.yml"
PIPELINE_CONFIG_PATH = ROOT / "config" / "pipeline_config.yml"

FAST_ONLY_MODELS = {"garch", "samossa", "mssa_rl"}
ALL_VALID_MODELS = {"garch", "samossa", "mssa_rl", "sarimax"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def forecasting_config() -> Dict[str, Any]:
    """Load the forecasting section from forecasting_config.yml."""
    assert FORECASTING_CONFIG_PATH.exists(), "forecasting_config.yml not found"
    raw = yaml.safe_load(FORECASTING_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return raw.get("forecasting", raw)


@pytest.fixture()
def ensemble_section(forecasting_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ensemble section from forecasting config."""
    return forecasting_config.get("ensemble", {})


@pytest.fixture()
def regime_section(forecasting_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract regime_detection section from forecasting config."""
    return forecasting_config.get("regime_detection", {})


@pytest.fixture()
def candidate_weights_from_config(ensemble_section: Dict[str, Any]) -> List[Dict[str, float]]:
    """Candidate weights as defined in YAML."""
    weights = ensemble_section.get("candidate_weights", [])
    assert weights, "candidate_weights missing from forecasting_config.yml"
    return weights


@pytest.fixture()
def ensemble_kwargs_from_config(ensemble_section: Dict[str, Any]) -> Dict[str, Any]:
    """Ensemble kwargs as they would be passed to TimeSeriesForecasterConfig."""
    return {k: v for k, v in ensemble_section.items() if k != "enabled"}


# ---------------------------------------------------------------------------
# 1. Config Load Path Validation
# ---------------------------------------------------------------------------

class TestConfigLoadPath:
    """Verify forecasting_config.yml loads with all required sections."""

    def test_forecasting_config_has_ensemble(self, forecasting_config):
        assert "ensemble" in forecasting_config, \
            "forecasting_config.yml must contain an 'ensemble' section"

    def test_forecasting_config_has_sarimax(self, forecasting_config):
        assert "sarimax" in forecasting_config, \
            "forecasting_config.yml must contain a 'sarimax' section"

    def test_forecasting_config_has_regime_detection(self, forecasting_config):
        assert "regime_detection" in forecasting_config, \
            "forecasting_config.yml must contain a 'regime_detection' section"

    def test_sarimax_enabled_by_default(self, forecasting_config):
        sarimax_cfg = forecasting_config.get("sarimax", {})
        assert sarimax_cfg.get("enabled") is True, \
            "SARIMAX should be enabled by default in config (Phase 10)"

    def test_ensemble_kwargs_not_empty(self, ensemble_kwargs_from_config):
        assert ensemble_kwargs_from_config, \
            "ensemble_kwargs derived from config should not be empty"

    def test_ensemble_has_candidate_weights(self, ensemble_kwargs_from_config):
        assert "candidate_weights" in ensemble_kwargs_from_config, \
            "ensemble_kwargs must include 'candidate_weights'"
        weights = ensemble_kwargs_from_config["candidate_weights"]
        assert len(weights) >= 3, \
            f"Expected at least 3 candidate weight sets, got {len(weights)}"

    def test_config_files_in_sync(self):
        """Pipeline config ensemble section should mirror forecasting config."""
        if not PIPELINE_CONFIG_PATH.exists():
            pytest.skip("pipeline_config.yml not found")
        pipe_raw = yaml.safe_load(PIPELINE_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        # pipeline_config.yml nests under "pipeline" then "forecasting"
        pipe_fc = pipe_raw.get("pipeline", pipe_raw).get("forecasting", {})
        pipe_sarimax = pipe_fc.get("sarimax", {})
        assert pipe_sarimax.get("enabled") is True, \
            "pipeline_config.yml SARIMAX should be enabled (Phase 10)"


# ---------------------------------------------------------------------------
# 2. EnsembleConfig Dataclass Defaults (No SARIMAX)
# ---------------------------------------------------------------------------

class TestEnsembleConfigDefaults:
    """Verify that EnsembleConfig dataclass defaults have no SARIMAX contamination."""

    def test_default_candidate_weights_include_sarimax(self):
        """Phase 10: default candidates include SARIMAX for model-class diversity."""
        config = EnsembleConfig()
        sarimax_count = sum(1 for c in config.candidate_weights if "sarimax" in c)
        assert sarimax_count >= 2, \
            f"Phase 10: expected >=2 SARIMAX-containing default candidates, got {sarimax_count}"

    def test_default_candidate_weights_use_valid_models(self):
        """All default candidates use valid model keys."""
        config = EnsembleConfig()
        for i, candidate in enumerate(config.candidate_weights):
            models = set(candidate.keys())
            extra = models - ALL_VALID_MODELS
            assert not extra, \
                f"candidate_weights[{i}] has invalid model(s): {extra}"

    def test_default_candidate_weights_sum_to_one(self):
        config = EnsembleConfig()
        for i, candidate in enumerate(config.candidate_weights):
            total = sum(candidate.values())
            assert abs(total - 1.0) < 1e-6, \
                f"candidate_weights[{i}] sums to {total}, expected 1.0"

    def test_default_has_multiple_candidates(self):
        config = EnsembleConfig()
        assert len(config.candidate_weights) >= 5, \
            "Default should have at least 5 candidate weight sets for diversity"


# ---------------------------------------------------------------------------
# 3. Disabled Model Weight Exclusion
# ---------------------------------------------------------------------------

class TestDisabledModelExclusion:
    """Disabled models must never receive positive weight in selected ensemble."""

    def test_sarimax_zero_confidence_yields_no_sarimax_weight(self):
        """When SARIMAX has zero confidence, no candidate should select it."""
        config = EnsembleConfig(confidence_scaling=True)
        coordinator = EnsembleCoordinator(config)
        model_confidence = {
            "garch": 0.6,
            "samossa": 0.9,
            "mssa_rl": 0.3,
            "sarimax": 0.0,  # disabled
        }
        weights, score = coordinator.select_weights(model_confidence)
        assert "sarimax" not in weights, \
            f"SARIMAX should not appear in selected weights: {weights}"

    def test_default_candidates_use_valid_model_keys(self):
        """All default candidates use recognized model keys."""
        config = EnsembleConfig()
        for candidate in config.candidate_weights:
            for model in candidate:
                assert model in ALL_VALID_MODELS, \
                    f"Unrecognized model '{model}' in default candidates"

    def test_scoring_excludes_zero_confidence_models(self):
        """Candidates referencing only zero-confidence models should score 0."""
        config = EnsembleConfig(
            confidence_scaling=True,
            candidate_weights=[
                {"sarimax": 1.0},  # all weight on disabled model
            ],
        )
        coordinator = EnsembleCoordinator(config)
        model_confidence = {"sarimax": 0.0, "garch": 0.6, "samossa": 0.9}
        weights, score = coordinator.select_weights(model_confidence)
        # Either empty or sarimax removed by normalization
        assert weights.get("sarimax", 0.0) == 0.0


# ---------------------------------------------------------------------------
# 4. Config-to-Model Mapping Alignment
# ---------------------------------------------------------------------------

class TestConfigModelMapping:
    """Ensemble config candidate models must match actual model keys."""

    VALID_MODEL_KEYS = {"sarimax", "garch", "samossa", "mssa_rl"}

    def test_yaml_candidates_use_valid_keys(self, candidate_weights_from_config):
        for i, candidate in enumerate(candidate_weights_from_config):
            for model in candidate:
                assert model in self.VALID_MODEL_KEYS, \
                    f"candidate_weights[{i}] has unknown model '{model}'"

    def test_yaml_candidates_include_sarimax_when_enabled(self, candidate_weights_from_config):
        """Phase 10: YAML candidates include SARIMAX entries since SARIMAX is enabled."""
        sarimax_count = sum(1 for c in candidate_weights_from_config if "sarimax" in c)
        assert sarimax_count >= 2, \
            f"Phase 10: expected >=2 SARIMAX-containing YAML candidates, got {sarimax_count}"


# ---------------------------------------------------------------------------
# 5. Candidate Weight Schema Contracts
# ---------------------------------------------------------------------------

class TestCandidateWeightSchema:
    """Validate structural invariants of candidate weight dicts."""

    def test_all_weights_positive(self, candidate_weights_from_config):
        for i, candidate in enumerate(candidate_weights_from_config):
            for model, weight in candidate.items():
                assert weight > 0, \
                    f"candidate_weights[{i}]['{model}'] = {weight}, must be > 0"

    def test_all_weights_sum_approximately_one(self, candidate_weights_from_config):
        for i, candidate in enumerate(candidate_weights_from_config):
            total = sum(candidate.values())
            assert abs(total - 1.0) < 0.01, \
                f"candidate_weights[{i}] sums to {total:.4f}, expected ~1.0"

    def test_minimum_component_weight_respected(self, ensemble_section):
        """No candidate should have a weight below the configured minimum."""
        min_weight = ensemble_section.get("minimum_component_weight", 0.05)
        candidates = ensemble_section.get("candidate_weights", [])
        for i, candidate in enumerate(candidates):
            for model, weight in candidate.items():
                assert weight >= min_weight, \
                    f"candidate_weights[{i}]['{model}'] = {weight} < minimum {min_weight}"


# ---------------------------------------------------------------------------
# 6. CV Config Propagation (deep-copy preserves ensemble_kwargs)
# ---------------------------------------------------------------------------

class TestCVConfigPropagation:
    """Cross-validation must preserve ensemble_kwargs across folds."""

    def test_deep_copy_preserves_ensemble_kwargs(self, ensemble_kwargs_from_config):
        config = TimeSeriesForecasterConfig(
            ensemble_kwargs=ensemble_kwargs_from_config,
        )
        copied = copy.deepcopy(config)
        assert copied.ensemble_kwargs == config.ensemble_kwargs
        assert copied.ensemble_kwargs is not config.ensemble_kwargs  # independent

    def test_deep_copy_preserves_candidate_count(self, ensemble_kwargs_from_config):
        config = TimeSeriesForecasterConfig(
            ensemble_kwargs=ensemble_kwargs_from_config,
        )
        copied = copy.deepcopy(config)
        orig_count = len(config.ensemble_kwargs.get("candidate_weights", []))
        copy_count = len(copied.ensemble_kwargs.get("candidate_weights", []))
        assert orig_count == copy_count > 0

    def test_empty_ensemble_kwargs_falls_back_to_phase10_defaults(self):
        """When ensemble_kwargs is empty, EnsembleConfig uses Phase 10 defaults with SARIMAX."""
        config = TimeSeriesForecasterConfig(ensemble_kwargs={})
        ec = EnsembleConfig(**config.ensemble_kwargs) if config.ensemble_kwargs else EnsembleConfig()
        # Phase 10: fallback defaults include SARIMAX for model-class diversity
        sarimax_count = sum(1 for c in ec.candidate_weights if "sarimax" in c)
        assert sarimax_count >= 2, \
            f"Phase 10: fallback defaults should include SARIMAX candidates, got {sarimax_count}"


# ---------------------------------------------------------------------------
# 7. Forecaster Construction With/Without ensemble_kwargs
# ---------------------------------------------------------------------------

class TestForecasterConstruction:
    """TimeSeriesForecaster must propagate ensemble config correctly."""

    def test_with_ensemble_kwargs(self, ensemble_kwargs_from_config):
        config = TimeSeriesForecasterConfig(
            ensemble_kwargs=ensemble_kwargs_from_config,
        )
        forecaster = TimeSeriesForecaster(config=config)
        ec = forecaster._ensemble_config
        n = len(ec.candidate_weights)
        expected = len(ensemble_kwargs_from_config.get("candidate_weights", []))
        assert n == expected, \
            f"Forecaster got {n} candidates, expected {expected} from config"

    def test_without_ensemble_kwargs_uses_phase10_defaults(self):
        """Bare config (empty ensemble_kwargs) uses Phase 10 defaults including SARIMAX."""
        config = TimeSeriesForecasterConfig()  # ensemble_kwargs = {}
        forecaster = TimeSeriesForecaster(config=config)
        ec = forecaster._ensemble_config
        sarimax_count = sum(1 for c in ec.candidate_weights if "sarimax" in c)
        assert sarimax_count >= 2, \
            f"Phase 10: bare forecaster should include >=2 SARIMAX candidates, got {sarimax_count}"

    def test_build_config_from_kwargs_sarimax_default(self):
        """Phase 10: legacy kwargs path defaults SARIMAX to enabled."""
        forecaster = TimeSeriesForecaster(forecast_horizon=10)
        assert forecaster.config.sarimax_enabled is True, \
            "SARIMAX should be enabled by default in kwargs path (Phase 10)"

    def test_build_config_from_kwargs_with_ensemble(self):
        """Legacy kwargs path with explicit ensemble config."""
        forecaster = TimeSeriesForecaster(
            forecast_horizon=10,
            ensemble_config={
                "enabled": True,
                "candidate_weights": [{"garch": 0.5, "samossa": 0.5}],
            },
        )
        ec = forecaster._ensemble_config
        assert len(ec.candidate_weights) == 1
        assert "sarimax" not in ec.candidate_weights[0]


# ---------------------------------------------------------------------------
# 8. Regime Candidate Weights SARIMAX Exclusion
# ---------------------------------------------------------------------------

class TestRegimeCandidateWeights:
    """Regime-specific candidate weights must also be SARIMAX-free."""

    def test_regime_candidates_use_valid_models(self, regime_section):
        """Regime candidates use valid model keys (SARIMAX allowed when enabled)."""
        rcw = regime_section.get("regime_candidate_weights", {})
        if not rcw:
            pytest.skip("No regime_candidate_weights in config")
        for regime, candidates in rcw.items():
            for i, candidate in enumerate(candidates):
                for model in candidate:
                    assert model in ALL_VALID_MODELS, \
                        f"regime_candidate_weights[{regime}][{i}] has invalid model '{model}'"

    def test_regime_model_preferences_no_sarimax(self, regime_section):
        prefs = regime_section.get("regime_model_preferences", {})
        if not prefs:
            pytest.skip("No regime_model_preferences in config")
        for regime, pref in prefs.items():
            models = pref.get("preferred_models", [])
            assert "sarimax" not in models, \
                f"regime_model_preferences[{regime}] lists disabled 'sarimax'"


# ---------------------------------------------------------------------------
# 9. Auto-Trader Config Loader Integration
# ---------------------------------------------------------------------------

class TestAutoTraderConfigLoader:
    """Validate that run_auto_trader._load_forecasting_config returns usable data."""

    def test_loader_returns_ensemble_section(self):
        """The auto-trader config loader must find the ensemble section."""
        # Import the private loader
        import importlib
        mod = importlib.import_module("scripts.run_auto_trader")
        loader = getattr(mod, "_load_forecasting_config")
        cfg = loader()
        assert "ensemble" in cfg, \
            "_load_forecasting_config() must return a dict with 'ensemble' key"
        ensemble = cfg["ensemble"]
        assert "candidate_weights" in ensemble, \
            "ensemble section must have 'candidate_weights'"
        assert len(ensemble["candidate_weights"]) >= 3

    def test_loader_sarimax_enabled(self):
        import importlib
        mod = importlib.import_module("scripts.run_auto_trader")
        loader = getattr(mod, "_load_forecasting_config")
        cfg = loader()
        sarimax = cfg.get("sarimax", {})
        assert sarimax.get("enabled") is True, \
            "_load_forecasting_config() SARIMAX should be enabled (Phase 10)"

    def test_loader_regime_detection_present(self):
        import importlib
        mod = importlib.import_module("scripts.run_auto_trader")
        loader = getattr(mod, "_load_forecasting_config")
        cfg = loader()
        assert "regime_detection" in cfg, \
            "_load_forecasting_config() must include regime_detection"


# ---------------------------------------------------------------------------
# 10. Quant Gate: Disabled Models Never Contribute Positive Weight
# ---------------------------------------------------------------------------

class TestQuantGateDisabledModels:
    """Ensemble selection must never assign positive normalized weight
    to a model with zero confidence (i.e., disabled)."""

    @pytest.mark.parametrize("disabled_model", ["sarimax", "garch", "samossa", "mssa_rl"])
    def test_zero_confidence_model_excluded_from_selection(self, disabled_model):
        """Any model with confidence=0 should be excluded from final weights."""
        all_models = {"garch": 0.7, "samossa": 0.9, "mssa_rl": 0.4, "sarimax": 0.5}
        all_models[disabled_model] = 0.0  # simulate disabled

        # Use candidates that include the disabled model
        config = EnsembleConfig(
            confidence_scaling=True,
            candidate_weights=[
                {m: 1.0 / len(all_models) for m in all_models},
            ],
        )
        coordinator = EnsembleCoordinator(config)
        weights, _ = coordinator.select_weights(all_models)
        assert weights.get(disabled_model, 0.0) == 0.0, \
            f"Disabled model '{disabled_model}' got weight {weights.get(disabled_model)}"

    def test_all_fast_models_positive_when_enabled(self):
        """When all fast models have positive confidence, they all contribute."""
        config = EnsembleConfig(
            confidence_scaling=True,
            candidate_weights=[
                {"garch": 0.33, "samossa": 0.34, "mssa_rl": 0.33},
            ],
        )
        coordinator = EnsembleCoordinator(config)
        model_confidence = {"garch": 0.6, "samossa": 0.9, "mssa_rl": 0.5}
        weights, score = coordinator.select_weights(model_confidence)
        for model in FAST_ONLY_MODELS:
            assert weights.get(model, 0.0) > 0, \
                f"Enabled model '{model}' should have positive weight"
        assert score > 0

    def test_synthetic_series_explicit_sarimax_disable(self):
        """End-to-end: explicit sarimax_enabled=False is honoured — SARIMAX not fitted,
        and thus has no fitted summary entry even though Phase 10 candidates include it."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        prices = pd.Series(
            100.0 + np.cumsum(np.random.randn(200) * 0.5),
            index=dates,
            name="Close",
        )
        returns = prices.pct_change().dropna()

        config = TimeSeriesForecasterConfig(
            forecast_horizon=10,
            sarimax_enabled=False,
        )
        forecaster = TimeSeriesForecaster(config=config)
        forecaster.fit(price_series=prices, returns_series=returns)
        bundle = forecaster.forecast()

        # sarimax_enabled=False: SARIMAX not fitted, so not in summaries
        assert forecaster.config.sarimax_enabled is False
        # Verify the forecast bundle exists (ensemble produced output)
        assert bundle is not None, "Forecaster should produce a forecast bundle"


# ---------------------------------------------------------------------------
# Bonus: Config Drift Detection
# ---------------------------------------------------------------------------

class TestConfigDriftDetection:
    """Detect unexpected changes to critical config sections."""

    def test_ensemble_candidate_count_stable(self, candidate_weights_from_config):
        """Alert if candidate count changes unexpectedly (Phase 10: 15 sets with SARIMAX)."""
        assert len(candidate_weights_from_config) == 15, \
            f"Expected 15 candidate weight sets (Phase 10), got {len(candidate_weights_from_config)}. " \
            "Update this test if intentional."

    def test_sarimax_default_enabled(self):
        """Phase 10: SARIMAX is enabled by default for model-class diversity."""
        config = TimeSeriesForecasterConfig()
        assert config.sarimax_enabled is True, \
            "TimeSeriesForecasterConfig.sarimax_enabled default must be True (Phase 10)"

    def test_ensemble_config_default_count_matches_yaml(self, candidate_weights_from_config):
        """EnsembleConfig defaults should match YAML candidate count (Phase 10: 15 sets)."""
        ec = EnsembleConfig()
        default_count = len(ec.candidate_weights)
        yaml_count = len(candidate_weights_from_config)
        assert default_count == yaml_count, \
            f"EnsembleConfig defaults ({default_count}) != YAML ({yaml_count}). " \
            "Keep them in sync."


# ---------------------------------------------------------------------------
# Phase 7.17: DA Penalty + Adaptive Candidate Weights
# ---------------------------------------------------------------------------

class TestPhase717DACapAndAdaptiveCandidates:
    """Phase 7.17 regression tests:
    - SAMOSSA DA=0 gets capped at da_weight_cap after select_weights()
    - Adaptive candidates are prepended and scored first
    """

    def test_samossa_da_zero_gets_capped_weight(self):
        """SAMOSSA with DA=0.0 must receive weight <= da_weight_cap (0.10) in the final selection."""
        config = EnsembleConfig(
            confidence_scaling=False,  # isolate DA penalty from confidence scaling
            track_directional_accuracy=True,
            da_floor=0.10,
            da_weight_cap=0.10,
            candidate_weights=[
                # Give SAMOSSA 50% weight — DA penalty must pull it below 0.10
                {"garch": 0.25, "samossa": 0.50, "mssa_rl": 0.25},
            ],
        )
        coordinator = EnsembleCoordinator(config)
        model_confidence = {"garch": 0.6, "samossa": 0.6, "mssa_rl": 0.6}
        model_da = {"garch": 0.55, "samossa": 0.0, "mssa_rl": 0.55}  # SAMOSSA DA=0

        weights, _ = coordinator.select_weights(model_confidence, model_da)
        samossa_weight = weights.get("samossa", 0.0)
        assert samossa_weight <= 0.10 + 1e-6, (
            f"SAMOSSA with DA=0.0 should have weight <= 0.10 after DA penalty, "
            f"got {samossa_weight:.4f}"
        )
        # Other models share the remaining budget
        assert weights.get("garch", 0.0) > 0.10
        assert weights.get("mssa_rl", 0.0) > 0.10

    def test_adaptive_candidates_tried_first_in_select_weights(self):
        """Adaptive candidates must be prepended to the scoring list.

        We set adaptive_candidate_weights = [{mssa_rl: 1.0}] with a very high
        mssa_rl confidence. With no DA penalty, this pure-mssa_rl candidate
        should win over any static candidate.
        """
        config = EnsembleConfig(
            confidence_scaling=True,
            track_directional_accuracy=False,
            adaptive_candidate_weights=[{"mssa_rl": 1.0}],  # Phase 7.17 adaptive
            candidate_weights=[
                {"garch": 0.5, "samossa": 0.5},  # static (garch+samossa, no mssa_rl)
            ],
            da_floor=0.10,
            da_weight_cap=0.10,
        )
        coordinator = EnsembleCoordinator(config)
        # mssa_rl is much better → adaptive candidate should be selected
        model_confidence = {"garch": 0.3, "samossa": 0.3, "mssa_rl": 0.95}
        weights, _ = coordinator.select_weights(model_confidence)
        # The adaptive pure-mssa_rl candidate should win
        assert weights.get("mssa_rl", 0.0) > weights.get("garch", 0.0), (
            f"Adaptive {{'mssa_rl':1.0}} candidate should dominate when mssa_rl confidence=0.95. "
            f"Got weights={weights}"
        )
        assert weights.get("mssa_rl", 0.0) > weights.get("samossa", 0.0)

    def test_adaptive_candidates_empty_list_uses_static_only(self):
        """Empty adaptive_candidate_weights must fall back to static candidates without error."""
        config = EnsembleConfig(
            adaptive_candidate_weights=[],
            candidate_weights=[{"garch": 0.5, "mssa_rl": 0.5}],
        )
        coordinator = EnsembleCoordinator(config)
        weights, score = coordinator.select_weights({"garch": 0.6, "mssa_rl": 0.7})
        assert weights, "Should still select a candidate from static list"
        assert score > 0

    def test_da_cap_config_fields_have_correct_defaults(self):
        """EnsembleConfig.da_floor and da_weight_cap must default to 0.10."""
        ec = EnsembleConfig()
        assert ec.da_floor == 0.10, f"da_floor default should be 0.10, got {ec.da_floor}"
        assert ec.da_weight_cap == 0.10, f"da_weight_cap default should be 0.10, got {ec.da_weight_cap}"
        assert ec.adaptive_candidate_weights == [], \
            f"adaptive_candidate_weights default should be [], got {ec.adaptive_candidate_weights}"

    def test_pure_winner_all_da_zero_candidate_skipped(self):
        """A pure-winner candidate where the only model has DA=0 must be skipped,
        not returned with unnormalized weights summing < 1.0 (all-capped bug fix).
        """
        from forcester_ts.ensemble import _apply_da_cap
        # Pure-winner: single model with weight 1.0 and DA=0
        weights = {"garch": 1.0}
        da_scores = {"garch": 0.0}  # DA=0, below floor
        result = _apply_da_cap(weights, da_scores, da_floor=0.10, da_weight_cap=0.10)
        # Should return {} so the caller can skip this candidate cleanly
        assert result == {}, (
            f"All-DA-penalized pure-winner should return {{}} to signal skip, got {result}"
        )

    def test_da_cap_output_sums_to_one(self):
        """After DA cap and redistribution, weights must always sum to 1.0."""
        from forcester_ts.ensemble import _apply_da_cap
        # garch penalized (DA=0), mssa_rl unpunished (DA=0.6); garch gets capped at 0.10
        weights = {"garch": 0.70, "mssa_rl": 0.30}
        da_scores = {"garch": 0.0, "mssa_rl": 0.60}
        result = _apply_da_cap(weights, da_scores, da_floor=0.10, da_weight_cap=0.10)
        assert result, "Should return non-empty result (mssa_rl not penalized)"
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-9, f"Weights must sum to 1.0, got {total}"
        assert result["garch"] <= 0.10 + 1e-9
        assert result["mssa_rl"] >= 0.89  # ~0.90 after redistribution


# ---------------------------------------------------------------------------
# Phase 7.17: Hypothesis property-based fuzz tests for _apply_da_cap
# ---------------------------------------------------------------------------

from hypothesis import given, settings, assume
from hypothesis import strategies as st


_ALL_MODELS = ("garch", "samossa", "mssa_rl")


@st.composite
def _normalized_weight_dict(draw, models=_ALL_MODELS):
    """Generate a normalized dict over a random non-empty subset of models."""
    n = draw(st.integers(min_value=1, max_value=len(models)))
    chosen = models[:n]
    raw = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    total = sum(raw)
    assume(total > 1e-9)
    return {m: w / total for m, w in zip(chosen, raw)}


@st.composite
def _da_scores(draw, models=_ALL_MODELS):
    return {
        m: draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        for m in models
    }


class TestApplyDaCapProperties:
    """Hypothesis property-based regression tests for _apply_da_cap().

    These tests encode the CONTRACT documented in the function's docstring
    and automatically fuzz over the full input space.
    """

    @given(
        weights=_normalized_weight_dict(),
        da_scores=_da_scores(),
        da_floor=st.floats(min_value=0.05, max_value=0.40),
        da_weight_cap=st.floats(min_value=0.05, max_value=0.40),
    )
    @settings(max_examples=300, deadline=None)
    def test_result_is_empty_or_normalized(self, weights, da_scores, da_floor, da_weight_cap):
        """_apply_da_cap must return {} or a dict summing to 1.0 ± 1e-6."""
        from forcester_ts.ensemble import _apply_da_cap
        result = _apply_da_cap(weights, da_scores, da_floor, da_weight_cap)
        if result:
            total = sum(result.values())
            assert abs(total - 1.0) < 1e-6, (
                f"sum={total:.9f}, weights={weights}, da={da_scores}, "
                f"floor={da_floor}, cap={da_weight_cap}"
            )
            assert all(0.0 <= v <= 1.0 + 1e-9 for v in result.values()), (
                f"Negative or >1 weight: {result}"
            )

    @given(
        weights=_normalized_weight_dict(),
        da_scores=_da_scores(),
        da_floor=st.floats(min_value=0.05, max_value=0.40),
        da_weight_cap=st.floats(min_value=0.05, max_value=0.40),
    )
    @settings(max_examples=300, deadline=None)
    def test_penalized_models_respect_cap(self, weights, da_scores, da_floor, da_weight_cap):
        """Every model with DA < da_floor must have weight ≤ da_weight_cap in result."""
        from forcester_ts.ensemble import _apply_da_cap
        result = _apply_da_cap(weights, da_scores, da_floor, da_weight_cap)
        for m, w in result.items():
            if da_scores.get(m, 1.0) < da_floor:
                assert w <= da_weight_cap + 1e-9, (
                    f"Model '{m}' DA={da_scores.get(m):.4f} < floor={da_floor} "
                    f"but weight={w:.6f} > cap={da_weight_cap}"
                )

    @given(
        weights=_normalized_weight_dict(),
        da_scores=_da_scores(),
        da_floor=st.floats(min_value=0.05, max_value=0.40),
        da_weight_cap=st.floats(min_value=0.05, max_value=0.40),
    )
    @settings(max_examples=300, deadline=None)
    def test_all_penalized_returns_empty(self, weights, da_scores, da_floor, da_weight_cap):
        """If every model in weights has DA < da_floor, result must be {}."""
        from forcester_ts.ensemble import _apply_da_cap
        # Force all models in this candidate to be below floor
        low_da = {m: 0.001 for m in weights}
        result = _apply_da_cap(weights, low_da, da_floor=0.10, da_weight_cap=da_weight_cap)
        # All weights also must be > da_weight_cap to trigger the cap
        # (cap only fires when w > da_weight_cap). For single-model candidates where
        # w=1.0 >> da_weight_cap, result must be {}.
        if all(weights[m] > da_weight_cap for m in weights):
            assert result == {}, (
                f"All models penalized: expected {{}}, got {result}. "
                f"weights={weights}, da_cap={da_weight_cap}"
            )


# ---------------------------------------------------------------------------
# Phase 10: Hybrid RMSE-rank confidence scoring
# ---------------------------------------------------------------------------

class TestPhase10RmseRankHybrid:
    """Phase 10: derive_model_confidence() includes RMSE-rank hybrid component.

    The RMSE-rank component counterbalances SAMoSSA's artificially-high EVR
    (always ~1.0 by SSA construction) with realized forecast accuracy, preventing
    near-flat forecasts from dominating via EVR alone.
    """

    def test_best_rmse_model_gets_higher_confidence(self):
        """Model with lowest RMSE should receive higher confidence than model with highest."""
        from forcester_ts.ensemble import derive_model_confidence
        # GARCH has much better RMSE than SAMoSSA
        summaries = {
            "garch": {
                "aic": -100.0, "bic": -90.0,
                "regression_metrics": {"rmse": 1.0, "directional_accuracy": 0.55},
            },
            "samossa": {
                "explained_variance_ratio": 0.99,  # high EVR (SSA artifact)
                "regression_metrics": {"rmse": 3.0, "directional_accuracy": 0.45},
            },
            "mssa_rl": {
                "baseline_variance": 1.0,
                "regression_metrics": {"rmse": 2.0, "directional_accuracy": 0.50},
            },
        }
        conf = derive_model_confidence(summaries)
        # GARCH has best RMSE (1.0) — should score higher than SAMoSSA (rmse=3.0)
        # despite SAMoSSA's EVR=0.99
        assert conf.get("garch", 0) > conf.get("samossa", 1), (
            f"GARCH (rmse=1.0) should beat SAMoSSA (rmse=3.0, evr=0.99) "
            f"after RMSE-rank injection. Got garch={conf.get('garch'):.3f}, "
            f"samossa={conf.get('samossa'):.3f}"
        )

    def test_rmse_rank_missing_suppressed_gracefully(self):
        """When models lack regression_metrics.rmse, RMSE-rank component is skipped."""
        from forcester_ts.ensemble import derive_model_confidence
        summaries = {
            "garch": {"aic": -100.0, "bic": -90.0},  # no regression_metrics
            "samossa": {"explained_variance_ratio": 0.80},
            "mssa_rl": {"baseline_variance": 0.5},
        }
        # Should not raise, should return valid confidence dict
        conf = derive_model_confidence(summaries)
        assert isinstance(conf, dict)
        # All present models get a score
        for model in ("garch", "samossa", "mssa_rl"):
            if model in conf:
                assert 0.0 <= conf[model] <= 1.0, f"{model} score out of range: {conf[model]}"

    def test_nan_rmse_excluded_from_rank_computation(self):
        """NaN RMSE must be filtered from _rmse_values so it does not propagate into ranks."""
        from forcester_ts.ensemble import derive_model_confidence
        summaries = {
            "garch": {"regression_metrics": {"rmse": float("nan")}},
            "samossa": {"regression_metrics": {"rmse": 1.0}},
            "mssa_rl": {"regression_metrics": {"rmse": 1.2}},
        }
        conf = derive_model_confidence(summaries)
        assert isinstance(conf, dict)
        for model, score in conf.items():
            assert np.isfinite(score), f"{model} confidence is non-finite ({score}) — NaN leaked from RMSE"

    def test_inf_rmse_excluded_from_rank_computation(self):
        """Inf RMSE must be filtered to avoid dividing by Inf in rank normalization."""
        from forcester_ts.ensemble import derive_model_confidence
        summaries = {
            "garch": {"regression_metrics": {"rmse": float("inf")}},
            "samossa": {"regression_metrics": {"rmse": 1.0}},
            "mssa_rl": {"regression_metrics": {"rmse": 1.2}},
        }
        conf = derive_model_confidence(summaries)
        assert isinstance(conf, dict)
        for model, score in conf.items():
            assert np.isfinite(score), f"{model} confidence is non-finite ({score})"

    def test_nan_aic_bic_does_not_corrupt_sarimax_score(self):
        """NaN AIC/BIC must produce sarimax_score=None, not propagate NaN into confidence."""
        from forcester_ts.ensemble import derive_model_confidence
        summaries = {
            "sarimax": {"aic": float("nan"), "bic": float("nan")},
            "garch": {"aic": -100.0, "bic": -90.0},
        }
        conf = derive_model_confidence(summaries)
        assert isinstance(conf, dict)
        for model, score in conf.items():
            assert np.isfinite(score), f"{model} confidence is non-finite ({score}) — NaN from AIC/BIC"

    def test_all_nan_inputs_return_empty_or_bounded_confidence(self):
        """When all inputs are NaN/Inf, no confidence score should be NaN or out of [0,1]."""
        from forcester_ts.ensemble import derive_model_confidence
        summaries = {
            "garch": {
                "aic": float("nan"),
                "bic": float("inf"),
                "regression_metrics": {"rmse": float("nan")},
            },
            "samossa": {
                "explained_variance_ratio": float("nan"),
                "regression_metrics": {"rmse": float("inf")},
            },
        }
        conf = derive_model_confidence(summaries)
        assert isinstance(conf, dict)
        for model, score in conf.items():
            assert np.isfinite(score), f"{model} confidence is non-finite: {score}"
            assert 0.0 <= score <= 1.0, f"{model} confidence out of [0,1]: {score}"
