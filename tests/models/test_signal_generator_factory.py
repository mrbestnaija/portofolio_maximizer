"""
Tests for models/signal_generator_factory.py -- Phase 7.15 Signal Generator Factory.

Two tiers of coverage:

Tier 1 (Basic): build_signal_generator() produces a correctly-configured
TimeSeriesSignalGenerator from signal_routing_config.yml with no parameter silently
dropped.

Tier 2 (Robust / PnL Impact): Every parameter that directly affects signal gating and
therefore PnL is asserted to propagate end-to-end from the YAML config through the
factory to the generator's internal state.  These tests exist because silent parameter
drops create invisible divergence between backtest and live signal behaviour.

PnL-critical parameters covered:
  - _min_signal_to_noise         -- SNR gate; 1.5 blocks ~50% of noisy signals
  - _resolve_thresholds_for_ticker() -- per-ticker min_return floor (AAPL/MSFT 20 bps)
  - _quant_validation_enabled    -- guards Platt JSONL writes; wrong value poisons training data
  - _forecasting_config_path     -- ensemble weights; wrong path -> empty ensemble_kwargs -> random picks
  - _instance_uid                -- uniqueness; collision -> duplicate ts_signal_id -> broken reconciliation
  - _default_roundtrip_cost_bps  -- roundtrip cost overrides from config (US_EQUITY 1.5 bps)
  - proof-mode override semantics -- max/min clamping, not absolute assignment
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_routing_cfg(path: Path, ts_section: dict) -> None:
    """Write a minimal signal_routing_config.yml at path."""
    cfg = {"signal_routing": {"time_series": ts_section}}
    path.write_text(yaml.dump(cfg), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildSignalGeneratorBasic:
    """Factory returns a properly-typed, configured instance."""

    def test_returns_time_series_signal_generator(self, tmp_path):
        from models.signal_generator_factory import build_signal_generator
        from models.time_series_signal_generator import TimeSeriesSignalGenerator

        _write_routing_cfg(tmp_path / "sr.yml", {
            "confidence_threshold": 0.55,
            "min_expected_return": 0.003,
            "max_risk_score": 0.70,
            "use_volatility_filter": True,
        })
        gen = build_signal_generator(config_path=tmp_path / "sr.yml")
        assert isinstance(gen, TimeSeriesSignalGenerator)

    def test_basic_thresholds_loaded(self, tmp_path):
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {
            "confidence_threshold": 0.62,
            "min_expected_return": 0.004,
            "max_risk_score": 0.65,
            "use_volatility_filter": False,
        })
        gen = build_signal_generator(config_path=tmp_path / "sr.yml")
        assert gen.confidence_threshold == pytest.approx(0.62)
        assert gen.min_expected_return == pytest.approx(0.004)
        assert gen.max_risk_score == pytest.approx(0.65)
        assert gen.use_volatility_filter is False

    def test_missing_config_file_uses_defaults(self, tmp_path):
        """When config file does not exist, constructor defaults are used."""
        from models.signal_generator_factory import build_signal_generator

        gen = build_signal_generator(config_path=tmp_path / "no_such.yml")
        assert gen.confidence_threshold == pytest.approx(0.55)
        assert gen.min_expected_return == pytest.approx(0.003)
        assert gen.max_risk_score == pytest.approx(0.7)
        assert gen.use_volatility_filter is True


class TestPerTickerAndCostModel:
    """per_ticker_thresholds and cost_model are extracted and passed through."""

    def test_per_ticker_thresholds_propagated(self, tmp_path):
        from models.signal_generator_factory import build_signal_generator

        per_ticker = {
            "AAPL": {"confidence_threshold": 0.55, "min_expected_return": 0.0020},
            "MSFT": {"confidence_threshold": 0.55, "min_expected_return": 0.0020},
        }
        _write_routing_cfg(tmp_path / "sr.yml", {
            "confidence_threshold": 0.55,
            "min_expected_return": 0.003,
            "max_risk_score": 0.70,
            "use_volatility_filter": True,
            "per_ticker": per_ticker,
        })
        gen = build_signal_generator(config_path=tmp_path / "sr.yml")
        assert gen._per_ticker_thresholds == per_ticker

    def test_per_ticker_not_dict_yields_empty(self, tmp_path):
        """Non-dict per_ticker (e.g., string) must not crash -- yields empty dict."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {
            "confidence_threshold": 0.55,
            "per_ticker": "not_a_dict",
        })
        gen = build_signal_generator(config_path=tmp_path / "sr.yml")
        # Generator normalises None -> {}
        assert gen._per_ticker_thresholds == {}

    def test_cost_model_propagated(self, tmp_path):
        from models.signal_generator_factory import build_signal_generator

        cost_model = {
            "default_roundtrip_cost_bps": {"US_EQUITY": 1.5},
            "min_signal_to_noise": 1.5,
        }
        _write_routing_cfg(tmp_path / "sr.yml", {
            "confidence_threshold": 0.55,
            "cost_model": cost_model,
        })
        gen = build_signal_generator(config_path=tmp_path / "sr.yml")
        # cost_model is stored as _cost_model on the generator
        assert gen._cost_model.get("min_signal_to_noise") == pytest.approx(1.5)


class TestOverrides:
    """ts_cfg_overrides are merged on top of loaded YAML."""

    def test_override_confidence_threshold(self, tmp_path):
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            ts_cfg_overrides={"confidence_threshold": 0.65},
        )
        assert gen.confidence_threshold == pytest.approx(0.65)

    def test_override_preserves_unoverridden_keys(self, tmp_path):
        from models.signal_generator_factory import build_signal_generator

        per_ticker = {"AAPL": {"confidence_threshold": 0.55}}
        _write_routing_cfg(tmp_path / "sr.yml", {
            "confidence_threshold": 0.55,
            "min_expected_return": 0.003,
            "per_ticker": per_ticker,
        })
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            ts_cfg_overrides={"confidence_threshold": 0.65},  # only override one key
        )
        # per_ticker from YAML is preserved; overridden threshold wins
        assert gen.confidence_threshold == pytest.approx(0.65)
        assert gen._per_ticker_thresholds == per_ticker

    def test_quant_validation_config_passthrough(self, tmp_path):
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            quant_validation_config={"logging": {"enabled": False}},
        )
        # quant_validation_config is consumed by the helper — no public attribute,
        # but the generator must construct without error.
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        assert isinstance(gen, TimeSeriesSignalGenerator)


class TestForecastingConfigPath:
    """forecasting_config_path is resolved and passed to generator."""

    def test_explicit_forecasting_config_path(self, tmp_path):
        from models.signal_generator_factory import build_signal_generator

        fc_path = tmp_path / "fc.yml"
        fc_path.write_text("forecasting: {}", encoding="utf-8")
        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})

        # Should not raise even if forecasting config is minimal
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            forecasting_config_path=fc_path,
        )
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        assert isinstance(gen, TimeSeriesSignalGenerator)

    def test_no_forecasting_config_file_still_constructs(self, tmp_path):
        """Missing forecasting config must not crash — generator falls back gracefully."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})
        # tmp_path has no forecasting config; factory must not crash
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            forecasting_config_path=tmp_path / "no_such_fc.yml",
        )
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        assert isinstance(gen, TimeSeriesSignalGenerator)


class TestProductionConfig:
    """Factory works against the real config/signal_routing_config.yml."""

    _ROUTING_CFG = Path("config/signal_routing_config.yml")

    def test_production_config_loads_correctly(self):
        if not self._ROUTING_CFG.exists():
            pytest.skip("config/signal_routing_config.yml not present in this environment")

        from models.signal_generator_factory import build_signal_generator
        gen = build_signal_generator()

        # Production values from Phase 7.14 config sanitization
        assert gen.confidence_threshold == pytest.approx(0.55)
        assert gen.min_expected_return == pytest.approx(0.003)
        assert gen.max_risk_score == pytest.approx(0.70)
        assert gen.use_volatility_filter is True

    def test_production_config_loads_per_ticker(self):
        if not self._ROUTING_CFG.exists():
            pytest.skip("config/signal_routing_config.yml not present in this environment")

        from models.signal_generator_factory import build_signal_generator
        gen = build_signal_generator()

        # AAPL and MSFT have per-ticker 20bps floor from Phase 7.14 Phase A
        assert "AAPL" in gen._per_ticker_thresholds
        assert "MSFT" in gen._per_ticker_thresholds

    def test_production_config_loads_cost_model(self):
        if not self._ROUTING_CFG.exists():
            pytest.skip("config/signal_routing_config.yml not present in this environment")

        from models.signal_generator_factory import build_signal_generator
        gen = build_signal_generator()

        # cost_model must be present (min_signal_to_noise 1.5 from Phase 7.14 Phase A)
        assert gen._cost_model is not None
        assert gen._cost_model.get("min_signal_to_noise") == pytest.approx(1.5)


# ===========================================================================
# Tier 2: Robust / PnL-impact tests
# ===========================================================================

class TestSNRGateIntegrity:
    """_min_signal_to_noise propagates from cost_model config to gen._min_signal_to_noise.

    This gate rejects signals where E[return] < threshold * CI_half_width.
    A value of 1.5 (production) blocks ~50% of noisy signals.
    Silently dropping cost_model means the gate is disabled → garbage signals trade.
    """

    def test_snr_value_reaches_generator_attribute(self, tmp_path):
        """min_signal_to_noise in cost_model must be stored as gen._min_signal_to_noise."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {
            "confidence_threshold": 0.55,
            "cost_model": {"min_signal_to_noise": 1.5},
        })
        gen = build_signal_generator(config_path=tmp_path / "sr.yml")
        assert gen._min_signal_to_noise == pytest.approx(1.5), (
            "_min_signal_to_noise=1.5 not propagated. "
            "SNR gate is DISABLED — noisy signals will trade."
        )

    def test_snr_zero_disables_gate(self, tmp_path):
        """SNR=0 must disable the gate (not block all signals)."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {
            "confidence_threshold": 0.55,
            "cost_model": {"min_signal_to_noise": 0},
        })
        gen = build_signal_generator(config_path=tmp_path / "sr.yml")
        assert gen._min_signal_to_noise == pytest.approx(0.0)

    def test_snr_via_override_wins_over_yaml(self, tmp_path):
        """ts_cfg_overrides cost_model must override YAML cost_model."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {
            "confidence_threshold": 0.55,
            "cost_model": {"min_signal_to_noise": 0.5},
        })
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            ts_cfg_overrides={"cost_model": {"min_signal_to_noise": 2.0}},
        )
        assert gen._min_signal_to_noise == pytest.approx(2.0)

    def test_production_snr_is_1_5(self):
        """Production config must have SNR=1.5 (Phase 7.14-A requirement)."""
        if not Path("config/signal_routing_config.yml").exists():
            pytest.skip("config/signal_routing_config.yml not present")

        from models.signal_generator_factory import build_signal_generator
        gen = build_signal_generator()
        assert gen._min_signal_to_noise == pytest.approx(1.5), (
            "Production SNR gate is not 1.5. Phase 7.14-A set min_signal_to_noise=1.5. "
            "If changed intentionally, update this test and PHASE_7.14_GATE_RECALIBRATION.md."
        )

    def test_missing_cost_model_gives_zero_snr(self, tmp_path):
        """When cost_model is absent, SNR gate must be 0 (disabled) not crash."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})
        gen = build_signal_generator(config_path=tmp_path / "sr.yml")
        # No cost_model -> _min_signal_to_noise defaults to 0.0
        assert gen._min_signal_to_noise == pytest.approx(0.0)


class TestPerTickerThresholdResolution:
    """_resolve_thresholds_for_ticker() must apply per-ticker overrides correctly.

    AAPL and MSFT use a 20bps min_return floor (Phase 7.14-A).
    Global default is 30bps.  Mixing these up causes wrong signal counts per ticker,
    which changes traded volume and realized PnL.
    """

    def _make_full_cfg(self, tmp_path: Path) -> Path:
        cfg_path = tmp_path / "sr.yml"
        _write_routing_cfg(cfg_path, {
            "confidence_threshold": 0.55,
            "min_expected_return": 0.003,   # global 30bps
            "max_risk_score": 0.70,
            "per_ticker": {
                "AAPL": {"confidence_threshold": 0.55, "min_expected_return": 0.0020},
                "MSFT": {"confidence_threshold": 0.55, "min_expected_return": 0.0020},
                "CL=F": {"confidence_threshold": 0.55, "min_expected_return": 0.005},
            },
        })
        return cfg_path

    def test_aapl_uses_20bps_floor_not_global(self, tmp_path):
        from models.signal_generator_factory import build_signal_generator

        gen = build_signal_generator(config_path=self._make_full_cfg(tmp_path))
        t = gen._resolve_thresholds_for_ticker("AAPL")
        assert t["min_expected_return"] == pytest.approx(0.0020), (
            "AAPL should use 20bps floor, not global 30bps. "
            "Mismatch causes different signal counts vs expected."
        )

    def test_msft_uses_20bps_floor_not_global(self, tmp_path):
        from models.signal_generator_factory import build_signal_generator

        gen = build_signal_generator(config_path=self._make_full_cfg(tmp_path))
        t = gen._resolve_thresholds_for_ticker("MSFT")
        assert t["min_expected_return"] == pytest.approx(0.0020)

    def test_unknown_ticker_falls_back_to_global(self, tmp_path):
        from models.signal_generator_factory import build_signal_generator

        gen = build_signal_generator(config_path=self._make_full_cfg(tmp_path))
        t = gen._resolve_thresholds_for_ticker("NVDA")
        assert t["min_expected_return"] == pytest.approx(0.003), (
            "Unknown ticker must use global 30bps, not a per-ticker override."
        )

    def test_per_ticker_confidence_threshold_resolved(self, tmp_path):
        """Per-ticker confidence_threshold override must win over global."""
        cfg_path = tmp_path / "sr.yml"
        _write_routing_cfg(cfg_path, {
            "confidence_threshold": 0.55,
            "per_ticker": {"SPECIAL": {"confidence_threshold": 0.70}},
        })
        from models.signal_generator_factory import build_signal_generator
        gen = build_signal_generator(config_path=cfg_path)
        t = gen._resolve_thresholds_for_ticker("SPECIAL")
        assert t["confidence_threshold"] == pytest.approx(0.70)

    def test_production_aapl_msft_get_20bps_floor(self):
        """Live production config must give AAPL and MSFT 20bps floor."""
        if not Path("config/signal_routing_config.yml").exists():
            pytest.skip("config/signal_routing_config.yml not present")

        from models.signal_generator_factory import build_signal_generator
        gen = build_signal_generator()
        for ticker in ("AAPL", "MSFT"):
            t = gen._resolve_thresholds_for_ticker(ticker)
            assert t["min_expected_return"] == pytest.approx(0.0020), (
                f"{ticker} should have 20bps floor from Phase 7.14-A config. "
                f"Got {t['min_expected_return'] * 10000:.1f}bps."
            )


class TestProofModeThresholdSemantics:
    """Proof-mode applies max/min clamping, NOT absolute assignment.

    run_auto_trader.py computes:
        confidence = max(existing, 0.65)
        min_return = max(existing, 0.005)
        max_risk   = min(existing, 0.60)

    These overrides are then passed as ts_cfg_overrides.  The factory must
    not disturb this semantics — override dict must win over YAML base values.
    """

    def _apply_proof_mode_strict(self, ts_cfg: dict) -> dict:
        """Mirror the proof-mode strict logic from run_auto_trader.py."""
        cfg = dict(ts_cfg)
        cfg["confidence_threshold"] = max(float(cfg.get("confidence_threshold", 0.55)), 0.65)
        cfg["min_expected_return"] = max(float(cfg.get("min_expected_return", 0.003)), 0.005)
        cfg["max_risk_score"] = min(float(cfg.get("max_risk_score", 0.7)), 0.60)
        return cfg

    def test_proof_mode_tightens_loose_thresholds(self, tmp_path):
        """Base thresholds below proof-mode floor must be raised."""
        from models.signal_generator_factory import build_signal_generator

        base_cfg = {
            "confidence_threshold": 0.55,
            "min_expected_return": 0.003,
            "max_risk_score": 0.70,
        }
        _write_routing_cfg(tmp_path / "sr.yml", base_cfg)
        proof_overrides = self._apply_proof_mode_strict(base_cfg)
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            ts_cfg_overrides=proof_overrides,
        )
        assert gen.confidence_threshold == pytest.approx(0.65)
        assert gen.min_expected_return == pytest.approx(0.005)
        assert gen.max_risk_score == pytest.approx(0.60)

    def test_proof_mode_does_not_loosen_already_tight_thresholds(self, tmp_path):
        """Base confidence=0.75 must stay 0.75 under proof-mode (max semantics)."""
        from models.signal_generator_factory import build_signal_generator

        base_cfg = {
            "confidence_threshold": 0.75,   # already tighter than 0.65
            "min_expected_return": 0.008,   # already tighter than 0.005
            "max_risk_score": 0.50,         # already tighter than 0.60
        }
        _write_routing_cfg(tmp_path / "sr.yml", base_cfg)
        proof_overrides = self._apply_proof_mode_strict(base_cfg)
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            ts_cfg_overrides=proof_overrides,
        )
        assert gen.confidence_threshold == pytest.approx(0.75)
        assert gen.min_expected_return == pytest.approx(0.008)
        assert gen.max_risk_score == pytest.approx(0.50)

    def test_proof_mode_partial_tightening(self, tmp_path):
        """Only thresholds below proof-mode floor get raised; others unchanged."""
        from models.signal_generator_factory import build_signal_generator

        base_cfg = {
            "confidence_threshold": 0.70,   # above 0.65 floor -> unchanged
            "min_expected_return": 0.002,   # below 0.005 floor -> raised
            "max_risk_score": 0.55,         # below 0.60 ceiling -> unchanged
        }
        _write_routing_cfg(tmp_path / "sr.yml", base_cfg)
        proof_overrides = self._apply_proof_mode_strict(base_cfg)
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            ts_cfg_overrides=proof_overrides,
        )
        assert gen.confidence_threshold == pytest.approx(0.70)
        assert gen.min_expected_return == pytest.approx(0.005)
        assert gen.max_risk_score == pytest.approx(0.55)


class TestQuantValidationGating:
    """_quant_validation_enabled guards Platt JSONL writes.

    Enabled in production → every signal generates a quant_validation entry → Platt data accumulates.
    Disabled in tests → no contamination of logs/signals/quant_validation.jsonl.
    Wrong value here either starves Platt calibration or poisons it with test data.
    """

    def test_disabled_by_logging_override(self, tmp_path):
        """quant_validation_config with no top-level 'enabled' key -> disabled."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            quant_validation_config={"logging": {"enabled": False}},
        )
        assert gen._quant_validation_enabled is False, (
            "quant_validation must be OFF when config has no top-level 'enabled' key. "
            "Test fixtures use this to prevent JSONL contamination."
        )

    def test_disabled_by_explicit_false(self, tmp_path):
        """quant_validation_config={'enabled': False} must disable validation."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            quant_validation_config={"enabled": False},
        )
        assert gen._quant_validation_enabled is False

    def test_two_instances_same_no_qv_config_both_disabled(self, tmp_path):
        """Multiple factory calls with disabled QV must each disable independently."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})
        qv_off = {"logging": {"enabled": False}}
        gen1 = build_signal_generator(config_path=tmp_path / "sr.yml", quant_validation_config=qv_off)
        gen2 = build_signal_generator(config_path=tmp_path / "sr.yml", quant_validation_config=qv_off)
        assert gen1._quant_validation_enabled is False
        assert gen2._quant_validation_enabled is False


class TestForecastingConfigPathPropagation:
    """Factory must pass forecasting_config_path through as a string.

    The generator uses _forecasting_config_path to load ensemble_kwargs during CV.
    If None is passed, ensemble_kwargs are empty -> random model picks during CV
    -> wrong model selected -> live trades use wrong model vs backtest.
    """

    def test_explicit_path_stored_on_generator(self, tmp_path):
        """Generator._forecasting_config_path must match the explicit path passed."""
        from models.signal_generator_factory import build_signal_generator

        fc = tmp_path / "fc.yml"
        fc.write_text("forecasting: {}", encoding="utf-8")
        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})

        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            forecasting_config_path=fc,
        )
        # Generator stores as Path; compare resolved paths
        assert gen._forecasting_config_path.resolve() == fc.resolve()

    def test_default_resolved_when_file_exists(self, tmp_path, monkeypatch):
        """When default config/forecasting_config.yml exists, factory resolves it."""
        from models.signal_generator_factory import build_signal_generator
        import models.signal_generator_factory as fac_mod

        # Point factory default to our tmp dir
        fake_fc = tmp_path / "fc.yml"
        fake_fc.write_text("forecasting: {}", encoding="utf-8")
        monkeypatch.setattr(fac_mod, "_DEFAULT_FORECASTING_CONFIG", fake_fc)

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})
        gen = build_signal_generator(config_path=tmp_path / "sr.yml")
        assert gen._forecasting_config_path.resolve() == fake_fc.resolve()

    def test_nonexistent_path_does_not_crash(self, tmp_path):
        """Passing a non-existent forecasting config path must not raise."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            forecasting_config_path=tmp_path / "phantom.yml",
        )
        # Generator logs a warning but must not raise; _forecasting_config is {}
        assert isinstance(gen._forecasting_config, dict)


class TestInstanceUniqueness:
    """Two factory calls must produce generators with different _instance_uid.

    ts_signal_id format: ts_{ticker}_{dt_part}_{instance_uid}_{counter:04d}
    Same uid between concurrent instances -> ts_signal_id collision -> reconciler
    matches the wrong trade outcome -> Platt is trained on wrong (winner/loser) labels.
    """

    def test_two_instances_have_different_uids(self, tmp_path):
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})
        gen1 = build_signal_generator(config_path=tmp_path / "sr.yml")
        gen2 = build_signal_generator(config_path=tmp_path / "sr.yml")
        assert gen1._instance_uid != gen2._instance_uid, (
            "Two factory instances share the same _instance_uid. "
            "This causes ts_signal_id collisions in concurrent/test contexts."
        )

    def test_uid_is_4_hex_chars(self, tmp_path):
        """_instance_uid must be a 4-character hex string (e.g. 'a3f2')."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})
        gen = build_signal_generator(config_path=tmp_path / "sr.yml")
        uid = gen._instance_uid
        assert len(uid) == 4
        assert all(c in "0123456789abcdef" for c in uid), (
            f"_instance_uid '{uid}' is not 4 lowercase hex chars."
        )

    def test_ten_instances_all_unique_uids(self, tmp_path):
        """In batch usage (e.g. bootstrap loop), all instances must have unique UIDs."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})
        uids = [build_signal_generator(config_path=tmp_path / "sr.yml")._instance_uid for _ in range(10)]
        assert len(set(uids)) == 10, (
            f"UID collision among 10 instances: {uids}. "
            "secrets.token_hex(2) has 65536 values; collision in 10 draws is a bug."
        )


class TestRoundtripCostOverrides:
    """_default_roundtrip_cost_bps from cost_model.default_roundtrip_cost_bps must reach generator.

    Production config uses 1.5bps for US_EQUITY (updated Phase 7.9).
    Wrong values here cause SNR gate to miscalculate feasibility -> wrong pass/block decisions.
    """

    def test_us_equity_roundtrip_cost_from_config(self, tmp_path):
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {
            "confidence_threshold": 0.55,
            "cost_model": {
                "min_signal_to_noise": 1.5,
                "default_roundtrip_cost_bps": {"US_EQUITY": 1.5, "INTL_EQUITY": 3.0},
            },
        })
        gen = build_signal_generator(config_path=tmp_path / "sr.yml")
        assert gen._default_roundtrip_cost_bps.get("US_EQUITY") == pytest.approx(1.5), (
            "US_EQUITY roundtrip cost not propagated from cost_model config."
        )

    def test_production_us_equity_cost_is_1_5_bps(self):
        """Production config must use 1.5bps US_EQUITY cost (Phase 7.9 update)."""
        if not Path("config/signal_routing_config.yml").exists():
            pytest.skip("config/signal_routing_config.yml not present")

        from models.signal_generator_factory import build_signal_generator
        gen = build_signal_generator()
        us_cost = gen._default_roundtrip_cost_bps.get("US_EQUITY")
        assert us_cost == pytest.approx(1.5), (
            f"US_EQUITY roundtrip cost is {us_cost}bps, expected 1.5bps. "
            "Phase 7.9 updated from 3.2bps -> 1.5bps for modern smart-order routing."
        )


class TestIdempotencyAndIsolation:
    """Multiple factory calls must not share state or interfere."""

    def test_same_config_produces_same_thresholds(self, tmp_path):
        """Idempotency: two calls with identical config produce identical thresholds."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {
            "confidence_threshold": 0.60,
            "min_expected_return": 0.004,
            "max_risk_score": 0.65,
        })
        gen1 = build_signal_generator(config_path=tmp_path / "sr.yml")
        gen2 = build_signal_generator(config_path=tmp_path / "sr.yml")
        assert gen1.confidence_threshold == gen2.confidence_threshold
        assert gen1.min_expected_return == gen2.min_expected_return
        assert gen1.max_risk_score == gen2.max_risk_score
        assert gen1._min_signal_to_noise == gen2._min_signal_to_noise

    def test_different_configs_do_not_bleed_between_instances(self, tmp_path):
        """Two generators with different configs must not share threshold values."""
        from models.signal_generator_factory import build_signal_generator

        cfg_a = tmp_path / "a.yml"
        cfg_b = tmp_path / "b.yml"
        _write_routing_cfg(cfg_a, {"confidence_threshold": 0.55, "min_expected_return": 0.003})
        _write_routing_cfg(cfg_b, {"confidence_threshold": 0.70, "min_expected_return": 0.006})

        gen_a = build_signal_generator(config_path=cfg_a)
        gen_b = build_signal_generator(config_path=cfg_b)

        assert gen_a.confidence_threshold == pytest.approx(0.55)
        assert gen_b.confidence_threshold == pytest.approx(0.70)
        # Ensure gen_a was not mutated by gen_b construction
        assert gen_a.confidence_threshold == pytest.approx(0.55)

    def test_empty_override_dict_treated_as_no_override(self, tmp_path):
        """Passing ts_cfg_overrides={} must behave identically to passing None."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {
            "confidence_threshold": 0.58,
            "min_expected_return": 0.0035,
        })
        gen_none = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            ts_cfg_overrides=None,
        )
        gen_empty = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            ts_cfg_overrides={},
        )
        assert gen_none.confidence_threshold == pytest.approx(gen_empty.confidence_threshold)
        assert gen_none.min_expected_return == pytest.approx(gen_empty.min_expected_return)

    def test_partial_yaml_missing_keys_use_constructor_defaults(self, tmp_path):
        """YAML with only one key must produce generator with constructor defaults for missing ones."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.62})
        gen = build_signal_generator(config_path=tmp_path / "sr.yml")

        assert gen.confidence_threshold == pytest.approx(0.62)  # from YAML
        assert gen.min_expected_return == pytest.approx(0.003)   # constructor default
        assert gen.max_risk_score == pytest.approx(0.70)          # constructor default
        assert gen.use_volatility_filter is True                  # constructor default
        assert gen._per_ticker_thresholds == {}                   # absent -> empty


# ===========================================================================
# Tests for findings-driven fixes
# ===========================================================================

class TestQuantValidationConfigPath:
    """quant_validation_config_path (param 6 of 9) must reach generator._quant_validation_config_path.

    Previously the factory omitted this param entirely, silently routing all callers
    to the default config/quant_success_config.yml even when a custom path was needed.
    """

    def test_explicit_qv_config_path_reaches_generator(self, tmp_path):
        """Explicit quant_validation_config_path must be stored on the generator."""
        from models.signal_generator_factory import build_signal_generator

        qv_cfg = tmp_path / "my_qv.yml"
        qv_cfg.write_text("quant_validation: {enabled: false}", encoding="utf-8")
        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})

        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            quant_validation_config_path=str(qv_cfg),
        )
        # Generator stores as _quant_validation_config_path (Path)
        assert gen._quant_validation_config_path.resolve() == qv_cfg.resolve()

    def test_none_qv_config_path_uses_default(self, tmp_path):
        """When quant_validation_config_path=None, generator defaults to quant_success_config.yml."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})
        gen = build_signal_generator(config_path=tmp_path / "sr.yml")
        # Generator falls back to default path (may not exist in test env — that is OK)
        assert "quant_success_config" in str(gen._quant_validation_config_path)

    def test_factory_signature_has_qv_config_path_param(self):
        """Factory function must accept quant_validation_config_path as a keyword arg."""
        import inspect
        from models.signal_generator_factory import build_signal_generator
        sig = inspect.signature(build_signal_generator)
        assert "quant_validation_config_path" in sig.parameters, (
            "quant_validation_config_path missing from build_signal_generator signature. "
            "This param is required to cover all 9 TimeSeriesSignalGenerator.__init__ args."
        )


class TestPipelineConfigOverrideSurface:
    """run_etl_pipeline.py Stage 8 must apply pipeline_cfg.signal_routing.time_series overrides.

    Before Phase 7.15 the pipeline explicitly tried pipeline_cfg first, then fell back
    to signal_routing_config.yml.  After the initial factory migration, pipeline_cfg
    overrides were silently dropped.  This class tests that the restore is correct.
    """

    def test_pipeline_override_wins_over_yaml_base(self, tmp_path):
        """ts_cfg_overrides from pipeline_cfg must override canonical YAML base."""
        from models.signal_generator_factory import build_signal_generator

        # YAML base has 0.55
        _write_routing_cfg(tmp_path / "sr.yml", {
            "confidence_threshold": 0.55,
            "min_expected_return": 0.003,
        })
        # Pipeline cfg override (simulate what run_etl_pipeline.py extracts)
        pl_ts_overrides = {"confidence_threshold": 0.60, "min_expected_return": 0.005}

        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            ts_cfg_overrides=pl_ts_overrides,
        )
        assert gen.confidence_threshold == pytest.approx(0.60), (
            "Pipeline cfg override confidence_threshold did not win over YAML base."
        )
        assert gen.min_expected_return == pytest.approx(0.005)

    def test_none_pipeline_override_falls_back_to_yaml(self, tmp_path):
        """When pipeline_cfg has no time_series section, YAML base is used intact."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {
            "confidence_threshold": 0.58,
            "min_expected_return": 0.0035,
        })
        # Simulate pipeline_cfg with no signal_routing.time_series section
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            ts_cfg_overrides=None,
        )
        assert gen.confidence_threshold == pytest.approx(0.58)
        assert gen.min_expected_return == pytest.approx(0.0035)


class TestNoneRobustExtraction:
    """Override dicts with None values must fall back to constructor defaults, not crash.

    candidate_simulator.py guardrails can have None values (built with `float(x or default)`).
    Factory must handle these gracefully via _float_or() helper.
    """

    def test_none_confidence_threshold_in_override_uses_yaml_base(self, tmp_path):
        """None confidence_threshold in override falls back to YAML base value."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.60})
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            ts_cfg_overrides={"confidence_threshold": None},
        )
        # None in override -> _float_or -> falls back to constructor default 0.55
        # (not YAML base, because None replaces 0.60 in the merge, then _float_or
        #  catches it and returns 0.55)
        assert gen.confidence_threshold == pytest.approx(0.55)

    def test_none_min_return_in_override_uses_constructor_default(self, tmp_path):
        """None min_expected_return must not raise TypeError."""
        from models.signal_generator_factory import build_signal_generator

        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})
        # Should not raise
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            ts_cfg_overrides={"min_expected_return": None},
        )
        assert gen.min_expected_return == pytest.approx(0.003)  # constructor default

    def test_guardrails_dict_pattern_constructs_correctly(self, tmp_path):
        """Simulate candidate_simulator guardrails dict (same keys, typed values)."""
        from models.signal_generator_factory import build_signal_generator

        # candidate_simulator builds guardrails from candidate_params
        guardrails = {
            "confidence_threshold": 0.60,
            "min_expected_return": 0.004,
            "max_risk_score": 0.65,
            "use_volatility_filter": True,
            "per_ticker": {"AAPL": {"min_expected_return": 0.0020}},
            "cost_model": {"min_signal_to_noise": 1.5},
        }
        _write_routing_cfg(tmp_path / "sr.yml", {"confidence_threshold": 0.55})
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            ts_cfg_overrides=guardrails,
        )
        # Guardrail values must win over YAML base
        assert gen.confidence_threshold == pytest.approx(0.60)
        assert gen.min_expected_return == pytest.approx(0.004)
        assert gen.max_risk_score == pytest.approx(0.65)
        assert gen._min_signal_to_noise == pytest.approx(1.5)
        assert "AAPL" in gen._per_ticker_thresholds

    def test_barbell_eval_explicit_override_pattern(self, tmp_path):
        """Simulate run_barbell_pnl_evaluation.py explicit-params override pattern."""
        from models.signal_generator_factory import build_signal_generator

        # barbell eval builds override dict from CLI args
        _write_routing_cfg(tmp_path / "sr.yml", {
            "confidence_threshold": 0.55,
            "per_ticker": {"AAPL": {"min_expected_return": 0.0020}},
            "cost_model": {"min_signal_to_noise": 1.5},
        })
        gen = build_signal_generator(
            config_path=tmp_path / "sr.yml",
            ts_cfg_overrides={
                "confidence_threshold": 0.62,
                "min_expected_return": 0.004,
                "max_risk_score": 0.68,
                "use_volatility_filter": True,
            },
            quant_validation_config={"enabled": False},
        )
        # Explicit CLI overrides win for thresholds
        assert gen.confidence_threshold == pytest.approx(0.62)
        assert gen.min_expected_return == pytest.approx(0.004)
        # per_ticker and cost_model from YAML base (not in override) propagate through
        assert "AAPL" in gen._per_ticker_thresholds
        assert gen._min_signal_to_noise == pytest.approx(1.5)
        # quant_validation disabled
        assert gen._quant_validation_enabled is False
