"""Unit tests for verifiable gate logging (confidence, SNR, quant).

Every gate must emit a structured record on the ``pmx.gates`` logger
with fields: gate, signal_id, ticker, value, threshold, result.

Tests verify:
  1. Gate pass/fail flips correctly at the declared threshold boundary.
  2. A structured log record is written for every gate traversal.
  3. The record's ``result`` field is PASS when value >= threshold
     and FAIL when value < threshold.
  4. ``signal_id`` in the log matches the signal object's ``signal_id``.
"""

import json
import logging
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from models.time_series_signal_generator import TimeSeriesSignalGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_forecast_bundle(
    *,
    forecast_price: float = 105.0,
    current_price: float = 100.0,
    horizon: int = 5,
) -> Dict[str, Any]:
    """Construct the smallest forecast bundle that passes primary-forecast resolution."""
    n = horizon + 1
    prices = np.linspace(current_price, forecast_price, n)
    lower = prices - 2.0
    upper = prices + 2.0
    return {
        "default_model": "SAMOSSA",
        "mean_forecast": {
            "forecast": list(prices),
            "lower_ci": list(lower),
            "upper_ci": list(upper),
            "horizon": horizon,
        },
        "execution_mode": "auto",
        "horizon": horizon,
    }


def _gate_records(caplog_records, gate: str) -> list:
    """Extract pmx.gates records matching a gate name from caplog."""
    results = []
    for r in caplog_records:
        if r.name != "pmx.gates":
            continue
        msg = r.getMessage()
        if not msg.startswith("GATE "):
            continue
        try:
            data = json.loads(msg[5:])  # strip "GATE "
        except json.JSONDecodeError:
            continue
        if data.get("gate") == gate:
            results.append(data)
    return results


def _make_generator(
    confidence_threshold: float = 0.55,
    min_expected_return: float = 0.003,
    max_risk_score: float = 0.70,
    min_snr: float = 0.0,
) -> TimeSeriesSignalGenerator:
    gen = TimeSeriesSignalGenerator(
        confidence_threshold=confidence_threshold,
        min_expected_return=min_expected_return,
        max_risk_score=max_risk_score,
        use_volatility_filter=False,
        quant_validation_config={"enabled": False},  # disable quant gate for routing tests
    )
    gen._min_signal_to_noise = min_snr
    return gen


# ---------------------------------------------------------------------------
# _determine_action — confidence gate boundary
# ---------------------------------------------------------------------------

class TestConfidenceGateBoundary:
    """Gate fires (FAIL) exactly when confidence < threshold, passes at threshold."""

    def test_confidence_equal_to_threshold_is_pass(self):
        gen = _make_generator(confidence_threshold=0.55)
        action, reason = gen._determine_action(
            expected_return=0.05,
            net_trade_return=0.04,
            confidence=0.55,
            risk_score=0.30,
            confidence_threshold=0.55,
            min_expected_return=0.001,
            max_risk_score=0.70,
        )
        assert action != "HOLD" or reason != "CONFIDENCE_BELOW_THRESHOLD", (
            "confidence == threshold must PASS the confidence gate"
        )

    def test_confidence_just_below_threshold_is_fail(self):
        gen = _make_generator(confidence_threshold=0.55)
        action, reason = gen._determine_action(
            expected_return=0.05,
            net_trade_return=0.04,
            confidence=0.5499,
            risk_score=0.30,
            confidence_threshold=0.55,
            min_expected_return=0.001,
            max_risk_score=0.70,
        )
        assert action == "HOLD"
        assert reason == "CONFIDENCE_BELOW_THRESHOLD"

    def test_confidence_well_above_threshold_is_pass(self):
        gen = _make_generator(confidence_threshold=0.55)
        action, reason = gen._determine_action(
            expected_return=0.05,
            net_trade_return=0.04,
            confidence=0.80,
            risk_score=0.30,
            confidence_threshold=0.55,
            min_expected_return=0.001,
            max_risk_score=0.70,
        )
        assert action in {"BUY", "SELL"}
        assert reason is None

    def test_confidence_gate_checked_before_return_gate(self):
        """Confidence gate fires first; min_return gate must not mask it."""
        gen = _make_generator()
        _, reason = gen._determine_action(
            expected_return=0.05,
            net_trade_return=0.04,
            confidence=0.30,  # below threshold
            risk_score=0.20,
            confidence_threshold=0.55,
            min_expected_return=0.001,
            max_risk_score=0.70,
        )
        assert reason == "CONFIDENCE_BELOW_THRESHOLD"


# ---------------------------------------------------------------------------
# _determine_action — min_return gate boundary
# ---------------------------------------------------------------------------

class TestMinReturnGateBoundary:
    def test_net_return_equal_threshold_is_pass(self):
        gen = _make_generator()
        action, reason = gen._determine_action(
            expected_return=0.05,
            net_trade_return=0.003,  # exactly at floor
            confidence=0.70,
            risk_score=0.30,
            confidence_threshold=0.55,
            min_expected_return=0.003,
            max_risk_score=0.70,
        )
        # net_trade_return + 1e-12 >= min_expected_return → PASS
        assert action in {"BUY", "SELL"}
        assert reason is None

    def test_net_return_just_below_threshold_is_fail(self):
        gen = _make_generator()
        action, reason = gen._determine_action(
            expected_return=0.05,
            net_trade_return=0.002,  # below floor
            confidence=0.70,
            risk_score=0.30,
            confidence_threshold=0.55,
            min_expected_return=0.003,
            max_risk_score=0.70,
        )
        assert action == "HOLD"
        assert reason == "MIN_RETURN"


# ---------------------------------------------------------------------------
# _log_gate_result — structured record schema
# ---------------------------------------------------------------------------

class TestLogGateResultSchema:
    """_log_gate_result emits a JSON-parseable record with all required fields."""

    def test_record_has_required_fields(self, caplog):
        with caplog.at_level(logging.INFO, logger="pmx.gates"):
            TimeSeriesSignalGenerator._log_gate_result(
                "confidence", "ts_AAPL_test_0001", "AAPL",
                value=0.62, threshold=0.55, result="PASS",
            )

        records = _gate_records(caplog.records, "confidence")
        assert len(records) == 1
        rec = records[0]
        assert rec["gate"] == "confidence"
        assert rec["signal_id"] == "ts_AAPL_test_0001"
        assert rec["ticker"] == "AAPL"
        assert rec["value"] == pytest.approx(0.62)
        assert rec["threshold"] == pytest.approx(0.55)
        assert rec["result"] == "PASS"

    def test_fail_record(self, caplog):
        with caplog.at_level(logging.INFO, logger="pmx.gates"):
            TimeSeriesSignalGenerator._log_gate_result(
                "snr", "ts_AAPL_test_0002", "AAPL",
                value=0.20, threshold=0.61, result="FAIL",
                horizon=30,
            )

        records = _gate_records(caplog.records, "snr")
        assert len(records) == 1
        assert records[0]["result"] == "FAIL"
        assert records[0]["horizon"] == 30

    def test_unknown_result_allowed(self, caplog):
        with caplog.at_level(logging.INFO, logger="pmx.gates"):
            TimeSeriesSignalGenerator._log_gate_result(
                "snr", "ts_AAPL_test_0003", "AAPL",
                value=None, threshold=0.61, result="UNKNOWN",
                gate_detail="ci_unavailable",
            )

        records = _gate_records(caplog.records, "snr")
        assert len(records) == 1
        assert records[0]["result"] == "UNKNOWN"
        assert records[0]["gate_detail"] == "ci_unavailable"

    def test_quant_criterion_record_has_criterion_field(self, caplog):
        with caplog.at_level(logging.INFO, logger="pmx.gates"):
            TimeSeriesSignalGenerator._log_gate_result(
                "quant_criterion", "ts_AAPL_test_0004", "AAPL",
                value=0.85, threshold=1.0, result="FAIL",
                criterion="omega_ratio", weight=0.20,
            )

        records = _gate_records(caplog.records, "quant_criterion")
        assert len(records) == 1
        assert records[0]["criterion"] == "omega_ratio"
        assert records[0]["weight"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Signal_id alignment: gate log signal_id == signal.signal_id
# ---------------------------------------------------------------------------

class TestSignalIdAlignment:
    """The signal_id in gate records must match signal.signal_id."""

    def _make_market_data(self, n: int = 120, price: float = 100.0) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        prices = np.full(n, price)
        return pd.DataFrame({"Close": prices, "Volume": np.full(n, 1_000_000)}, index=idx)

    def test_confidence_gate_log_uses_same_signal_id_as_signal(self, caplog):
        gen = _make_generator(confidence_threshold=0.55, min_expected_return=0.001)
        bundle = _minimal_forecast_bundle(forecast_price=105.0, current_price=100.0)
        md = self._make_market_data()

        with caplog.at_level(logging.INFO, logger="pmx.gates"):
            signal = gen.generate_signal(bundle, current_price=100.0, ticker="AAPL", market_data=md)

        confidence_records = _gate_records(caplog.records, "confidence")
        assert len(confidence_records) >= 1, "No confidence gate record emitted"
        assert confidence_records[-1]["signal_id"] == signal.signal_id, (
            f"Gate log signal_id {confidence_records[-1]['signal_id']!r} "
            f"does not match signal.signal_id {signal.signal_id!r}"
        )
        assert confidence_records[-1]["ticker"] == "AAPL"
        assert confidence_records[-1]["result"] in {"PASS", "FAIL"}

    def test_confidence_gate_fail_emits_fail_record(self, caplog):
        """When a signal is demoted to HOLD by confidence gate, record shows FAIL."""
        # Use a very high threshold so all signals fail
        gen = _make_generator(confidence_threshold=0.99, min_expected_return=0.001)
        bundle = _minimal_forecast_bundle(forecast_price=101.0, current_price=100.0)
        md = self._make_market_data()

        with caplog.at_level(logging.INFO, logger="pmx.gates"):
            signal = gen.generate_signal(bundle, current_price=100.0, ticker="AAPL", market_data=md)

        assert signal.action == "HOLD"
        conf_records = _gate_records(caplog.records, "confidence")
        assert len(conf_records) >= 1
        assert conf_records[-1]["result"] == "FAIL"

    def test_snr_gate_fail_emits_fail_record(self, caplog):
        """When SNR gate blocks a signal, record shows FAIL and gate=snr."""
        # Force SNR gate to fire: set very high threshold, give forecast small edge
        gen = _make_generator(
            confidence_threshold=0.55,
            min_expected_return=0.001,
            min_snr=999.0,  # impossibly high SNR threshold
        )
        bundle = _minimal_forecast_bundle(forecast_price=101.0, current_price=100.0)
        md = self._make_market_data()

        with caplog.at_level(logging.INFO, logger="pmx.gates"):
            signal = gen.generate_signal(bundle, current_price=100.0, ticker="AAPL", market_data=md)

        snr_records = _gate_records(caplog.records, "snr")
        assert len(snr_records) >= 1
        # At least one SNR record must be FAIL (the gate fired)
        fail_records = [r for r in snr_records if r["result"] == "FAIL"]
        assert len(fail_records) >= 1, f"Expected FAIL SNR record, got: {snr_records}"
