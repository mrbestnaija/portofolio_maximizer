from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from models.signal_router import SignalBundle
from scripts.run_auto_trader import (
    _attach_signal_context_to_forecast_audit,
    _execute_signal,
)


def _market_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex(["2026-04-09T09:00:00+00:00"])
    return pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [1000],
        },
        index=idx,
    )


def _audit_payload() -> dict:
    return {
        "dataset": {
            "ticker": "MSFT",
            "start": "2025-01-01",
            "end": "2026-04-09",
            "length": 260,
            "forecast_horizon": 30,
        },
        "signal_context": {},
        "results": {},
    }


def _hold_primary_signal() -> dict:
    return {
        "ticker": "MSFT",
        "action": "HOLD",
        "confidence": 0.38,
        "expected_return": 0.0019,
        "expected_return_net": -0.0011,
        "gross_trade_return": 0.0019,
        "net_trade_return": -0.0011,
        "roundtrip_cost_fraction": 0.0030,
        "roundtrip_cost_bps": 30.0,
        "risk_score": 0.31,
        "risk_level": "low",
        "reasoning": "QuantValidation=FAIL (min_expected_profit)",
        "signal_type": "TIME_SERIES",
        "volatility": 0.20,
        "model_type": "ENSEMBLE",
        "source": "TIME_SERIES",
        "signal_id": None,
        "ts_signal_id": "ts_MSFT_20260409T090000Z_abcd_0001",
        "confidence_calibrated": 0.41,
        "provenance": {
            "hold_reason": "SNR_GATE",
            "snr_gate_blocked": True,
            "snr_gate_threshold": 1.5,
            "decision_context_snr": 0.84,
            "decision_context": {
                "signal_to_noise": 0.84,
                "expected_return_net": -0.0011,
                "gross_trade_return": 0.0019,
                "net_trade_return": -0.0011,
                "roundtrip_cost_fraction": 0.0030,
                "roundtrip_cost_bps": 30.0,
            },
            "quant_validation": {
                "status": "FAIL",
                "failed_criteria": ["min_expected_profit"],
                "utility_score": 0.21,
                "objective_mode": "domain_utility",
            },
        },
    }


def test_execute_signal_derives_rejection_taxonomy_from_provenance() -> None:
    """routing_reason/SNR must come from TS provenance, not dead top-level placeholders."""
    router = MagicMock()
    router.route_signal.return_value = SignalBundle(primary_signal=_hold_primary_signal())

    trading_engine = MagicMock()
    trading_engine.execute_signal.return_value = SimpleNamespace(
        status="REJECTED",
        reason="Non-actionable signal",
        validation_warnings=["hold"],
        trade=None,
        portfolio=None,
    )

    with patch("scripts.run_auto_trader._write_funnel_audit_entry") as funnel_audit:
        report = _execute_signal(
            router=router,
            trading_engine=trading_engine,
            ticker="MSFT",
            forecast_bundle={"horizon": 30},
            current_price=100.0,
            market_data=_market_frame(),
        )

    assert report is not None
    assert report["status"] == "REJECTED"
    assert report["routing_reason"] == "SNR_GATE"
    assert report["hold_reason"] == "SNR_GATE"
    assert report["snr"] == pytest.approx(0.84)
    assert report["quant_validation_status"] == "FAIL"
    assert report["quant_validation_failed_criteria"] == ["min_expected_profit"]

    signal_snapshot = report["signal_snapshot"]
    assert signal_snapshot["routing_reason"] == "SNR_GATE"
    assert signal_snapshot["hold_reason"] == "SNR_GATE"
    assert signal_snapshot["snr"] == pytest.approx(0.84)
    assert signal_snapshot["quant_validation"]["status"] == "FAIL"
    assert signal_snapshot["quant_validation"]["objective_mode"] == "domain_utility"

    funnel_audit.assert_called_once()
    assert funnel_audit.call_args.kwargs["reason"] == "SNR_GATE"
    assert funnel_audit.call_args.kwargs["snr"] == pytest.approx(0.84)


def test_attach_signal_context_persists_routed_signal_snapshot() -> None:
    """The patched audit JSON must carry a usable routed-signal snapshot for live funnel analysis."""
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        audit_file = tmp_path / "forecast_audit_20260409_090000.json"
        audit_file.write_text(json.dumps(_audit_payload()), encoding="utf-8")

        execution_report = {
            "ts_signal_id": "ts_MSFT_20260409T090000Z_abcd_0001",
            "signal_timestamp": "2026-04-09T09:00:00+00:00",
            "executed": False,
            "status": "REJECTED",
            "reason": "Non-actionable signal",
            "action": "HOLD",
            "signal_confidence": 0.38,
            "confidence_calibrated": 0.41,
            "expected_return": 0.0019,
            "expected_return_net": -0.0011,
            "gross_trade_return": 0.0019,
            "net_trade_return": -0.0011,
            "roundtrip_cost_fraction": 0.0030,
            "roundtrip_cost_bps": 30.0,
            "routing_reason": "SNR_GATE",
            "hold_reason": "SNR_GATE",
            "snr": 0.84,
            "directional_gate_applied": False,
            "quant_validation_status": "FAIL",
            "quant_validation_failed_criteria": ["min_expected_profit"],
            "signal_snapshot": {
                "ticker": "MSFT",
                "signal_source": "TIME_SERIES",
                "ts_signal_id": "ts_MSFT_20260409T090000Z_abcd_0001",
                "action": "HOLD",
                "confidence": 0.38,
                "confidence_calibrated": 0.41,
                "expected_return": 0.0019,
                "expected_return_net": -0.0011,
                "gross_trade_return": 0.0019,
                "net_trade_return": -0.0011,
                "roundtrip_cost_fraction": 0.0030,
                "roundtrip_cost_bps": 30.0,
                "hold_reason": "SNR_GATE",
                "routing_reason": "SNR_GATE",
                "snr": 0.84,
                "quant_validation": {
                    "status": "FAIL",
                    "failed_criteria": ["min_expected_profit"],
                    "utility_score": 0.21,
                    "objective_mode": "domain_utility",
                },
            },
        }

        with patch("scripts.run_auto_trader.ROOT_PATH", tmp_path):
            _attach_signal_context_to_forecast_audit(
                forecast_bundle={"horizon": 30, "forecast_audit_path": str(audit_file)},
                execution_report=execution_report,
                ticker="MSFT",
                run_id="20260409_090000",
            )

        payload = json.loads(audit_file.read_text(encoding="utf-8"))
        signal_context = payload["signal_context"]
        assert signal_context["routing_reason"] == "SNR_GATE"
        assert signal_context["hold_reason"] == "SNR_GATE"
        assert signal_context["confidence"] == pytest.approx(0.38)
        assert signal_context["snr"] == pytest.approx(0.84)
        assert signal_context["quant_validation_status"] == "FAIL"
        assert signal_context["quant_validation_failed_criteria"] == ["min_expected_profit"]

        signal = payload["signal"]
        assert signal["action"] == "HOLD"
        assert signal["routing_reason"] == "SNR_GATE"
        assert signal["hold_reason"] == "SNR_GATE"
        assert signal["snr"] == pytest.approx(0.84)
        assert signal["quant_validation"]["status"] == "FAIL"
        assert signal["quant_validation"]["failed_criteria"] == ["min_expected_profit"]
