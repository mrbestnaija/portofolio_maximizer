from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from risk.barbell_promotion_gate import (
    BarbellPromotionDecision,
    decide_promotion_from_report,
    load_promotion_evidence,
    write_promotion_evidence,
)
from risk.barbell_policy import BarbellConfig
from risk.barbell_sizing import BarbellMarketContext, apply_barbell_confidence, build_barbell_market_context


def test_apply_barbell_confidence_tags_bucket() -> None:
    cfg = BarbellConfig.from_yaml()
    res = apply_barbell_confidence(ticker="MSFT", base_confidence=0.8, cfg=cfg)
    assert res.bucket in {"safe", "core", "spec", "other"}
    assert 0.0 <= res.effective_confidence <= 1.0


def test_apply_barbell_confidence_penalizes_costly_high_gap_speculative_trade() -> None:
    cfg = BarbellConfig.from_yaml()
    ctx = BarbellMarketContext(
        expected_return_net=0.01,
        forecast_horizon_bars=5,
        roundtrip_cost_bps=80.0,
        gap_risk_pct=0.02,
        leverage=1.0,
        funding_bps_per_day=0.0,
        regime="CRISIS",
    )
    res = apply_barbell_confidence(ticker="BTC-USD", base_confidence=0.8, cfg=cfg, context=ctx)
    assert res.bucket == "spec"
    assert res.multiplier < 0.8
    assert res.market_multiplier < 1.0
    assert res.regime_multiplier <= 1.0


def test_apply_barbell_confidence_penalizes_thin_liquidity_relative_to_order_size() -> None:
    cfg = BarbellConfig.from_yaml()
    ctx = BarbellMarketContext(
        expected_return_net=0.012,
        forecast_horizon_bars=5,
        roundtrip_cost_bps=10.0,
        gap_risk_pct=0.002,
        leverage=1.0,
        funding_bps_per_day=0.0,
        depth_notional=5000.0,
        order_notional=2500.0,
        regime="MODERATE_TRENDING",
    )
    res = apply_barbell_confidence(ticker="AAPL", base_confidence=0.8, cfg=cfg, context=ctx)
    assert res.bucket == "spec"
    assert res.market_multiplier < 1.0
    assert res.diagnostics["liquidity_to_depth"] == pytest.approx(0.5)


def test_apply_barbell_confidence_keeps_safe_bucket_near_base_in_calm_market() -> None:
    cfg = BarbellConfig.from_yaml()
    ctx = BarbellMarketContext(
        expected_return_net=0.004,
        forecast_horizon_bars=5,
        roundtrip_cost_bps=5.0,
        gap_risk_pct=0.001,
        leverage=1.0,
        funding_bps_per_day=0.0,
        regime="LIQUID_RANGEBOUND",
    )
    res = apply_barbell_confidence(ticker="SHY", base_confidence=0.8, cfg=cfg, context=ctx)
    assert res.bucket == "safe"
    assert res.effective_confidence == pytest.approx(0.8, rel=1e-6)


def test_build_barbell_market_context_extracts_gap_risk_and_regime() -> None:
    market_data = pd.DataFrame(
        {
            "Open": [100.0, 97.0, 101.0, 94.0],
            "Close": [99.0, 102.0, 95.0, 96.0],
            "Depth": [100000.0, 90000.0, 85000.0, 80000.0],
            "FundingBps": [0.0, 0.0, 0.0, 1.5],
        }
    )
    ctx = build_barbell_market_context(
        signal_payload={
            "expected_return_net": 0.015,
            "forecast_horizon": 5,
            "roundtrip_cost_bps": 24.0,
            "position_value": 2500.0,
        },
        market_data=market_data,
        detected_regime="HIGH_VOL_TRENDING",
    )

    assert ctx.expected_return_net == pytest.approx(0.015)
    assert ctx.roundtrip_cost_bps == pytest.approx(24.0)
    assert ctx.gap_risk_pct is not None and ctx.gap_risk_pct > 0.0
    assert ctx.depth_notional == pytest.approx(80000.0)
    assert ctx.order_notional == pytest.approx(2500.0)
    assert ctx.funding_bps_per_day == pytest.approx(1.5)
    assert ctx.regime == "HIGH_VOL_TRENDING"


def _full_barbell_metrics(**overrides):
    """Return a barbell_sized metrics dict with all required evidence fields."""
    base = {
        "total_trades": 40,
        "losing_trades": 10,
        "profit_factor": 1.3,
        "max_drawdown": 0.18,
        "total_return_pct": 0.06,
        "expected_shortfall": -0.025,
        # barbell robustness evidence
        "omega_robustness_score": 0.55,
        "payoff_asymmetry_support_ok": True,
        "payoff_asymmetry_effective": 1.50,
        "winner_concentration_ratio": 0.45,
        # path-risk evidence
        "path_risk_trade_count": 30,
        "path_risk_ok_rate": 0.85,
    }
    base.update(overrides)
    return base


def test_decide_promotion_requires_trades_and_losses() -> None:
    payload = {
        "evidence_source": "trade_history",
        "metrics": {
            "ts_only": {
                "total_trades": 40, "losing_trades": 10, "profit_factor": 1.2,
                "max_drawdown": 0.2, "total_return_pct": 0.05,
                "expected_shortfall": -0.030,
            },
            "barbell_sized": _full_barbell_metrics(),
            "delta": {"profit_factor": 0.1, "max_drawdown": -0.02, "total_return_pct": 0.01},
        },
    }
    decision = decide_promotion_from_report(payload)
    assert decision.passed is True, f"Expected PASS, got: {decision.reason}\nchecks: {decision.checks}"

    payload["metrics"]["barbell_sized"]["total_trades"] = 10
    payload["metrics"]["delta"]["total_return_pct"] = 0.01
    decision2 = decide_promotion_from_report(payload)
    assert decision2.passed is False
    assert "trade_support" in decision2.reason


def test_promotion_evidence_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    decision = BarbellPromotionDecision(passed=True, reason="ok", evidence_source="walk_forward")
    write_promotion_evidence(path=path, decision=decision)
    loaded = load_promotion_evidence(path)
    assert loaded.passed is True
    assert loaded.reason == "ok"

    archive_dir = tmp_path / "evidence_history"
    archived = sorted(archive_dir.glob("evidence_*.json"))
    assert archived, "expected immutable archive copy of promotion evidence"
    assert (archive_dir / "manifest.jsonl").exists(), "expected archive manifest"
    archived_loaded = load_promotion_evidence(archived[-1])
    assert archived_loaded.passed is True
    assert archived_loaded.reason == "ok"
