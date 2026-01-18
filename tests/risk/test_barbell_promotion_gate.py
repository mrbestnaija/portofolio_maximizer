from __future__ import annotations

from pathlib import Path

import pytest

from risk.barbell_promotion_gate import (
    BarbellPromotionDecision,
    decide_promotion_from_report,
    load_promotion_evidence,
    write_promotion_evidence,
)
from risk.barbell_policy import BarbellConfig
from risk.barbell_sizing import apply_barbell_confidence


def test_apply_barbell_confidence_tags_bucket() -> None:
    cfg = BarbellConfig.from_yaml()
    res = apply_barbell_confidence(ticker="MSFT", base_confidence=0.8, cfg=cfg)
    assert res.bucket in {"safe", "core", "spec", "other"}
    assert 0.0 <= res.effective_confidence <= 1.0


def test_decide_promotion_requires_trades_and_losses() -> None:
    payload = {
        "evidence_source": "trade_history",
        "metrics": {
            "ts_only": {"total_trades": 40, "losing_trades": 10, "profit_factor": 1.2, "max_drawdown": 0.2, "total_return_pct": 0.05},
            "barbell_sized": {"total_trades": 40, "losing_trades": 10, "profit_factor": 1.3, "max_drawdown": 0.18, "total_return_pct": 0.06},
            "delta": {"profit_factor": 0.1, "max_drawdown": -0.02, "total_return_pct": 0.01},
        },
    }
    decision = decide_promotion_from_report(payload)
    assert decision.passed is True

    payload["metrics"]["barbell_sized"]["total_trades"] = 10
    payload["metrics"]["delta"]["total_return_pct"] = 0.01
    decision2 = decide_promotion_from_report(payload)
    assert decision2.passed is False
    assert "insufficient trades" in decision2.reason


def test_promotion_evidence_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    decision = BarbellPromotionDecision(passed=True, reason="ok", evidence_source="walk_forward")
    write_promotion_evidence(path=path, decision=decision)
    loaded = load_promotion_evidence(path)
    assert loaded.passed is True
    assert loaded.reason == "ok"
