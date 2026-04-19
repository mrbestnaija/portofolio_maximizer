from __future__ import annotations

from scripts import apply_nav_reallocation as mod


def _good_metrics() -> dict:
    return {
        "coverage_ratio": 0.82,
        "missing_metrics_fraction": 0.0,
        "imputed_fraction": 0.0,
        "padding_fraction": 0.0,
        "oos_source_kind": "GENUINE_OOS",
        "provenance_trusted": True,
        "data_source": "production_closed_trades",
        "status": "HEALTHY",
    }


def test_shadow_first_plan_blocks_promotions_but_allows_demotions() -> None:
    current_config = {
        "barbell": {
            "safe_bucket": {"symbols": ["CASH"]},
            "core_bucket": {"symbols": ["AAPL"]},
            "speculative_bucket": {"symbols": ["NVDA"]},
        }
    }
    plan = {
        "_source_plan": "logs/automation/nav_rebalance_plan_latest.json",
        "_rollout": {"live_apply_allowed": False},
        "promotions": [{"ticker": "NVDA", "metrics": _good_metrics(), "reason": "rolling_pf_wr_ok"}],
        "demotions": [{"ticker": "AAPL", "metrics": _good_metrics(), "reason": "rolling_pf_wr_below_floor"}],
    }

    artifact = mod.apply_reallocation(plan, current_config, mod.AllocationConstraints())

    assert artifact.applied_promotions == []
    assert artifact.applied_demotions == plan["demotions"]
    assert artifact.skipped_moves and artifact.skipped_moves[0]["ticker"] == "NVDA"
    assert "LIVE_APPLY_BLOCKED" in artifact.skipped_moves[0]["skip_reason"]
    assert artifact.evidence_gate["NVDA"]["passed"] is False
    assert "LIVE_APPLY_BLOCKED" in artifact.evidence_gate["NVDA"]["blocking_reasons"]
    assert "AAPL" in artifact.sleeve_allocations["speculative"]["symbols"]
    assert artifact.sleeve_allocations["core"]["symbols"] == []


def test_live_apply_allowed_promotes_when_evidence_is_green() -> None:
    current_config = {
        "barbell": {
            "safe_bucket": {"symbols": ["CASH"]},
            "core_bucket": {"symbols": ["AAPL"]},
            "speculative_bucket": {"symbols": ["NVDA"]},
        }
    }
    plan = {
        "_source_plan": "logs/automation/nav_rebalance_plan_latest.json",
        "_rollout": {"live_apply_allowed": True},
        "promotions": [{"ticker": "NVDA", "metrics": _good_metrics(), "reason": "rolling_pf_wr_ok"}],
        "demotions": [{"ticker": "AAPL", "metrics": _good_metrics(), "reason": "rolling_pf_wr_below_floor"}],
    }

    artifact = mod.apply_reallocation(plan, current_config, mod.AllocationConstraints())

    assert artifact.applied_promotions == plan["promotions"]
    assert artifact.applied_demotions == plan["demotions"]
    assert "NVDA" in artifact.sleeve_allocations["core"]["symbols"]
    assert "AAPL" in artifact.sleeve_allocations["speculative"]["symbols"]
    assert artifact.evidence_gate["NVDA"]["passed"] is True
