from __future__ import annotations

import sqlite3

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


def test_apply_reallocation_leaves_trade_execution_rows_untouched(tmp_path) -> None:
    db_path = tmp_path / "portfolio.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE trade_executions (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                is_close INTEGER DEFAULT 0,
                is_synthetic INTEGER DEFAULT 0
            )
            """
        )
        conn.executemany(
            "INSERT INTO trade_executions (id, ticker, is_close, is_synthetic) VALUES (?, ?, 0, 0)",
            [
                (1, "AAPL"),
                (2, "AAPL"),
                (3, "NVDA"),
            ],
        )
        conn.commit()

        before = conn.execute(
            "SELECT COUNT(*) FROM trade_executions WHERE ticker='AAPL' AND is_close=0"
        ).fetchone()[0]

        artifact = mod.apply_reallocation(
            {
                "_source_plan": "logs/automation/nav_rebalance_plan_latest.json",
                "_rollout": {"live_apply_allowed": True},
                "promotions": [{"ticker": "NVDA", "metrics": _good_metrics(), "reason": "rolling_pf_wr_ok"}],
                "demotions": [{"ticker": "AAPL", "metrics": _good_metrics(), "reason": "rolling_pf_wr_below_floor"}],
            },
            {
                "barbell": {
                    "safe_bucket": {"symbols": ["CASH"]},
                    "core_bucket": {"symbols": ["AAPL"]},
                    "speculative_bucket": {"symbols": ["NVDA"]},
                }
            },
            mod.AllocationConstraints(),
        )

        after = conn.execute(
            "SELECT COUNT(*) FROM trade_executions WHERE ticker='AAPL' AND is_close=0"
        ).fetchone()[0]
    finally:
        conn.close()

    assert before == after == 2
    assert artifact.applied_demotions == [{"ticker": "AAPL", "metrics": _good_metrics(), "reason": "rolling_pf_wr_below_floor"}]
