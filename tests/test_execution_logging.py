import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_auto_trader
from scripts.evaluate_sleeve_promotions import PromotionRules, evaluate_promotions


def test_compute_mid_price_prefers_bid_ask():
    frame = pd.DataFrame(
        [{"Bid": 10.0, "Ask": 12.0, "High": 15.0, "Low": 9.0, "Close": 11.0}],
        index=[pd.Timestamp("2024-01-01")],
    )
    assert run_auto_trader._compute_mid_price(frame) == pytest.approx(11.0)


def test_log_execution_event_jsonl(tmp_path: Path):
    log_path = tmp_path / "exec.jsonl"
    run_auto_trader.EXECUTION_LOG_PATH = log_path
    run_auto_trader._log_execution_event(
        "run1",
        2,
        {"ticker": "AAPL", "status": "EXECUTED", "entry_price": 10.0, "mid_price": 9.5},
    )
    payload = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert payload["run_id"] == "run1"
    assert payload["cycle"] == 2
    assert payload["mid_price"] == 9.5


def test_evaluate_promotions_promotes_and_demotes():
    summary = [
        {"ticker": "MTN", "bucket": "speculative", "total_trades": 12, "win_rate": 0.6, "profit_factor": 1.3},
        {"ticker": "CL=F", "bucket": "core", "total_trades": 15, "win_rate": 0.4, "profit_factor": 0.8},
    ]
    rules = PromotionRules(min_trades=10, promote_win_rate=0.55, promote_profit_factor=1.2)
    plan = evaluate_promotions(summary, bucket_map={}, rules=rules)
    promoted = [p["ticker"] for p in plan["promotions"]]
    demoted = [d["ticker"] for d in plan["demotions"]]
    assert "MTN" in promoted
    assert "CL=F" in demoted
