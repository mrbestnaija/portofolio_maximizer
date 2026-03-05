from __future__ import annotations


def test_thresholds_match_capital_readiness_contract() -> None:
    from scripts.capital_readiness_check import (
        R3_MIN_PROFIT_FACTOR,
        R3_MIN_TRADES,
        R3_MIN_WIN_RATE,
        R4_MAX_BRIER,
    )
    from scripts.robustness_thresholds import threshold_map

    thresholds = threshold_map()
    assert thresholds["r3_min_trades"] == R3_MIN_TRADES
    assert thresholds["r3_min_win_rate"] == R3_MIN_WIN_RATE
    assert thresholds["r3_min_profit_factor"] == R3_MIN_PROFIT_FACTOR
    assert thresholds["r4_max_brier"] == R4_MAX_BRIER


def test_thresholds_include_source_paths_and_hashes() -> None:
    from scripts.robustness_thresholds import threshold_map

    thresholds = threshold_map()
    assert "source_paths" in thresholds
    assert "source_hashes" in thresholds
    assert thresholds["source_paths"]["capital_readiness_check"]
    assert thresholds["source_paths"]["forecaster_monitoring"]
