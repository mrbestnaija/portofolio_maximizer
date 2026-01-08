from __future__ import annotations

from execution.lob_simulator import LOBConfig, simulate_market_order_fill


def test_lob_fill_scales_with_order_size():
    cfg = LOBConfig(levels=5, tick_size_bps=1.0, alpha=0.5, max_exhaust_levels=10)
    mid = 100.0
    half_spread = 0.05
    depth = 50000.0

    small = simulate_market_order_fill(
        side="BUY",
        mid_price=mid,
        half_spread=half_spread,
        depth_notional=depth,
        shares=10,
        baseline_slippage=half_spread,
        config=cfg,
    )
    large = simulate_market_order_fill(
        side="BUY",
        mid_price=mid,
        half_spread=half_spread,
        depth_notional=depth,
        shares=1000,
        baseline_slippage=half_spread,
        config=cfg,
    )

    assert large.vwap_price >= small.vwap_price
    assert large.levels_consumed >= small.levels_consumed


def test_lob_fill_marks_exhaustion_when_depth_too_small():
    cfg = LOBConfig(levels=3, tick_size_bps=1.0, alpha=0.5, max_exhaust_levels=5)
    fill = simulate_market_order_fill(
        side="BUY",
        mid_price=100.0,
        half_spread=0.1,
        depth_notional=1000.0,
        shares=10_000,
        baseline_slippage=0.1,
        config=cfg,
    )
    assert fill.exhausted is True
    assert fill.levels_consumed >= 3


def test_lob_fill_handles_tail_depth_for_large_orders():
    cfg = LOBConfig(levels=5, tick_size_bps=0.5, alpha=0.2, max_exhaust_levels=8, tail_depth_multiplier=3.0)
    fill = simulate_market_order_fill(
        side="BUY",
        mid_price=50.0,
        half_spread=0.05,
        depth_notional=5000.0,
        shares=50_000,
        baseline_slippage=0.05,
        config=cfg,
    )
    assert fill.exhausted
    assert fill.levels_consumed >= cfg.levels
    assert fill.mid_slippage_bps > 0


def test_lob_fill_handles_notional_and_profiles():
    cfg = LOBConfig(
        levels=4,
        tick_size_bps=0.5,
        alpha=0.3,
        max_exhaust_levels=8,
        default_order_value=20000.0,
        tail_depth_multiplier=2.0,
        depth_profiles={
            "US_EQUITY": {"depth_notional": 100000.0, "half_spread_bps": 1.0, "order_value": 5000.0}
        },
    )

    fill = simulate_market_order_fill(
        side="BUY",
        mid_price=50.0,
        half_spread=None,
        depth_notional=None,
        order_notional=25000.0,
        baseline_slippage=0.0,
        asset_class="US_equity",
        config=cfg,
    )

    assert fill.requested_shares > 0  # derived from notional
    assert fill.levels_consumed > 0
    assert fill.mid_slippage_bps >= 0
