from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional


Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class LOBConfig:
    levels: int = 10
    tick_size_bps: float = 1.0
    alpha: float = 0.8
    max_exhaust_levels: int = 25
    default_order_value: float = 10_000.0
    depth_profiles: Optional[Dict[str, Dict[str, float]]] = None
    tail_depth_multiplier: float = 1.0


@dataclass(frozen=True)
class LOBFill:
    requested_shares: float
    vwap_price: float
    mid_price: float
    start_price: float
    levels_consumed: int
    exhausted: bool

    @property
    def mid_slippage_bps(self) -> float:
        if self.mid_price <= 0:
            return 0.0
        return ((self.vwap_price - self.mid_price) / self.mid_price) * 1e4


def _resolve_profile(
    cfg: LOBConfig,
    asset_class: Optional[str],
) -> Dict[str, float]:
    if not cfg.depth_profiles or not asset_class:
        return {}
    key = asset_class.upper()
    profile = cfg.depth_profiles.get(key) or cfg.depth_profiles.get(key.lower())
    if not isinstance(profile, dict):
        return {}
    return {k: float(v) for k, v in profile.items() if v is not None}


def simulate_market_order_fill(
    *,
    side: Side,
    mid_price: float,
    half_spread: Optional[float] = None,
    depth_notional: Optional[float] = None,
    shares: Optional[float] = None,
    order_notional: Optional[float] = None,
    baseline_slippage: float = 0.0,
    asset_class: Optional[str] = None,
    config: Optional[LOBConfig] = None,
) -> LOBFill:
    cfg = config or LOBConfig()
    mid = float(mid_price or 0.0)
    if mid <= 0:
        return LOBFill(
            requested_shares=0.0,
            vwap_price=mid,
            mid_price=mid,
            start_price=mid,
            levels_consumed=0,
            exhausted=False,
        )

    profile = _resolve_profile(cfg, asset_class)
    resolved_half_spread = half_spread
    if resolved_half_spread is None:
        if "half_spread_bps" in profile:
            resolved_half_spread = (mid * profile["half_spread_bps"]) / 1e4
        else:
            resolved_half_spread = 0.0
    depth_total = float(depth_notional) if depth_notional is not None else float(profile.get("depth_notional", 0.0))

    order_value = order_notional
    if order_value is None and shares is not None:
        order_value = float(shares) * mid
    if order_value is None:
        order_value = float(profile.get("order_value", cfg.default_order_value))

    requested = float(shares) if shares is not None else float(order_value / mid) if mid > 0 else 0.0
    if requested <= 0:
        return LOBFill(
            requested_shares=0.0,
            vwap_price=mid,
            mid_price=mid,
            start_price=mid,
            levels_consumed=0,
            exhausted=False,
        )

    levels = max(1, int(cfg.levels))
    alpha = max(0.0, float(cfg.alpha))
    tick_size_bps = max(0.0, float(cfg.tick_size_bps))
    tick = (mid * tick_size_bps) / 1e4
    if tick <= 0:
        tick = max(1e-9, mid * 1e-4)

    half_spread_abs = abs(float(resolved_half_spread or 0.0))
    baseline_abs = abs(float(baseline_slippage or 0.0))

    if side == "BUY":
        start_price = mid + half_spread_abs + baseline_abs
    else:
        start_price = mid - half_spread_abs - baseline_abs
    start_price = max(start_price, tick)

    if depth_total <= 0:
        return LOBFill(
            requested_shares=requested,
            vwap_price=start_price,
            mid_price=mid,
            start_price=start_price,
            levels_consumed=1,
            exhausted=True,
        )

    weights = [math.exp(-alpha * i) for i in range(levels)]
    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        weights = [1.0 for _ in range(levels)]
        weight_sum = float(levels)

    remaining = requested
    total_cost = 0.0
    levels_consumed = 0
    for i in range(levels):
        if remaining <= 0:
            break
        px = start_price + i * tick if side == "BUY" else max(tick, start_price - i * tick)
        notional_here = depth_total * (weights[i] / weight_sum)
        shares_here = notional_here / px if px > 0 else 0.0
        if shares_here <= 0:
            continue
        fill = remaining if remaining <= shares_here else shares_here
        remaining -= fill
        total_cost += fill * px
        if fill > 0:
            levels_consumed = i + 1

    exhausted = remaining > 0
    if exhausted:
        tail_levels = max(levels, int(cfg.max_exhaust_levels * max(1.0, float(cfg.tail_depth_multiplier))))
        px = start_price + tail_levels * tick if side == "BUY" else max(tick, start_price - tail_levels * tick)
        total_cost += remaining * px
        remaining = 0.0
        levels_consumed = max(levels_consumed, levels)

    vwap = total_cost / requested if requested > 0 else start_price
    return LOBFill(
        requested_shares=requested,
        vwap_price=float(vwap),
        mid_price=mid,
        start_price=start_price,
        levels_consumed=levels_consumed,
        exhausted=exhausted,
    )


__all__ = ["LOBConfig", "LOBFill", "simulate_market_order_fill"]

