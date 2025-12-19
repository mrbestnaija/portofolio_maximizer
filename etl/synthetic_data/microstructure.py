from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def simulate_microstructure(
    price: float,
    inst_vol: float,
    shock: float,
    regime: str,
    micro_cfg: Dict[str, any],
    rng: np.random.Generator,
    order_size: float = 1.0,
) -> Tuple[float, float, float, float, float, float]:
    spread_cfg = micro_cfg.get("spread", {})
    slippage_cfg = micro_cfg.get("slippage", {})
    depth_cfg = micro_cfg.get("depth", {})
    imbalance_cfg = micro_cfg.get("order_flow", {})
    base_spread_bps = float(spread_cfg.get("bps", 5.0))
    vol_beta = float(spread_cfg.get("vol_beta", 20.0))
    regime_spread = float(spread_cfg.get("regime_widen", 1.5 if "high" in regime else 1.0))
    spread = price * (base_spread_bps / 10_000.0) * (1 + vol_beta * inst_vol) * regime_spread

    base_slip_bps = float(slippage_cfg.get("bps", 3.0))
    shock_beta = float(slippage_cfg.get("shock_beta", 10.0))
    size_beta = float(slippage_cfg.get("size_beta", 0.1))
    slippage = price * (base_slip_bps / 10_000.0) * (1 + shock_beta * abs(shock) + size_beta * order_size)

    base_depth = float(depth_cfg.get("base", 1_000_000.0))
    vol_beta_depth = float(depth_cfg.get("vol_beta", 15.0))
    min_depth = float(depth_cfg.get("min_depth", 50_000.0))
    depth = max(base_depth * (1 - vol_beta_depth * inst_vol), min_depth)

    imbalance_sigma = float(imbalance_cfg.get("sigma", 0.15))
    shock_beta_imb = float(imbalance_cfg.get("shock_beta", 0.8))
    regime_bias = float(imbalance_cfg.get("regime_bias", -0.1 if "high" in regime else 0.0))
    order_imbalance = rng.normal(regime_bias, imbalance_sigma) + shock_beta_imb * shock

    exec_cost_bps = (spread + slippage) / price * 10_000.0
    impact = float(max(0.0, slippage / max(depth, 1.0))) * 10000.0
    return spread, slippage, depth, order_imbalance, exec_cost_bps, impact
