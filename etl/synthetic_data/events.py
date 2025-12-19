from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def apply_events(cfg: Dict, rng: np.random.Generator, index: int, price: float, inst_vol: float, base_drift: float) -> Tuple[float, float, str | None]:
    """Apply configured events and return updated price/vol plus event tag."""
    fired = None
    flash_cfg = cfg.get("flash_crash", {})
    if flash_cfg.get("enabled") and rng.random() < float(flash_cfg.get("prob", 0.0)):
        impact = float(flash_cfg.get("impact_pct", 0.08))
        price = max(price * (1 - abs(impact)), 1e-6)
        inst_vol = inst_vol * (1 + abs(impact) * 5)
        fired = "flash_crash"

    vol_cfg = cfg.get("vol_spike", {})
    if vol_cfg.get("enabled") and rng.random() < float(vol_cfg.get("prob", 0.0)):
        multiplier = float(vol_cfg.get("multiplier", 2.5))
        inst_vol = inst_vol * max(multiplier, 1.0)
        fired = fired or "vol_spike"

    gap_cfg = cfg.get("gap_risk", {})
    if gap_cfg.get("enabled") and rng.random() < float(gap_cfg.get("prob", 0.0)):
        impact = float(gap_cfg.get("impact_pct", 0.03))
        direction = -1 if rng.random() < 0.6 else 1
        price = max(price * (1 + direction * impact), 1e-6)
        inst_vol = inst_vol * (1 + abs(impact) * 3)
        fired = fired or "gap_risk"

    bear_cfg = cfg.get("bear_run", {})
    if bear_cfg.get("enabled") and rng.random() < float(bear_cfg.get("prob", 0.0)):
        drift_shift = float(bear_cfg.get("drift_shift", -0.001))
        price = max(price * (1 + drift_shift), 1e-6)
        fired = fired or "bear_run"

    liquidity_cfg = cfg.get("liquidity_shock", {})
    if liquidity_cfg.get("enabled") and rng.random() < float(liquidity_cfg.get("prob", 0.0)):
        impact = float(liquidity_cfg.get("impact_pct", 0.05))
        price = max(price * (1 - abs(impact)), 1e-6)
        inst_vol = inst_vol * float(liquidity_cfg.get("vol_multiplier", 3.0))
        fired = fired or "liquidity_shock"

    macro_cfg = cfg.get("macro_regime_change", {})
    if macro_cfg.get("enabled") and rng.random() < float(macro_cfg.get("prob", 0.0)):
        drift_shift = float(macro_cfg.get("drift_shift", -0.0005))
        vol_mult = float(macro_cfg.get("vol_multiplier", 1.5))
        price = max(price * (1 + drift_shift), 1e-6)
        inst_vol = inst_vol * vol_mult
        fired = fired or "macro_regime_change"

    return price, inst_vol, fired
