"""
Lightweight NAV allocator scaffold (feature-flagged off by default).

When enabled, scales per-bucket relative weights by NAV budgets from
config/risk_buckets.yml. With enabled=False (default), this is a pass-through.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping


@dataclass
class BucketBudgets:
    enabled: bool
    base_nav_frac: Dict[str, float]
    min_nav_frac: Dict[str, float]
    max_nav_frac: Dict[str, float]


def load_budgets(config_path: Path = Path("config/risk_buckets.yml")) -> BucketBudgets:
    raw = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    cfg = raw.get("risk_buckets") or {}
    return BucketBudgets(
        enabled=bool(cfg.get("enabled", False)),
        base_nav_frac=cfg.get("base_nav_frac", {}),
        min_nav_frac=cfg.get("min_nav_frac", {}),
        max_nav_frac=cfg.get("max_nav_frac", {}),
    )


def apply_nav_allocator(
    symbol_weights: Mapping[str, float],
    bucket_for_symbol: Mapping[str, str],
    budgets: BucketBudgets,
    nav: float,
) -> Dict[str, float]:
    """
    Apply NAV budgets to symbol weights by bucket. Pass-through when disabled.
    This is a placeholder until integrated + tested; callers should guard with budgets.enabled.
    """
    if not budgets.enabled:
        return dict(symbol_weights)

    # Aggregate per-bucket relative weights
    bucket_rel: Dict[str, float] = {}
    for symbol, w in symbol_weights.items():
        bucket = bucket_for_symbol.get(symbol, "ts_core")
        bucket_rel[bucket] = bucket_rel.get(bucket, 0.0) + max(w, 0.0)

    # Compute effective bucket budgets
    target_bucket_nav: Dict[str, float] = {}
    for bucket, rel_weight in bucket_rel.items():
        base_frac = budgets.base_nav_frac.get(bucket, 0.0)
        min_frac = budgets.min_nav_frac.get(bucket, 0.0)
        max_frac = budgets.max_nav_frac.get(bucket, 1.0)
        target = base_frac * nav
        target = max(target, min_frac * nav)
        target = min(target, max_frac * nav)
        target_bucket_nav[bucket] = target if rel_weight > 0 else 0.0

    # Distribute bucket NAV to symbols proportionally to their relative weights within the bucket
    adjusted: Dict[str, float] = {}
    for symbol, w in symbol_weights.items():
        bucket = bucket_for_symbol.get(symbol, "ts_core")
        bucket_total = bucket_rel.get(bucket, 0.0)
        if bucket_total <= 0:
            adjusted[symbol] = 0.0
            continue
        bucket_nav = target_bucket_nav.get(bucket, 0.0)
        # convert NAV allocation back to weight fraction of total NAV
        adjusted[symbol] = (w / bucket_total) * (bucket_nav / nav if nav > 0 else 0.0)
    return adjusted
