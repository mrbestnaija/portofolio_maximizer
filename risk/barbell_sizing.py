"""
risk.barbell_sizing
-------------------

Small, shared helpers for barbell sizing overlays.

Design goals:
- Feature-flagged: callers decide when/if to apply.
- Audit-friendly: return bucket + multiplier for provenance.
- Deterministic: no randomness, config-driven via BarbellConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from risk.barbell_policy import BarbellConfig


@dataclass(frozen=True)
class BarbellSizingResult:
    bucket: str
    multiplier: float
    effective_confidence: float


def barbell_bucket(ticker: str, cfg: BarbellConfig) -> str:
    sym = str(ticker).upper()
    if sym in set(cfg.safe_symbols):
        return "safe"
    if sym in set(cfg.core_symbols):
        return "core"
    if sym in set(cfg.speculative_symbols):
        return "spec"
    return "other"


def barbell_confidence_multipliers(cfg: BarbellConfig) -> Dict[str, float]:
    """
    Map barbell per-position caps to a simple confidence multiplier heuristic.

    Uses safe max_per_position as the reference scale.
    """
    safe_max_per_position = 0.50
    core_mult = float(cfg.core_max_per) / safe_max_per_position if safe_max_per_position else 0.2
    spec_mult = float(cfg.spec_max_per) / safe_max_per_position if safe_max_per_position else 0.1
    return {
        "safe": 1.0,
        "core": max(0.0, min(1.0, core_mult)),
        "spec": max(0.0, min(1.0, spec_mult)),
        "other": 1.0,
    }


def apply_barbell_confidence(
    *,
    ticker: str,
    base_confidence: float,
    cfg: BarbellConfig,
) -> BarbellSizingResult:
    bucket = barbell_bucket(ticker, cfg)
    multipliers = barbell_confidence_multipliers(cfg)
    multiplier = float(multipliers.get(bucket, 1.0))
    conf = max(0.0, min(1.0, float(base_confidence)))
    effective = max(0.0, min(1.0, conf * multiplier))
    return BarbellSizingResult(bucket=bucket, multiplier=multiplier, effective_confidence=effective)
