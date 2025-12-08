"""
risk.barbell_policy
-------------------

Lightweight barbell allocation helper.

This module is intentionally self-contained and configuration-driven. It reads
`config/barbell.yml` and exposes small helpers that downstream components
(optimisers, allocators, risk managers) can use to:

  - Compute safe / risk / other bucket weights for a given portfolio weight
    vector, and
  - Project raw weights into the barbell-feasible region when the feature flag
    is enabled.

By design, importing this module and shipping `config/barbell.yml` does not
change behaviour anywhere until callers explicitly opt in via
`enable_barbell_allocation: true`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml


ROOT_PATH = Path(__file__).resolve().parent.parent
BARBELL_CONFIG_PATH = ROOT_PATH / "config" / "barbell.yml"


@dataclass
class BarbellConfig:
  enable_barbell_allocation: bool
  enable_barbell_validation: bool
  enable_antifragility_tests: bool
  safe_min: float
  safe_max: float
  risk_max: float
  safe_symbols: Iterable[str]
  core_symbols: Iterable[str]
  speculative_symbols: Iterable[str]
  core_max: float
  core_max_per: float
  spec_max: float
  spec_max_per: float
  risk_symbols: Iterable[str] | None = None

  @classmethod
  def from_yaml(cls, path: Path = BARBELL_CONFIG_PATH) -> "BarbellConfig":
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    b = raw.get("barbell") or {}
    sb = b.get("safe_bucket") or {}
    core = b.get("core_bucket") or {}
    spec = b.get("speculative_bucket") or {}
    return cls(
      enable_barbell_allocation=bool(b.get("enable_barbell_allocation", False)),
      enable_barbell_validation=bool(b.get("enable_barbell_validation", False)),
      enable_antifragility_tests=bool(b.get("enable_antifragility_tests", False)),
      safe_min=float(sb.get("min_weight", 0.0)),
      safe_max=float(sb.get("max_weight", 1.0)),
      risk_max=float((b.get("risk_bucket") or {}).get("max_weight", 1.0)),
      safe_symbols=list(sb.get("symbols") or []),
      core_symbols=list(core.get("symbols") or []),
      speculative_symbols=list(spec.get("symbols") or []),
      core_max=float(core.get("max_weight", 0.0)),
      core_max_per=float(core.get("max_per_position", 0.0)),
      spec_max=float(spec.get("max_weight", 0.0)),
      spec_max_per=float(spec.get("max_per_position", 0.0)),
      risk_symbols=list((core.get("symbols") or []) + (spec.get("symbols") or [])),
    )


class BarbellConstraint:
  """
  Stateless barbell helper that can be used to enforce safe vs risk bucket
  weights at the portfolio level.
  """

  def __init__(self, cfg: BarbellConfig | None = None):
    self.cfg = cfg or BarbellConfig.from_yaml()

  def bucket_weights(self, weights: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Compute (safe_weight, core_weight, speculative_weight, other_weight) given a mapping of
    symbol -> weight.
    """
    safe_set = set(self.cfg.safe_symbols)
    core_set = set(self.cfg.core_symbols)
    spec_set = set(self.cfg.speculative_symbols)
    w_safe = w_core = w_spec = w_other = 0.0
    for symbol, w in weights.items():
      if symbol in safe_set:
        w_safe += float(w)
      elif symbol in core_set:
        w_core += float(w)
      elif symbol in spec_set:
        w_spec += float(w)
      else:
        w_other += float(w)
    return w_safe, w_core, w_spec, w_other

  def project_to_feasible(self, weights: Dict[str, float]) -> Dict[str, float]:
    """
    Project weights into the barbell-feasible region.

    If barbell allocation is disabled, this is a no-op.
    """
    if not self.cfg.enable_barbell_allocation:
      # Do nothing when feature flag is off.
      return dict(weights)

    safe_set = set(self.cfg.safe_symbols)
    core_set = set(self.cfg.core_symbols)
    spec_set = set(self.cfg.speculative_symbols)
    risk_set = set(self.cfg.risk_symbols or (list(core_set) + list(spec_set)))
    # Copy to avoid mutating caller input.
    w = dict(weights)

    total = float(sum(w.values()) or 1.0)
    safe, core, spec, other = self.bucket_weights(w)

    # Enforce risk_max: scale down core+spec proportionally if needed.
    risk = core + spec
    if risk > self.cfg.risk_max:
      scale = self.cfg.risk_max / max(risk, 1e-12)
      for s in list(core_set | spec_set):
        if s in w:
          w[s] *= scale
      safe, core, spec, other = self.bucket_weights(w)
      risk = core + spec

    # Enforce per-bucket caps.
    if core > self.cfg.core_max > 0:
      scale = self.cfg.core_max / max(core, 1e-12)
      for s in core_set:
        if s in w:
          w[s] *= scale
    if spec > self.cfg.spec_max > 0:
      scale = self.cfg.spec_max / max(spec, 1e-12)
      for s in spec_set:
        if s in w:
          w[s] *= scale
    # Recompute after bucket caps.
    safe, core, spec, other = self.bucket_weights(w)
    risk = core + spec

    # Enforce safe_min: if safe too small, scale up safe bucket and scale down
    # risk+other to compensate.
    if safe < self.cfg.safe_min:
      deficit = self.cfg.safe_min - safe
      # Take proportionally from risk+other.
      donor_total = total - safe
      if donor_total > 0:
        frac = deficit / donor_total
        for s in w:
          if s not in safe_set:
            w[s] *= max(0.0, 1.0 - frac)
        # Redistribute deficit to safe symbols.
        safe_current = sum(w[s] for s in safe_set if s in w)
        if safe_current > 0:
          boost = deficit / safe_current
          for s in safe_set:
            if s in w:
              w[s] *= 1.0 + boost

    # Final normalisation to preserve total weight sum.
    new_total = float(sum(w.values()) or 1.0)
    if new_total != total:
      scale = total / new_total
      for s in w:
        w[s] *= scale

    return w
