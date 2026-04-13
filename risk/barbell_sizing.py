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

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from risk.barbell_policy import BarbellConfig


@dataclass(frozen=True)
class BarbellSizingResult:
    bucket: str
    multiplier: float
    effective_confidence: float
    bucket_multiplier: float = 1.0
    regime_multiplier: float = 1.0
    market_multiplier: float = 1.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BarbellMarketContext:
    expected_return_net: Optional[float] = None
    forecast_horizon_bars: Optional[int] = None
    roundtrip_cost_bps: Optional[float] = None
    gap_risk_pct: Optional[float] = None
    leverage: Optional[float] = None
    funding_bps_per_day: Optional[float] = None
    depth_notional: Optional[float] = None
    order_notional: Optional[float] = None
    regime: Optional[str] = None


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _estimate_gap_risk_pct(market_data: Any, lookback: int = 20) -> Optional[float]:
    if market_data is None or not isinstance(market_data, pd.DataFrame) or market_data.empty:
        return None

    open_col = "Open" if "Open" in market_data.columns else None
    close_col = "Close" if "Close" in market_data.columns else None
    if not open_col or not close_col:
        return None

    closes = pd.to_numeric(market_data[close_col], errors="coerce")
    opens = pd.to_numeric(market_data[open_col], errors="coerce")
    prev_close = closes.shift(1)
    gaps = ((opens - prev_close).abs() / prev_close.abs()).replace([pd.NA, pd.NaT], pd.NA).dropna()
    if gaps.empty:
        return None
    tail = gaps.tail(max(int(lookback), 1))
    try:
        return float(tail.quantile(0.9))
    except Exception:
        return float(tail.max())


def build_barbell_market_context(
    *,
    signal_payload: Mapping[str, Any],
    market_data: Any = None,
    detected_regime: Optional[str] = None,
) -> BarbellMarketContext:
    last_row = None
    if isinstance(market_data, pd.DataFrame) and not market_data.empty:
        last_row = market_data.iloc[-1]

    expected_return_net = _safe_float(
        signal_payload.get("expected_return_net")
        or signal_payload.get("net_trade_return")
        or signal_payload.get("expected_return")
    )
    forecast_horizon_bars = _safe_float(
        signal_payload.get("forecast_horizon") or signal_payload.get("horizon")
    )
    roundtrip_cost_bps = _safe_float(signal_payload.get("roundtrip_cost_bps"))
    leverage = _safe_float(signal_payload.get("leverage")) or 1.0
    funding_bps = _safe_float(signal_payload.get("funding_bps_per_day"))
    depth_notional = _safe_float(last_row.get("Depth")) if last_row is not None and hasattr(last_row, "get") else None
    order_notional = _safe_float(
        signal_payload.get("order_notional")
        or signal_payload.get("position_value")
        or signal_payload.get("target_value")
    )
    if funding_bps is None and last_row is not None and hasattr(last_row, "get"):
        funding_bps = _safe_float(last_row.get("FundingBps") or last_row.get("FundingRateBps"))

    regime = str(
        detected_regime
        or signal_payload.get("detected_regime")
        or signal_payload.get("regime")
        or ""
    ).strip().upper() or None

    return BarbellMarketContext(
        expected_return_net=expected_return_net,
        forecast_horizon_bars=int(forecast_horizon_bars) if forecast_horizon_bars else None,
        roundtrip_cost_bps=roundtrip_cost_bps,
        gap_risk_pct=_estimate_gap_risk_pct(market_data),
        leverage=leverage,
        funding_bps_per_day=funding_bps,
        depth_notional=depth_notional,
        order_notional=order_notional,
        regime=regime,
    )


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
    Resolve per-bucket confidence multipliers.

    Prefer explicit config when present. Fall back to the legacy per-position-cap
    heuristic so historical behavior remains reproducible.
    """
    if cfg.bucket_multipliers:
        resolved = {str(k).strip().lower(): float(v) for k, v in cfg.bucket_multipliers.items()}
        for key in ("safe", "core", "spec", "other"):
            resolved.setdefault(key, 1.0 if key == "safe" else 0.85)
        return resolved

    safe_max_per_position = 0.50
    core_mult = float(cfg.core_max_per) / safe_max_per_position if safe_max_per_position else 0.2
    spec_mult = float(cfg.spec_max_per) / safe_max_per_position if safe_max_per_position else 0.1
    return {
        "safe": 1.0,
        "core": max(0.0, min(1.0, core_mult)),
        "spec": max(0.0, min(1.0, spec_mult)),
        "other": 1.0,
    }


def _ratio_penalty(
    *,
    ratio: Optional[float],
    soft_cap: float,
    floor_multiplier: float,
) -> float:
    if ratio is None or soft_cap <= 0 or ratio <= soft_cap:
        return 1.0
    excess = max(0.0, (float(ratio) - float(soft_cap)) / float(soft_cap))
    penalty = 1.0 / (1.0 + excess)
    return max(float(floor_multiplier), min(1.0, float(penalty)))


def _regime_multiplier(bucket: str, cfg: BarbellConfig, regime: Optional[str]) -> float:
    if not regime:
        return 1.0
    mapping = cfg.regime_bucket_multipliers or {}
    regime_map = mapping.get(str(regime).upper()) or {}
    value = regime_map.get(bucket)
    try:
        return float(value) if value is not None else 1.0
    except (TypeError, ValueError):
        return 1.0


def _market_context_multiplier(context: Optional[BarbellMarketContext], cfg: BarbellConfig) -> tuple[float, Dict[str, Any]]:
    if context is None:
        return 1.0, {}

    floors = cfg.overlay_floors or {}
    caps = cfg.edge_ratio_soft_caps or {}
    diagnostics: Dict[str, Any] = {}
    multipliers: list[float] = []

    edge = abs(float(context.expected_return_net or 0.0))
    edge_bps = edge * 1e4 if edge > 0 else None
    diagnostics["expected_return_net_bps"] = edge_bps

    if edge_bps and context.roundtrip_cost_bps is not None:
        ratio = float(context.roundtrip_cost_bps) / edge_bps if edge_bps > 0 else None
        diagnostics["roundtrip_cost_to_edge"] = ratio
        multipliers.append(
            _ratio_penalty(
                ratio=ratio,
                soft_cap=float(caps.get("roundtrip_cost_to_edge", 0.35)),
                floor_multiplier=float(floors.get("cost", 0.55)),
            )
        )

    if edge and context.gap_risk_pct is not None:
        ratio = float(context.gap_risk_pct) / edge if edge > 0 else None
        diagnostics["gap_risk_to_edge"] = ratio
        multipliers.append(
            _ratio_penalty(
                ratio=ratio,
                soft_cap=float(caps.get("gap_risk_to_edge", 0.75)),
                floor_multiplier=float(floors.get("gap", 0.50)),
            )
        )

    if edge_bps and context.funding_bps_per_day is not None:
        horizon = max(int(context.forecast_horizon_bars or 1), 1)
        total_funding_bps = float(context.funding_bps_per_day) * horizon
        ratio = total_funding_bps / edge_bps if edge_bps > 0 else None
        diagnostics["funding_to_edge"] = ratio
        diagnostics["funding_bps_total_horizon"] = total_funding_bps
        multipliers.append(
            _ratio_penalty(
                ratio=ratio,
                soft_cap=float(caps.get("funding_to_edge", 0.25)),
                floor_multiplier=float(floors.get("funding", 0.70)),
            )
        )

    if context.depth_notional is not None and context.order_notional is not None and context.depth_notional > 0:
        ratio = float(context.order_notional) / float(context.depth_notional)
        diagnostics["liquidity_to_depth"] = ratio
        diagnostics["order_notional"] = float(context.order_notional)
        multipliers.append(
            _ratio_penalty(
                ratio=ratio,
                soft_cap=float(caps.get("liquidity_to_depth", 0.10)),
                floor_multiplier=float(floors.get("liquidity", 0.65)),
            )
        )

    leverage = float(context.leverage or 1.0)
    diagnostics["leverage"] = leverage
    if leverage > 1.0:
        multipliers.append(
            max(float(floors.get("leverage", 0.60)), min(1.0, 1.0 / leverage))
        )

    if context.depth_notional is not None:
        diagnostics["depth_notional"] = float(context.depth_notional)
    if context.gap_risk_pct is not None:
        diagnostics["gap_risk_pct"] = float(context.gap_risk_pct)
    if context.roundtrip_cost_bps is not None:
        diagnostics["roundtrip_cost_bps"] = float(context.roundtrip_cost_bps)
    if context.regime:
        diagnostics["regime"] = context.regime

    if not multipliers:
        return 1.0, diagnostics
    return float(max(0.0, min(1.0, min(multipliers)))), diagnostics


def evaluate_barbell_path_risk(
    *,
    context: Optional[BarbellMarketContext],
    cfg: BarbellConfig,
) -> Dict[str, Any]:
    """
    Evaluate whether a trade's implementation path is still barbell-safe.

    This is stricter than the confidence overlay multiplier. The multiplier may
    soften sizing when path risk is high; this helper answers the binary audit
    question: did the trade clear the configured path-risk bounds at all?
    """
    _, diagnostics = _market_context_multiplier(context, cfg)
    caps = cfg.edge_ratio_soft_caps or {}
    floors = cfg.overlay_floors or {}
    checks: Dict[str, Optional[bool]] = {}

    def _check_upper(key: str, cap_key: str) -> None:
        ratio = diagnostics.get(key)
        if ratio is None:
            checks[key] = None
            return
        try:
            cap = float(caps.get(cap_key))
        except (TypeError, ValueError):
            checks[key] = None
            return
        checks[key] = bool(float(ratio) <= cap)

    _check_upper("roundtrip_cost_to_edge", "roundtrip_cost_to_edge")
    _check_upper("gap_risk_to_edge", "gap_risk_to_edge")
    _check_upper("funding_to_edge", "funding_to_edge")
    _check_upper("liquidity_to_depth", "liquidity_to_depth")

    leverage = diagnostics.get("leverage")
    if leverage is not None:
        leverage_cap = float(caps.get("leverage", 1.0 / max(float(floors.get("leverage", 0.60)), 1e-6)))
        checks["leverage"] = bool(float(leverage) <= leverage_cap)
    else:
        checks["leverage"] = None

    active_checks = [bool(value) for value in checks.values() if value is not None]
    path_risk_ok = all(active_checks) if active_checks else True
    return {
        "barbell_path_risk_ok": bool(path_risk_ok),
        "path_risk_checks": checks,
        "diagnostics": diagnostics,
    }


def apply_barbell_confidence(
    *,
    ticker: str,
    base_confidence: float,
    cfg: BarbellConfig,
    context: Optional[BarbellMarketContext] = None,
) -> BarbellSizingResult:
    bucket = barbell_bucket(ticker, cfg)
    multipliers = barbell_confidence_multipliers(cfg)
    bucket_multiplier = float(multipliers.get(bucket, 1.0))
    regime_multiplier = _regime_multiplier(bucket, cfg, context.regime if context is not None else None)
    market_multiplier, diagnostics = _market_context_multiplier(context, cfg)
    path_risk = evaluate_barbell_path_risk(context=context, cfg=cfg)
    multiplier = float(bucket_multiplier * regime_multiplier * market_multiplier)
    conf = max(0.0, min(1.0, float(base_confidence)))
    effective = max(0.0, min(1.0, conf * multiplier))
    diagnostics.update(
        {
            "bucket_multiplier": bucket_multiplier,
            "regime_multiplier": regime_multiplier,
            "market_multiplier": market_multiplier,
            "strategy_style": cfg.strategy_style,
            "rebalance_frequency_bars": cfg.rebalance_frequency_bars,
            "barbell_path_risk_ok": path_risk.get("barbell_path_risk_ok"),
            "path_risk_checks": path_risk.get("path_risk_checks"),
        }
    )
    return BarbellSizingResult(
        bucket=bucket,
        multiplier=multiplier,
        effective_confidence=effective,
        bucket_multiplier=bucket_multiplier,
        regime_multiplier=regime_multiplier,
        market_multiplier=market_multiplier,
        diagnostics=diagnostics,
    )
