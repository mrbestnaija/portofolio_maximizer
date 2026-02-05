"""
etl.synthetic_pricer
--------------------

Helpers for pricing synthetic instruments defined via synthetic_legs.

Design goals:
- Keep this module lightweight and DB-agnostic (it only needs a sqlite3
  connection object and plain dict/Dataclass structures).
- Defer detailed MTM policy choices to the caller; this module simply
  computes a neutral mark-to-market value for a synthetic structure by
  summing leg values using spot/option heuristics.

This is used by scripts/liquidate_open_trades.py to provide a more realistic
diagnostic MTM for instrument_type='synthetic' trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import sqlite3


@dataclass
class SyntheticLeg:
    """Single leg in a synthetic structure."""

    id: int
    synthetic_trade_id: int
    leg_type: str
    ticker: Optional[str]
    underlying_ticker: Optional[str]
    direction: int
    quantity: float
    strike: Optional[float]
    expiry: Optional[str]
    multiplier: float = 1.0


def load_synthetic_legs(
    conn: sqlite3.Connection,
    synthetic_trade_id: int,
) -> List[SyntheticLeg]:
    """
    Load all legs for a given synthetic trade id.

    The schema is defined in DatabaseManager._initialize_schema() and mirrors
    the design in Documentation/MTM_AND_LIQUIDATION_IMPLEMENTATION_PLAN.md.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            id,
            synthetic_trade_id,
            leg_type,
            ticker,
            underlying_ticker,
            direction,
            quantity,
            strike,
            expiry,
            multiplier
        FROM synthetic_legs
        WHERE synthetic_trade_id = ?
        """,
        (synthetic_trade_id,),
    )
    rows = cur.fetchall()
    legs: List[SyntheticLeg] = []
    for row in rows:
        # Row can be either a tuple or sqlite3.Row; use indices for robustness.
        (
            leg_id,
            trade_id,
            leg_type,
            ticker,
            underlying_ticker,
            direction,
            quantity,
            strike,
            expiry,
            multiplier,
        ) = row
        legs.append(
            SyntheticLeg(
                id=int(leg_id),
                synthetic_trade_id=int(trade_id),
                leg_type=str(leg_type or "").lower(),
                ticker=str(ticker) if ticker is not None else None,
                underlying_ticker=str(underlying_ticker) if underlying_ticker is not None else None,
                direction=int(direction or 0),
                quantity=float(quantity or 0.0),
                strike=float(strike) if strike is not None else None,
                expiry=str(expiry) if expiry is not None else None,
                multiplier=float(multiplier or 1.0),
            )
        )
    return legs


def _option_intrinsic(spot: float, strike: float, option_type: str) -> float:
    """Plain intrinsic value for a call/put leg."""
    if option_type == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def compute_synthetic_mtm(
    legs: List[SyntheticLeg],
    spot_prices: Dict[str, float],
) -> float:
    """
    Compute a neutral MTM value for a synthetic position as the sum of leg
    values:

        MTM_synthetic = sum_i direction_i * quantity_i * unit_value_i

    where:
    - spot / index / crypto legs use the provided spot_prices,
    - option legs use intrinsic value (per leg_type) with multiplier.

    This function intentionally does not apply conservative clamps or
    Black–Scholes; those policy choices are handled by the caller at the
    parent trade level.
    """
    if not legs:
        return 0.0

    total = 0.0
    for leg in legs:
        if not leg.quantity:
            continue
        base_ticker = (leg.underlying_ticker or leg.ticker or "").upper()
        if not base_ticker:
            continue
        spot = spot_prices.get(base_ticker)
        if spot is None:
            # Without a spot reference this leg cannot contribute to MTM.
            continue
        spot = float(spot)
        leg_type = leg.leg_type.lower()
        multiplier = float(leg.multiplier or 1.0)

        if leg_type in ("spot", "cash"):
            unit_value = spot * multiplier
        elif leg_type in ("call", "put"):
            k = float(leg.strike or 0.0)
            if k <= 0 or spot <= 0:
                unit_value = 0.0
            else:
                unit_value = _option_intrinsic(spot, k, leg_type) * multiplier
        else:
            # Unknown leg type – treat as zero-valued to avoid fantasy PnL.
            unit_value = 0.0

        direction = 1 if leg.direction >= 0 else -1
        qty = float(leg.quantity or 0.0)
        total += direction * qty * unit_value

    return total

