#!/usr/bin/env python3
"""
liquidate_open_trades.py
------------------------

Diagnostic liquidation helper to convert open trades (`realized_pnl IS NULL`)
into realized trades using configurable mark-to-market heuristics.

The original version was **spot-only and yfinance-dependent**. As the project
expands to options, crypto, and synthetic exposures, this script now:

- Supports basic asset class / instrument type hints when available
  (`asset_class`, `instrument_type`, `underlying_ticker`, `strike`, `expiry`,
  `multiplier` columns on `trade_executions` when present).
- Uses a multi-step MTM hierarchy:
  - Latest close from the local OHLCV database (`ohlcv_data`) when possible.
  - yfinance as a best-effort vendor quote (if available).
  - Entry price as the final fallback.
- Applies a simple, pluggable pricing policy:
  - Equities/crypto spot: neutral or conservative MTM.
  - Options: intrinsic-only (with Black–Scholes reserved for a future upgrade).
  - Synthetic/unknown instruments: fall back to entry price.

This remains an **evidence-only tool** (for research and diagnostics); it is
not intended for authoritative PnL reporting.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import click
import math
import datetime
import yaml

from etl.synthetic_pricer import load_synthetic_legs, compute_synthetic_mtm

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None


@dataclass
class TradeRow:
    """Lightweight view over a trade_executions row."""

    trade_id: int
    ticker: str
    action: str
    shares: float
    price: float
    commission: float
    asset_class: str = "equity"
    instrument_type: str = "spot"
    underlying_ticker: Optional[str] = None
    strike: Optional[float] = None
    expiry: Optional[str] = None
    multiplier: float = 1.0


def _fetch_vendor_prices(tickers: Dict[str, float]) -> Dict[str, float]:
    """Fetch latest close from yfinance, falling back to caller defaults."""
    prices: Dict[str, float] = {}
    if not yf or not tickers:
        return prices
    try:
        data = yf.download(
            list(tickers.keys()),
            period="1d",
            progress=False,
            threads=False,
        )
        if data.empty:
            return prices
        if "Close" not in data.columns:
            return prices
        close = data["Close"]
        for t in tickers:
            try:
                # Handle multi-index when multiple tickers requested
                series = close[t] if hasattr(close, "columns") and t in close.columns else close
                val = series.dropna().iloc[-1]
            except Exception:
                continue
            prices[t] = float(val)
    except Exception:
        return prices
    return prices


def _load_last_close_from_db(conn: sqlite3.Connection, tickers: List[str]) -> Dict[str, float]:
    """Load latest close price per ticker from the local OHLCV table, if present."""
    if not tickers:
        return {}
    try:
        cur = conn.cursor()
        placeholders = ",".join("?" for _ in tickers)
        query = f"""
            SELECT o.ticker, o.close
            FROM ohlcv_data o
            JOIN (
                SELECT ticker, MAX(date) AS max_date
                FROM ohlcv_data
                WHERE ticker IN ({placeholders})
                GROUP BY ticker
            ) m
            ON o.ticker = m.ticker AND o.date = m.max_date
        """
        cur.execute(query, tickers)
        rows = cur.fetchall()
    except Exception:
        return {}
    return {str(ticker): float(close) for (ticker, close) in rows}


def _build_spot_price_map(conn: sqlite3.Connection, trades: List[TradeRow]) -> Dict[str, float]:
    """
    Construct a mapping ticker -> spot price using:
    1) Local OHLCV DB, 2) yfinance, 3) entry price fallback.
    """
    # Base tickers: use underlying when present, otherwise the trade ticker.
    base_tickers: Dict[str, float] = {}
    for t in trades:
        key = t.underlying_ticker or t.ticker
        base_tickers[key] = float(t.price)

    tickers = list(base_tickers.keys())
    from_db = _load_last_close_from_db(conn, tickers)
    remaining = {t: base_tickers[t] for t in tickers if t not in from_db}
    from_vendor = _fetch_vendor_prices(remaining)

    prices: Dict[str, float] = {}
    for t in tickers:
        if t in from_db:
            prices[t] = from_db[t]
        elif t in from_vendor:
            prices[t] = from_vendor[t]
        else:
            prices[t] = base_tickers[t]
    return prices


def _load_price_history_from_db(
    conn: sqlite3.Connection,
    ticker: str,
    lookback_days: int = 60,
) -> List[float]:
    """Load recent close prices for realised volatility estimation."""
    try:
        cur = conn.cursor()
        end = datetime.date.today()
        start = end - datetime.timedelta(days=max(lookback_days, 2))
        cur.execute(
            """
            SELECT close
            FROM ohlcv_data
            WHERE ticker = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
            """,
            (ticker, start.isoformat(), end.isoformat()),
        )
        rows = cur.fetchall()
        return [float(v[0]) for v in rows if v[0] is not None]
    except Exception:
        return []


def _estimate_realised_vol(returns: List[float]) -> Optional[float]:
    """Estimate realised volatility from a list of returns."""
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / max(len(returns) - 1, 1)
    return math.sqrt(var)


def _load_risk_free_rate(root: Path) -> float:
    """Best-effort risk-free rate from quant_success_config.yml, with sane default."""
    cfg_path = root / "config" / "quant_success_config.yml"
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        qv = raw.get("quant_validation") or {}
        rfr = qv.get("risk_free_rate")
        return float(rfr) if isinstance(rfr, (int, float)) else 0.02
    except Exception:
        return 0.02


def _black_scholes_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    option_type: str,
) -> float:
    """Plain Black–Scholes call/put price using continuous compounding."""
    if spot <= 0 or strike <= 0 or vol <= 0 or time_to_expiry <= 0:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * time_to_expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t

    # Standard normal CDF via error function.
    def _phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    if option_type == "call":
        return spot * _phi(d1) - strike * math.exp(-rate * time_to_expiry) * _phi(d2)
    else:
        return strike * math.exp(-rate * time_to_expiry) * _phi(-d2) - spot * _phi(-d1)


def _mark_to_market(
    trade: TradeRow,
    spot_prices: Dict[str, float],
    pricing_policy: str,
) -> float:
    """
    Return a synthetic close price per *unit* of the trade instrument.

    - For spot (equity/crypto): applies neutral or conservative MTM.
    - For options (call/put): intrinsic-only MTM for now (BS reserved for future).
    - For synthetic/unknown: fall back to entry price.
    """
    policy = pricing_policy.lower()
    action = trade.action.upper()
    base_ticker = trade.underlying_ticker or trade.ticker
    spot = float(spot_prices.get(base_ticker, trade.price))
    entry = float(trade.price)

    asset_class = (trade.asset_class or "equity").lower()
    instrument = (trade.instrument_type or "spot").lower()

    # Spot equities / ETFs / crypto
    if instrument in ("spot", "") and asset_class in ("equity", "etf", "crypto", "fx", "forex"):
        if policy == "conservative":
            # Do not mark unrealised gains: clamp toward entry depending on side.
            if action == "BUY":
                return min(spot, entry)
            return max(spot, entry)
        # Neutral policy: latest spot
        return spot

    # Simple options (calls/puts) on equities/ETFs/indices
    if instrument in ("call", "put"):
        k = float(trade.strike or 0.0)
        intrinsic: float
        if instrument == "call":
            intrinsic = max(spot - k, 0.0)
        else:
            intrinsic = max(k - spot, 0.0)

        if policy == "intrinsic":
            return intrinsic

        if policy == "bs_model":
            # Black–Scholes pricing using realised volatility and risk-free rate.
            root = Path(__file__).resolve().parent.parent
            rate = _load_risk_free_rate(root)
            # Load recent underlying history for realised vol.
            # NOTE: we rely on the caller to have spot_prices built from DB so
            # the same connection can be reused for history; when unavailable,
            # this branch falls back to intrinsic.
            # This helper is intentionally simple and defensive.
            # We approximate 60-day realised volatility.
            try:
                # We assume spot_prices came from DB/vendor; for volatility we
                # reload DB via a short-lived connection.
                conn = sqlite3.connect(root / "data" / "portfolio_maximizer.db")
            except Exception:
                return intrinsic
            try:
                prices = _load_price_history_from_db(conn, base_ticker, lookback_days=60)
            finally:
                conn.close()

            if len(prices) < 2:
                return intrinsic
            rets = [
                math.log(max(prices[i] / prices[i - 1], 1e-12))
                for i in range(1, len(prices))
                if prices[i - 1] > 0
            ]
            vol = _estimate_realised_vol(rets)
            if vol is None or vol <= 0:
                return intrinsic

            # Time to expiry in years (ACT/365).
            try:
                if trade.expiry:
                    expiry_dt = datetime.date.fromisoformat(str(trade.expiry))
                else:
                    return intrinsic
            except Exception:
                return intrinsic
            today = datetime.date.today()
            days = max((expiry_dt - today).days, 0)
            if days <= 0:
                return intrinsic
            t_years = days / 365.0

            bs_price = _black_scholes_price(
                spot=spot,
                strike=k,
                rate=rate,
                vol=vol,
                time_to_expiry=t_years,
                option_type=instrument,
            )
            return max(bs_price, 0.0)

        # Default: last known premium (entry price per contract)
        return entry

    # Synthetic instruments: decompose into legs when definitions exist.
    if instrument == "synthetic":
        root = Path(__file__).resolve().parent.parent
        db_path = root / "data" / "portfolio_maximizer.db"
        try:
            conn = sqlite3.connect(db_path)
        except Exception:
            # If we cannot open the DB, fall back to entry price.
            return entry

        try:
            legs = load_synthetic_legs(conn, trade.trade_id)
        except Exception:
            conn.close()
            return entry

        try:
            if not legs:
                # No leg definitions – keep behaviour explicit and non-fantasy.
                return entry

            leg_tickers = sorted(
                {
                    (leg.underlying_ticker or leg.ticker or "").upper()
                    for leg in legs
                    if (leg.underlying_ticker or leg.ticker)
                }
            )
            if not leg_tickers:
                return entry

            # Build a spot map for leg underlyings using the same DB/vendor
            # hierarchy as regular spot MTM, but scoped to leg tickers.
            from_db = _load_last_close_from_db(conn, leg_tickers)
            remaining = {t: spot_prices.get(t, entry) for t in leg_tickers if t not in from_db}
            from_vendor = _fetch_vendor_prices(remaining)

            leg_spots: Dict[str, float] = {}
            for t in leg_tickers:
                if t in from_db:
                    leg_spots[t] = from_db[t]
                elif t in from_vendor:
                    leg_spots[t] = from_vendor[t]
                else:
                    leg_spots[t] = remaining.get(t, entry)

            synthetic_value = compute_synthetic_mtm(legs, leg_spots)
            # Interpret compute_synthetic_mtm output as a per-unit synthetic
            # price, consistent with trade.price semantics.
            return float(synthetic_value)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # Fallback: keep entry as MTM for unknown instruments.
    return entry


@click.command()
@click.option(
    "--db-path",
    default="data/portfolio_maximizer.db",
    show_default=True,
    help="SQLite database path used by the trading engine.",
)
@click.option(
    "--pricing-policy",
    type=click.Choice(["neutral", "conservative", "intrinsic", "bs_model"]),
    default="neutral",
    show_default=True,
    help=(
        "Mark-to-market policy: 'neutral' or 'conservative' for spot; "
        "'intrinsic' / 'bs_model' currently behave as intrinsic for options."
    ),
)
def main(db_path: str, pricing_policy: str) -> None:
    path = Path(db_path)
    if not path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Discover optional columns so the script remains compatible with older schemas.
    cur.execute("PRAGMA table_info(trade_executions)")
    cols = {row["name"] for row in cur.fetchall()}

    required = ["id", "ticker", "action", "shares", "price", "commission"]
    optional = [
        "asset_class",
        "instrument_type",
        "underlying_ticker",
        "strike",
        "expiry",
        "multiplier",
    ]
    select_cols = required + [c for c in optional if c in cols]
    cur.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM trade_executions
        WHERE realized_pnl IS NULL
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("No open trades found.")
        conn.close()
        return

    trades: List[TradeRow] = []
    for row in rows:
        data = dict(row)
        trades.append(
            TradeRow(
                trade_id=int(data["id"]),
                ticker=str(data["ticker"]),
                action=str(data["action"]),
                shares=float(data["shares"]),
                price=float(data["price"]),
                commission=float(data.get("commission") or 0.0),
                asset_class=str(data.get("asset_class") or "equity"),
                instrument_type=str(data.get("instrument_type") or "spot"),
                underlying_ticker=data.get("underlying_ticker"),
                strike=float(data["strike"]) if data.get("strike") is not None else None,
                expiry=str(data["expiry"]) if data.get("expiry") is not None else None,
                multiplier=float(data.get("multiplier") or 1.0),
            )
        )

    spot_prices = _build_spot_price_map(conn, trades)

    updated = 0
    for trade in trades:
        entry = float(trade.price)
        mtm = float(_mark_to_market(trade, spot_prices, pricing_policy=pricing_policy))
        shares = float(trade.shares)
        commission = float(trade.commission or 0.0)

        if trade.action.upper() == "BUY":
            pnl = (mtm - entry) * shares - commission
        else:  # SELL
            pnl = (entry - mtm) * shares - commission
        pnl_pct = pnl / (entry * shares) if entry * shares else 0.0

        cur.execute(
            """
            UPDATE trade_executions
            SET realized_pnl = ?, realized_pnl_pct = ?, holding_period_days = 0
            WHERE id = ?
            """,
            (pnl, pnl_pct, trade.trade_id),
        )
        updated += 1

    conn.commit()
    conn.close()
    print(f"Liquidated {updated} open trades using pricing_policy={pricing_policy}.")


if __name__ == "__main__":
    main()
