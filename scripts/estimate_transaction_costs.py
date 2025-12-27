#!/usr/bin/env python3
"""
estimate_transaction_costs.py
-----------------------------

Estimate empirical transaction costs (slippage + commission) from realised
trades and emit grouped statistics for use in config tuning.

Design goals:
- Read-only helper over trade_executions; does not modify schema or configs.
- Group results by ticker or simple derived groups (e.g. prefix-based
  asset buckets) so callers can map them onto execution / TS configs.
- Produce JSON outputs that automation or notebooks can consume.

NOTE:
- The current schema does not store explicit mid-price-at-fill, so this
  scaffold treats `commission` as the direct transaction cost signal and
  reports realised_pnl distributions separately. A future refinement can
  extend trade_executions with mid-price snapshots to measure slippage.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import pandas as pd
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in __import__("sys").modules["sys"].path:
    __import__("sys").modules["sys"].path.insert(0, str(ROOT_PATH))

from etl.database_manager import DatabaseManager  # noqa: E402

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@dataclass
class CostStats:
    group: str
    trades: int
    commission_median: float
    commission_mean: float
    commission_p95: float
    commission_median_bps: float
    commission_mean_bps: float
    commission_p95_bps: float
    slippage_median_bps: float
    slippage_mean_bps: float
    slippage_p95_bps: float
    total_cost_median_bps: float
    total_cost_mean_bps: float
    total_cost_p95_bps: float
    roundtrip_cost_median_bps: float
    roundtrip_cost_mean_bps: float
    roundtrip_cost_p95_bps: float
    pnl_median: float
    pnl_mean: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group,
            "trades": self.trades,
            "commission_median": self.commission_median,
            "commission_mean": self.commission_mean,
            "commission_p95": self.commission_p95,
            "commission_median_bps": self.commission_median_bps,
            "commission_mean_bps": self.commission_mean_bps,
            "commission_p95_bps": self.commission_p95_bps,
            "slippage_median_bps": self.slippage_median_bps,
            "slippage_mean_bps": self.slippage_mean_bps,
            "slippage_p95_bps": self.slippage_p95_bps,
            "total_cost_median_bps": self.total_cost_median_bps,
            "total_cost_mean_bps": self.total_cost_mean_bps,
            "total_cost_p95_bps": self.total_cost_p95_bps,
            "roundtrip_cost_median_bps": self.roundtrip_cost_median_bps,
            "roundtrip_cost_mean_bps": self.roundtrip_cost_mean_bps,
            "roundtrip_cost_p95_bps": self.roundtrip_cost_p95_bps,
            "pnl_median": self.pnl_median,
            "pnl_mean": self.pnl_mean,
        }


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    q = max(0.0, min(100.0, q))
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    weight = pos - lo
    return xs[lo] * (1 - weight) + xs[hi] * weight


def _group_key(ticker: str, mode: str) -> str:
    ticker = (ticker or "").upper()
    if mode == "ticker":
        return ticker or "UNKNOWN"
    # Simple prefix-based asset buckets; callers can override by
    # post-processing the JSON if they prefer more nuanced classes.
    if mode == "asset_class":
        if ticker.endswith("=X"):
            return "FX"
        if ticker.endswith("-USD") or ticker in {"BTC", "ETH"}:
            return "CRYPTO"
        if "^" in ticker:
            return "INDEX"
        if any(ticker.endswith(suffix) for suffix in (".NS", ".TW", ".L")):
            return "INTL_EQUITY"
        return "US_EQUITY"
    return "UNKNOWN"


def _load_trades(
    db: DatabaseManager,
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[Dict[str, Any]]:
    query = """
        SELECT ticker, trade_date, commission, total_value, price, mid_price, mid_slippage_bps, realized_pnl
        FROM trade_executions
    """
    params: List[Any] = []
    if start_date:
        query += " AND trade_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND trade_date <= ?"
        params.append(end_date)
    query += " ORDER BY trade_date ASC, id ASC"

    db.cursor.execute(query, params)
    rows = db.cursor.fetchall()
    return [dict(row) for row in rows]


def _compute_group_stats(records: Dict[str, List[Dict[str, Any]]]) -> List[CostStats]:
    stats: List[CostStats] = []
    for group, rows in records.items():
        commissions: List[float] = []
        commissions_bps: List[float] = []
        slippages_bps: List[float] = []
        total_costs_bps: List[float] = []
        pnls: List[float] = []
        for row in rows:
            commission = float(row.get("commission") or 0.0)
            commissions.append(commission)
            total_value = float(row.get("total_value") or 0.0)
            commission_bps = (commission / total_value) * 1e4 if total_value > 0 else 0.0
            commissions_bps.append(commission_bps)

            slippage_bps = row.get("mid_slippage_bps")
            if slippage_bps is None:
                mid_price = row.get("mid_price")
                price = row.get("price")
                try:
                    mid_f = float(mid_price) if mid_price is not None else None
                    px_f = float(price) if price is not None else None
                except (TypeError, ValueError):
                    mid_f = None
                    px_f = None
                if mid_f and px_f and mid_f > 0:
                    slippage_bps = ((px_f - mid_f) / mid_f) * 1e4
            try:
                slippage_bps_f = float(slippage_bps) if slippage_bps is not None else 0.0
            except (TypeError, ValueError):
                slippage_bps_f = 0.0
            slippage_abs = abs(slippage_bps_f)
            slippages_bps.append(slippage_abs)

            total_cost_bps = commission_bps + slippage_abs
            total_costs_bps.append(total_cost_bps)

            pnl_raw = row.get("realized_pnl")
            if pnl_raw is not None:
                pnls.append(float(pnl_raw or 0.0))

        trades = len(rows)
        if trades == 0:
            continue
        commission_mean = sum(commissions) / trades if trades else 0.0
        commission_mean_bps = sum(commissions_bps) / trades if trades else 0.0
        slippage_mean_bps = sum(slippages_bps) / trades if trades else 0.0
        total_cost_mean_bps = sum(total_costs_bps) / trades if trades else 0.0
        pnl_count = len(pnls)
        pnl_mean = sum(pnls) / pnl_count if pnl_count else 0.0
        roundtrip_cost_mean_bps = 2.0 * total_cost_mean_bps
        stats.append(
            CostStats(
                group=group,
                trades=trades,
                commission_median=_percentile(commissions, 50.0),
                commission_mean=commission_mean,
                commission_p95=_percentile(commissions, 95.0),
                commission_median_bps=_percentile(commissions_bps, 50.0),
                commission_mean_bps=commission_mean_bps,
                commission_p95_bps=_percentile(commissions_bps, 95.0),
                slippage_median_bps=_percentile(slippages_bps, 50.0),
                slippage_mean_bps=slippage_mean_bps,
                slippage_p95_bps=_percentile(slippages_bps, 95.0),
                total_cost_median_bps=_percentile(total_costs_bps, 50.0),
                total_cost_mean_bps=total_cost_mean_bps,
                total_cost_p95_bps=_percentile(total_costs_bps, 95.0),
                roundtrip_cost_median_bps=2.0 * _percentile(total_costs_bps, 50.0),
                roundtrip_cost_mean_bps=roundtrip_cost_mean_bps,
                roundtrip_cost_p95_bps=2.0 * _percentile(total_costs_bps, 95.0),
                pnl_median=_percentile(pnls, 50.0),
                pnl_mean=pnl_mean,
            )
        )
    return stats


def _compute_from_synthetic(dataset_path: str, grouping: str) -> Dict[str, Any]:
    path = Path(dataset_path)
    if path.name == "latest.json" and path.exists():
        try:
            payload = json.loads(path.read_text())
            dataset_path = payload.get("dataset_path") or dataset_path
            path = Path(dataset_path)
        except Exception:
            pass
    frames: List[pd.DataFrame] = []
    if path.is_dir():
        frames = [pd.read_parquet(pq) for pq in path.glob("*.parquet")]
    elif path.suffix == ".parquet":
        frames = [pd.read_parquet(path)]
    if not frames:
        return {}
    df = pd.concat(frames).sort_index()
    if "TxnCostBps" not in df.columns:
        return {}
    stats = []
    for grp, df_g in df.groupby("ticker" if grouping == "ticker" else None):
        if df_g.empty:
            continue
        txn = df_g["TxnCostBps"].dropna().astype(float)
        impact = df_g["ImpactBps"].dropna().astype(float) if "ImpactBps" in df_g.columns else pd.Series(dtype=float)
        stats.append(
            {
                "group": grp if grouping == "ticker" else "synthetic",
                "trades": int(len(txn)),
                "txn_cost_median_bps": float(txn.median()) if len(txn) else 0.0,
                "txn_cost_mean_bps": float(txn.mean()) if len(txn) else 0.0,
                "impact_median_bps": float(impact.median()) if len(impact) else 0.0,
                "impact_mean_bps": float(impact.mean()) if len(impact) else 0.0,
            }
        )
    return {"grouping": grouping, "stats": stats}


@click.command()
@click.option("--db-path", default="data/portfolio_maximizer.db", show_default=True, help="SQLite database path used by the trading engine.")
@click.option("--lookback-days", default=365, show_default=True, help="Number of days to look back from --as-of for trade samples.")
@click.option("--as-of", default=None, help="Reference date in YYYY-MM-DD (default: today UTC).")
@click.option("--grouping", type=click.Choice(["ticker", "asset_class"], case_sensitive=False), default="asset_class", show_default=True, help="Grouping mode for cost statistics.")
@click.option("--min-trades", default=5, show_default=True, help="Minimum trades per group required to emit stats.")
@click.option("--output", default="logs/automation/transaction_costs.json", show_default=True, help="Path to write JSON summary (directories are created as needed).")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
@click.option("--synthetic-dataset-path", default=None, help="Optional synthetic dataset path/latest.json to derive txn cost proxies.")
def main(
    db_path: str,
    lookback_days: int,
    as_of: Optional[str],
    grouping: str,
    min_trades: int,
    output: str,
    verbose: bool,
    synthetic_dataset_path: Optional[str],
) -> None:
    """
    Estimate transaction cost statistics from realised trades.

    The output is a JSON document with per-group commission statistics and
    realised PnL distributions. Callers can use this to calibrate
    friction buffers and minimum expected returns in configs.
    """
    _configure_logging(verbose)

    if as_of:
        ref_date = datetime.fromisoformat(as_of).date()
    else:
        ref_date = datetime.utcnow().date()
    start = ref_date - timedelta(days=int(lookback_days))
    start_iso = start.isoformat()
    end_iso = ref_date.isoformat()

    synthetic_stats = _compute_from_synthetic(synthetic_dataset_path, grouping) if synthetic_dataset_path else {}

    with DatabaseManager(db_path=db_path) as db:
        rows = _load_trades(db, start_iso, end_iso)
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = _group_key(str(row.get("ticker") or ""), grouping.lower())
            grouped[key].append(row)

        filtered = {g: recs for g, recs in grouped.items() if len(recs) >= int(min_trades)}
        stats = _compute_group_stats(filtered) if filtered else []

    payload = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "db_path": db_path,
            "as_of": ref_date.isoformat(),
            "lookback_days": lookback_days,
            "grouping": grouping,
            "min_trades": min_trades,
        },
        "groups": [s.to_dict() for s in stats],
        "synthetic": synthetic_stats,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Transaction cost summary written to %s (%s groups)",
        out_path,
        len(stats),
    )


if __name__ == "__main__":
    main()
