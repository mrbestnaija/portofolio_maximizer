#!/usr/bin/env python3
"""
update_regime_state.py
----------------------

Compute simple per-ticker regime / exploration state from realised trades
and persist it to a lightweight YAML file that the execution layer can
consult for risk scaling.

This is intentionally simple:
  - For each ticker, look at the last N closed trades (with realised_pnl).
  - If n_trades < N_min   -> mode = "exploration"
  - Else compute a Sharpe-like score S_N = mean / (std + eps) over those
    trades and classify:
        S_N > S_high       -> state = "green"
        S_N < S_low        -> state = "red"
        otherwise          -> state = "neutral"

The output is written to config/regime_state.yml:

regime_state:
  AAPL:
    n_trades: 12
    sharpe_N: 0.85
    mode: exploration|exploitation
    state: green|red|neutral

The paper trading engine can then scale per-trade risk based on this file
without changing any upstream ETL or TS routing logic.
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml

from etl.database_manager import DatabaseManager

ROOT_PATH = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = ROOT_PATH / "config" / "regime_state.yml"


def _compute_regime_state(
    db: DatabaseManager,
    lookback_trades: int = 20,
    min_trades_for_exploitation: int = 20,
    s_high: float = 0.5,
    s_low: float = -0.2,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute regime/exploration state per ticker from realised PnL.

    Parameters roughly align with the critical review: we want at least
    ~20 trades before trusting Sharpe; below that we stay in exploration.
    """
    cursor = db.cursor.execute(
        """
        SELECT ticker, realized_pnl
        FROM trade_executions
        WHERE realized_pnl IS NOT NULL
        ORDER BY trade_date DESC, rowid DESC
        """
    )

    pnl_buckets: Dict[str, List[float]] = defaultdict(list)
    for row in cursor.fetchall():
        ticker = row["ticker"]
        pnl = row["realized_pnl"]
        if pnl is None:
            continue
        if len(pnl_buckets[ticker]) < lookback_trades:
            pnl_buckets[ticker].append(float(pnl))

    regime_state: Dict[str, Dict[str, Any]] = {}
    eps = 1e-8
    for ticker, pnls in pnl_buckets.items():
        n = len(pnls)
        if n == 0:
            continue
        if n < min_trades_for_exploitation:
            regime_state[ticker] = {
                "n_trades": n,
                "sharpe_N": None,
                "mode": "exploration",
                "state": "neutral",
            }
            continue

        mean_pnl = statistics.mean(pnls)
        std_pnl = statistics.pstdev(pnls) if n > 1 else 0.0
        sharpe_like = mean_pnl / (std_pnl + eps)

        if sharpe_like > s_high:
            state = "green"
        elif sharpe_like < s_low:
            state = "red"
        else:
            state = "neutral"

        regime_state[ticker] = {
            "n_trades": n,
            "sharpe_N": sharpe_like,
            "mode": "exploitation",
            "state": state,
        }

    return regime_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update per-ticker regime/exploration state from realised PnL."
    )
    parser.add_argument(
        "--db-path",
        default="data/portfolio_maximizer.db",
        help="SQLite database path (default: data/portfolio_maximizer.db)",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to write regime_state.yml (default: config/regime_state.yml)",
    )
    parser.add_argument(
        "--lookback-trades",
        type=int,
        default=20,
        help="Number of recent trades per ticker to use for regime calculation.",
    )
    parser.add_argument(
        "--min-trades-for-exploitation",
        type=int,
        default=20,
        help="Minimum closed trades required before switching from exploration.",
    )
    args = parser.parse_args()

    db = DatabaseManager(db_path=args.db_path)
    state = _compute_regime_state(
        db,
        lookback_trades=args.lookback_trades,
        min_trades_for_exploitation=args.min_trades_for_exploitation,
    )
    db.close()

    payload = {"regime_state": state}
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    print(f"Wrote regime state for {len(state)} tickers to {out_path}")


if __name__ == "__main__":
    main()

