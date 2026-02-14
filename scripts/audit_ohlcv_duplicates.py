"""
Audit duplicate OHLCV rows in the SQLite DB and optionally export a deduped snapshot.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrity.sqlite_guardrails import guarded_sqlite_connect


DEFAULT_DB = Path("data/portfolio_maximizer.db")
DEFAULT_TABLE = "ohlcv_data"
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]


def _parse_tickers(raw: Optional[str]) -> List[str]:
    if not raw:
        return DEFAULT_TICKERS
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _load_ohlcv(conn: sqlite3.Connection, table: str, ticker: str) -> pd.DataFrame:
    query = f"""
        SELECT date as Date, open as Open, high as High, low as Low, close as Close, volume as Volume, ticker
        FROM {table}
        WHERE ticker = ?
        ORDER BY date
    """
    df = pd.read_sql_query(query, conn, params=(ticker,), parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def _dedupe(df: pd.DataFrame, *, strategy: str) -> pd.DataFrame:
    if not df.index.has_duplicates:
        return df
    if strategy == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        agg = {col: "mean" for col in numeric_cols}
        agg["ticker"] = "last"
        return df.groupby(level=0).agg(agg)
    # Default: keep last row for duplicate timestamps.
    return df[~df.index.duplicated(keep="last")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit duplicate OHLCV rows in SQLite DB.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--table", type=str, default=DEFAULT_TABLE)
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS))
    parser.add_argument(
        "--export-deduped",
        type=Path,
        default=None,
        help="Optional output directory for deduped parquet snapshots.",
    )
    parser.add_argument(
        "--dedupe-strategy",
        choices=["last", "mean"],
        default="last",
        help="Deduplication strategy when exporting (default: last).",
    )
    args = parser.parse_args()

    tickers = _parse_tickers(args.tickers)
    if args.export_deduped:
        args.export_deduped.mkdir(parents=True, exist_ok=True)

    conn = guarded_sqlite_connect(str(args.db))
    try:
        for ticker in tickers:
            df = _load_ohlcv(conn, args.table, ticker)
            total_rows = len(df)
            unique_dates = df.index.nunique()
            dupes = total_rows - unique_dates

            print("\n" + "=" * 80)
            print(f"{ticker} :: rows={total_rows} unique_dates={unique_dates} duplicates={dupes}")
            if total_rows:
                print(f"Range: {df.index.min().date()} -> {df.index.max().date()}")

            if args.export_deduped:
                deduped = _dedupe(df, strategy=args.dedupe_strategy)
                out_path = args.export_deduped / f"{ticker}_deduped.parquet"
                deduped.to_parquet(out_path)
                print(f"[OK] Deduped snapshot saved -> {out_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
