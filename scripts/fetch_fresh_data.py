"""Fetch fresh data directly from yfinance for multi-ticker validation."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import yfinance as yf


DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]
DEFAULT_START = "2024-07-01"
DEFAULT_END = "2026-01-18"
DEFAULT_OUTPUT_DIR = Path("data/raw")


def _parse_tickers(raw: str | None) -> List[str]:
    if not raw:
        return DEFAULT_TICKERS
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_frame(frame, output_dir: Path, ticker: str, fmt: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if fmt == "csv":
        dest = output_dir / f"{ticker}_fresh_{ts}.csv"
        frame.to_csv(dest)
        return dest
    dest = output_dir / f"{ticker}_fresh_{ts}.parquet"
    frame.to_parquet(dest)
    return dest


def _configure_cache_dir(cache_dir: str | None) -> None:
    if not cache_dir:
        return
    os.environ["YFINANCE_CACHE_DIR"] = cache_dir
    os.environ["YF_CACHE_DIR"] = cache_dir


def fetch_fresh_data(
    tickers: Iterable[str],
    *,
    start_date: str,
    end_date: str,
    output_dir: Path,
    fmt: str,
    cache_dir: str | None,
) -> None:
    _configure_cache_dir(cache_dir)
    output_dir = _ensure_output_dir(output_dir)

    tickers = list(tickers)
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")

    for ticker in tickers:
        print(f"\n[{ticker}] Fetching...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            print(f"  [ERROR] No data returned for {ticker}")
            continue

        dest = _save_frame(df, output_dir, ticker, fmt)
        print(f"  [OK] Fetched {len(df)} rows, saved to {dest.name}")

    print("\n[COMPLETE] Fresh data fetched successfully")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch fresh data from yfinance.")
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated ticker list (default: AAPL,MSFT,NVDA).",
    )
    parser.add_argument(
        "--start",
        dest="start_date",
        type=str,
        default=DEFAULT_START,
        help=f"Start date (YYYY-MM-DD). Default: {DEFAULT_START}",
    )
    parser.add_argument(
        "--end",
        dest="end_date",
        type=str,
        default=DEFAULT_END,
        help=f"End date (YYYY-MM-DD). Default: {DEFAULT_END}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store fetched data (default: data/raw).",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (default: parquet).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional yfinance cache directory override.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    tickers = _parse_tickers(args.tickers)
    fetch_fresh_data(
        tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        fmt=args.format,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
