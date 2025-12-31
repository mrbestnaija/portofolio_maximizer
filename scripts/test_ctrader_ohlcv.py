#!/usr/bin/env python3
"""
Quick cTrader OHLCV sanity checker.

This helper uses your local .env credentials + config/ctrader_config.yml
to fetch a small OHLCV window from the cTrader Open API and then passes
it through CTraderExtractor so you can verify:

  - Authentication + account_id wiring
  - REST bars endpoint shape
  - Normalised OHLCV mapping (index + columns)

Usage examples (from repo root, with simpleTrader_env activated):

  python scripts/test_ctrader_ohlcv.py --symbol AAPL
  python scripts/test_ctrader_ohlcv.py --symbol EURUSD --days 10 --timeframe H1

Once you're happy with the mapping, you can flip DATA_SOURCE=ctrader in
your ETL/trader launchers or raise ctrader's priority in
config/data_sources_config.yml.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UTC = timezone.utc

from execution.ctrader_client import (  # noqa: E402
    CTraderClient,
    CTraderClientConfig,
    CTraderClientError,
)
from etl.ctrader_extractor import CTraderExtractor  # noqa: E402


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and inspect OHLCV bars from cTrader via CTraderClient + CTraderExtractor."
    )
    parser.add_argument(
        "--symbol",
        default="AAPL",
        help="Symbol/instrument name to request (e.g. AAPL, EURUSD).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Lookback window in days for OHLCV bars (default: 30).",
    )
    parser.add_argument(
        "--timeframe",
        default="D1",
        help="cTrader timeframe identifier for bars (default: D1).",
    )
    args = parser.parse_args(argv)

    symbol: str = args.symbol
    days: int = max(1, args.days)
    timeframe: str = args.timeframe

    now = datetime.now(UTC)
    end_date = now.date()
    start_date = end_date - timedelta(days=days)

    _print_header("cTrader configuration")
    try:
        cfg = CTraderClientConfig.from_env(environment="demo")
    except Exception as exc:
        print("Config/auth error while loading CTraderClientConfig.from_env(environment='demo'):")
        print(f"  {type(exc).__name__}: {exc}")
        print("\nCheck that USERNAME_CTRADER/EMAIL_CTRADER, PASSWORD_CTRADER, APPLICATION_NAME_CTRADER")
        print("and CTRADER_ACCOUNT_ID are set in your environment or .env file.")
        return 1

    print(f"Environment : {'demo' if cfg.is_demo else 'live'}")
    print(f"Account ID  : {cfg.account_id!r}")
    print(f"API base    : {cfg.api_base}")

    if cfg.account_id is None:
        print("\nERROR: CTRADER_ACCOUNT_ID is missing or invalid. Set it in .env before continuing.")
        return 1

    client = CTraderClient(config=cfg)

    _print_header(f"Raw bars from cTrader for {symbol} ({timeframe})")
    try:
        start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=UTC)
        end_dt = datetime.combine(end_date, datetime.min.time(), tzinfo=UTC)
        bars: List[Dict[str, Any]] = client.get_bars(
            symbol=symbol,
            start=start_dt,
            end=end_dt,
            timeframe=timeframe,
        )
        print(f"Bars payload type: {type(bars).__name__}, length={len(bars)}")
        if bars:
            print("First record keys:", sorted(bars[0].keys()))
            print("First record sample:", bars[0])
        else:
            print("No bars returned for the requested window.")
    except CTraderClientError as exc:
        print(f"cTraderClientError while requesting bars for {symbol}: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - upstream dependent
        print(f"Unexpected error while requesting bars: {type(exc).__name__}: {exc}")
        return 1

    _print_header(f"Normalised OHLCV via CTraderExtractor for {symbol}")
    try:
        extractor = CTraderExtractor(name="ctrader_test")
        frame = extractor.extract_ohlcv(
            tickers=[symbol],
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        print("DataFrame shape :", frame.shape)
        print("Columns         :", list(frame.columns))
        if not frame.empty:
            print("\nHead:")
            print(frame.head())
    except Exception as exc:
        print(f"Error while normalising bars via CTraderExtractor: {type(exc).__name__}: {exc}")
        return 1

    print("\nSUCCESS: cTrader OHLCV path appears wired. Review the samples above,")
    print("then consider setting DATA_SOURCE=ctrader or raising ctrader's priority")
    print("in config/data_sources_config.yml when you are ready to exercise it in ETL/trader runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
