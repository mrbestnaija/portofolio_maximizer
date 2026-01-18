"""
Seed a small cached OHLCV dataset to enable a no-network pipeline dry-run.

Creates a synthetic AAPL dataset in `data/raw` via DataStorage so that the
YFinance extractor's cache-first logic will hit the local cache.

Usage:
  python scripts/seed_cache_for_dry_run.py
"""
from datetime import datetime
import numpy as np
import pandas as pd

from etl.data_storage import DataStorage


def make_synthetic_ohlcv(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start=start, end=end, freq="B")
    n = len(dates)
    # Simple synthetic price path
    base = 100.0
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.0005, scale=0.01, size=n)
    prices = base * np.cumprod(1 + returns)

    close = prices
    open_ = close * (1 + rng.normal(0, 0.002, size=n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, size=n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, size=n)))
    volume = (1_000_000 * (1 + rng.normal(0, 0.05, size=n))).astype(int)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "Adj Close": close,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


if __name__ == "__main__":
    # Pick a stable historical window; file mtime will be "now" so cache is fresh
    start = "2024-01-02"
    end = "2024-01-19"
    df = make_synthetic_ohlcv(start, end)

    storage = DataStorage(base_path="data")
    path = storage.save(df, stage="raw", symbol="AAPL")
    print(f"Seeded cache at: {path}")
