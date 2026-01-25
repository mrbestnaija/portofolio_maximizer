"""Fetch fresh data directly from yfinance for multi-ticker validation."""
import yfinance as yf
from datetime import datetime
from pathlib import Path

tickers = ["AAPL", "MSFT", "NVDA"]
start_date = "2024-07-01"
end_date = "2026-01-18"

print(f"Fetching data for {tickers} from {start_date} to {end_date}...")

for ticker in tickers:
    print(f"\n[{ticker}] Fetching...")
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)

    if df.empty:
        print(f"  [ERROR] No data returned for {ticker}")
        continue

    # Save to cache directory
    cache_path = Path(f"data/raw/{ticker}_fresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
    df.to_parquet(cache_path)
    print(f"  [OK] Fetched {len(df)} rows, saved to {cache_path.name}")

print("\n[COMPLETE] Fresh data fetched successfully")
