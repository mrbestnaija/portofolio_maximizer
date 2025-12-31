import io
from pathlib import Path

import pandas as pd
import pytest

from etl.ticker_discovery.alpha_vantage_loader import AlphaVantageTickerLoader
from etl.ticker_discovery.ticker_validator import TickerValidator
from etl.ticker_discovery.ticker_universe import TickerUniverseManager


CSV_SAMPLE = """symbol,name,exchange,assetType,ipoDate,delistingDate,status
AAPL,Apple Inc.,NASDAQ,Common Stock,1980-12-12,,Active
MSFT,Microsoft Corporation,NASDAQ,Common Stock,1986-03-13,,Active
BRK.B,Berkshire Hathaway Inc.,NYSE,Common Stock,1996-05-10,,Active
XYZQ,Example Delisted,NASDAQ,Common Stock,2020-01-01,2021-01-01,Delisted
SPY,SPDR S&P 500 ETF Trust,NYSEARCA,ETF,1993-01-22,,Active
bad_ticker,Bad Name,NASDAQ,Common Stock,2020-01-01,,Active
"""


@pytest.fixture()
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture()
def fallback_csv(tmp_path: Path) -> Path:
    path = tmp_path / "listings.csv"
    path.write_text(CSV_SAMPLE)
    return path


def test_alpha_vantage_loader_filters_and_caches(cache_dir: Path, fallback_csv: Path):
    loader = AlphaVantageTickerLoader(api_key=None, cache_dir=str(cache_dir))
    df = loader.download_listings(force=True, fallback_csv=fallback_csv)

    assert set(df.columns) == {"symbol", "name", "exchange", "asset_type", "status"}
    assert set(df["symbol"]) == {"AAPL", "MSFT", "BRK.B", "SPY"}
    cached = loader.download_listings(force=False)
    assert len(cached) == len(df)


def test_ticker_universe_manager_refresh_and_load(cache_dir: Path, fallback_csv: Path):
    loader = AlphaVantageTickerLoader(cache_dir=str(cache_dir))
    validator = TickerValidator(disallowed_prefixes=("BAD",))
    manager = TickerUniverseManager(loader=loader, validator=validator)

    universe = manager.refresh_universe(force_download=True, fallback_csv=fallback_csv)
    assert universe.tickers == ["AAPL", "BRK.B", "MSFT", "SPY"]
    assert manager.universe_path.exists()

    cached = manager.load_universe()
    assert cached.tickers == universe.tickers
