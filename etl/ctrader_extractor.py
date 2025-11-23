"""
cTrader-backed extractor placeholder that proxies to yfinance for OHLCV data.

Used as a substitution path when Finnhub credentials are missing so the pipeline
retains a third provider for breadth/failover.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pandas as pd

from etl.base_extractor import BaseExtractor, ExtractorMetadata
from etl.yfinance_extractor import YFinanceExtractor

logger = logging.getLogger(__name__)


class CTraderExtractor(BaseExtractor):
    """Proxy extractor that reuses yfinance while honoring the BaseExtractor API."""

    def __init__(self, name: str = "ctrader", storage=None, cache_hours: int = 24, **kwargs):
        super().__init__(name=name, cache_hours=cache_hours, storage=storage, **kwargs)
        self._delegate = YFinanceExtractor(name=f"{name}_proxy", storage=storage, cache_hours=cache_hours)

    def extract_ohlcv(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        logger.info("cTrader proxy delegating OHLCV fetch to yfinance for: %s", ", ".join(tickers))
        frame = self._delegate.extract_ohlcv(tickers, start_date, end_date)
        frame["source"] = self.name
        return frame

    def validate_data(self, data: pd.DataFrame) -> bool:
        return self._delegate.validate_data(data)

    def get_metadata(self, ticker: str, data: pd.DataFrame, cache_hit: bool) -> ExtractorMetadata:
        return self._delegate.get_metadata(ticker, data, cache_hit)

    def get_cache_stats(self) -> Dict[str, int]:
        return self._delegate.get_cache_stats()
