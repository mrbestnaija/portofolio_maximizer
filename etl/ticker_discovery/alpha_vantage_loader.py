"""Alpha Vantage ticker discovery loader."""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Optional, List

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

from .base_ticker_loader import BaseTickerLoader
from .ticker_validator import TickerValidator

logger = logging.getLogger(__name__)

if load_dotenv:
    load_dotenv()


class AlphaVantageTickerLoader(BaseTickerLoader):
    """Download and cache ticker listings from Alpha Vantage.

    The loader prefers cached artefacts and only attempts to hit the live API
    when both an API key and a requests session are supplied.  This design keeps
    the module fully testable/offline.  Tests typically provide a ``fallback_csv``
    path which is then parsed via :meth:`download_listings`.
    """

    CACHE_FILE_NAME = "alpha_vantage_listings.csv"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        session: Optional["requests.Session"] = None,
        listings_url: str = "https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={api_key}",
    ) -> None:
        super().__init__(cache_dir=cache_dir)
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.session = session
        self.listings_url = listings_url

    def download_listings(
        self,
        force: bool = False,
        fallback_csv: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Fetch the latest Alpha Vantage listings."""
        if not force:
            cached = self.load_cached_listings(self.CACHE_FILE_NAME)
            if cached is not None:
                logger.info("Loaded Alpha Vantage listings from cache (%s rows)", len(cached))
                return cached

        if fallback_csv is not None:
            data = pd.read_csv(fallback_csv)
            logger.info("Loaded Alpha Vantage listings from fallback CSV: %s", fallback_csv)
        elif self.api_key:
            data = self._download_from_api()
        else:
            raise RuntimeError(
                "Alpha Vantage listings unavailable. Provide a fallback CSV or configure an API key."
            )

        cleaned = self._clean_listings(data)
        self.save_cached_listings(self.CACHE_FILE_NAME, cleaned)
        return cleaned

    def _download_from_api(self) -> pd.DataFrame:
        import requests  # Local import to keep dependency optional

        if self.session is None:
            self.session = requests.Session()

        url = self.listings_url.format(api_key=self.api_key)
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))

    @staticmethod
    def _clean_listings(df: pd.DataFrame) -> pd.DataFrame:
        """Standardise column names and filter active equities/ETFs."""
        expected_columns = {"symbol", "name", "exchange", "assetType", "status"}
        missing = expected_columns.difference(df.columns)
        if missing:
            raise ValueError(f"Alpha Vantage listings missing required columns: {missing}")

        cleaned = df.copy()
        cleaned.columns = [c.strip() for c in cleaned.columns]
        cleaned = cleaned.rename(columns={"assetType": "asset_type"})
        cleaned["symbol"] = cleaned["symbol"].str.upper().str.strip()
        cleaned["status"] = cleaned["status"].str.lower().str.strip()
        cleaned["asset_type"] = cleaned["asset_type"].str.lower().str.strip()

        active_mask = cleaned["status"] == "active"
        equity_mask = cleaned["asset_type"].isin({"common stock", "etf"})
        filtered = cleaned.loc[active_mask & equity_mask].reset_index(drop=True)

        validator = TickerValidator()
        valid_symbols = validator.filter_valid(filtered["symbol"].tolist())

        if not valid_symbols:
            filtered = filtered.iloc[0:0]
        else:
            filtered = (
                filtered
                .drop_duplicates(subset="symbol")
                .set_index("symbol")
                .loc[valid_symbols]
                .reset_index()
            )

        logger.info(
            "Filtered Alpha Vantage listings from %s to %s active equities/ETFs",
            len(cleaned),
            len(filtered),
        )
        return filtered[["symbol", "name", "exchange", "asset_type", "status"]]

    def get_active_equities(self, listings: Optional[pd.DataFrame] = None) -> List[str]:
        """Return a list of validated ticker symbols."""
        if listings is None:
            listings = self.download_listings()
        return [sym for sym in listings["symbol"].tolist() if sym]
