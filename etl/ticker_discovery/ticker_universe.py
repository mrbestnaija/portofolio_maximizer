"""Ticker universe maintenance utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, List

import pandas as pd

from .base_ticker_loader import BaseTickerLoader
from .ticker_validator import TickerValidator

logger = logging.getLogger(__name__)


@dataclass
class TickerUniverse:
    tickers: List[str]
    metadata: pd.DataFrame


class TickerUniverseManager:
    """Manage a cached universe of validated tickers."""

    def __init__(
        self,
        loader: BaseTickerLoader,
        validator: Optional[TickerValidator] = None,
        universe_path: Optional[str] = None,
    ) -> None:
        self.loader = loader
        self.validator = validator or TickerValidator()
        self.universe_path = Path(universe_path or loader.cache_dir / "ticker_universe.csv")
        self.universe_path.parent.mkdir(parents=True, exist_ok=True)

    def refresh_universe(
        self,
        force_download: bool = False,
        fallback_csv: Optional[Path] = None,
    ) -> TickerUniverse:
        """Refresh the ticker universe and persist it to disk."""
        listings = self.loader.download_listings(force=force_download, fallback_csv=fallback_csv)
        tickers = self.validator.filter_valid(listings["symbol"])
        metadata = listings.loc[listings["symbol"].isin(tickers)].copy()
        metadata.drop_duplicates(subset="symbol", inplace=True)
        metadata.sort_values("symbol", inplace=True)
        sorted_tickers = metadata["symbol"].tolist()
        self._write_universe(metadata)
        logger.info("Recorded %s tickers to %s", len(sorted_tickers), self.universe_path)
        return TickerUniverse(tickers=sorted_tickers, metadata=metadata)

    def load_universe(self) -> TickerUniverse:
        """Load the cached universe from disk."""
        if not self.universe_path.exists():
            logger.info("Ticker universe cache missing (%s)", self.universe_path)
            return TickerUniverse(tickers=[], metadata=pd.DataFrame(columns=["symbol"]))

        metadata = pd.read_csv(self.universe_path)
        tickers = self.validator.filter_valid(metadata.get("symbol", []))
        return TickerUniverse(tickers=tickers, metadata=metadata)

    def _write_universe(self, metadata: pd.DataFrame) -> None:
        metadata.to_csv(self.universe_path, index=False)
