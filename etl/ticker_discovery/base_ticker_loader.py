"""Base classes for ticker discovery loaders."""
from __future__ import annotations

import abc
import os
from pathlib import Path
from typing import Optional

import pandas as pd


class BaseTickerLoader(abc.ABC):
    """Abstract base class for ticker loaders.

    Implementations must provide a ``download_listings`` method that returns a
    ``pandas.DataFrame`` containing, at minimum, a ``symbol`` column.  The base
    class offers simple caching helpers so that concrete loaders can avoid
    repetitive network calls.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = Path(cache_dir or os.getenv("TICKER_CACHE_DIR", "data/tickers"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def download_listings(self, force: bool = False, fallback_csv: Optional[Path] = None) -> pd.DataFrame:
        """Return a dataframe of ticker listings.

        Args:
            force: When ``True`` the loader must refresh cached artefacts.
            fallback_csv: Optional path to a locally stored CSV that can be used
                instead of performing a network request (helpful for tests/offline).
        """

    def load_cached_listings(self, file_name: str) -> Optional[pd.DataFrame]:
        """Load cached listings if they exist."""
        path = self.cache_dir / file_name
        if not path.exists():
            return None
        return pd.read_csv(path)

    def save_cached_listings(self, file_name: str, data: pd.DataFrame) -> Path:
        """Persist listings to the cache directory."""
        path = self.cache_dir / file_name
        data.to_csv(path, index=False)
        return path
