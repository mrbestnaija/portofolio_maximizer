"""Ticker validation utilities."""
from __future__ import annotations

import logging
import re
from typing import Iterable, List, Sequence

logger = logging.getLogger(__name__)


class TickerValidator:
    """Lightweight ticker validation with configurable heuristics.

    The validator avoids network calls so it can run inside unit tests and CI
    without external dependencies.  Validation rules are intentionally simple
    and can be extended later (for example by sampling data using the existing
    extractors).
    """

    TICKER_RE = re.compile(r"^[A-Z0-9\.\-]{1,10}$")

    def __init__(self, disallowed_prefixes: Sequence[str] | None = None) -> None:
        self.disallowed_prefixes = tuple(disallowed_prefixes or [])

    def is_valid(self, ticker: str) -> bool:
        """Return True when a ticker passes basic validation rules."""
        if not ticker:
            return False
        ticker = ticker.upper().strip()
        if not self.TICKER_RE.match(ticker):
            return False
        if any(ticker.startswith(prefix) for prefix in self.disallowed_prefixes):
            return False
        return True

    def filter_valid(self, tickers: Iterable[str]) -> List[str]:
        """Filter an iterable of tickers, returning only the valid entries."""
        unique = []
        seen = set()
        for ticker in tickers:
            if not ticker:
                continue
            normalized = ticker.upper().strip()
            if normalized in seen:
                continue
            if self.is_valid(normalized):
                unique.append(normalized)
                seen.add(normalized)
            else:
                logger.debug("Filtered invalid ticker: %s", ticker)
        return unique
