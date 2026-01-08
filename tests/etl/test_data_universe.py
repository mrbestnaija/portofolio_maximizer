"""Tests for data‑source‑aware ticker universe resolution."""
from __future__ import annotations

from typing import Sequence

from etl.data_universe import ResolvedUniverse, resolve_ticker_universe


def test_resolve_ticker_universe_explicit_frontier_passthrough(monkeypatch):
    """Explicit tickers + frontier overlay should be preserved."""

    # Use a tiny base list and disable frontier for a deterministic check.
    base: Sequence[str] = ["aapl", "MSFT", " aapl "]

    universe = resolve_ticker_universe(
        base_tickers=base,
        include_frontier=False,
        active_source="yfinance",
        use_discovery=True,
    )

    assert isinstance(universe, ResolvedUniverse)
    # merge_frontier_tickers normalises and de‑duplicates while preserving order
    assert universe.tickers == ["AAPL", "MSFT"]
    assert universe.active_source == "yfinance"
    assert universe.universe_source == "explicit+frontier"


def test_resolve_ticker_universe_empty_no_discovery():
    """With no base tickers and discovery disabled, return an empty universe."""

    universe = resolve_ticker_universe(
        base_tickers=[],
        include_frontier=False,
        active_source="yfinance",
        use_discovery=False,
    )

    assert isinstance(universe, ResolvedUniverse)
    assert universe.tickers == []
    assert universe.universe_source in ("none", "")

