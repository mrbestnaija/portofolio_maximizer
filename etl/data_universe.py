"""Data‑source‑aware ticker universe resolution.

This module is intentionally thin and reuses existing components:

- etl.frontier_markets.merge_frontier_tickers
- etl.ticker_discovery.TickerUniverseManager
- etl.ticker_discovery.AlphaVantageTickerLoader

It provides a single helper that callers (ETL, auto‑trader, backtests) can use
to resolve a concrete ticker list in a configuration‑driven way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from etl.frontier_markets import merge_frontier_tickers
from etl.ticker_discovery.ticker_universe import TickerUniverseManager
from etl.ticker_discovery.alpha_vantage_loader import AlphaVantageTickerLoader


@dataclass
class ResolvedUniverse:
    """Result of resolving a ticker universe for a given data source."""

    tickers: List[str]
    active_source: str
    universe_source: str
    notes: Dict[str, str] = field(default_factory=dict)


def resolve_ticker_universe(
    base_tickers: Sequence[str],
    include_frontier: bool,
    active_source: str,
    use_discovery: bool = True,
) -> ResolvedUniverse:
    """Resolve a concrete ticker universe without duplicating discovery logic.

    Behaviour:
    - If ``base_tickers`` is non‑empty, always return ``merge_frontier_tickers`` as
      today (explicit + optional frontier overlay).
    - If ``base_tickers`` is empty and ``use_discovery`` is True, attempt to use a
      provider‑specific universe for supported sources (currently Alpha Vantage).
    - All provider‑specific listing logic lives in the existing ticker_discovery
      loaders and TickerUniverseManager.
    """
    # 1) Current default path: explicit tickers + optional frontier overlay.
    merged = merge_frontier_tickers(base_tickers, include_frontier=include_frontier)
    if merged:
        return ResolvedUniverse(
            tickers=merged,
            active_source=active_source,
            universe_source="explicit+frontier",
        )

    notes: Dict[str, str] = {}
    discovered: List[str] = []
    universe_source = "none"

    # 2) Optional provider‑level discovery when caller did not specify tickers.
    if use_discovery and active_source == "alpha_vantage":
        loader = AlphaVantageTickerLoader()
        manager = TickerUniverseManager(loader=loader)

        universe = manager.load_universe()
        if not universe.tickers:
            try:
                universe = manager.refresh_universe()
            except Exception as exc:  # pragma: no cover - defensive path
                notes["warning"] = f"alpha_vantage_discovery_failed: {exc}"
            else:
                notes["info"] = "alpha_vantage_universe_refreshed"

        discovered = list(universe.tickers)
        universe_source = "alpha_vantage_universe" if discovered else "none"

    return ResolvedUniverse(
        tickers=discovered,
        active_source=active_source,
        universe_source=universe_source,
        notes=notes,
    )

