"""Tests for data-source-aware ticker universe resolution."""
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
    # merge_frontier_tickers normalises and de-duplicates while preserving order
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


def test_resolve_ticker_universe_filters_provider_blocklist():
    """Provider-specific blocklists should drop unsupported tickers for yfinance."""

    universe = resolve_ticker_universe(
        base_tickers=["MSFT", "HNB", "SAMP"],
        include_frontier=False,
        active_source="yfinance",
        use_discovery=False,
    )

    assert universe.tickers == ["MSFT"]
    assert "provider_blocklist" in universe.notes
    assert "HNB" in universe.notes["provider_blocklist"]


def test_resolve_ticker_universe_uses_discovery_config_and_refresh(monkeypatch):
    """Alpha Vantage discovery should be driven through the shared resolver."""

    captured: dict[str, object] = {}

    class _FakeLoader:
        def __init__(self, api_key=None, cache_dir=None):
            captured["api_key"] = api_key
            captured["cache_dir"] = cache_dir

    class _FakeManager:
        def __init__(self, loader, validator=None, universe_path=None):
            captured["universe_path"] = universe_path

        def refresh_universe(self, force_download=False, fallback_csv=None):
            captured["force_download"] = force_download
            captured["fallback_csv"] = str(fallback_csv) if fallback_csv else None
            return type("Universe", (), {"tickers": ["AAPL", "MSFT"]})()

        def load_universe(self):
            return type("Universe", (), {"tickers": []})()

    monkeypatch.setattr("etl.data_universe.AlphaVantageTickerLoader", _FakeLoader)
    monkeypatch.setattr("etl.data_universe.TickerUniverseManager", _FakeManager)

    universe = resolve_ticker_universe(
        base_tickers=[],
        include_frontier=False,
        active_source="alpha_vantage",
        use_discovery=True,
        discovery_config={
            "api_key": "test-key",
            "cache_dir": "cache/alpha",
            "universe_path": "cache/universe.csv",
            "fallback_csv": "cache/fallback.csv",
        },
        refresh_discovery=True,
    )

    assert universe.tickers == ["AAPL", "MSFT"]
    assert universe.universe_source == "alpha_vantage_universe"
    assert universe.notes["info"] == "alpha_vantage_universe_refreshed"
    assert captured["api_key"] == "test-key"
    assert captured["cache_dir"] == "cache/alpha"
    assert captured["universe_path"] == "cache/universe.csv"
    assert captured["force_download"] is True
    assert str(captured["fallback_csv"]).endswith("fallback.csv")
