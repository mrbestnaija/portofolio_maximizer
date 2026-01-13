# Data‑Source‑Aware Ticker Universe (Migration‑Friendly Design)

This document defines how ticker / instrument discovery should work in a **data‑source‑aware**, **configuration‑driven** way, reusing existing building blocks instead of duplicating logic. The goal is to make it easy to migrate from `yfinance` as the default source to `ctrader` or any other brokerage feed.

## Goals

- **Source‑aware universes**  
  - Know which tickers / instruments are actually available on a given provider (yfinance, Alpha Vantage, cTrader proxy, future real brokers).  
  - Support asset‑class filters (equities, FX, commodities, crypto) where the provider exposes them.

- **Config‑driven, not hardcoded lists**  
  - Keep explicit user tickers and the existing frontier‑market list, but allow discovery from provider‑native listings when desired.  
  - All behaviour toggled via config (data source config + future universe config), not via hardcoded rules.

- **Reuse existing components**  
  - Build on top of:
    - `etl.frontier_markets.merge_frontier_tickers` for the curated frontier universe.  
    - `etl.ticker_discovery.TickerUniverseManager` + `AlphaVantageTickerLoader` for listings‑based universes.  
    - `etl.data_source_manager.DataSourceManager` for active source selection.
  - Avoid introducing a parallel “universe” stack; instead, provide a thin orchestration helper.

## Existing Building Blocks

- **Frontier markets list**  
  - Module: `etl/frontier_markets.py`  
  - Provides `FRONTIER_MARKET_TICKERS_BY_REGION`, flattened `FRONTIER_MARKET_TICKERS`, and helper `merge_frontier_tickers(base_tickers, include_frontier)`.

- **Ticker discovery / universe caching**  
  - Package: `etl/ticker_discovery/`  
  - `BaseTickerLoader`: abstract base for provider‑specific listing loaders.  
  - `AlphaVantageTickerLoader`: downloads & cleans Alpha Vantage listings (equities/ETFs).  
  - `TickerValidator`: basic symbol hygiene / filtering.  
  - `TickerUniverseManager`: loads/refreshes a cached `TickerUniverse` (`tickers` + `metadata`).

- **Data source orchestration**  
  - Module: `etl/data_source_manager.py`  
  - Reads `config/data_sources_config.yml` and instantiates enabled providers.  
  - Tracks `active_extractor` and exposes `get_active_source()` and `get_extractor(name)`.

These are the only components the new helper should rely on; no second universe stack should be introduced.

## Helper API (Minimal, Reusable)

New module: `etl/data_universe.py`  

```python
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ResolvedUniverse:
    tickers: List[str]
    active_source: str
    universe_source: str  # e.g. "explicit+frontier", "alpha_vantage_universe"
    notes: Dict[str, str]
```

Core function:

```python
def resolve_ticker_universe(
    base_tickers: Sequence[str],
    include_frontier: bool,
    active_source: str,
    use_discovery: bool = True,
) -> ResolvedUniverse:
    ...
```

Behaviour:

1. **Explicit + frontier (current default)**  
   - Always call `merge_frontier_tickers(base_tickers, include_frontier)` first.  
   - If the merged list is non‑empty, return it directly with:
     - `universe_source="explicit+frontier"`  
     - `active_source` passed through unchanged.  
   - This preserves today's behaviour for `run_auto_trader.py` and CLI tools.
   - Apply provider-aware filters to drop tickers unsupported by the active source (e.g., yfinance lacks Sri Lanka CSE coverage) so live runs avoid repeated missing-data failures.

2. **Optional provider‑level discovery** (only when no explicit tickers)  
   - If `base_tickers` is empty and `use_discovery` is `True`:
     - If `active_source == "alpha_vantage"`:  
       - Build `AlphaVantageTickerLoader` and `TickerUniverseManager`.  
       - Prefer `TickerUniverseManager.load_universe()`; if empty, attempt `refresh_universe()` (which uses cached CSV or live API depending on setup).  
       - Return `ResolvedUniverse(tickers=universe.tickers, universe_source="alpha_vantage_universe", active_source=active_source)`.  
     - For other sources (e.g. `ctrader`), defer to future provider‑specific loaders without changing this interface.
   - Any network / API errors should be caught and recorded in `notes["warning"]`, returning an empty ticker list instead of failing the caller.

3. **No hardcoded cTrader logic (yet)**  
   - The cTrader extractor is currently a yfinance proxy. The helper should not hardcode any broker‑specific symbol rules; that belongs in a future `CTraderTickerLoader` plus `TickerUniverseManager`.

## Integration Plan (Phase 1 – Minimal Wiring)

**1. Add `etl/data_universe.py`**  
- Implement `ResolvedUniverse` and `resolve_ticker_universe` as described above.  
- Import and reuse only:
  - `merge_frontier_tickers` from `etl.frontier_markets`.  
  - `TickerUniverseManager` and `AlphaVantageTickerLoader` from `etl.ticker_discovery.*`.  
- Do not add any new network code; all downloads are delegated to existing loaders.

**2. Wire `run_auto_trader.py` through the helper**  
- Replace the direct call to `merge_frontier_tickers` with:
  - Instantiate `DataSourceManager` first to get `active_source = data_source_manager.get_active_source()`.  
  - Call `resolve_ticker_universe(base_tickers, include_frontier_tickers, active_source)`.  
  - Use `universe.tickers` as `ticker_list`.  
  - Log `universe.universe_source` and `universe.active_source` to make the data source choice visible.
- Behaviour with explicit tickers stays identical; the helper simply centralises the logic.

**3. Future: ETL / backtest integration (Phase 2)**  
- `scripts/run_etl_pipeline.py` and `backtesting/candidate_backtester.py` can later import `resolve_ticker_universe` to:
  - Avoid repeating frontier‑merging logic.  
  - Switch between “explicit+frontier” vs “provider‑universe” modes by config.

## Testing Strategy

- New tests in `tests/etl/test_data_universe.py`:
  - **Explicit path**:  
    - Given `base_tickers=["AAPL", "MSFT"]`, `include_frontier=False`, `active_source="yfinance"`, verify:
      - `tickers` preserves ordering and case‑normalisation from `merge_frontier_tickers`.  
      - `universe_source == "explicit+frontier"`.
  - **Empty + non‑discovery path**:  
    - Given `base_tickers=[]`, `include_frontier=False`, `use_discovery=False`, verify `tickers == []` and no exception is raised.
  - **Alpha Vantage discovery (offline)**:  
    - Monkeypatch `TickerUniverseManager.load_universe()` to return a small fake universe and verify that `resolve_ticker_universe` returns those tickers when `active_source="alpha_vantage"` and `base_tickers=[]`.

These tests rely only on the existing discovery plumbing (`TickerUniverseManager`) and do not introduce new network dependencies.

## Migration Notes (yfinance → cTrader and Others)

- Migration will be primarily **configuration‑driven**:
  - `config/data_sources_config.yml` chooses the active provider (`selection_strategy` + `providers[].enabled/priority`).  
  - `resolve_ticker_universe` respects whichever provider is currently active but does not hardcode provider‑specific logic.
- When a real `CTraderTickerLoader` exists:
  - Implement it under `etl/ticker_discovery/` using the same `BaseTickerLoader` pattern.  
  - Register a new branch in `resolve_ticker_universe` for `active_source == "ctrader"` that uses `TickerUniverseManager` with the new loader.  
  - Keep all symbol‑mapping specifics inside the loader; the resolver stays thin and generic.

This keeps ticker & instrument discovery centralised, testable, and extensible without duplicating logic or bloating the project.***
