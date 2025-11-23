"""Shared definitions for frontier market tickers used during training/tests.

This module centralizes the list so both CLI tooling and documentation can
reference the same symbols without drifting. The regions and tickers follow
the guide in Documentation/arch_tree.md under the Frontier Market section.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

FRONTIER_MARKET_TICKERS_BY_REGION = {
    "Africa - Nigeria (NGX)": [
        "MTNN",
        "AIRTELAFRI",
        "ZENITHBANK",
        "GUARANTY",
        "FBNH",
    ],
    "Africa - Kenya (NSE)": [
        "EABL",
        "KCB",
        "SCANGROUP",
        "COOP",
    ],
    "Africa - South Africa (JSE)": [
        "NPN",
        "BIL",
        "SAB",
        "SOL",
        "MTN",
    ],
    "Asia - Vietnam (HOSE)": [
        "VHM",
        "GAS",
        "BID",
        "SSI",
    ],
    "Asia - Bangladesh (DSE)": [
        "BRACBANK",
        "LAFSURCEML",
        "IFADAUTOS",
        "RELIANCE",
    ],
    "Asia - Sri Lanka (CSE)": [
        "COMBANK",
        "HNB",
        "SAMP",
        "LOLC",
    ],
    "Asia - Pakistan (PSX)": [
        "OGDC",
        "MEBL",
        "LUCK",
        "UBL",
    ],
    "Middle East - Kuwait (KSE)": [
        "ZAIN",
        "NBK",
        "KFH",
        "MAYADEEN",
    ],
    "Middle East - Qatar (QSE)": [
        "QNBK",
        "DUQM",
        "QISB",
        "QAMC",
    ],
    "Central/Eastern Europe - Romania (BVB)": [
        "SIF1",
        "TGN",
        "BRD",
        "TLV",
    ],
    "Central/Eastern Europe - Bulgaria (BSE)": [
        "5EN",
        "BGO",
        "AIG",
        "SYN",
    ],
}

# Flattened view preserves the ordering from the user guide.
FRONTIER_MARKET_TICKERS: List[str] = [
    ticker
    for region in FRONTIER_MARKET_TICKERS_BY_REGION.values()
    for ticker in region
]


def _normalize(symbols: Iterable[str]) -> List[str]:
    """Normalize arbitrary ticker strings into uppercase symbols."""
    normalized: List[str] = []
    for symbol in symbols:
        candidate = symbol.strip().upper()
        if candidate:
            normalized.append(candidate)
    return normalized


def merge_frontier_tickers(
    base_tickers: Sequence[str],
    include_frontier: bool = True,
) -> List[str]:
    """Return a de-duplicated ticker list with optional frontier coverage."""

    normalized_base = _normalize(base_tickers)
    seen = set()
    merged: List[str] = []

    # Preserve caller ordering first.
    for symbol in normalized_base:
        if symbol not in seen:
            merged.append(symbol)
            seen.add(symbol)

    if not include_frontier:
        return merged

    for symbol in FRONTIER_MARKET_TICKERS:
        norm_symbol = symbol.strip().upper()
        if norm_symbol and norm_symbol not in seen:
            merged.append(norm_symbol)
            seen.add(norm_symbol)

    return merged


def frontier_ticker_string(base_tickers: Sequence[str]) -> str:
    """Comma-separated helper for CLI surfaces."""
    return ",".join(merge_frontier_tickers(base_tickers, include_frontier=True))
