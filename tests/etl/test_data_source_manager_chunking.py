from __future__ import annotations

import pandas as pd

from etl.data_source_manager import DataSourceManager


class _DummyExtractor:
    def __init__(self) -> None:
        self.name = "dummy"
        self.calls: list[list[str]] = []

    def extract_ohlcv(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append(list(tickers))
        frames = []
        for t in tickers:
            idx = pd.date_range(start_date, periods=2, freq="D")
            frames.append(
                pd.DataFrame(
                    {"Open": [1.0, 2.0], "Close": [1.0, 2.0], "ticker": t},
                    index=idx,
                )
            )
        return pd.concat(frames)

    def get_cache_statistics(self) -> dict:
        return {}


def test_extract_ohlcv_chunks_batches_and_concatenates() -> None:
    manager: DataSourceManager = object.__new__(DataSourceManager)  # type: ignore[assignment]
    manager.config = {"selection_strategy": {"mode": "priority"}}
    extractor = _DummyExtractor()
    manager.extractors = {"dummy": extractor}
    manager.active_extractor = extractor
    manager._failover_extraction = lambda *args, **kwargs: pd.DataFrame()  # type: ignore[assignment]

    tickers = [f"T{i}" for i in range(5)]
    df = manager.extract_ohlcv(tickers, "2024-01-01", "2024-01-10", chunk_size=2)

    # 5 tickers with chunk_size=2 => 3 calls (2,2,1)
    assert len(extractor.calls) == 3
    assert set(df["ticker"].unique()) == set(tickers)
    assert len(df) == len(tickers) * 2
