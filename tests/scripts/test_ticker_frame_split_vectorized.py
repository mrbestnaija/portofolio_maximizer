from __future__ import annotations

import pandas as pd

from scripts import run_auto_trader


def test_build_ticker_frame_map_vectorized_slice_sorted() -> None:
    idx = pd.DatetimeIndex(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-01",  # out of order on purpose
            "2024-01-04",
        ]
    )
    raw = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "aapl", "MSFT"],
            "Open": [10.0, 20.0, 11.0, 21.0],
            "Close": [11.0, 21.0, 12.0, 22.0],
        },
        index=idx,
    )

    frames = run_auto_trader._build_ticker_frame_map(raw, ["aapl", "msft"])

    assert set(frames.keys()) == {"AAPL", "MSFT"}

    aapl_frame = frames["AAPL"]
    msft_frame = frames["MSFT"]

    # Ticker column removed and index sorted
    assert "ticker" not in aapl_frame.columns
    assert aapl_frame.index.is_monotonic_increasing
    assert msft_frame.index.is_monotonic_increasing

    # Values preserved per ticker in chronological order
    assert list(aapl_frame["Open"]) == [11.0, 10.0]
    assert list(msft_frame["Close"]) == [21.0, 22.0]
