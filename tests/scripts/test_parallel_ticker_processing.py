from __future__ import annotations

import pandas as pd

from scripts import run_auto_trader


class _DummyPreprocessor:
    def handle_missing(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame


def _make_frame(seed: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    base = 100 + seed
    return pd.DataFrame(
        {
            "Open": [base] * 5,
            "High": [base + 1] * 5,
            "Low": [base - 1] * 5,
            "Close": [base, base + 0.5, base + 1.0, base + 1.5, base + 2.0],
            "Volume": [100, 100, 100, 100, 100],
        },
        index=idx,
    )


def test_parallel_candidates_match_sequential() -> None:
    frames = [_make_frame(i) for i in range(6)]
    entries = [{"ticker": f"T{i}", "frame": frames[i], "order": i} for i in range(len(frames))]
    preprocessor = _DummyPreprocessor()

    seq = run_auto_trader._build_ticker_candidates(
        entries,
        preprocessor=preprocessor,
        parallel=False,
        max_workers=None,
    )
    par = run_auto_trader._build_ticker_candidates(
        entries,
        preprocessor=preprocessor,
        parallel=True,
        max_workers=3,
    )

    assert [c["ticker"] for c in seq] == [c["ticker"] for c in par]
    for left, right in zip(seq, par):
        assert left["quality"]["quality_score"] == right["quality"]["quality_score"]
        assert left["mid_price"] == right["mid_price"]
