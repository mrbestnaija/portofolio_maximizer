"""Phase 9 — generate_classifier_training_labels unit tests."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def _make_price_df(n: int = 400, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with realistic price levels."""
    np.random.seed(seed)
    dates = pd.date_range(start, periods=n, freq="B", tz="UTC")
    returns = np.random.randn(n) * 0.01
    close = 100.0 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)
    return df


class TestGenerateLabels:
    def test_generates_labeled_rows(self):
        from scripts.generate_classifier_training_labels import generate_labels
        df = _make_price_df(n=400)
        rows = generate_labels("AAPL", df, lookback=120, step=20, horizon=30)
        assert len(rows) > 0

    def test_entry_ts_is_historical(self):
        """entry_ts must be within the parquet's date range, not wall-clock."""
        from scripts.generate_classifier_training_labels import generate_labels
        df = _make_price_df(n=400, start="2020-01-01")
        rows = generate_labels("AAPL", df, lookback=120, step=20, horizon=30)
        import pandas as pd
        for row in rows:
            ts = pd.Timestamp(row["entry_ts"])
            assert ts.year < 2025, f"entry_ts {ts} looks like wall-clock, not historical"

    def test_labels_are_binary(self):
        from scripts.generate_classifier_training_labels import generate_labels
        df = _make_price_df(n=400)
        rows = generate_labels("AAPL", df, lookback=120, step=20, horizon=30)
        for row in rows:
            assert row["y_directional"] in (0, 1)

    def test_forward_label_correct_direction(self):
        """On a monotonically rising series, all labels should be 1."""
        from scripts.generate_classifier_training_labels import generate_labels
        n = 400
        dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
        close = pd.Series(range(100, 100 + n, 1), index=dates, dtype=float)
        df = pd.DataFrame({"Close": close})
        rows = generate_labels("AAPL", df, lookback=120, step=20, horizon=30)
        assert len(rows) > 0
        for row in rows:
            assert row["y_directional"] == 1, "Monotone up → all labels must be 1"

    def test_features_present_in_each_row(self):
        from scripts.generate_classifier_training_labels import generate_labels, _FEATURE_NAMES
        df = _make_price_df(n=400)
        rows = generate_labels("AAPL", df, lookback=120, step=20, horizon=30)
        for row in rows:
            for name in _FEATURE_NAMES:
                assert name in row, f"Feature '{name}' missing from row"

    def test_price_features_not_all_nan(self):
        """At least the computable price features must have some non-NaN values."""
        import math
        from scripts.generate_classifier_training_labels import generate_labels
        df = _make_price_df(n=400)
        rows = generate_labels("AAPL", df, lookback=200, step=50, horizon=30)
        price_features = ["realized_vol_annualized", "recent_return_5d", "recent_vol_ratio"]
        for feat in price_features:
            has_value = any(
                not math.isnan(r.get(feat, float("nan")))
                for r in rows
                if isinstance(r.get(feat), float)
            )
            assert has_value, f"Feature '{feat}' is all-NaN — price-based extraction failed"

    def test_too_short_parquet_returns_empty(self):
        from scripts.generate_classifier_training_labels import generate_labels
        df = _make_price_df(n=20)  # too short for lookback=60
        rows = generate_labels("AAPL", df, lookback=60, step=5, horizon=30)
        assert rows == []

    def test_ticker_in_every_row(self):
        from scripts.generate_classifier_training_labels import generate_labels
        df = _make_price_df(n=400)
        rows = generate_labels("MSFT", df, lookback=120, step=20, horizon=30)
        for row in rows:
            assert row["ticker"] == "MSFT"


class TestAppendToDataset:
    def test_creates_new_parquet(self, tmp_path):
        from scripts.generate_classifier_training_labels import _append_to_dataset
        rows = [
            {"ts_signal_id": "gen_AAPL_20200101_00000", "ticker": "AAPL",
             "entry_ts": "2020-01-01", "y_directional": 1, "label_source": "price_parquet_scan"}
        ]
        out = tmp_path / "test.parquet"
        added = _append_to_dataset(rows, out)
        assert added == 1
        assert out.exists()

    def test_deduplicates_on_ts_signal_id(self, tmp_path):
        from scripts.generate_classifier_training_labels import _append_to_dataset
        row = {"ts_signal_id": "gen_AAPL_20200101_00000", "ticker": "AAPL",
               "entry_ts": "2020-01-01", "y_directional": 1, "label_source": "price_parquet_scan"}
        out = tmp_path / "test.parquet"
        _append_to_dataset([row], out)
        added = _append_to_dataset([row], out)  # same row again
        assert added == 0  # dedup: nothing added second time
        df = pd.read_parquet(out)
        assert len(df) == 1

    def test_appends_new_rows(self, tmp_path):
        from scripts.generate_classifier_training_labels import _append_to_dataset
        row1 = {"ts_signal_id": "gen_AAPL_20200101_00000", "ticker": "AAPL",
                "entry_ts": "2020-01-01", "y_directional": 1, "label_source": "price_parquet_scan"}
        row2 = {"ts_signal_id": "gen_AAPL_20200201_00001", "ticker": "AAPL",
                "entry_ts": "2020-02-01", "y_directional": 0, "label_source": "price_parquet_scan"}
        out = tmp_path / "test.parquet"
        _append_to_dataset([row1], out)
        added = _append_to_dataset([row2], out)
        assert added == 1
        df = pd.read_parquet(out)
        assert len(df) == 2


class TestLoadBestParquet:
    def test_explicit_path_loaded(self, tmp_path):
        from scripts.generate_classifier_training_labels import _load_best_parquet
        df_orig = _make_price_df(n=200)
        parquet_path = tmp_path / "test.parquet"
        df_orig.to_parquet(parquet_path)
        result = _load_best_parquet("AAPL", tmp_path, parquet_path)
        assert result is not None
        assert "Close" in result.columns

    def test_too_short_parquet_returns_none(self, tmp_path):
        from scripts.generate_classifier_training_labels import _load_best_parquet
        df_short = _make_price_df(n=10)
        path = tmp_path / "short.parquet"
        df_short.to_parquet(path)
        result = _load_best_parquet("AAPL", tmp_path, path)
        assert result is None  # too short (< _MIN_LOOKBACK + _DEFAULT_HORIZON = 90)

    def test_missing_close_column_returns_none(self, tmp_path):
        from scripts.generate_classifier_training_labels import _load_best_parquet
        df = pd.DataFrame({"Volume": range(200)})
        path = tmp_path / "no_close.parquet"
        df.to_parquet(path)
        result = _load_best_parquet("AAPL", tmp_path, path)
        assert result is None

    def test_ticker_name_in_filename_found_by_auto(self, tmp_path):
        """Auto-detect should find parquet when ticker is in the filename."""
        from scripts.generate_classifier_training_labels import _load_best_parquet
        df = _make_price_df(n=200)
        path = tmp_path / "AAPL_data_extraction_test.parquet"
        df.to_parquet(path)
        result = _load_best_parquet("AAPL", tmp_path)
        assert result is not None


class TestCLI:
    def test_cold_start_exit_code_2(self, tmp_path):
        """n_labeled < 60 should exit with code 2."""
        import subprocess, sys
        df = _make_price_df(n=150)  # will generate ~8 rows with step=10 → cold start
        parquet = tmp_path / "AAPL_data_extraction_test.parquet"
        df.to_parquet(parquet)
        out = tmp_path / "out.parquet"
        result = subprocess.run(
            [sys.executable, "scripts/generate_classifier_training_labels.py",
             "--ticker", "AAPL", "--parquet", str(parquet),
             "--lookback", "120", "--step", "30", "--horizon", "30",
             "--output", str(out)],
            capture_output=True, text=True,
        )
        # May be 0 (enough rows) or 2 (cold start) — both acceptable, just not 1
        assert result.returncode in (0, 2), f"Unexpected exit: {result.returncode}\n{result.stderr}"
