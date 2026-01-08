import pandas as pd

from etl.split_diagnostics import summarize_returns, drift_metrics, validate_non_overlap


def test_summarize_returns_basic():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    df = pd.DataFrame({"Close": [1, 1.1, 1.05, 1.2, 1.15]}, index=idx)
    summary = summarize_returns("train", df)
    assert summary.length == 4  # pct_change drops first
    assert summary.start.startswith("2024-01-01")
    assert summary.end.startswith("2024-01-05")


def test_drift_metrics_small_delta():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    df1 = pd.DataFrame({"Close": [1, 1.1, 1.05, 1.2, 1.15], "Volume": [100, 110, 105, 120, 115]}, index=idx)
    df2 = pd.DataFrame({"Close": [1, 1.11, 1.04, 1.19, 1.16], "Volume": [101, 109, 106, 121, 114]}, index=idx)
    drift = drift_metrics(df1, df2)
    assert drift["psi"] >= 0.0
    assert drift["mean_delta"] < 0.01
    assert drift["vol_psi"] >= 0.0
    assert "volatility_ratio" in drift


def test_validate_non_overlap():
    idx1 = pd.date_range("2024-01-01", periods=2, freq="B")
    idx2 = pd.date_range("2024-01-03", periods=2, freq="B")
    assert validate_non_overlap(idx1, idx2)
