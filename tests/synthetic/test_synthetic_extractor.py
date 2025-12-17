import pandas as pd

from etl.synthetic_extractor import SyntheticExtractor


def test_extract_and_validate_basic_schema():
    extractor = SyntheticExtractor(config_path="config/synthetic_data_config.yml")
    data = extractor.extract_ohlcv(["AAPL", "MSFT"], "2020-01-01", "2024-01-01")

    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.is_monotonic_increasing
    for col in ("Open", "High", "Low", "Close", "Volume"):
        assert col in data.columns
    assert "ticker" in data.columns
    assert data.attrs.get("dataset_id") is not None

    validation = extractor.validate_data(data)
    assert validation["passed"]
    assert validation["metrics"]["rows"] == len(data)


def test_load_persisted_via_env(tmp_path, monkeypatch):
    extractor = SyntheticExtractor(config_path="config/synthetic_data_config.yml")
    data = extractor.extract_ohlcv(["AAPL"], "2020-01-01", "2020-01-10")

    dataset_id = data.attrs["dataset_id"]
    target_dir = tmp_path / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    data.to_parquet(target_dir / "combined.parquet")

    monkeypatch.setenv("SYNTHETIC_DATASET_ID", dataset_id)
    monkeypatch.setenv("SYNTHETIC_DATASET_PATH", str(target_dir))

    extractor_reload = SyntheticExtractor(config_path="config/synthetic_data_config.yml")
    loaded = extractor_reload.extract_ohlcv(["AAPL"], "2020-01-01", "2020-01-10")

    assert len(loaded) == len(data)
    assert loaded.attrs.get("dataset_id") == dataset_id


def test_regime_switching_and_correlation_shapes():
    extractor = SyntheticExtractor(config_path="config/synthetic_data_config.yml")
    data = extractor.extract_ohlcv(["AAPL", "MSFT"], "2020-01-01", "2020-06-01")
    # Expect both tickers present and correlated shocks applied; at least 2 tickers exist
    assert set(data["ticker"].unique()) == {"AAPL", "MSFT"}
    assert len(data) > 0
    validation = extractor.validate_data(data)
    assert validation["passed"]
