import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from etl.synthetic_extractor import SyntheticExtractor
from pandas.testing import assert_frame_equal


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


def _load_legacy_generator():
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    scripts_path = str(scripts_dir)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    import run_etl_pipeline  # type: ignore

    return run_etl_pipeline.generate_synthetic_ohlcv


def test_legacy_v0_matches_run_etl_pipeline_helper(tmp_path):
    config_path = tmp_path / "syn.yml"
    config_path.write_text(
        "\n".join(
            [
                "synthetic:",
                "  generator_version: v0",
                "  seed: 42",
                "  start_date: 2020-01-01",
                "  end_date: 2020-01-10",
                "  frequency: B",
                "  tickers:",
                "    - AAPL",
            ]
        )
    )

    extractor = SyntheticExtractor(config_path=str(config_path))
    data = extractor.extract_ohlcv(["AAPL"], "2020-01-01", "2020-01-10")

    legacy_fn = _load_legacy_generator()
    legacy = legacy_fn(["AAPL"], "2020-01-01", "2020-01-10", seed=42)

    assert data.attrs["dataset_id"] == legacy.attrs["dataset_id"]
    assert data.attrs["generator_version"] == "v0"
    assert_frame_equal(data[legacy.columns], legacy[legacy.columns])


@pytest.mark.parametrize("price_model", ["gbm", "ou", "jump_diffusion", "heston", "hybrid"])
def test_price_models_emit_correlated_positive_series(tmp_path, price_model):
    config = {
        "synthetic": {
            "generator_version": "v1",
            "seed": 321,
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "frequency": "B",
            "price_model": price_model,
            "volatility_model": "stochastic_vol" if price_model == "heston" else "none",
            "tickers": ["AAPL", "MSFT"],
            "regimes": {
                "enabled": True,
                "names": ["low_vol", "high_vol"],
                "transition_matrix": [[0.9, 0.1], [0.2, 0.8]],
                "params": {
                    "low_vol": {"drift": 0.0004, "vol": 0.01},
                    "high_vol": {"drift": -0.0002, "vol": 0.03, "jump_intensity": 0.05},
                },
            },
            "correlation": {"mode": "static", "target_matrix": [[1.0, 0.8], [0.8, 1.0]]},
            "jump_diffusion": {"enabled": True, "intensity": 0.05, "jump_mean": -0.01, "jump_std": 0.02},
        }
    }
    config_path = tmp_path / f"{price_model}.yml"
    config_path.write_text(yaml.safe_dump(config))

    extractor = SyntheticExtractor(config_path=str(config_path))
    data = extractor.extract_ohlcv(["AAPL", "MSFT"], "2020-01-01", "2020-12-31")

    validation = extractor.validate_data(data)
    assert validation["passed"]
    assert data["Volume"].min() >= 0
    assert set(data["ticker"].unique()) == {"AAPL", "MSFT"}

    wide = data.pivot_table(index=data.index, columns="ticker", values="Close")
    corr = wide.pct_change().dropna().corr().iloc[0, 1]
    assert corr > 0.15
    regimes_used = data.attrs.get("regimes_used", [])
    assert len(regimes_used) >= 1
