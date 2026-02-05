import numpy as np
import pandas as pd
import pytest
import yaml

from etl.synthetic_extractor import SyntheticExtractor


def _write_config(tmp_path, cfg: dict) -> str:
    path = tmp_path / "synthetic.yml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_gbm_mean_and_variance_within_tolerance(tmp_path):
    cfg = {
        "synthetic": {
            "generator_version": "v1",
            "seed": 42,
            "start_date": "2020-01-01",
            "end_date": "2022-12-31",
            "frequency": "B",
            "tickers": ["AAPL"],
            "market_condition": "efficient",
            "price_model": "gbm",
            "volatility_model": "none",
            "jump_diffusion": {"enabled": False},
            "regimes": {"enabled": False},
            "event_library": {},
            "microstructure": {},
            "market_hours": {},
        }
    }
    extractor = SyntheticExtractor(config_path=_write_config(tmp_path, cfg))
    data = extractor.extract_ohlcv(["AAPL"], "2020-01-01", "2022-12-31")

    rets = np.log(data["Close"]).diff().dropna()
    mean_target = 0.0005  # base drift for efficient market condition
    var_target = 0.01**2  # base_vol^2 when volatility_model is "none"

    assert abs(rets.mean() - mean_target) < 5e-4
    assert abs(rets.var() - var_target) < 2.5e-5


def test_ou_path_stays_near_long_term_mean(tmp_path):
    long_term_price = 120.0
    cfg = {
        "synthetic": {
            "generator_version": "v1",
            "seed": 7,
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "frequency": "B",
            "tickers": ["AAPL"],
            "price_model": "ou",
            "volatility_model": "none",
            "regimes": {"enabled": False, "params": {"base": {"ou_speed": 0.8, "ou_mean": float(np.log(long_term_price)), "vol": 0.01}}},
            "jump_diffusion": {"enabled": False},
            "event_library": {},
            "microstructure": {},
            "market_hours": {},
        }
    }
    extractor = SyntheticExtractor(config_path=_write_config(tmp_path, cfg))
    data = extractor.extract_ohlcv(["AAPL"], "2020-01-01", "2023-12-31")

    rolling_mean = data["Close"].iloc[len(data) // 2 :].mean()
    assert np.isfinite(rolling_mean)
    assert abs(rolling_mean - long_term_price) < 20.0


def test_jump_diffusion_has_heavier_tails_than_baseline(tmp_path):
    base_cfg = {
        "synthetic": {
            "generator_version": "v1",
            "seed": 99,
            "start_date": "2020-01-01",
            "end_date": "2021-12-31",
            "frequency": "B",
            "tickers": ["AAPL"],
            "price_model": "jump_diffusion",
            "volatility_model": "none",
            "regimes": {"enabled": False},
            "event_library": {},
            "microstructure": {},
            "market_hours": {},
            "jump_diffusion": {"enabled": False},
        }
    }
    jump_cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    jump_cfg["synthetic"]["jump_diffusion"] = {"enabled": True, "intensity": 0.15, "jump_mean": -0.02, "jump_std": 0.05}

    base_extractor = SyntheticExtractor(config_path=_write_config(tmp_path, base_cfg))
    jump_extractor = SyntheticExtractor(config_path=_write_config(tmp_path, jump_cfg))

    base = base_extractor.extract_ohlcv(["AAPL"], "2020-01-01", "2021-12-31")
    jump = jump_extractor.extract_ohlcv(["AAPL"], "2020-01-01", "2021-12-31")

    base_kurtosis = np.log(base["Close"]).diff().dropna().kurtosis()
    jump_kurtosis = np.log(jump["Close"]).diff().dropna().kurtosis()
    assert jump_kurtosis > base_kurtosis + 0.5


def test_correlation_tracks_target_matrix(tmp_path):
    cfg = {
        "synthetic": {
            "generator_version": "v1",
            "seed": 314,
            "start_date": "2020-01-01",
            "end_date": "2022-12-31",
            "frequency": "B",
            "tickers": ["AAPL", "MSFT"],
            "price_model": "gbm",
            "volatility_model": "none",
            "regimes": {"enabled": False},
            "event_library": {},
            "microstructure": {},
            "market_hours": {},
            "correlation": {"mode": "static", "target_matrix": [[1.0, 0.7], [0.7, 1.0]]},
            "jump_diffusion": {"enabled": False},
        }
    }
    extractor = SyntheticExtractor(config_path=_write_config(tmp_path, cfg))
    data = extractor.extract_ohlcv(["AAPL", "MSFT"], "2020-01-01", "2022-12-31")

    price_matrix = data.pivot_table(index=data.index, columns="ticker", values="Close").sort_index()
    wide = price_matrix.pct_change().dropna()
    corr = wide.corr().iloc[0, 1]
    assert abs(corr - 0.7) < 0.2


def test_heston_shows_volatility_clustering(tmp_path):
    cfg = {
        "synthetic": {
            "generator_version": "v1",
            "seed": 2024,
            "start_date": "2020-01-01",
            "end_date": "2022-12-31",
            "frequency": "B",
            "tickers": ["AAPL"],
            "price_model": "heston",
            "volatility_model": "stochastic_vol",
            "regimes": {
                "enabled": False,
                "params": {
                    "base": {
                        "heston_kappa": 0.35,
                        "heston_theta": 0.0004,
                        "heston_sigma": 0.25,
                        "heston_rho": -0.2,
                        "vol": 0.02,
                        "drift": 0.0003,
                    }
                },
            },
            "event_library": {},
            "microstructure": {},
            "market_hours": {},
            "jump_diffusion": {"enabled": False},
        }
    }
    extractor = SyntheticExtractor(config_path=_write_config(tmp_path, cfg))
    data = extractor.extract_ohlcv(["AAPL"], "2020-01-01", "2022-12-31")

    squared_returns = data["Close"].pct_change().dropna() ** 2
    ac1 = squared_returns.autocorr(lag=1)
    assert ac1 > 0.05
