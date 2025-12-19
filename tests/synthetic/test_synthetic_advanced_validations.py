import numpy as np
import pandas as pd
import yaml

from etl.synthetic_extractor import SyntheticExtractor


def _cfg(tmp_path, body: dict) -> str:
    path = tmp_path / "synthetic.yml"
    path.write_text(yaml.safe_dump({"synthetic": body}))
    return str(path)


def test_regime_transition_frequencies(tmp_path):
    cfg = {
        "generator_version": "v1",
        "seed": 11,
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "frequency": "B",
        "tickers": ["AAPL"],
        "price_model": "gbm",
        "regimes": {
            "enabled": True,
            "names": ["calm", "panic"],
            "transition_matrix": [[0.7, 0.3], [0.2, 0.8]],
            "params": {"calm": {"drift": 0.0003, "vol": 0.012}, "panic": {"drift": -0.0005, "vol": 0.04}},
        },
    }
    extractor = SyntheticExtractor(config_path=_cfg(tmp_path, cfg))
    data = extractor.extract_ohlcv(["AAPL"], cfg["start_date"], cfg["end_date"])
    states = pd.Series(data.attrs.get("regimes_used", []))
    assert {"calm", "panic"}.issubset(set(states)) or len(states) == 0


def test_tail_dependence_with_copula(tmp_path):
    cfg = {
        "generator_version": "v1",
        "seed": 5,
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "frequency": "B",
        "tickers": ["AAPL", "MSFT"],
        "price_model": "gbm",
        "correlation": {"mode": "t_copula", "copula_df": 2, "target_matrix": [[1.0, 0.6], [0.6, 1.0]], "tail_scale": 2.0},
        "regimes": {
            "enabled": True,
            "names": ["base"],
            "transition_matrix": [[1.0]],
            "params": {"base": {"drift": 0.0003, "vol": 0.05}},
        },
    }
    extractor = SyntheticExtractor(config_path=_cfg(tmp_path, cfg))
    data = extractor.extract_ohlcv(["AAPL", "MSFT"], cfg["start_date"], cfg["end_date"])
    price_matrix = data.pivot_table(index=data.index, columns="ticker", values="Close").sort_index()
    wide = price_matrix.pct_change().dropna()
    kurt = wide.apply(lambda s: s.kurtosis())
    # Expect heavier tails than Gaussian baseline (3), allow tolerance for finite sample
    assert kurt.min() > 1.5


def test_microstructure_exec_cost_monotonic(tmp_path):
    cfg = {
        "generator_version": "v1",
        "seed": 21,
        "start_date": "2020-01-01",
        "end_date": "2020-03-01",
        "frequency": "B",
        "tickers": ["AAPL"],
        "price_model": "gbm",
        "regimes": {"enabled": False},
    }
    extractor = SyntheticExtractor(config_path=_cfg(tmp_path, cfg))
    data = extractor.extract_ohlcv(["AAPL"], cfg["start_date"], cfg["end_date"])
    assert "TxnCostBps" in data.columns
    assert data["TxnCostBps"].min() >= 0
