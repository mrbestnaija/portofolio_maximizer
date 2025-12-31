"""
Split diagnostics and drift checks for time-series CV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd


@dataclass
class SplitSummary:
    name: str
    length: int
    start: str
    end: str
    mean: float
    std: float
    skew: float
    kurtosis: float


def summarize_returns(name: str, frame: pd.DataFrame) -> SplitSummary:
    series = pd.Series(dtype=float)
    if frame is not None and not frame.empty and "Close" in frame.columns:
        series = frame["Close"].astype(float).pct_change().dropna()
    return SplitSummary(
        name=name,
        length=len(series),
        start=str(frame.index.min()) if frame is not None and not frame.empty else "n/a",
        end=str(frame.index.max()) if frame is not None and not frame.empty else "n/a",
        mean=float(series.mean() or 0),
        std=float(series.std() or 0),
        skew=float(series.skew() or 0),
        kurtosis=float(series.kurtosis() or 0),
    )


def _calc_psi(base: pd.Series, other: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two return series."""
    if base.empty or other.empty:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = base.quantile(quantiles).drop_duplicates().values
    cuts[0], cuts[-1] = -np.inf, np.inf
    base_counts, _ = np.histogram(base, bins=cuts)
    other_counts, _ = np.histogram(other, bins=cuts)
    base_perc = base_counts / base_counts.sum() if base_counts.sum() else np.zeros_like(base_counts)
    other_perc = other_counts / other_counts.sum() if other_counts.sum() else np.zeros_like(other_counts)
    psi = np.sum((base_perc - other_perc) * np.log((base_perc + 1e-9) / (other_perc + 1e-9)))
    return float(psi)


def drift_metrics(train: pd.DataFrame, other: pd.DataFrame) -> Dict[str, float]:
    base = train["Close"].astype(float).pct_change().dropna() if "Close" in train.columns else pd.Series(dtype=float)
    cmp = other["Close"].astype(float).pct_change().dropna() if "Close" in other.columns else pd.Series(dtype=float)
    psi = _calc_psi(base, cmp)
    mean_delta = float(abs(base.mean() - cmp.mean())) if not base.empty and not cmp.empty else 0.0
    std_delta = float(abs(base.std() - cmp.std())) if not base.empty and not cmp.empty else 0.0
    volatility_ratio = float(cmp.std() / base.std()) if not base.empty and base.std() > 0 else 1.0
    # Volume drift
    base_vol = train["Volume"].astype(float) if "Volume" in train.columns else pd.Series(dtype=float)
    cmp_vol = other["Volume"].astype(float) if "Volume" in other.columns else pd.Series(dtype=float)
    vol_psi = _calc_psi(base_vol, cmp_vol) if not base_vol.empty and not cmp_vol.empty else 0.0
    vol_delta = float(abs(base_vol.mean() - cmp_vol.mean())) if not base_vol.empty and not cmp_vol.empty else 0.0

    return {
        "psi": psi,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "vol_psi": vol_psi,
        "vol_delta": vol_delta,
        "volatility_ratio": volatility_ratio,
    }


def validate_non_overlap(indices_a: pd.Index, indices_b: pd.Index) -> bool:
    return indices_a.intersection(indices_b).empty
