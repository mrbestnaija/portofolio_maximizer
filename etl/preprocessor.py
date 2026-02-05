"""Data preprocessing with vectorized transformations."""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """Vectorized data preprocessing for time series."""

    def __init__(self):
        pass

    @staticmethod
    def _ticker_groupby(data: pd.DataFrame):
        """Return a groupby object for per-ticker operations when possible."""
        if isinstance(data, pd.DataFrame) and "ticker" in data.columns:
            tickers = data["ticker"].astype(str).str.upper().str.strip()
            if not tickers.empty and tickers.nunique(dropna=False) > 0:
                return data.groupby(tickers, group_keys=False)
        return None

    def handle_missing(self, data: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
        """Handle missing data with temporal interpolation."""
        if data is None or data.empty:
            return data.copy()

        filled = data.sort_index().copy()
        ticker_group = self._ticker_groupby(filled)

        if method == 'forward':
            filled = ticker_group.ffill() if ticker_group is not None else filled.ffill()
        elif method == 'backward':
            filled = ticker_group.bfill() if ticker_group is not None else filled.bfill()
        elif method == 'hybrid':
            filled = ticker_group.ffill().bfill() if ticker_group is not None else filled.ffill().bfill()
        elif method == 'interpolate':
            if ticker_group is not None:
                filled = ticker_group.apply(lambda df: df.interpolate(method='linear'))
            else:
                filled = filled.interpolate(method='linear')
        elif method == 'drop':
            filled = filled.dropna()
            logger.info("Dropped rows with missing values")
            return filled
        else:
            raise ValueError(f"Unknown method: {method}")

        # Fill remaining NaN with mean (only numeric columns; per-ticker when possible)
        numeric_cols = filled.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if ticker_group is not None:
                group_means = filled.groupby(filled["ticker"].astype(str).str.upper().str.strip())[numeric_cols].transform("mean")
                filled[numeric_cols] = filled[numeric_cols].fillna(group_means)
            filled[numeric_cols] = filled[numeric_cols].fillna(filled[numeric_cols].mean())

        logger.info(f"Filled missing data using {method}")
        return filled

    def normalize(self, data: pd.DataFrame, method: str = 'zscore',
                 columns: Optional[list] = None) -> Tuple[pd.DataFrame, Dict]:
        """Normalize data."""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        normalized = data.copy()
        stats = {}

        for col in columns:
            if col not in data.columns:
                continue

            if method == 'zscore':
                ticker_group = self._ticker_groupby(data)
                if ticker_group is None:
                    mean = data[col].mean()
                    std = data[col].std()
                    if std > 0:
                        normalized[col] = (data[col] - mean) / std
                        stats[col] = {'mean': mean, 'std': std}
                    continue

                tickers = data["ticker"].astype(str).str.upper().str.strip()
                mean = data.groupby(tickers)[col].transform("mean")
                std = data.groupby(tickers)[col].transform("std")
                std_safe = std.replace(0, np.nan)
                normalized[col] = (data[col] - mean) / std_safe
                normalized[col] = normalized[col].fillna(0.0)

                # Store per-ticker stats for inverse transforms/debugging.
                try:
                    per_ticker = (
                        data.assign(_ticker=tickers)
                        .groupby("_ticker")[col]
                        .agg(["mean", "std"])
                        .to_dict(orient="index")
                    )
                    if len(per_ticker) == 1:
                        # Backward-compatible scalar stats when only one ticker exists.
                        only = next(iter(per_ticker.values()))
                        stats[col] = {"mean": only.get("mean"), "std": only.get("std")}
                    else:
                        stats[col] = {
                            "mean": float(data[col].mean()),
                            "std": float(data[col].std()),
                            "per_ticker": per_ticker,
                        }
                except Exception:  # pragma: no cover - stats best effort
                    stats[col] = {}

        logger.info(f"Normalized {len(columns)} columns")
        return normalized, stats

    def apply_normalization(
        self,
        data: pd.DataFrame,
        stats: Dict,
        method: str = "zscore",
        columns: Optional[list] = None,
    ) -> pd.DataFrame:
        """Apply previously computed normalization stats to a new frame (no refit)."""
        if data is None or data.empty:
            return data.copy()
        if not stats:
            return data.copy()

        if columns is None:
            columns = [col for col in stats.keys() if col in data.columns]

        normalized = data.copy()
        if method != "zscore":
            raise ValueError(f"Unknown method: {method}")

        tickers = None
        if "ticker" in data.columns:
            tickers = data["ticker"].astype(str).str.upper().str.strip()

        for col in columns:
            if col not in normalized.columns:
                continue
            col_stats = stats.get(col) or {}
            per_ticker = col_stats.get("per_ticker")
            if isinstance(per_ticker, dict) and tickers is not None:
                mean_map = {t: v.get("mean") for t, v in per_ticker.items()}
                std_map = {t: v.get("std") for t, v in per_ticker.items()}
                mean = tickers.map(mean_map)
                std = tickers.map(std_map)

                global_mean = col_stats.get("mean")
                global_std = col_stats.get("std")
                if global_mean is not None:
                    mean = mean.fillna(float(global_mean))
                if global_std is not None:
                    std = std.fillna(float(global_std))

                std_safe = std.replace(0, np.nan)
                normalized[col] = (data[col] - mean) / std_safe
                normalized[col] = normalized[col].fillna(0.0)
                continue

            mean = col_stats.get("mean")
            std = col_stats.get("std")
            if mean is None or std is None:
                continue
            std = float(std)
            if std == 0 or not np.isfinite(std):
                normalized[col] = 0.0
                continue
            normalized[col] = (data[col] - float(mean)) / std

        logger.info("Applied normalization to %s columns", len(columns))
        return normalized
