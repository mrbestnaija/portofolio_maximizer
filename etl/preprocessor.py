"""Data preprocessing with vectorized transformations."""
import pandas as pd
import numpy as np
from typing import Any, Tuple, Dict, Optional
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

    @staticmethod
    def _data_columns(data: pd.DataFrame) -> list[str]:
        """Return the value-bearing columns used for preprocessing health checks."""
        if data is None or data.empty:
            return []

        candidate_numeric = [
            col
            for col in data.columns
            if pd.api.types.is_numeric_dtype(data[col]) and str(col).lower() not in {"ticker", "symbol"}
        ]
        if candidate_numeric:
            return candidate_numeric

        excluded = {"ticker", "symbol", "date", "datetime", "timestamp"}
        return [col for col in data.columns if str(col).lower() not in excluded]

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

    def validate_post_preprocess(
        self,
        raw_data: pd.DataFrame,
        processed_data: pd.DataFrame,
        *,
        min_usable_bars: int = 120,
        max_imputed_fraction: float = 0.30,
        max_padding_fraction: float = 0.20,
        on_failure: str = "warn",
    ) -> Dict[str, Any]:
        """Validate the post-fill / post-pad frame and classify production readiness.

        The validator is intentionally conservative:
        - structural issues fail closed;
        - sparse / heavily imputed / over-padded frames remain research-usable
          but cannot promote to live capital.
        """
        raw = raw_data.copy() if isinstance(raw_data, pd.DataFrame) else pd.DataFrame()
        processed = processed_data.copy() if isinstance(processed_data, pd.DataFrame) else pd.DataFrame()

        raw_rows = int(len(raw))
        processed_rows = int(len(processed))
        data_columns = [col for col in self._data_columns(processed) if col in raw.columns] if not raw.empty else self._data_columns(processed)
        if not data_columns and not processed.empty:
            data_columns = self._data_columns(processed)

        raw_missing_cells = 0.0
        if raw_rows > 0 and data_columns:
            try:
                raw_missing_cells = float(raw.loc[:, data_columns].isna().sum().sum())
            except Exception:
                raw_missing_cells = float(raw.reindex(columns=data_columns).isna().sum().sum())

        padding_rows = max(0, processed_rows - raw_rows)
        total_cells = float(processed_rows * len(data_columns)) if processed_rows > 0 and data_columns else 0.0
        imputed_cells = float(raw_missing_cells + (padding_rows * len(data_columns)))
        imputed_fraction = (imputed_cells / total_cells) if total_cells > 0 else 0.0
        padding_fraction = (padding_rows / processed_rows) if processed_rows > 0 else 0.0

        missing_after = 0
        non_finite_values = 0
        numeric_cols = [
            col for col in data_columns if col in processed.columns and pd.api.types.is_numeric_dtype(processed[col])
        ]
        if processed_rows > 0 and data_columns:
            try:
                missing_after = int(processed.loc[:, data_columns].isna().sum().sum())
            except Exception:
                missing_after = int(processed.reindex(columns=data_columns).isna().sum().sum())
        if processed_rows > 0 and numeric_cols:
            try:
                numeric_values = processed.loc[:, numeric_cols].to_numpy(dtype=float, copy=True)
                non_finite_values = int((~np.isfinite(numeric_values)).sum())
            except Exception:
                non_finite_values = 0

        duplicate_index_rows = int(processed.index.duplicated().sum()) if processed_rows > 0 else 0
        duplicate_rows = int(processed.duplicated().sum()) if processed_rows > 0 else 0

        monotonic_dates = True
        if processed_rows > 0:
            try:
                parsed_index = pd.to_datetime(processed.index, errors="coerce")
                monotonic_dates = bool(len(parsed_index) == processed_rows and not pd.isna(parsed_index).any() and pd.Index(parsed_index).is_monotonic_increasing)
            except Exception:
                monotonic_dates = False

        structural_failures: list[str] = []
        if processed_rows <= 0:
            structural_failures.append("EMPTY_FRAME")
        if duplicate_index_rows > 0 or duplicate_rows > 0:
            structural_failures.append("DUPLICATE_ROWS")
        if not monotonic_dates:
            structural_failures.append("NON_MONOTONIC_DATES")
        if non_finite_values > 0:
            structural_failures.append("INFINITE_VALUES")
        if missing_after > 0:
            structural_failures.append("MISSING_VALUES_AFTER_PREPROCESS")
        if processed_rows < int(min_usable_bars):
            structural_failures.append("MIN_BARS_UNMET")

        quality_warnings: list[str] = []
        if raw_rows < int(min_usable_bars):
            quality_warnings.append(
                f"SPARSE_DATA:raw_rows={raw_rows} < min_usable_bars={int(min_usable_bars)}"
            )
        if imputed_fraction > float(max_imputed_fraction):
            quality_warnings.append(
                f"HIGH_IMPUTE:imputed_fraction={imputed_fraction:.1%} > {float(max_imputed_fraction):.0%}"
            )
        if padding_fraction > float(max_padding_fraction):
            quality_warnings.append(
                f"HIGH_IMPUTE:padding_fraction={padding_fraction:.1%} > {float(max_padding_fraction):.0%}"
            )

        status = "PASS"
        if structural_failures:
            status = "FAIL"
        elif quality_warnings:
            status = "WARN"

        production_ok = status == "PASS"
        research_ok = status != "FAIL"
        quality_tag = "CLEAN"
        if quality_warnings:
            quality_tag = "HIGH_IMPUTE" if any(token.startswith("HIGH_IMPUTE") for token in quality_warnings) else "SPARSE_DATA"
        if structural_failures:
            quality_tag = "BLOCKED"

        reason_parts = structural_failures if structural_failures else quality_warnings
        reason = "; ".join(reason_parts) if reason_parts else "CLEAN"

        return {
            "status": status,
            "reason": reason,
            "quality_tag": quality_tag,
            "production_ok": production_ok,
            "research_ok": research_ok,
            "raw_length": raw_rows,
            "processed_length": processed_rows,
            "usable_bars": processed_rows,
            "min_usable_bars": int(min_usable_bars),
            "data_columns": data_columns,
            "imputed_cells": int(imputed_cells),
            "imputed_fraction": float(imputed_fraction),
            "padding_rows": int(padding_rows),
            "padding_fraction": float(padding_fraction),
            "duplicate_rows": duplicate_rows,
            "duplicate_index_rows": duplicate_index_rows,
            "non_finite_values": int(non_finite_values),
            "missing_values_after": int(missing_after),
            "monotonic_dates": monotonic_dates,
            "max_imputed_fraction": float(max_imputed_fraction),
            "max_padding_fraction": float(max_padding_fraction),
            "on_failure": str(on_failure or "warn").lower(),
        }

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
