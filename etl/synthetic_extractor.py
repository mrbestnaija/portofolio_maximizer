"""Synthetic data extractor (Phase 0 scaffolding).

Provides a config-driven, deterministic synthetic OHLCV generator that plugs
into DataSourceManager without changing existing behaviour. Defaults mirror
the legacy generate_synthetic_ohlcv helper until richer generators land.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from etl.base_extractor import BaseExtractor, ExtractorMetadata

logger = logging.getLogger(__name__)


@dataclass
class SyntheticConfig:
    generator_version: str = "v0"
    dataset_id_strategy: str = "hash"
    seed: int = 123
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    frequency: str = "B"
    tickers: Sequence[str] = ("AAPL", "MSFT")
    market_condition: str = "efficient"
    persistence_root: Path = Path("data/synthetic")
    partitioning: str = "by_ticker"
    keep_last: int = 3
    validation_checks: Sequence[str] = (
        "schema",
        "monotonic_index",
        "ohlcv_sanity",
        "non_negative_prices",
    )
    dataset_id_override: Optional[str] = None
    dataset_path_override: Optional[Path] = None
    raw: Dict[str, Any] = None


class SyntheticExtractor(BaseExtractor):
    """Config-driven synthetic extractor."""

    def __init__(
        self,
        name: str = "synthetic",
        config_path: str = "config/synthetic_data_config.yml",
        timeout: int = 30,
        cache_hours: int = 24,
        storage=None,
        **kwargs,
    ):
        super().__init__(name=name, timeout=timeout, cache_hours=cache_hours, storage=storage, **kwargs)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.dataset_id_override = os.getenv("SYNTHETIC_DATASET_ID")
        dataset_path_env = os.getenv("SYNTHETIC_DATASET_PATH")
        self.dataset_path_override = Path(dataset_path_env) if dataset_path_env else None
        logger.info("Synthetic extractor ready (version=%s)", self.config.generator_version)

    def extract_ohlcv(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        if not tickers:
            raise ValueError("At least one ticker is required for synthetic extraction")

        start, end = self._resolve_dates(start_date, end_date)
        tickers_resolved = [t.strip() for t in tickers if t.strip()]
        if not tickers_resolved:
            raise ValueError("No valid tickers provided after normalization")

        data = self._load_persisted_dataset(tickers_resolved)
        dataset_id = None

        if data is None:
            if self.config.generator_version == "v0":
                data = self._generate_v0(tickers_resolved, start, end, self.config.frequency, self.config.seed)
            else:
                data = self._generate_v1(tickers_resolved, start, end)
            dataset_id = self._compute_dataset_id(tickers_resolved, start, end, self.config.seed)
        else:
            dataset_id = data.attrs.get("dataset_id") or self.dataset_id_override or "synthetic_loaded"

        data.attrs["source"] = self.name
        data.attrs["dataset_id"] = dataset_id
        data.attrs["generator_version"] = self.config.generator_version

        if self.config.partitioning == "by_ticker":
            # Ensure ticker column exists for downstream slicing
            if "ticker" not in data.columns:
                data["ticker"] = np.repeat(tickers_resolved[0], len(data))

        return data

    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []

        if data is None or data.empty:
            errors.append("Empty dataset")
            return {
                "passed": False,
                "errors": errors,
                "warnings": warnings,
                "quality_score": 0.0,
                "metrics": {"missing_rate": 1.0, "outlier_count": 0, "gap_count": 0},
            }

        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        missing = required_cols - set(data.columns)
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")

        inferred_freq = None
        ticker_col_present = "ticker" in data.columns
        if not isinstance(data.index, pd.DatetimeIndex):
            errors.append("Index is not DatetimeIndex")
        else:
            if ticker_col_present:
                # Allow duplicate dates across tickers but require uniqueness per (date, ticker)
                multi = pd.MultiIndex.from_frame(pd.DataFrame({"date": data.index, "ticker": data["ticker"]}))
                if multi.has_duplicates:
                    errors.append("Duplicate (date, ticker) combinations detected")
                # Check monotonic within each ticker
                for t, df_t in data.groupby("ticker"):
                    if not df_t.index.is_monotonic_increasing:
                        errors.append(f"Index not monotonic for ticker {t}")
            else:
                if not data.index.is_monotonic_increasing:
                    errors.append("Index not monotonic increasing")
                if data.index.has_duplicates:
                    errors.append("Index has duplicates")

            try:
                inferred_freq = pd.infer_freq(data.index)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                inferred_freq = None
            if inferred_freq and self.config.frequency and inferred_freq != self.config.frequency:
                warnings.append(f"Inferred frequency {inferred_freq} differs from config {self.config.frequency}")

        price_fields = ["Open", "High", "Low", "Close"]
        for field in price_fields:
            if field in data.columns and (data[field] <= 0).any():
                errors.append(f"Non-positive values in {field}")
                break

        if "High" in data.columns and "Low" in data.columns:
            if (data["High"] < data["Low"]).any():
                errors.append("High < Low detected")

        if "Volume" in data.columns and (data["Volume"] < 0).any():
            errors.append("Negative values in Volume")

        metrics = {
            "missing_rate": float(data.isna().mean().mean()),
            "outlier_count": int(((data.select_dtypes(include=[float, int]) > 1e9)).sum().sum()),
            "gap_count": self._count_gaps(data.index, self.config.frequency) if isinstance(data.index, pd.DatetimeIndex) else 0,
            "rows": len(data),
        }

        passed = len(errors) == 0
        quality_score = max(0.0, 1.0 - metrics["missing_rate"])

        return {
            "passed": passed,
            "errors": errors,
            "warnings": warnings,
            "quality_score": quality_score,
            "metrics": metrics,
        }

    def get_metadata(self, ticker: str, data: pd.DataFrame) -> ExtractorMetadata:
        if data is None or data.empty:
            raise ValueError("Cannot build metadata for empty data")
        idx = data.index
        extraction_timestamp = datetime.utcnow()
        data_start_date = idx.min().to_pydatetime()
        data_end_date = idx.max().to_pydatetime()
        row_count = len(data)
        return ExtractorMetadata(
            ticker=ticker,
            source=self.name,
            extraction_timestamp=extraction_timestamp,
            data_start_date=data_start_date,
            data_end_date=data_end_date,
            row_count=row_count,
            cache_hit=False,
        )

    # --- Helpers ---

    def _load_config(self) -> SyntheticConfig:
        if not self.config_path.exists():
            logger.warning("Synthetic config not found at %s; using defaults", self.config_path)
            return SyntheticConfig(raw={})

        try:
            cfg = yaml.safe_load(self.config_path.read_text()) or {}
            root = cfg.get("synthetic", {}) or {}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load synthetic config (%s); using defaults", exc)
            return SyntheticConfig(raw={})

        return SyntheticConfig(
            generator_version=str(root.get("generator_version", "v0")),
            dataset_id_strategy=str(root.get("dataset_id_strategy", "hash")),
            seed=int(root.get("seed", 123)),
            start_date=str(root.get("start_date", "2020-01-01")),
            end_date=str(root.get("end_date", "2024-01-01")),
            frequency=str(root.get("frequency", "B")),
            tickers=tuple(root.get("tickers", ["AAPL", "MSFT"])),
            market_condition=str(root.get("market_condition", "efficient")),
            persistence_root=Path(root.get("persistence", {}).get("output_root", "data/synthetic")),
            partitioning=str(root.get("persistence", {}).get("partitioning", "by_ticker")),
            keep_last=int(root.get("persistence", {}).get("keep_last", 3)),
            validation_checks=tuple((root.get("validation", {}) or {}).get("checks", [])),
            raw=root,
            dataset_id_override=os.getenv("SYNTHETIC_DATASET_ID"),
            dataset_path_override=Path(os.getenv("SYNTHETIC_DATASET_PATH")) if os.getenv("SYNTHETIC_DATASET_PATH") else None,
        )

    def _resolve_dates(self, start_date: Optional[str], end_date: Optional[str]) -> (str, str):
        start = start_date or self.config.start_date
        end = end_date or self.config.end_date
        return start, end

    def _load_persisted_dataset(self, tickers: Sequence[str]) -> Optional[pd.DataFrame]:
        dataset_id = self.dataset_id_override
        if not dataset_id and not self.dataset_path_override:
            return None

        base_path = self.dataset_path_override or (self.config.persistence_root / dataset_id)
        if not base_path.exists():
            logger.warning("Requested synthetic dataset path %s does not exist; falling back to generation", base_path)
            return None

        frames = []
        if self.config.partitioning == "by_ticker":
            for ticker in tickers:
                candidate = base_path / f"{ticker}.parquet"
                if candidate.exists():
                    frames.append(pd.read_parquet(candidate))
        else:
            combined = base_path / "combined.parquet"
            if combined.exists():
                frames.append(pd.read_parquet(combined))

        if not frames:
            logger.warning("No persisted synthetic parquet files found under %s", base_path)
            return None

        data = pd.concat(frames).sort_index()
        data.attrs["dataset_id"] = dataset_id or base_path.name
        return data

    def _compute_dataset_id(self, tickers: Sequence[str], start: str, end: str, seed: int) -> str:
        payload = {
            "generator_version": self.config.generator_version,
            "tickers": list(tickers),
            "start": start,
            "end": end,
            "seed": seed,
        }
        payload_str = json.dumps(payload, sort_keys=True)
        digest = hashlib.sha1(payload_str.encode("utf-8")).hexdigest()  # nosec B303
        if self.config.dataset_id_strategy == "timestamped_hash":
            suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            return f"syn_{digest[:12]}_{suffix}"
        return f"syn_{digest[:12]}"

    def _count_gaps(self, index: pd.DatetimeIndex, expected_freq: Optional[str]) -> int:
        if expected_freq is None or len(index) < 2:
            return 0
        expected = pd.date_range(start=index.min(), end=index.max(), freq=expected_freq)
        missing = expected.difference(index)
        return len(missing)

    # --- Core generators (Phase 1) ---

    def _generate_v1(self, tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
        date_index = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq=self.config.frequency)
        if len(date_index) == 0:
            return pd.DataFrame()

        rng_global = np.random.default_rng(self.config.seed)
        regime_enabled = self.config.raw.get("regimes", {}).get("enabled", False)
        regimes = self.config.raw.get("regimes", {})
        names = regimes.get("names") or ["base"]
        transition = np.array(regimes.get("transition_matrix") or [[1.0]])
        params = regimes.get("params") or {}
        correlation_cfg = self.config.raw.get("correlation", {}) or {}
        corr_matrix = self._prepare_correlation_matrix(correlation_cfg, len(tickers))

        n = len(date_index)
        shocks = rng_global.multivariate_normal(np.zeros(len(tickers)), corr_matrix, size=n)
        frames = []

        regime_states = self._simulate_regimes(rng_global, n, names, transition) if regime_enabled else [names[0]] * n

        for idx_t, ticker in enumerate(tickers):
            prices, volume = self._simulate_price_path(
                rng_global=rng_global,
                shocks=shocks[:, idx_t],
                date_index=date_index,
                ticker=ticker,
                regime_states=regime_states,
                regime_params=params,
                base_drift=0.0005,
                base_vol=0.01,
            )
            frame = pd.DataFrame(
                {
                    "Open": prices["open"],
                    "High": prices["high"],
                    "Low": prices["low"],
                    "Close": prices["close"],
                    "Volume": volume,
                    "Adj Close": prices["close"],
                    "ticker": ticker,
                },
                index=date_index,
            )
            frame.index.name = "Date"
            frames.append(frame)

        return pd.concat(frames).sort_index()

    def _simulate_regimes(self, rng: np.random.Generator, n: int, names: Sequence[str], transition: np.ndarray) -> List[str]:
        if transition.shape != (len(names), len(names)):
            raise ValueError("Transition matrix shape does not match regime names")
        states = []
        current = 0
        for _ in range(n):
            states.append(names[current])
            probs = transition[current]
            current = rng.choice(len(names), p=probs)
        return states

    def _simulate_price_path(
        self,
        rng_global: np.random.Generator,
        shocks: np.ndarray,
        date_index: pd.DatetimeIndex,
        ticker: str,
        regime_states: Sequence[str],
        regime_params: Dict[str, Any],
        base_drift: float,
        base_vol: float,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        n = len(date_index)
        open_arr = np.zeros(n)
        high_arr = np.zeros(n)
        low_arr = np.zeros(n)
        close_arr = np.zeros(n)
        volume_arr = np.zeros(n, dtype=int)

        price = 100.0 * (1 + rng_global.normal(0, 0.01))
        for i in range(n):
            regime = regime_states[i]
            rp = regime_params.get(regime, {})
            drift = rp.get("drift", base_drift)
            vol = rp.get("vol", base_vol)
            jump_intensity = rp.get("jump_intensity", 0.0)
            jump_mean = rp.get("jump_mean", 0.0)
            jump_std = rp.get("jump_std", 0.0)

            ret = drift + vol * shocks[i]
            if jump_intensity > 0 and rng_global.uniform() < jump_intensity:
                ret += rng_global.normal(jump_mean, jump_std)

            price = price * (1 + ret)
            close = price
            open_ = close * (1 + rng_global.normal(0, 0.002))
            high = max(open_, close) * (1 + abs(rng_global.normal(0, 0.003)))
            low = min(open_, close) * (1 - abs(rng_global.normal(0, 0.003)))
            volume = int(1_000_000 * (1 + rng_global.normal(0, 0.05)))

            open_arr[i] = open_
            high_arr[i] = high
            low_arr[i] = low
            close_arr[i] = close
            volume_arr[i] = max(volume, 0)

        return {"open": open_arr, "high": high_arr, "low": low_arr, "close": close_arr}, volume_arr

    def _prepare_correlation_matrix(self, cfg: Dict[str, Any], n_assets: int) -> np.ndarray:
        target = cfg.get("target_matrix") or []
        if not target:
            return np.eye(n_assets)
        mat = np.array(target, dtype=float)
        if mat.shape != (n_assets, n_assets):
            return np.eye(n_assets)
        # Ensure PSD by small diagonal bump if needed
        try:
            np.linalg.cholesky(mat)
            return mat
        except np.linalg.LinAlgError:
            eps = 1e-4
            bumped = mat + np.eye(n_assets) * eps
            return bumped

    def _generate_v0(self, tickers: Sequence[str], start: str, end: str, frequency: str, seed: int) -> pd.DataFrame:
        date_index = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq=frequency)
        if len(date_index) == 0:
            return pd.DataFrame()

        frames = []
        base_rng = np.random.default_rng(seed)
        for i, ticker in enumerate(tickers):
            rng = np.random.default_rng(base_rng.integers(0, 1_000_000) + i)
            n = len(date_index)
            rets = rng.normal(0.0005, 0.01, size=n)
            price = 100.0 * np.cumprod(1 + rets)
            open_ = price * (1 + rng.normal(0, 0.002, size=n))
            close = price
            high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, size=n)))
            low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, size=n)))
            volume = (1_000_000 * (1 + rng.normal(0, 0.05, size=n))).astype(int)
            frame = pd.DataFrame(
                {
                    "Open": open_,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Volume": volume,
                    "Adj Close": close,
                    "ticker": ticker,
                },
                index=date_index,
            )
            frame.index.name = "Date"
            frames.append(frame)

        return pd.concat(frames).sort_index()
