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
from dataclasses import dataclass, field
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
    # TODO: Add a way to override Synthetic data Parameters including the start and end date from the environment/default range is 10 years min
    start_date: str = "2014-01-01"
    end_date: str = "2024-01-01"
    frequency: str = "B"
    tickers: Sequence[str] = ("AAPL", "MSFT")
    market_condition: str = "efficient"
    price_model: str = "gbm"
    volatility_model: str = "none"
    correlation_mode: str = "static"
    correlation_target: Optional[Sequence[Sequence[float]]] = None
    jump_diffusion_params: Dict[str, Any] = field(default_factory=dict)
    regimes_enabled: bool = False
    regime_names: Sequence[str] = ("base",)
    regime_transition: Sequence[Sequence[float]] = field(default_factory=list)
    regime_params: Dict[str, Any] = field(default_factory=dict)
    event_library: Dict[str, Any] = field(default_factory=dict)
    market_hours: Dict[str, Any] = field(default_factory=dict)
    microstructure: Dict[str, Any] = field(default_factory=dict)
    calibration: Dict[str, Any] = field(default_factory=dict)
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
    raw: Dict[str, Any] = field(default_factory=dict)


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

        # Microstructure sanity when present
        micro_fields = {
            "Spread": lambda s: (s < 0).any(),
            "Slippage": lambda s: (s < 0).any(),
            "Depth": lambda s: (s <= 0).any(),
            "OrderImbalance": lambda s: (~np.isfinite(s)).any(),
        }
        for field, bad_fn in micro_fields.items():
            if field in data.columns:
                try:
                    if bad_fn(data[field]):
                        errors.append(f"Invalid microstructure values in {field}")
                except Exception:
                    warnings.append(f"Could not validate microstructure field {field}")

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

        regimes_cfg = root.get("regimes", {}) or {}
        correlation_cfg = root.get("correlation", {}) or {}
        jump_cfg = root.get("jump_diffusion", {}) or {}
        event_cfg = root.get("event_library", {}) or {}
        market_cfg = root.get("market_hours", {}) or {}
        micro_cfg = root.get("microstructure", {}) or {}
        calib_cfg = root.get("calibration", {}) or {}

        return SyntheticConfig(
            generator_version=str(root.get("generator_version", "v0")),
            dataset_id_strategy=str(root.get("dataset_id_strategy", "hash")),
            seed=int(root.get("seed", 123)),
            start_date=str(root.get("start_date", "2020-01-01")),
            end_date=str(root.get("end_date", "2024-01-01")),
            frequency=str(root.get("frequency", "B")),
            tickers=tuple(root.get("tickers", ["AAPL", "MSFT"])),
            market_condition=str(root.get("market_condition", "efficient")),
            price_model=str(root.get("price_model", "gbm")),
            volatility_model=str(root.get("volatility_model", "none")),
            correlation_mode=str(correlation_cfg.get("mode", "static")),
            correlation_target=correlation_cfg.get("target_matrix"),
            jump_diffusion_params={
                "enabled": bool(jump_cfg.get("enabled", False)),
                "intensity": float(jump_cfg.get("intensity", 0.0)),
                "jump_mean": float(jump_cfg.get("jump_mean", 0.0)),
                "jump_std": float(jump_cfg.get("jump_std", 0.0)),
            },
            regimes_enabled=bool(regimes_cfg.get("enabled", False)),
            regime_names=tuple(regimes_cfg.get("names") or ["base"]),
            regime_transition=regimes_cfg.get("transition_matrix") or [[1.0]],
            regime_params=regimes_cfg.get("params") or {},
            event_library=event_cfg,
            market_hours=market_cfg,
            microstructure=micro_cfg,
            calibration=calib_cfg,
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
        dataset_path = self.dataset_path_override
        pointer_payload = None

        if dataset_id and dataset_id.lower() == "latest":
            pointer_payload = self._load_latest_pointer()
        elif dataset_path:
            if dataset_path.is_file() and dataset_path.name == "latest.json":
                pointer_payload = self._load_latest_pointer(pointer_path=dataset_path)
            elif dataset_path.is_dir() and dataset_path.name == "latest":
                pointer_payload = self._load_latest_pointer(pointer_path=dataset_path / "latest.json")

        if pointer_payload:
            dataset_id = pointer_payload.get("dataset_id") or dataset_id
            pointer_path = pointer_payload.get("dataset_path")
            if pointer_path:
                dataset_path = Path(pointer_path)

        if not dataset_id and not dataset_path:
            return None

        base_path = dataset_path or (self.config.persistence_root / dataset_id)
        if base_path.is_file():
            base_path = base_path.parent
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

    def _load_latest_pointer(self, pointer_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        path = pointer_path or (self.config.persistence_root / "latest.json")
        if not path.exists():
            logger.warning("Synthetic latest pointer not found at %s", path)
            return None
        try:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                return None
            return payload
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read synthetic latest pointer %s: %s", path, exc)
            return None

    def _compute_dataset_id(self, tickers: Sequence[str], start: str, end: str, seed: int) -> str:
        payload = {
            "generator_version": self.config.generator_version,
            "tickers": list(tickers),
            "start": start,
            "end": end,
            "seed": seed,
        }
        if self.config.generator_version != "v0":
            payload["config_hash"] = hashlib.sha1(json.dumps(self.config.raw, sort_keys=True).encode("utf-8")).hexdigest()  # nosec B303
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
        correlation_cfg = self.config.raw.get("correlation", {}) or {}
        corr_matrix = self._prepare_correlation_matrix(correlation_cfg, len(tickers))

        n = len(date_index)
        shocks = rng_global.multivariate_normal(np.zeros(len(tickers)), corr_matrix, size=n)
        regime_states = (
            self._simulate_regimes(rng_global, n, self.config.regime_names, np.array(self.config.regime_transition))
            if self.config.regimes_enabled
            else [self.config.regime_names[0]] * n
        )

        frames = []
        price_model = (self.config.price_model or "gbm").lower()
        for idx_t, ticker in enumerate(tickers):
            prices, volume, micro = self._simulate_price_path(
                price_model=price_model,
                rng_global=rng_global,
                shocks=shocks[:, idx_t],
                date_index=date_index,
                ticker=ticker,
                regime_states=regime_states,
            )
            frame = pd.DataFrame(
                {
                    "Open": prices["open"],
                    "High": prices["high"],
                    "Low": prices["low"],
                    "Close": prices["close"],
                    "Volume": volume,
                    "Spread": micro.get("spread"),
                    "Slippage": micro.get("slippage"),
                    "Depth": micro.get("depth"),
                    "OrderImbalance": micro.get("order_imbalance"),
                    "Adj Close": prices["close"],
                    "ticker": ticker,
                },
                index=date_index,
            )
            frame.index.name = "Date"
            frames.append(frame)

        data = pd.concat(frames).sort_index()
        data.attrs["regimes_used"] = sorted(set(regime_states))
        data.attrs["correlation_mode"] = self.config.correlation_mode
        return data

    def _simulate_regimes(self, rng: np.random.Generator, n: int, names: Sequence[str], transition: np.ndarray) -> List[str]:
        if transition.shape != (len(names), len(names)):
            raise ValueError("Transition matrix shape does not match regime names")
        states: List[str] = []
        current = 0
        for _ in range(n):
            states.append(names[current])
            probs = transition[current].astype(float)
            probs = probs / probs.sum() if probs.sum() else np.ones(len(names)) / len(names)
            current = rng.choice(len(names), p=probs)
        return states

    def _simulate_price_path(
        self,
        price_model: str,
        rng_global: np.random.Generator,
        shocks: np.ndarray,
        date_index: pd.DatetimeIndex,
        ticker: str,
        regime_states: Sequence[str],
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
        n = len(date_index)
        open_arr = np.zeros(n)
        high_arr = np.zeros(n)
        low_arr = np.zeros(n)
        close_arr = np.zeros(n)
        volume_arr = np.zeros(n, dtype=int)
        spread_arr = np.zeros(n)
        slippage_arr = np.zeros(n)
        depth_arr = np.zeros(n)
        imbalance_arr = np.zeros(n)

        base_drift = {"efficient": 0.0005, "mixed": 0.0003, "inefficient": -0.0001}.get(
            self.config.market_condition, 0.0005
        )
        drift_multiplier = float(self.config.calibration.get("drift_multiplier", 1.0))
        vol_multiplier = float(self.config.calibration.get("vol_multiplier", 1.0))

        base_drift = base_drift * drift_multiplier
        base_vol = (0.01 if self.config.volatility_model == "none" else 0.012) * vol_multiplier
        jump_cfg = self.config.jump_diffusion_params or {}
        regime_params = self.config.regime_params or {}
        regime_names = self.config.regime_names or ("base",)
        micro_cfg = self.config.microstructure or {}

        price = 100.0 * (1 + rng_global.normal(0, 0.01))
        current_var = max(base_vol**2, 1e-6)
        long_term_log = np.log(max(price, 1e-6))

        for i in range(n):
            regime = regime_states[i] if i < len(regime_states) else regime_names[0]
            rp = regime_params.get(regime, {})
            drift = float(rp.get("drift", base_drift))
            vol = float(rp.get("vol", base_vol))
            jump_intensity = float(rp.get("jump_intensity", jump_cfg.get("intensity", 0.0)))
            jump_mean = float(rp.get("jump_mean", jump_cfg.get("jump_mean", 0.0)))
            jump_std = float(rp.get("jump_std", jump_cfg.get("jump_std", 0.0)))

            active_model = price_model
            if active_model == "hybrid":
                if self.config.volatility_model == "stochastic_vol":
                    active_model = "heston"
                elif jump_cfg.get("enabled"):
                    active_model = "jump_diffusion"
                else:
                    active_model = "gbm"

            inst_vol = vol
            if active_model == "ou":
                log_price = np.log(max(price, 1e-6))
                theta = float(rp.get("ou_speed", 0.05))
                long_term = float(rp.get("ou_mean", long_term_log))
                log_price = log_price + theta * (long_term - log_price) + vol * shocks[i]
                price = float(np.exp(log_price))
                inst_vol = vol
            elif active_model == "jump_diffusion":
                ret = drift + vol * shocks[i]
                if (jump_cfg.get("enabled") or jump_intensity > 0) and rng_global.random() < jump_intensity:
                    ret += rng_global.normal(jump_mean, jump_std)
                price = max(price * (1 + ret), 1e-6)
                inst_vol = max(abs(ret), vol)
            elif active_model == "heston":
                kappa = float(rp.get("heston_kappa", 1.2))
                theta_v = float(rp.get("heston_theta", (vol**2)))
                sigma_v = float(rp.get("heston_sigma", 0.1))
                rho = float(rp.get("heston_rho", -0.3))
                z2 = rng_global.normal()
                z_var = rho * shocks[i] + np.sqrt(max(1 - rho**2, 0.0)) * z2
                current_var = max(
                    current_var + kappa * (theta_v - current_var) + sigma_v * np.sqrt(max(current_var, 1e-8)) * z_var,
                    1e-8,
                )
                inst_vol = np.sqrt(current_var)
                ret = drift + inst_vol * shocks[i]
                if (jump_cfg.get("enabled") or jump_intensity > 0) and rng_global.random() < jump_intensity:
                    ret += rng_global.normal(jump_mean, jump_std)
                price = max(price * np.exp(ret), 1e-6)
            else:  # gbm or fallback
                ret = drift + vol * shocks[i]
                price = max(price * (1 + ret), 1e-6)
                inst_vol = max(abs(ret), vol)

            price, inst_vol = self._apply_event_impacts(
                rng=rng_global,
                index=i,
                price=price,
                inst_vol=inst_vol,
                base_drift=drift,
            )

            # Market hours / seasonality adjustments (apply to volume + volatility)
            vol_multiplier = self._seasonality_multiplier(date_index[i])
            inst_vol = inst_vol * vol_multiplier
            volume_scale = vol_multiplier

            spread_arr[i], slippage_arr[i], depth_arr[i], imbalance_arr[i] = self._simulate_microstructure(
                price=price,
                inst_vol=inst_vol,
                shock=shocks[i],
                regime=regime,
                micro_cfg=micro_cfg,
                rng=rng_global,
            )

            open_price = price * (1 + rng_global.normal(0, 0.002))
            high_price = max(open_price, price) * (1 + abs(rng_global.normal(0, 0.003 + inst_vol * 0.1)))
            low_price = min(open_price, price) * (1 - abs(rng_global.normal(0, 0.003 + inst_vol * 0.1)))
            volume_scale = volume_scale * (1 + inst_vol * 5 * abs(shocks[i]))
            volume_noise = rng_global.normal(0, 0.05)
            volume = int(max(1_000_000 * (1 + volume_noise) * volume_scale, 0))

            open_arr[i] = open_price
            high_arr[i] = high_price
            low_arr[i] = low_price
            close_arr[i] = price
            volume_arr[i] = volume

        micro = {
            "spread": spread_arr,
            "slippage": slippage_arr,
            "depth": depth_arr,
            "order_imbalance": imbalance_arr,
        }
        return {"open": open_arr, "high": high_arr, "low": low_arr, "close": close_arr}, volume_arr, micro

    def _apply_event_impacts(
        self,
        rng: np.random.Generator,
        index: int,
        price: float,
        inst_vol: float,
        base_drift: float,
    ) -> Tuple[float, float]:
        """
        Apply crisis/event library shocks to the current price/vol state.

        Events are simple, deterministic-probability hooks to model:
        - flash crashes (down moves with vol spike)
        - volatility spikes
        - gap risk (up/down jumps)
        - bear run drift shifts
        """
        cfg = self.config.event_library or {}

        flash_cfg = cfg.get("flash_crash", {})
        if flash_cfg.get("enabled") and rng.random() < float(flash_cfg.get("prob", 0.0)):
            impact = float(flash_cfg.get("impact_pct", 0.08))
            price = max(price * (1 - abs(impact)), 1e-6)
            inst_vol = inst_vol * (1 + abs(impact) * 5)

        vol_cfg = cfg.get("vol_spike", {})
        if vol_cfg.get("enabled") and rng.random() < float(vol_cfg.get("prob", 0.0)):
            multiplier = float(vol_cfg.get("multiplier", 2.5))
            inst_vol = inst_vol * max(multiplier, 1.0)

        gap_cfg = cfg.get("gap_risk", {})
        if gap_cfg.get("enabled") and rng.random() < float(gap_cfg.get("prob", 0.0)):
            impact = float(gap_cfg.get("impact_pct", 0.03))
            direction = -1 if rng.random() < 0.6 else 1
            price = max(price * (1 + direction * impact), 1e-6)
            inst_vol = inst_vol * (1 + abs(impact) * 3)

        bear_cfg = cfg.get("bear_run", {})
        if bear_cfg.get("enabled") and rng.random() < float(bear_cfg.get("prob", 0.0)):
            drift_shift = float(bear_cfg.get("drift_shift", -0.001))
            price = max(price * (1 + drift_shift), 1e-6)

        liquidity_cfg = cfg.get("liquidity_shock", {})
        if liquidity_cfg.get("enabled") and rng.random() < float(liquidity_cfg.get("prob", 0.0)):
            impact = float(liquidity_cfg.get("impact_pct", 0.05))
            price = max(price * (1 - abs(impact)), 1e-6)
            inst_vol = inst_vol * float(liquidity_cfg.get("vol_multiplier", 3.0))

        return price, inst_vol

    def _seasonality_multiplier(self, ts: pd.Timestamp) -> float:
        season_cfg = self.config.market_hours.get("seasonality") if self.config.market_hours else {}
        if not season_cfg:
            return 1.0
        dow = ts.dayofweek
        weights = season_cfg.get("day_weights", {})
        return float(weights.get(str(dow), 1.0))

    def _simulate_microstructure(
        self,
        price: float,
        inst_vol: float,
        shock: float,
        regime: str,
        micro_cfg: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Tuple[float, float, float, float]:
        spread_cfg = micro_cfg.get("spread", {})
        slippage_cfg = micro_cfg.get("slippage", {})
        depth_cfg = micro_cfg.get("depth", {})
        imbalance_cfg = micro_cfg.get("order_flow", {})
        base_spread_bps = float(spread_cfg.get("bps", 5.0))
        vol_beta = float(spread_cfg.get("vol_beta", 20.0))
        regime_spread = float(spread_cfg.get("regime_widen", 1.5 if "high" in regime else 1.0))
        spread = price * (base_spread_bps / 10_000.0) * (1 + vol_beta * inst_vol) * regime_spread

        base_slip_bps = float(slippage_cfg.get("bps", 3.0))
        shock_beta = float(slippage_cfg.get("shock_beta", 10.0))
        slippage = price * (base_slip_bps / 10_000.0) * (1 + shock_beta * abs(shock))
        base_depth = float(depth_cfg.get("base", 1_000_000.0))
        depth_vol_beta = float(depth_cfg.get("vol_beta", 15.0))
        min_depth = float(depth_cfg.get("min_depth", 50_000.0))
        depth = max(base_depth * (1 - depth_vol_beta * inst_vol), min_depth)

        imbalance_sigma = float(imbalance_cfg.get("sigma", 0.15))
        shock_beta_imb = float(imbalance_cfg.get("shock_beta", 0.8))
        regime_bias = float(imbalance_cfg.get("regime_bias", -0.1 if "high" in regime else 0.0))
        order_imbalance = rng.normal(regime_bias, imbalance_sigma) + shock_beta_imb * shock
        return spread, slippage, depth, order_imbalance

    def _prepare_correlation_matrix(self, cfg: Dict[str, Any], n_assets: int) -> np.ndarray:
        mode = str(cfg.get("mode", "static"))
        target = cfg.get("target_matrix") or []
        if mode == "factor":
            strength = float(cfg.get("factor_strength", 0.4))
            strength = float(np.clip(strength, -0.95, 0.95))
            mat = np.full((n_assets, n_assets), strength)
            np.fill_diagonal(mat, 1.0)
            return self._ensure_psd(mat, n_assets)

        if mode == "rolling":
            jitter = float(cfg.get("jitter", 0.01))
            base = np.array(target, dtype=float) if target else np.eye(n_assets)
            mat = base + np.eye(n_assets) * jitter
            return self._ensure_psd(mat, n_assets)

        if not target:
            return np.eye(n_assets)
        mat = np.array(target, dtype=float)
        return self._ensure_psd(mat, n_assets)

    def _ensure_psd(self, matrix: np.ndarray, n_assets: int) -> np.ndarray:
        if matrix.shape != (n_assets, n_assets):
            return np.eye(n_assets)
        try:
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            eps = 1e-4
            return matrix + np.eye(n_assets) * eps

    def _generate_v0(self, tickers: Sequence[str], start: str, end: str, frequency: str, seed: int) -> pd.DataFrame:
        date_index = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq=frequency)
        if len(date_index) == 0:
            return pd.DataFrame()

        frames = []
        rng = np.random.default_rng(seed)
        for ticker in tickers:
            n = len(date_index)
            base = 100.0 * (1 + rng.normal(0, 0.01))
            rets = rng.normal(0.0005, 0.01, size=n)
            prices = base * np.cumprod(1 + rets)
            open_ = prices * (1 + rng.normal(0, 0.002, size=n))
            close = prices
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
