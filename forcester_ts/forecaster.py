"""
Unified time-series forecaster orchestrating SARIMAX, GARCH, SAMOSSA,
and the new MSSA-RL variant in parallel.
"""

from __future__ import annotations

import inspect
import logging
import os
import warnings
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ._freq_compat import normalize_freq

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tools.sm_exceptions import InterpolationWarning  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    adfuller = None
    kpss = None
    InterpolationWarning = None

try:
    from etl.warning_recorder import log_warning_records
except Exception:  # pragma: no cover - logging helper optional in standalone runs
    def log_warning_records(records, context):
        return

from .ensemble import (
    EnsembleConfig,
    EnsembleCoordinator,
    canonical_model_key,
    derive_model_confidence,
)
from .garch import GARCHForecaster
from .instrumentation import ModelInstrumentation
from .mssa_rl import MSSARLConfig, MSSARLForecaster
from .samossa import SAMOSSAForecaster
from .sarimax import SARIMAXForecaster
from .metrics import compute_regression_metrics

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesForecasterConfig:
    sarimax_enabled: bool = True
    garch_enabled: bool = True
    samossa_enabled: bool = True
    mssa_rl_enabled: bool = True
    ensemble_enabled: bool = True
    forecast_horizon: int = 10
    sarimax_kwargs: Dict[str, Any] = field(default_factory=dict)
    garch_kwargs: Dict[str, Any] = field(default_factory=dict)
    samossa_kwargs: Dict[str, Any] = field(default_factory=dict)
    mssa_rl_kwargs: Dict[str, Any] = field(default_factory=dict)
    ensemble_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Phase 7.5: Regime detection for adaptive model selection
    regime_detection_enabled: bool = False
    regime_detection_kwargs: Dict[str, Any] = field(default_factory=dict)


class TimeSeriesForecaster:
    """
    Coordinates all available forecasters and merges their outputs for downstream
    dashboards or automated trading workflows.
    """

    def __init__(
        self,
        config: Optional[TimeSeriesForecasterConfig] = None,
        *,
        forecast_horizon: Optional[int] = None,
        sarimax_config: Optional[Dict[str, Any]] = None,
        garch_config: Optional[Dict[str, Any]] = None,
        samossa_config: Optional[Dict[str, Any]] = None,
        mssa_rl_config: Optional[Dict[str, Any]] = None,
        ensemble_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is None:
            config = self._build_config_from_kwargs(
                forecast_horizon=forecast_horizon,
                sarimax_config=sarimax_config,
                garch_config=garch_config,
                samossa_config=samossa_config,
                mssa_rl_config=mssa_rl_config,
                ensemble_config=ensemble_config,
            )
        elif forecast_horizon is not None:
            config.forecast_horizon = forecast_horizon
        self.config = config
        self._sarimax: Optional[SARIMAXForecaster] = None
        self._garch: Optional[GARCHForecaster] = None
        self._samossa: Optional[SAMOSSAForecaster] = None
        self._mssa: Optional[MSSARLForecaster] = None

        # Phase 7.3 DEBUG: Log ensemble_kwargs to verify candidate_weights are loaded
        if self.config.ensemble_enabled:
            logger.info(
                "Creating EnsembleConfig with kwargs keys: %s, candidate_weights count: %s",
                list(self.config.ensemble_kwargs.keys()),
                len(self.config.ensemble_kwargs.get('candidate_weights', [])),
            )

        self._ensemble_config = (
            EnsembleConfig(**self.config.ensemble_kwargs)
            if self.config.ensemble_enabled
            else EnsembleConfig(enabled=False)
        )

        # Phase 7.5: Initialize regime detector if enabled
        self._regime_detector: Optional['RegimeDetector'] = None
        self._regime_candidate_weights: Dict[str, list[Dict[str, float]]] = {}
        if self.config.regime_detection_enabled:
            from forcester_ts.regime_detector import RegimeDetector, RegimeConfig
            # Filter regime_detection_kwargs to only include RegimeConfig fields.
            # Additional keys are handled locally (e.g., regime_candidate_weights).
            regime_config_fields = {
                'enabled', 'lookback_window', 'vol_threshold_low', 'vol_threshold_high',
                'trend_threshold_weak', 'trend_threshold_strong'
            }
            regime_config_kwargs = {
                k: v for k, v in self.config.regime_detection_kwargs.items()
                if k in regime_config_fields
            }
            regime_config = RegimeConfig(**regime_config_kwargs)
            self._regime_detector = RegimeDetector(regime_config)
            logger.info(
                "[TS_MODEL] REGIME_DETECTION enabled :: lookback=%d, vol_thresholds=(%.2f,%.2f), trend_thresholds=(%.2f,%.2f)",
                regime_config.lookback_window,
                regime_config.vol_threshold_low,
                regime_config.vol_threshold_high,
                regime_config.trend_threshold_weak,
                regime_config.trend_threshold_strong,
            )
            self._regime_candidate_weights = self._coerce_regime_candidate_weights(
                self.config.regime_detection_kwargs.get("regime_candidate_weights")
            )

        self._model_summaries: Dict[str, Dict[str, Any]] = {}
        self._latest_results: Dict[str, Any] = {}
        self._latest_metrics: Dict[str, Dict[str, float]] = {}
        self._model_errors: Dict[str, str] = {}
        self._model_events: list[Dict[str, Any]] = []
        self._series_diagnostics: Dict[str, Any] = {}
        self._instrumentation = ModelInstrumentation()
        self._sarimax_exog_last_row: Optional[pd.Series] = None
        self._sarimax_exog_columns: list[str] = []
        audit_dir = None
        if isinstance(self.config.ensemble_kwargs, dict) and "audit_log_dir" in self.config.ensemble_kwargs:
            audit_dir = self.config.ensemble_kwargs.get("audit_log_dir")
        else:
            audit_dir = os.environ.get("TS_FORECAST_AUDIT_DIR")
        self._audit_dir: Optional[Path] = Path(audit_dir).expanduser() if audit_dir else None
        cfg_path = os.environ.get("TS_FORECAST_MONITOR_CONFIG", "config/forecaster_monitoring.yml")
        self._rmse_monitor_cfg: Dict[str, Any] = self._load_rmse_monitoring_config(Path(cfg_path))

    def _build_sarimax_exogenous(
        self,
        *,
        price_series: pd.Series,
        returns_series: Optional[pd.Series],
    ) -> pd.DataFrame:
        """
        Build a minimal SARIMAX-X feature set from the observed window.

        Feature names are treated as part of the instrumentation contract:
        ["ret_1", "vol_10", "mom_5", "ema_gap_10", "zscore_20"].
        """
        returns = returns_series if returns_series is not None else price_series.pct_change()
        returns = returns.reindex(price_series.index)

        ret_1 = returns.shift(1)
        vol_10 = returns.rolling(10).std()
        mom_5 = price_series.pct_change(5)

        ema_10 = price_series.ewm(span=10, adjust=False).mean()
        ema_gap_10 = (price_series - ema_10) / ema_10.replace(0.0, pd.NA)

        mean_20 = price_series.rolling(20).mean()
        std_20 = price_series.rolling(20).std()
        zscore_20 = (price_series - mean_20) / std_20.replace(0.0, pd.NA)

        exog = pd.DataFrame(
            {
                "ret_1": ret_1,
                "vol_10": vol_10,
                "mom_5": mom_5,
                "ema_gap_10": ema_gap_10,
                "zscore_20": zscore_20,
            },
            index=price_series.index,
        )
        exog = exog.astype(float).replace([pd.NA, float("inf"), float("-inf")], 0.0).fillna(0.0)
        return exog

    def _build_sarimax_forecast_exogenous(self, horizon: int) -> Optional[pd.DataFrame]:
        if not self._sarimax_exog_columns or self._sarimax_exog_last_row is None:
            return None
        horizon = int(horizon)
        if horizon <= 0:
            return None
        last = self._sarimax_exog_last_row.reindex(self._sarimax_exog_columns).astype(float)
        repeated = pd.DataFrame([last.to_dict()] * horizon, columns=self._sarimax_exog_columns)
        return repeated

    def _construct_with_filtered_kwargs(self, cls: type, kwargs: Optional[Dict[str, Any]]) -> Any:
        """
        Instantiate cls with kwargs filtered to only parameters accepted by cls.__init__.

        - If cls.__init__ accepts **kwargs, forward everything.
        - Otherwise, drop unsupported keys to avoid TypeError from config drift.
        """
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            return cls()

        try:
            sig = inspect.signature(cls.__init__)
            accepts_varkw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if accepts_varkw:
                return cls(**kwargs)

            accepted = {
                name
                for name, p in sig.parameters.items()
                if name != "self"
                and p.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
            filtered = {k: v for k, v in kwargs.items() if k in accepted}
            removed = sorted(set(kwargs.keys()) - set(filtered.keys()))
            if removed:
                logger.debug(
                    "Filtered unsupported kwargs for %s: %s",
                    getattr(cls, "__name__", str(cls)),
                    removed,
                )
            return cls(**filtered)
        except Exception:
            return cls(**kwargs)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _record_model_event(
        self,
        model: str,
        phase: str,
        *,
        level: int = logging.INFO,
        **context: Any,
    ) -> None:
        """Track model-level events for downstream diagnostics/logging."""
        payload = {
            "model": model.upper(),
            "phase": phase,
            **context,
        }
        # Keep the most recent 200 events to avoid unbounded growth.
        self._model_events.append(payload)
        if len(self._model_events) > 200:
            self._model_events = self._model_events[-200:]

        context_str = ", ".join(f"{k}={v}" for k, v in context.items()) if context else ""
        logger.log(
            level,
            "[TS_MODEL] %s %s%s",
            model.upper(),
            phase,
            f" :: {context_str}" if context_str else "",
        )

    def _handle_model_failure(self, model: str, phase: str, exc: Exception) -> None:
        """Centralised error logging so failures are captured but do not crash the entire ensemble."""
        err_msg = f"{phase} failed: {exc}"
        self._model_errors[model] = err_msg
        self._record_model_event(model, f"{phase}_failed", level=logging.ERROR, error=str(exc))

    def _ensure_series(self, series: pd.Series) -> pd.Series:
        if not isinstance(series, pd.Series):
            raise TypeError("TimeSeriesForecaster expects a pandas Series")
        if series.isna().all():
            raise ValueError("Series contains only NaN values")
        if series.index.has_duplicates:
            logger.debug("Dropping duplicate index values for forecasting series")
        series = series[~series.index.duplicated(keep="last")]
        series = series.sort_index()
        # Normalise index to naive timestamps for statsmodels compatibility
        normalized_index = pd.DatetimeIndex(series.index).tz_localize(None)
        series = series.copy()
        series.index = normalized_index
        median_seconds: Optional[float] = None
        if len(series.index) >= 3:
            try:
                diffs = pd.Series(series.index).diff().dropna()
                if not diffs.empty:
                    median_seconds = float(diffs.dt.total_seconds().median())
            except Exception:
                median_seconds = None

        try:
            inferred_freq = pd.infer_freq(series.index)
        except Exception:  # pragma: no cover - inference best effort
            inferred_freq = None

        freq_hint: Optional[str] = None
        if inferred_freq:
            freq_hint = normalize_freq(str(inferred_freq))
        elif median_seconds is not None and median_seconds > 0:
            if median_seconds < 3600:
                minutes = max(1, int(round(median_seconds / 60.0)))
                freq_hint = f"{minutes}min"
            elif median_seconds < 86400:
                hours = max(1, int(round(median_seconds / 3600.0)))
                freq_hint = f"{hours}h"
            else:
                freq_hint = "B"

        if freq_hint:
            series.attrs["_pm_freq_hint"] = freq_hint

        # Only coerce to a fixed frequency for daily-or-slower series. Intraday
        # gaps (overnights/weekends) would otherwise be padded into synthetic bars.
        if freq_hint and median_seconds is not None and median_seconds >= 20 * 3600:
            try:
                series = series.asfreq(freq_hint, method="pad")
            except Exception:  # pragma: no cover - defensive; do not fail forecasting
                logger.debug("Unable to enforce frequency %s on series", freq_hint)
        return series

    def _capture_series_diagnostics(self, series: pd.Series) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {
            "length": int(len(series)),
            "start": str(series.index[0]),
            "end": str(series.index[-1]),
        }
        values = series.dropna()
        if adfuller is not None and len(values) > 10:
            try:
                adf_stat, adf_pvalue, _, _, crit, _ = adfuller(values, autolag="AIC")
                diagnostics["adf_stat"] = float(adf_stat)
                diagnostics["adf_pvalue"] = float(adf_pvalue)
                diagnostics["adf_critical"] = {k: float(v) for k, v in crit.items()}
                diagnostics["is_adf_stationary"] = adf_pvalue < 0.05
            except Exception as exc:  # pragma: no cover - diagnostics only
                diagnostics["adf_error"] = str(exc)
        else:
            diagnostics["adf_available"] = False

        if kpss is not None and len(values) > 10:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    if InterpolationWarning is not None:
                        warnings.simplefilter("always", InterpolationWarning)
                    kpss_stat, kpss_pvalue, _, crit = kpss(values, nlags="auto")
                log_warning_records(caught, "TimeSeriesForecaster.kpss")
                diagnostics["kpss_stat"] = float(kpss_stat)
                diagnostics["kpss_pvalue"] = float(kpss_pvalue)
                diagnostics["kpss_critical"] = {k: float(v) for k, v in crit.items()}
                diagnostics["is_kpss_stationary"] = kpss_pvalue > 0.05
            except Exception as exc:  # pragma: no cover - diagnostics only
                diagnostics["kpss_error"] = str(exc)
        else:
            diagnostics["kpss_available"] = diagnostics.get("kpss_error") is None and kpss is not None

        self._record_model_event(
            "series",
            "diagnostics",
            adf_pvalue=diagnostics.get("adf_pvalue"),
            kpss_pvalue=diagnostics.get("kpss_pvalue"),
            length=diagnostics.get("length"),
        )
        return diagnostics

    @staticmethod
    def _enforce_convexity(weights: Dict[str, float]) -> Dict[str, float]:
        if not weights:
            return {}
        positive = {model: max(float(w), 0.0) for model, w in weights.items()}
        total = sum(positive.values())
        if total <= 0.0:
            return {}
        normalized = {model: value / total for model, value in positive.items()}
        normal_sum = sum(normalized.values())
        if abs(normal_sum - 1.0) > 1e-6 and normal_sum > 0:
            normalized = {model: value / normal_sum for model, value in normalized.items()}
        return normalized

    def fit(
        self,
        price_series: pd.Series,
        returns_series: Optional[pd.Series] = None,
    ) -> "TimeSeriesForecaster":
        price_series = self._ensure_series(price_series)
        self._series_diagnostics = self._capture_series_diagnostics(price_series)
        self._model_events.clear()
        self._model_errors.clear()
        self._instrumentation.reset()
        self._instrumentation.set_dataset_metadata(
            length=len(price_series),
            start=str(price_series.index.min()),
            end=str(price_series.index.max()),
            frequency=normalize_freq(str(getattr(price_series.index, "freqstr", None) or price_series.attrs.get("_pm_freq_hint"))),
        )
        if self._series_diagnostics:
            self._instrumentation.record_artifact("series_diagnostics", self._series_diagnostics)
        self._instrumentation.record_series_snapshot("price_series", price_series)

        # Phase 7.5: Detect market regime before fitting models
        self._regime_result: Optional[Dict[str, Any]] = None
        if self._regime_detector:
            try:
                self._record_model_event("regime", "detect_start")
                with self._instrumentation.track("regime", "detect", points=len(price_series)) as meta:
                    self._regime_result = self._regime_detector.detect_regime(
                        price_series,
                        returns_series
                    )
                    meta["regime"] = self._regime_result["regime"]
                    meta["confidence"] = self._regime_result["confidence"]
                    meta["features"] = self._regime_result["features"]
                self._record_model_event(
                    "regime",
                    "detect_complete",
                    regime=self._regime_result["regime"],
                    confidence=self._regime_result["confidence"],
                    features=self._regime_result["features"],
                )
                logger.info(
                    "[TS_MODEL] REGIME detected :: regime=%s, confidence=%.3f, vol=%.3f, trend=%.3f, hurst=%.3f",
                    self._regime_result["regime"],
                    self._regime_result["confidence"],
                    self._regime_result["features"]["realized_volatility"],
                    self._regime_result["features"]["trend_strength"],
                    self._regime_result["features"]["hurst_exponent"],
                )
            except Exception as exc:
                logger.warning(
                    "[TS_MODEL] REGIME detection failed: %s (falling back to static ensemble)",
                    exc,
                    exc_info=True
                )
                self._regime_result = None
                self._handle_model_failure("regime", "detect", exc)

        if self.config.sarimax_enabled:
            self._record_model_event(
                "sarimax",
                "fit_start",
                points=len(price_series),
                first=str(price_series.index.min()),
                last=str(price_series.index.max()),
            )
            try:
                with self._instrumentation.track("sarimax", "fit", points=len(price_series)) as meta:
                    self._sarimax = self._construct_with_filtered_kwargs(
                        SARIMAXForecaster, self.config.sarimax_kwargs
                    )
                    exog = self._build_sarimax_exogenous(
                        price_series=price_series,
                        returns_series=returns_series,
                    )
                    self._sarimax_exog_columns = list(exog.columns)
                    self._sarimax_exog_last_row = exog.iloc[-1] if not exog.empty else None
                    self._instrumentation.record_artifact(
                        "sarimax_exogenous",
                        {"columns": self._sarimax_exog_columns, "row_count": int(len(exog))},
                    )
                    self._sarimax.fit(price_series, exogenous=exog)
                    meta["order"] = getattr(self._sarimax, "best_order", None)
                    meta["seasonal"] = getattr(self._sarimax, "best_seasonal_order", None)
                self._record_model_event(
                    "sarimax",
                    "fit_complete",
                    order=getattr(self._sarimax, "best_order", None),
                    seasonal=getattr(self._sarimax, "best_seasonal_order", None),
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._sarimax = None
                self._sarimax_exog_columns = []
                self._sarimax_exog_last_row = None
                self._handle_model_failure("sarimax", "fit", exc)

        if self.config.samossa_enabled:
            self._record_model_event(
                "samossa",
                "fit_start",
                points=len(price_series),
                first=str(price_series.index.min()),
                last=str(price_series.index.max()),
            )
            try:
                with self._instrumentation.track("samossa", "fit", points=len(price_series)) as meta:
                    self._samossa = self._construct_with_filtered_kwargs(
                        SAMOSSAForecaster, self.config.samossa_kwargs
                    )
                    self._samossa.fit(price_series)
                    summary = self._samossa.get_model_summary()
                    meta["explained_variance"] = summary.get("explained_variance_ratio")
                    meta["components"] = summary.get("n_components")
                    meta["window"] = summary.get("window_length_used")
                self._record_model_event(
                    "samossa",
                    "fit_complete",
                    explained_variance=summary.get("explained_variance_ratio"),
                    components=summary.get("n_components"),
                    window=summary.get("window_length_used"),
                )
            except Exception as exc:
                self._samossa = None
                self._handle_model_failure("samossa", "fit", exc)

        if self.config.mssa_rl_enabled:
            self._record_model_event(
                "mssa_rl",
                "fit_start",
                points=len(price_series),
            )
            try:
                with self._instrumentation.track("mssa_rl", "fit", points=len(price_series)) as meta:
                    mssa_config = MSSARLConfig(**self.config.mssa_rl_kwargs)
                    self._mssa = MSSARLForecaster(config=mssa_config)
                    self._mssa.fit(price_series)
                    diagnostics = self._mssa.get_diagnostics()
                    meta["change_points"] = len(diagnostics.get("change_points", []))
                    meta["rank"] = diagnostics.get("rank")
                self._record_model_event(
                    "mssa_rl",
                    "fit_complete",
                    change_points=len(diagnostics.get("change_points", [])),
                    rank=diagnostics.get("rank"),
                )
            except Exception as exc:
                self._mssa = None
                self._handle_model_failure("mssa_rl", "fit", exc)

        if self.config.garch_enabled:
            if returns_series is None:
                returns_series = price_series.pct_change().dropna()
            else:
                returns_series = returns_series.dropna()
            self._instrumentation.record_series_snapshot("returns_series", returns_series)

            if returns_series.empty:
                self._handle_model_failure(
                    "garch",
                    "fit",
                    ValueError("Returns series required for GARCH is empty after dropna"),
                )
            else:
                self._record_model_event(
                    "garch",
                    "fit_start",
                    points=len(returns_series),
                )
                try:
                    with self._instrumentation.track("garch", "fit", points=len(returns_series)) as meta:
                        self._garch = self._construct_with_filtered_kwargs(
                            GARCHForecaster, self.config.garch_kwargs
                        )
                        self._garch.fit(returns_series)
                        summary = self._garch.get_model_summary()
                        meta["order"] = {"p": self._garch.p, "q": self._garch.q}
                        meta["aic"] = summary.get("aic")
                        meta["bic"] = summary.get("bic")
                    self._record_model_event(
                        "garch",
                        "fit_complete",
                        order={"p": self._garch.p, "q": self._garch.q},
                        aic=summary.get("aic"),
                        bic=summary.get("bic"),
                    )
                except Exception as exc:
                    self._garch = None
                    self._handle_model_failure("garch", "fit", exc)

        self._model_summaries = self.get_component_summaries()
        if self._series_diagnostics:
            self._model_summaries["series_diagnostics"] = self._series_diagnostics
        if self._model_summaries:
            self._instrumentation.record_artifact("component_summaries", self._model_summaries)
        return self

    def forecast(self, *, steps: Optional[int] = None, alpha: float = 0.05) -> Dict[str, Any]:
        horizon = int(steps) if steps is not None else self.config.forecast_horizon
        results: Dict[str, Any] = {"horizon": horizon}
        self._instrumentation.set_dataset_metadata(forecast_horizon=horizon)

        if self._sarimax:
            try:
                self._record_model_event("sarimax", "forecast_start", horizon=horizon)
                with self._instrumentation.track("sarimax", "forecast", horizon=horizon) as meta:
                    exog = self._build_sarimax_forecast_exogenous(horizon)
                    results["sarimax_forecast"] = self._sarimax.forecast(
                        steps=horizon,
                        exogenous=exog,
                        alpha=alpha,
                    )
                    meta["confidence_interval"] = alpha
                self._record_model_event("sarimax", "forecast_complete")
            except Exception as exc:
                results["sarimax_forecast"] = None
                self._handle_model_failure("sarimax", "forecast", exc)
        else:
            results["sarimax_forecast"] = None

        if self._samossa:
            try:
                self._record_model_event("samossa", "forecast_start", horizon=horizon)
                with self._instrumentation.track("samossa", "forecast", horizon=horizon) as meta:
                    samossa_result = self._samossa.forecast(steps=horizon)
                    results["samossa_forecast"] = samossa_result
                    summary = self._samossa.get_model_summary()
                    meta["trend_strength"] = summary.get("trend_strength")
                    meta["seasonal_strength"] = summary.get("seasonal_strength")
                self._record_model_event("samossa", "forecast_complete")
            except Exception as exc:
                results["samossa_forecast"] = None
                self._handle_model_failure("samossa", "forecast", exc)
        else:
            results["samossa_forecast"] = None

        if self._mssa:
            try:
                self._record_model_event("mssa_rl", "forecast_start", horizon=horizon)
                with self._instrumentation.track("mssa_rl", "forecast", horizon=horizon) as meta:
                    mssa_output = self._mssa.forecast(steps=horizon)
                    diagnostics = self._mssa.get_diagnostics()
                    results["mssa_rl_forecast"] = mssa_output
                    results["mssa_rl_diagnostics"] = diagnostics
                    meta["change_points"] = len(diagnostics.get("change_points", []))
                self._record_model_event("mssa_rl", "forecast_complete")
            except Exception as exc:
                results["mssa_rl_forecast"] = None
                results["mssa_rl_diagnostics"] = {}
                self._handle_model_failure("mssa_rl", "forecast", exc)
        else:
            results["mssa_rl_forecast"] = None
            results["mssa_rl_diagnostics"] = {}

        if self._garch:
            try:
                self._record_model_event("garch", "forecast_start", horizon=horizon)
                with self._instrumentation.track("garch", "forecast", horizon=horizon) as meta:
                    garch_result = self._garch.forecast(steps=horizon)
                    results["garch_forecast"] = garch_result
                    volatility_payload = {
                        "volatility": garch_result.get("volatility"),
                        "variance": garch_result.get("variance_forecast"),
                        "model_order": {"p": self._garch.p, "q": self._garch.q},
                        "aic": garch_result.get("aic"),
                        "bic": garch_result.get("bic"),
                    }
                    results["volatility_forecast"] = volatility_payload
                    meta.update(volatility_payload)
                self._record_model_event("garch", "forecast_complete")
            except Exception as exc:
                results["garch_forecast"] = None
                results["volatility_forecast"] = None
                self._handle_model_failure("garch", "forecast", exc)
        else:
            results["garch_forecast"] = None
            results["volatility_forecast"] = None

        self._model_summaries = self.get_component_summaries()
        with self._instrumentation.track("ensemble", "build", horizon=horizon) as ensemble_phase:
            ensemble = self._build_ensemble(results)
            ensemble_phase["models_considered"] = sum(
                1
                for key in ("sarimax_forecast", "samossa_forecast", "garch_forecast", "mssa_rl_forecast")
                if results.get(key) is not None
            )
        if ensemble:
            results["ensemble_forecast"] = ensemble["forecast_bundle"]
            results["ensemble_metadata"] = ensemble["metadata"]
            results["mean_forecast"] = ensemble["forecast_bundle"]
            self._instrumentation.record_artifact(
                "ensemble_weights", ensemble["metadata"].get("weights", {})
            )
            ensemble_meta = ensemble["metadata"]
            self._record_model_event(
                "ensemble",
                "build_complete",
                weights=ensemble_meta.get("weights"),
                confidence=ensemble_meta.get("confidence"),
            )
        else:
            results["ensemble_forecast"] = None
            results["ensemble_metadata"] = {}
            # Prefer SAMOSSA as the default TS baseline when available,
            # falling back to SARIMAX for backward compatibility.
            if results.get("samossa_forecast") is not None:
                results["mean_forecast"] = results.get("samossa_forecast")
            else:
                results["mean_forecast"] = results.get("sarimax_forecast")

        # Phase 7.5: Add regime metadata to results
        if self._regime_result:
            results["regime"] = self._regime_result["regime"]
            results["regime_confidence"] = self._regime_result["confidence"]
            results["regime_features"] = self._regime_result["features"]
            results["regime_recommendations"] = self._regime_result["recommendations"]
        else:
            results["regime"] = "STATIC"  # No regime detection or disabled
            results["regime_confidence"] = None
            results["regime_features"] = None
            results["regime_recommendations"] = None

        self._latest_results = results
        results["model_errors"] = dict(self._model_errors)
        results["model_events"] = list(self._model_events)
        results["instrumentation_report"] = self._instrumentation.export()
        if self._audit_dir:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            audit_path = self._audit_dir / f"forecast_audit_{timestamp}.json"
            self.save_audit_report(audit_path)
        return results

    def get_component_summaries(self) -> Dict[str, Any]:
        summaries = {
            "sarimax": self._sarimax.get_model_summary() if self._sarimax else {},
            "samossa": self._samossa.get_model_summary() if self._samossa else {},
            "garch": self._garch.get_model_summary() if self._garch else {},
            "mssa_rl": self._mssa.get_diagnostics() if self._mssa else {},
            "errors": dict(self._model_errors),
            "events": list(self._model_events),
        }
        canonicalized: Dict[str, Any] = {}
        for key, value in summaries.items():
            canonicalized[canonical_model_key(key)] = value
        return canonicalized

    def get_instrumentation_report(self) -> Dict[str, Any]:
        """Expose the detailed model instrumentation payload."""
        return self._instrumentation.export()

    def save_audit_report(self, output_path: Path) -> None:
        """Persist the instrumentation report to disk for interpretable-AI auditing."""
        self._instrumentation.dump_json(output_path)

    def _build_config_from_kwargs(
        self,
        *,
        forecast_horizon: Optional[int],
        sarimax_config: Optional[Dict[str, Any]],
        garch_config: Optional[Dict[str, Any]],
        samossa_config: Optional[Dict[str, Any]],
        mssa_rl_config: Optional[Dict[str, Any]],
        ensemble_config: Optional[Dict[str, Any]],
    ) -> TimeSeriesForecasterConfig:
        config = TimeSeriesForecasterConfig()
        if forecast_horizon is not None:
            config.forecast_horizon = int(forecast_horizon)

        if sarimax_config is not None:
            config.sarimax_enabled = bool(sarimax_config.get("enabled", True))
            config.sarimax_kwargs = {
                k: v for k, v in sarimax_config.items() if k != "enabled"
            }

        if garch_config is not None:
            config.garch_enabled = bool(garch_config.get("enabled", True))
            config.garch_kwargs = {
                k: v for k, v in garch_config.items() if k != "enabled"
            }

        if samossa_config is not None:
            config.samossa_enabled = bool(samossa_config.get("enabled", True))
            config.samossa_kwargs = {
                k: v for k, v in samossa_config.items() if k != "enabled"
            }

        if mssa_rl_config is not None:
            config.mssa_rl_enabled = bool(mssa_rl_config.get("enabled", True))
            config.mssa_rl_kwargs = {
                k: v for k, v in mssa_rl_config.items() if k != "enabled"
            }

        if ensemble_config is not None:
            config.ensemble_enabled = bool(ensemble_config.get("enabled", True))
            config.ensemble_kwargs = {
                k: v for k, v in ensemble_config.items() if k != "enabled"
            }

        return config

    @staticmethod
    def _coerce_regime_candidate_weights(value: Any) -> Dict[str, list[Dict[str, float]]]:
        """
        Normalise regime-specific candidate weights from config.

        Expected shape:
            {
                "LIQUID_RANGEBOUND": [{"sarimax": 0.6, "samossa": 0.4}, ...],
                "CRISIS": [{"sarimax": 1.0}, ...],
            }
        """
        if not isinstance(value, dict):
            return {}

        coerced: Dict[str, list[Dict[str, float]]] = {}
        for regime, raw_candidates in value.items():
            if regime is None:
                continue
            regime_key = str(regime)

            if isinstance(raw_candidates, dict):
                candidates_list = [raw_candidates]
            elif isinstance(raw_candidates, list):
                candidates_list = raw_candidates
            else:
                continue

            cleaned_candidates: list[Dict[str, float]] = []
            for candidate in candidates_list:
                if not isinstance(candidate, dict):
                    continue
                cleaned: Dict[str, float] = {}
                for model_key, weight in candidate.items():
                    canon = canonical_model_key(str(model_key))
                    try:
                        weight_val = float(weight)
                    except Exception:
                        continue
                    if weight_val <= 0.0:
                        continue
                    cleaned[canon] = weight_val
                if cleaned:
                    cleaned_candidates.append(cleaned)

            if cleaned_candidates:
                coerced[regime_key] = cleaned_candidates

        return coerced

    def _build_ensemble(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._ensemble_config.enabled:
            return None

        original_candidates = self._ensemble_config.candidate_weights
        try:
            regime_name = None
            if self._regime_result:
                regime_name = self._regime_result.get("regime")

            # Phase 7.6: If regime-specific candidate weights exist, use them
            # instead of the Phase 7.5 reorder heuristic.
            override_candidates = None
            if regime_name and self._regime_candidate_weights:
                override_candidates = self._regime_candidate_weights.get(str(regime_name))

            if override_candidates:
                logger.info(
                    "[TS_MODEL] REGIME candidate_override :: regime=%s, override_top=%s",
                    regime_name,
                    override_candidates[0] if override_candidates else None,
                )
                self._ensemble_config.candidate_weights = override_candidates
            elif self._regime_result and self._regime_detector and original_candidates:
                # Phase 7.5: Reorder candidates based on regime detection (if enabled)
                try:
                    preferred_candidates = self._regime_detector.get_preferred_candidates(
                        self._regime_result,
                        original_candidates
                    )
                    logger.info(
                        "[TS_MODEL] REGIME candidate_reorder :: regime=%s, original_top=%s, preferred_top=%s",
                        self._regime_result["regime"],
                        original_candidates[0] if original_candidates else None,
                        preferred_candidates[0] if preferred_candidates else None,
                    )
                    # Temporarily use regime-preferred candidates
                    self._ensemble_config.candidate_weights = preferred_candidates
                except Exception as exc:
                    logger.warning(
                        "[TS_MODEL] REGIME candidate reordering failed: %s (using original order)",
                        exc
                    )

            coordinator = EnsembleCoordinator(self._ensemble_config)
            confidence = derive_model_confidence(self._model_summaries)
            weights, score = coordinator.select_weights(confidence)
            weights = self._enforce_convexity(weights)
        finally:
            # Restore original candidates after ensemble build
            self._ensemble_config.candidate_weights = original_candidates
        if not weights:
            return None

        forecasts = {
            "sarimax": self._extract_series(results.get("sarimax_forecast")),
            "garch": self._extract_series(results.get("garch_forecast")),
            "samossa": self._extract_series(results.get("samossa_forecast")),
            "mssa_rl": self._extract_series(results.get("mssa_rl_forecast")),
        }
        lowers = {
            "sarimax": self._extract_series(results.get("sarimax_forecast"), "lower_ci"),
            "garch": self._extract_series(results.get("garch_forecast"), "lower_ci"),
            "samossa": self._extract_series(results.get("samossa_forecast"), "lower_ci"),
            "mssa_rl": self._extract_series(results.get("mssa_rl_forecast"), "lower_ci"),
        }
        uppers = {
            "sarimax": self._extract_series(results.get("sarimax_forecast"), "upper_ci"),
            "garch": self._extract_series(results.get("garch_forecast"), "upper_ci"),
            "samossa": self._extract_series(results.get("samossa_forecast"), "upper_ci"),
            "mssa_rl": self._extract_series(results.get("mssa_rl_forecast"), "upper_ci"),
        }

        blended = coordinator.blend_forecasts(forecasts, lowers, uppers)
        if not blended:
            return None

        primary_model = None
        if weights:
            try:
                primary_model = max(weights.items(), key=lambda item: item[1])[0].upper()
            except Exception:  # pragma: no cover - defensive
                primary_model = None

        forecast_bundle = {
            "forecast": blended["forecast"],
            "lower_ci": blended.get("lower_ci"),
            "upper_ci": blended.get("upper_ci"),
            "weights": weights,
            "confidence": confidence,
            "selection_score": score,
            "primary_model": primary_model,
        }
        metadata = {
            "weights": weights,
            "confidence": confidence,
            "selection_score": score,
            "primary_model": primary_model,
        }
        return {"forecast_bundle": forecast_bundle, "metadata": metadata}

    @staticmethod
    def _extract_series(forecast_payload: Optional[Dict[str, Any]], key: str = "forecast") -> Optional[pd.Series]:
        if not isinstance(forecast_payload, dict):
            return None
        series = forecast_payload.get(key)
        return series if isinstance(series, pd.Series) else None

    @staticmethod
    def _rmse_from_metrics(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
        if not isinstance(metrics, dict):
            return None
        val = metrics.get("rmse")
        try:
            return float(val) if isinstance(val, (int, float)) else None
        except Exception:
            return None

    @staticmethod
    def _best_single_from_metrics(
        metrics_map: Dict[str, Dict[str, Any]]
    ) -> tuple[Optional[str], Optional[float]]:
        best_model: Optional[str] = None
        best_rmse: Optional[float] = None
        for name in ("sarimax", "samossa", "mssa_rl"):
            rmse_val = TimeSeriesForecaster._rmse_from_metrics(metrics_map.get(name))
            if rmse_val is None:
                continue
            if best_rmse is None or rmse_val < best_rmse:
                best_rmse = rmse_val
                best_model = name
        return best_model, best_rmse

    @staticmethod
    def _load_rmse_monitoring_config(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            import yaml  # type: ignore
        except Exception:
            return {}
        try:
            raw = yaml.safe_load(path.read_text()) or {}
            fm = raw.get("forecaster_monitoring") or {}
            return fm.get("regression_metrics") or {}
        except Exception:
            return {}

    def _audit_history_stats(
        self,
        current_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """Summarize recent audit performance for holdout-based model selection."""
        ratios: list[float] = []
        cfg = self._rmse_monitor_cfg or {}
        max_ratio = float(cfg.get("max_rmse_ratio_vs_baseline", 1.1))
        min_lift_rmse_ratio = float(cfg.get("min_lift_rmse_ratio", 0.0) or 0.0)

        def _append_from_metrics(metrics_map: Dict[str, Dict[str, Any]]) -> None:
            ensemble_rmse = self._rmse_from_metrics(metrics_map.get("ensemble"))
            best_model, best_rmse = self._best_single_from_metrics(metrics_map)
            if ensemble_rmse is None or best_rmse is None or best_rmse <= 0:
                return
            ratios.append(ensemble_rmse / best_rmse)

        def _extract_from_audit(path: Path) -> None:
            try:
                audit = json.loads(path.read_text())
            except Exception:
                return
            artifacts = audit.get("artifacts") or {}
            eval_metrics = artifacts.get("evaluation_metrics") or {}
            if not isinstance(eval_metrics, dict):
                return
            metrics_map = {
                key: val for key, val in eval_metrics.items() if isinstance(val, dict)
            }
            _append_from_metrics(metrics_map)

        if self._audit_dir and self._audit_dir.exists():
            files = sorted(
                self._audit_dir.glob("forecast_audit_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            seen_windows: set[tuple[Any, ...]] = set()
            for path in files[:limit]:
                try:
                    audit = json.loads(path.read_text())
                except Exception:
                    continue
                dataset = audit.get("dataset") or {}
                window = (
                    dataset.get("start"),
                    dataset.get("end"),
                    dataset.get("length"),
                    dataset.get("forecast_horizon"),
                )
                if window in seen_windows:
                    continue
                seen_windows.add(window)
                _extract_from_audit(path)

        if current_metrics:
            _append_from_metrics(current_metrics)

        effective_n = len(ratios)
        violation_rate = (
            sum(1 for ratio in ratios if ratio > max_ratio) / effective_n
            if effective_n
            else 0.0
        )
        lift_threshold = 1.0 - min_lift_rmse_ratio
        lift_fraction = (
            sum(1 for ratio in ratios if ratio < lift_threshold) / effective_n
            if effective_n
            else 0.0
        )
        return {
            "effective_n": effective_n,
            "violation_rate": violation_rate,
            "lift_fraction": lift_fraction,
            "ratios": ratios,
        }

    def evaluate(self, actual_series: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Compute regression metrics for the latest forecasts using realised prices.

        Args:
            actual_series: Series containing realised prices for the forecast horizon.

        Returns:
            Mapping of model name -> metrics dict.
        """

        if not self._latest_results:
            raise RuntimeError("Call forecast() before evaluate().")

        actual = self._ensure_series(actual_series)
        self._instrumentation.record_series_snapshot("actual_realized", actual)
        metrics_map: Dict[str, Dict[str, float]] = {}

        def _evaluate_model(name: str, payload: Optional[Dict[str, Any]]) -> None:
            forecast_series = self._extract_series(payload)
            if forecast_series is None:
                return
            metrics = compute_regression_metrics(actual, forecast_series)
            if metrics:
                metrics_map[name] = metrics

        _evaluate_model("sarimax", self._latest_results.get("sarimax_forecast"))
        _evaluate_model("garch", self._latest_results.get("garch_forecast"))  # Phase 7.3: Add GARCH to metrics
        _evaluate_model("samossa", self._latest_results.get("samossa_forecast"))
        _evaluate_model("mssa_rl", self._latest_results.get("mssa_rl_forecast"))

        ensemble_payload = self._latest_results.get("ensemble_forecast")
        if isinstance(ensemble_payload, dict):
            _evaluate_model("ensemble", ensemble_payload)

        rmse_cfg = self._rmse_monitor_cfg or {}
        history_stats = self._audit_history_stats(metrics_map)

        def _maybe_reweight_ensemble_from_holdout() -> None:
            metadata = self._latest_results.get("ensemble_metadata")
            if not isinstance(metadata, dict):
                return
            if not any(model in metrics_map for model in ("sarimax", "samossa", "mssa_rl")):
                return
            required = max(
                int(rmse_cfg.get("holding_period_audits", 0) or 0),
                int(rmse_cfg.get("min_effective_audits", 0) or 0),
                0,
            )
            if required > 0 and history_stats.get("effective_n", 0) < required:
                self._record_model_event(
                    "ensemble",
                    "holdout_wait",
                    effective_audits=history_stats.get("effective_n", 0),
                    required_audits=required,
                )
                return

            rmse_by_model: Dict[str, float] = {}
            for model in ("sarimax", "garch", "samossa", "mssa_rl"):
                rmse_val = (metrics_map.get(model) or {}).get("rmse")
                if isinstance(rmse_val, (int, float)) and float(rmse_val) >= 0:
                    rmse_by_model[model] = float(rmse_val)
            if not rmse_by_model:
                return

            best_model, best_rmse = min(rmse_by_model.items(), key=lambda item: item[1])
            # Only blend models that are "close enough" to the best performer on
            # the evaluated window. Otherwise, pick the best model outright so
            # the ensemble cannot be worse than the best available forecaster.
            #
            # This implements the policy "model with least forecasting error is
            # weightier" in a way that's robust for production gates.
            relative_band = 0.05
            eligible = {
                model: rmse
                for model, rmse in rmse_by_model.items()
                if rmse <= best_rmse * (1.0 + relative_band)
            }
            if not eligible:
                eligible = {best_model: best_rmse}

            eps = 1e-12
            raw_weights = {model: 1.0 / (rmse + eps) for model, rmse in eligible.items()}
            total = sum(raw_weights.values())
            if total <= 0:
                return
            weights = {model: val / total for model, val in raw_weights.items()}

            # Rebuild the ensemble forecast using these RMSE-derived weights so
            # the lowest error models are weightier on the evaluated window.
            model_series: Dict[str, pd.Series] = {}
            lower_bounds: Dict[str, Optional[pd.Series]] = {}
            upper_bounds: Dict[str, Optional[pd.Series]] = {}
            for model in weights.keys():
                payload = self._latest_results.get(f"{model}_forecast")
                if not isinstance(payload, dict):
                    continue
                forecast_series = self._extract_series(payload)
                if forecast_series is None:
                    continue
                model_series[model] = forecast_series
                lower_bounds[model] = payload.get("lower_ci") if isinstance(payload.get("lower_ci"), pd.Series) else None
                upper_bounds[model] = payload.get("upper_ci") if isinstance(payload.get("upper_ci"), pd.Series) else None

            if not model_series:
                return

            coordinator = EnsembleCoordinator(self._ensemble_config)
            coordinator.selected_weights = dict(weights)
            blended = coordinator.blend_forecasts(model_series, lower_bounds, upper_bounds)
            if not blended:
                return

            self._latest_results["ensemble_forecast"] = blended
            metadata["weights"] = dict(weights)
            metadata["confidence"] = dict(weights)
            self._instrumentation.record_artifact("ensemble_weights", dict(weights))
            self._record_model_event(
                "ensemble",
                "reweighted_from_holdout",
                weights=metadata.get("weights"),
            )

            # Recompute ensemble metrics on the reweighted forecast bundle.
            ensemble_series = blended.get("forecast")
            if isinstance(ensemble_series, pd.Series):
                new_metrics = compute_regression_metrics(actual, ensemble_series)
                if new_metrics:
                    metrics_map["ensemble"] = new_metrics

        _maybe_reweight_ensemble_from_holdout()
        self._enforce_ensemble_safety(metrics_map, history_stats, rmse_cfg)

        self._latest_metrics = metrics_map
        self._latest_results.setdefault("regression_metrics", {}).update(metrics_map)
        # Surface metrics in summaries/metadata for downstream consumers (dashboard/DB)
        for model, metrics in metrics_map.items():
            summary = self._model_summaries.setdefault(model, {})
            summary["regression_metrics"] = metrics
        ensemble_meta = self._latest_results.setdefault("ensemble_metadata", {})
        if metrics_map.get("ensemble"):
            ensemble_meta["regression_metrics"] = metrics_map["ensemble"]
        self._instrumentation.record_artifact("evaluation_metrics", metrics_map)
        for model, metrics in metrics_map.items():
            self._instrumentation.record_model_metrics(
                model,
                metrics,
                horizon=self.config.forecast_horizon,
                n_observations=metrics.get("n_observations"),
            )

        for model, metrics in metrics_map.items():
            summary = self._model_summaries.setdefault(model, {})
            summary["regression_metrics"] = metrics

        metadata = self._latest_results.get("ensemble_metadata")
        if metadata and "ensemble" in metrics_map:
            metadata["regression_metrics"] = metrics_map["ensemble"]

        if self._audit_dir:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            audit_path = self._audit_dir / f"forecast_audit_{timestamp}.json"
            self.save_audit_report(audit_path)

        return metrics_map

    def _enforce_ensemble_safety(
        self,
        metrics_map: Dict[str, Dict[str, Any]],
        history_stats: Dict[str, Any],
        rmse_cfg: Dict[str, Any],
    ) -> None:
        metadata = self._latest_results.setdefault("ensemble_metadata", {})
        ensemble_rmse = self._rmse_from_metrics(metrics_map.get("ensemble"))
        best_model, best_rmse = self._best_single_from_metrics(metrics_map)
        if ensemble_rmse is None or best_rmse is None or best_rmse <= 0:
            metadata.setdefault("ensemble_status", "NO_ENSEMBLE_EVIDENCE")
            return

        max_ratio = float(rmse_cfg.get("max_rmse_ratio_vs_baseline", 1.1))
        promotion_margin = float(rmse_cfg.get("promotion_margin", 0.0) or 0.0)
        min_lift_fraction = float(rmse_cfg.get("min_lift_fraction", 0.0) or 0.0)
        min_effective = int(rmse_cfg.get("min_effective_audits", 0) or 0)
        holding_period = int(rmse_cfg.get("holding_period_audits", 0) or 0)
        disable_if_no_lift = bool(rmse_cfg.get("disable_ensemble_if_no_lift", False))
        effective_n = history_stats.get("effective_n", 0) or 0
        lift_fraction = history_stats.get("lift_fraction", 0.0) or 0.0
        required = max(min_effective, holding_period, 0)

        decision = "KEEP"
        reason = "ensemble within tolerance"
        ratio = ensemble_rmse / best_rmse

        metadata["rmse_ratio"] = ratio
        metadata["ensemble_rmse"] = ensemble_rmse
        metadata["best_model_rmse"] = best_rmse
        metadata["best_model"] = best_model

        if ratio > max_ratio:
            decision = "DISABLE_DEFAULT"
            reason = f"rmse regression (ratio={ratio:.3f} > {max_ratio:.3f})"
        elif disable_if_no_lift and required > 0 and effective_n >= required and lift_fraction < min_lift_fraction:
            decision = "DISABLE_DEFAULT"
            reason = (
                f"insufficient lift over baseline (lift_fraction={lift_fraction:.3f} "
                f"< {min_lift_fraction:.3f})"
            )
        elif promotion_margin > 0 and ratio > (1.0 - promotion_margin):
            decision = "RESEARCH_ONLY"
            reason = f"no margin lift (required >= {promotion_margin:.3f})"

        metadata["ensemble_status"] = decision
        metadata["ensemble_decision_reason"] = reason
        if decision != "KEEP":
            metadata["default_model"] = (best_model or "").upper() or metadata.get("primary_model")
            self._latest_results["default_model"] = metadata.get("default_model")
            self._record_model_event(
                "ensemble",
                "policy_decision",
                status=decision,
                reason=reason,
                ratio=ratio,
                effective_audits=effective_n,
                required_audits=required,
            )
        self._instrumentation.record_artifact(
            "ensemble_policy_decision",
            {
                "status": decision,
                "reason": reason,
                "ratio": ratio,
                "effective_audits": effective_n,
                "required_audits": required,
            },
        )
