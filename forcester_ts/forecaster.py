"""
Unified time-series forecaster orchestrating SARIMAX, GARCH, SAMOSSA,
and the new MSSA-RL variant in parallel.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

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

from .ensemble import EnsembleConfig, EnsembleCoordinator, derive_model_confidence
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
        self._ensemble_config = (
            EnsembleConfig(**self.config.ensemble_kwargs)
            if self.config.ensemble_enabled
            else EnsembleConfig(enabled=False)
        )
        self._model_summaries: Dict[str, Dict[str, Any]] = {}
        self._latest_results: Dict[str, Any] = {}
        self._latest_metrics: Dict[str, Dict[str, float]] = {}
        self._model_errors: Dict[str, str] = {}
        self._model_events: list[Dict[str, Any]] = []
        self._series_diagnostics: Dict[str, Any] = {}
        self._instrumentation = ModelInstrumentation()
        audit_dir = self.config.ensemble_kwargs.get("audit_log_dir") if isinstance(self.config.ensemble_kwargs, dict) else None
        if not audit_dir:
            audit_dir = os.environ.get("TS_FORECAST_AUDIT_DIR")
        self._audit_dir: Optional[Path] = Path(audit_dir).expanduser() if audit_dir else None

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
        try:
            inferred_freq = pd.infer_freq(series.index)
        except Exception:  # pragma: no cover - inference best effort
            inferred_freq = None
        freq_to_use = inferred_freq or ("B" if len(series) > 3 else None)
        if freq_to_use:
            series.attrs["_pm_freq_hint"] = freq_to_use
            try:
                # Enforce a concrete DatetimeIndex frequency to keep statsmodels quiet.
                series = series.asfreq(freq_to_use, method="pad")
            except Exception:  # pragma: no cover - defensive; do not fail forecasting
                logger.debug("Unable to enforce frequency %s on series", freq_to_use)
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
            frequency=str(getattr(price_series.index, "freqstr", None) or price_series.attrs.get("_pm_freq_hint")),
        )
        if self._series_diagnostics:
            self._instrumentation.record_artifact("series_diagnostics", self._series_diagnostics)
        self._instrumentation.record_series_snapshot("price_series", price_series)

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
                    self._sarimax = SARIMAXForecaster(**self.config.sarimax_kwargs)
                    self._sarimax.fit(price_series)
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
                    self._samossa = SAMOSSAForecaster(**self.config.samossa_kwargs)
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
                        self._garch = GARCHForecaster(**self.config.garch_kwargs)
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
                    results["sarimax_forecast"] = self._sarimax.forecast(steps=horizon, alpha=alpha)
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
        return {
            "sarimax": self._sarimax.get_model_summary() if self._sarimax else {},
            "samossa": self._samossa.get_model_summary() if self._samossa else {},
            "garch": self._garch.get_model_summary() if self._garch else {},
            "mssa_rl": self._mssa.get_diagnostics() if self._mssa else {},
            "errors": dict(self._model_errors),
            "events": list(self._model_events),
        }

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

    def _build_ensemble(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._ensemble_config.enabled:
            return None

        coordinator = EnsembleCoordinator(self._ensemble_config)
        confidence = derive_model_confidence(self._model_summaries)
        weights, score = coordinator.select_weights(confidence)
        weights = self._enforce_convexity(weights)
        if not weights:
            return None

        forecasts = {
            "sarimax": self._extract_series(results.get("sarimax_forecast")),
            "samossa": self._extract_series(results.get("samossa_forecast")),
            "mssa_rl": self._extract_series(results.get("mssa_rl_forecast")),
        }
        lowers = {
            "sarimax": self._extract_series(results.get("sarimax_forecast"), "lower_ci"),
            "samossa": self._extract_series(results.get("samossa_forecast"), "lower_ci"),
            "mssa_rl": self._extract_series(results.get("mssa_rl_forecast"), "lower_ci"),
        }
        uppers = {
            "sarimax": self._extract_series(results.get("sarimax_forecast"), "upper_ci"),
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
        _evaluate_model("samossa", self._latest_results.get("samossa_forecast"))
        _evaluate_model("mssa_rl", self._latest_results.get("mssa_rl_forecast"))

        ensemble_payload = self._latest_results.get("ensemble_forecast")
        if isinstance(ensemble_payload, dict):
            _evaluate_model("ensemble", ensemble_payload)

        self._latest_metrics = metrics_map
        self._latest_results.setdefault("regression_metrics", {}).update(metrics_map)
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

        return metrics_map
