"""
Unified time-series forecaster orchestrating SARIMAX, GARCH, SAMOSSA,
and the new MSSA-RL variant in parallel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd
from pandas.tseries.frequencies import to_offset

from .ensemble import EnsembleConfig, EnsembleCoordinator, derive_model_confidence
from .garch import GARCHForecaster
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
        self._ensemble_config = EnsembleConfig(**self.config.ensemble_kwargs) if self.config.ensemble_enabled else EnsembleConfig(enabled=False)
        self._model_summaries: Dict[str, Dict[str, Any]] = {}
        self._latest_results: Dict[str, Any] = {}
        self._latest_metrics: Dict[str, Dict[str, float]] = {}
        self._model_errors: Dict[str, str] = {}
        self._model_events: list[Dict[str, Any]] = []

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
        inferred_freq = pd.infer_freq(series.index)
        if inferred_freq:
            try:
                series = series.asfreq(inferred_freq)
            except ValueError:
                logger.debug("Unable to reindex series to inferred freq %s", inferred_freq)
        else:
            # Fallback: coerce to a business-day offset so statsmodels sees a freq
            fallback_offset = to_offset("B")
            try:
                series.index.freq = fallback_offset
            except ValueError:
                logger.debug("Unable to assign fallback freq %s to index", fallback_offset)
        return series

    def fit(
        self,
        price_series: pd.Series,
        returns_series: Optional[pd.Series] = None,
    ) -> "TimeSeriesForecaster":
        price_series = self._ensure_series(price_series)
        self._model_events.clear()
        self._model_errors.clear()

        if self.config.sarimax_enabled:
            self._record_model_event(
                "sarimax",
                "fit_start",
                points=len(price_series),
                first=str(price_series.index.min()),
                last=str(price_series.index.max()),
            )
            try:
                self._sarimax = SARIMAXForecaster(**self.config.sarimax_kwargs)
                self._sarimax.fit(price_series)
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
                self._samossa = SAMOSSAForecaster(**self.config.samossa_kwargs)
                self._samossa.fit(price_series)
                summary = self._samossa.get_model_summary()
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
                mssa_config = MSSARLConfig(**self.config.mssa_rl_kwargs)
                self._mssa = MSSARLForecaster(config=mssa_config)
                self._mssa.fit(price_series)
                diagnostics = self._mssa.get_diagnostics()
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
                    self._garch = GARCHForecaster(**self.config.garch_kwargs)
                    self._garch.fit(returns_series)
                    summary = self._garch.get_model_summary()
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
        return self

    def forecast(self, *, steps: Optional[int] = None, alpha: float = 0.05) -> Dict[str, Any]:
        horizon = int(steps) if steps is not None else self.config.forecast_horizon
        results: Dict[str, Any] = {"horizon": horizon}

        if self._sarimax:
            try:
                self._record_model_event("sarimax", "forecast_start", horizon=horizon)
                results["sarimax_forecast"] = self._sarimax.forecast(steps=horizon, alpha=alpha)
                self._record_model_event("sarimax", "forecast_complete")
            except Exception as exc:
                results["sarimax_forecast"] = None
                self._handle_model_failure("sarimax", "forecast", exc)
        else:
            results["sarimax_forecast"] = None

        if self._samossa:
            try:
                self._record_model_event("samossa", "forecast_start", horizon=horizon)
                results["samossa_forecast"] = self._samossa.forecast(steps=horizon)
                self._record_model_event("samossa", "forecast_complete")
            except Exception as exc:
                results["samossa_forecast"] = None
                self._handle_model_failure("samossa", "forecast", exc)
        else:
            results["samossa_forecast"] = None

        if self._mssa:
            try:
                self._record_model_event("mssa_rl", "forecast_start", horizon=horizon)
                mssa_output = self._mssa.forecast(steps=horizon)
                results["mssa_rl_forecast"] = mssa_output
                results["mssa_rl_diagnostics"] = self._mssa.get_diagnostics()
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
                garch_result = self._garch.forecast(steps=horizon)
                results["garch_forecast"] = garch_result
                results["volatility_forecast"] = {
                    "volatility": garch_result.get("volatility"),
                    "variance": garch_result.get("variance_forecast"),
                    "model_order": {"p": self._garch.p, "q": self._garch.q},
                    "aic": garch_result.get("aic"),
                    "bic": garch_result.get("bic"),
                }
                self._record_model_event("garch", "forecast_complete")
            except Exception as exc:
                results["garch_forecast"] = None
                results["volatility_forecast"] = None
                self._handle_model_failure("garch", "forecast", exc)
        else:
            results["garch_forecast"] = None
            results["volatility_forecast"] = None

        self._model_summaries = self.get_component_summaries()
        ensemble = self._build_ensemble(results)
        if ensemble:
            results["ensemble_forecast"] = ensemble["forecast_bundle"]
            results["ensemble_metadata"] = ensemble["metadata"]
            results["mean_forecast"] = ensemble["forecast_bundle"]
            self._record_model_event(
                "ensemble",
                "build_complete",
                weights=ensemble["metadata"].get("weights"),
                confidence=ensemble["metadata"].get("confidence"),
            )
        else:
            results["ensemble_forecast"] = None
            results["ensemble_metadata"] = {}
            results["mean_forecast"] = results.get("sarimax_forecast")

        self._latest_results = results
        results["model_errors"] = dict(self._model_errors)
        results["model_events"] = list(self._model_events)
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

        forecast_bundle = {
            "forecast": blended["forecast"],
            "lower_ci": blended.get("lower_ci"),
            "upper_ci": blended.get("upper_ci"),
            "weights": weights,
            "confidence": confidence,
            "selection_score": score,
        }
        metadata = {
            "weights": weights,
            "confidence": confidence,
            "selection_score": score,
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

        for model, metrics in metrics_map.items():
            summary = self._model_summaries.setdefault(model, {})
            summary["regression_metrics"] = metrics

        metadata = self._latest_results.get("ensemble_metadata")
        if metadata and "ensemble" in metrics_map:
            metadata["regression_metrics"] = metrics_map["ensemble"]

        return metrics_map
