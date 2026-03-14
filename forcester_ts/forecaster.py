"""
Unified time-series forecaster orchestrating SARIMAX, GARCH, SAMOSSA,
and the new MSSA-RL variant in parallel.
"""

from __future__ import annotations

import inspect
import hashlib
import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

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
from utils.evidence_io import atomic_write_json, build_manifest_entry, upsert_jsonl_record

logger = logging.getLogger(__name__)


def _series_to_plain_list(s: "Optional[pd.Series]") -> "Optional[list[float]]":
    """Convert a pd.Series to a plain list[float] for JSON-safe artifact emission.

    Agent B's _parse_optional_float_list() requires a plain list; the
    instrumentation's _make_json_safe() converts Series to {"index":â€¦,"values":â€¦}
    which the parser rejects.  This helper is called before record_artifact().

    Returns None when input is None or conversion fails â€” never returns [].
    """
    if s is None:
        return None
    try:
        result = [float(v) for v in s if pd.notna(v)]
        return result if result else None
    except Exception:
        return None


@dataclass
class TimeSeriesForecasterConfig:
    sarimax_enabled: bool = False  # Off by default; slow SARIMAX grid search
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

    # Phase 7.16: Auto-learning order cache + snapshot store
    order_learning_config: Dict[str, Any] = field(default_factory=dict)
    order_learning_db_path: str = ""  # path to portfolio_maximizer.db
    monte_carlo_config: Dict[str, Any] = field(default_factory=dict)

    # EXP-R5-001: Residual ensemble research experiment (Agent A).
    # OFF by default â€” enabling this flag does NOT change any gate, default
    # model, or live strategy.  It only adds the `residual_experiment` artifact
    # to forecast() results for audit/research pipelines.
    residual_experiment_enabled: bool = False


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
        monte_carlo_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is None:
            config = self._build_config_from_kwargs(
                forecast_horizon=forecast_horizon,
                sarimax_config=sarimax_config,
                garch_config=garch_config,
                samossa_config=samossa_config,
                mssa_rl_config=mssa_rl_config,
                ensemble_config=ensemble_config,
                monte_carlo_config=monte_carlo_config,
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

        # Strip forecaster-routing keys (e.g. audit_log_dir) that are not
        # EnsembleConfig dataclass fields.  Callers may embed routing hints
        # inside ensemble_kwargs for convenience; the EnsembleConfig boundary
        # must not receive unknown kwargs or a TypeError silently kills the
        # whole forecaster while the pipeline reports success with 0 forecasts.
        import dataclasses as _dc
        _ensemble_valid_fields = {f.name for f in _dc.fields(EnsembleConfig)}
        _ensemble_kw = {
            k: v for k, v in self.config.ensemble_kwargs.items()
            if k in _ensemble_valid_fields
        }
        self._ensemble_config = (
            EnsembleConfig(**_ensemble_kw)
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

        # Phase 7.16: Construct OrderLearner + ModelSnapshotStore if enabled.
        self._order_learner: Optional[Any] = None
        self._snapshot_store: Optional[Any] = None
        ol_cfg = self.config.order_learning_config or {}
        if ol_cfg.get("enabled", False):
            _db_path = self.config.order_learning_db_path or "data/portfolio_maximizer.db"
            try:
                from forcester_ts.order_learner import OrderLearner
                from forcester_ts.model_snapshot_store import ModelSnapshotStore
                self._order_learner = OrderLearner(db_path=_db_path, config=ol_cfg)
                self._snapshot_store = ModelSnapshotStore()
                logger.info(
                    "[TS_MODEL] OrderLearner + ModelSnapshotStore initialised (db=%s)", _db_path
                )
            except Exception as _ole:
                logger.warning(
                    "[TS_MODEL] OrderLearner/SnapshotStore init failed: %s", _ole
                )

        self._model_summaries: Dict[str, Dict[str, Any]] = {}
        self._latest_results: Dict[str, Any] = {}
        self._latest_metrics: Dict[str, Dict[str, float]] = {}
        self._model_errors: Dict[str, str] = {}
        self._model_events: list[Dict[str, Any]] = []
        self._regime_result: Optional[Dict[str, Any]] = None

        # EXP-R5-001: residual model placeholder â€” Phase 2 will wire a fitted
        # ResidualModel here.  None = experiment runs with no residual correction.
        self._residual_model: Optional[Any] = None
        self._residual_model_oos_n: Optional[int] = None  # RC3 observability
        self._residual_model_reason_code: Optional[str] = None
        self._residual_model_skip_reason: Optional[str] = None
        self._series_diagnostics: Dict[str, Any] = {}
        self._instrumentation = ModelInstrumentation()
        self._sarimax_exog_last_row: Optional[pd.Series] = None
        self._sarimax_exog_columns: list[str] = []
        self._macro_context: Optional[pd.DataFrame] = None  # Signal Quality A
        self._last_price: Optional[float] = None
        self._last_timestamp: Optional[pd.Timestamp] = None
        self._series_freq_hint: Optional[str] = None
        audit_dir = None
        if isinstance(self.config.ensemble_kwargs, dict) and "audit_log_dir" in self.config.ensemble_kwargs:
            audit_dir = self.config.ensemble_kwargs.get("audit_log_dir")
        else:
            audit_dir = os.environ.get("TS_FORECAST_AUDIT_DIR")
        if audit_dir is None:
            # Production evidence must route deterministically when the production subdir
            # already exists; only explicit research routing should bypass it.
            prod_dir = Path("logs/forecast_audits/production")
            self._audit_dir = prod_dir if prod_dir.exists() else Path("logs/forecast_audits")
        else:
            audit_text = str(audit_dir).strip()
            if audit_text.lower() in {"", "0", "off", "none", "false"}:
                self._audit_dir = None
            else:
                self._audit_dir = Path(audit_text).expanduser()
        cfg_path = os.environ.get("TS_FORECAST_MONITOR_CONFIG", "config/forecaster_monitoring.yml")
        self._rmse_monitor_cfg: Dict[str, Any] = self._load_rmse_monitoring_config(Path(cfg_path))

    # Signal Quality A: macro columns sourced from macro_context
    _MACRO_EXOG_COLUMNS = ("vix_level", "yield_spread_10y_2y", "sector_momentum_5d")
    _DEFAULT_MC_PATHS = 1000
    _GENERIC_TICKER_NAMES = {
        "close",
        "adj close",
        "adj_close",
        "open",
        "high",
        "low",
        "price",
        "returns",
        "return",
    }

    def _build_sarimax_exogenous(
        self,
        *,
        price_series: "pd.Series",
        returns_series: "Optional[pd.Series]",
        macro_context: "Optional[pd.DataFrame]" = None,
    ) -> "pd.DataFrame":
        """
        Build a SARIMAX-X feature set from the observed window.

        Core features (always present):
          ["ret_1", "vol_10", "mom_5", "ema_gap_10", "zscore_20"]

        Macro features (Signal Quality A â€” merged when macro_context is provided):
          ["vix_level", "yield_spread_10y_2y", "sector_momentum_5d"]

        Macro columns absent from macro_context are silently omitted.
        All values forward-filled then back-filled to handle business-day gaps.
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

        exog_dict: Dict[str, "pd.Series"] = {
            "ret_1": ret_1,
            "vol_10": vol_10,
            "mom_5": mom_5,
            "ema_gap_10": ema_gap_10,
            "zscore_20": zscore_20,
        }

        # Signal Quality A: merge optional macro context columns
        if macro_context is not None and not macro_context.empty:
            for col in self._MACRO_EXOG_COLUMNS:
                if col in macro_context.columns:
                    aligned = (
                        macro_context[col]
                        .reindex(price_series.index)
                        .ffill()
                        .bfill()
                        .fillna(0.0)
                    )
                    exog_dict[col] = aligned.astype(float)

        exog = pd.DataFrame(exog_dict, index=price_series.index)
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

    @staticmethod
    def _hash_snapshot_inputs(
        series: pd.Series,
        *,
        exogenous: Optional[pd.DataFrame] = None,
    ) -> str:
        hasher = hashlib.sha256()
        values = np.asarray(series.astype(float, copy=False), dtype=np.float64)
        hasher.update(values.tobytes())

        try:
            if isinstance(series.index, pd.DatetimeIndex):
                hasher.update(series.index.view("int64").tobytes())
            else:
                for item in series.index:
                    hasher.update(str(item).encode("utf-8", errors="replace"))
                    hasher.update(b"\x1f")
        except Exception:
            hasher.update(str(list(series.index)).encode("utf-8", errors="replace"))

        if exogenous is not None and not exogenous.empty:
            hasher.update(b"|exogenous|")
            hasher.update(",".join(str(col) for col in exogenous.columns).encode("utf-8", errors="replace"))
            exog_values = np.asarray(exogenous.astype(float).to_numpy(copy=True), dtype=np.float64)
            hasher.update(exog_values.tobytes())

        return hasher.hexdigest()

    @staticmethod
    def _snapshot_loaded(component: str, forecaster: Any) -> bool:
        if component == "samossa":
            return bool(getattr(forecaster, "_fitted", False))
        return getattr(forecaster, "fitted_model", None) is not None

    def _maybe_restore_snapshot(
        self,
        *,
        component: str,
        model_type: str,
        forecaster: Any,
        series: pd.Series,
        exogenous: Optional[pd.DataFrame],
        ticker: str,
        regime: Optional[str],
    ) -> bool:
        if self._snapshot_store is None or not ticker:
            return False
        try:
            data_hash = self._hash_snapshot_inputs(series, exogenous=exogenous)
            snapshot = self._snapshot_store.load(
                ticker=ticker,
                model_type=model_type,
                regime=regime,
                current_n_obs=int(len(series)),
                current_data_hash=data_hash,
                max_obs_delta=0,
                strict_hash=True,
            )
            if snapshot is None:
                return False
            forecaster.load_fitted(snapshot)
            if not self._snapshot_loaded(component, forecaster):
                return False
            self._record_restored_order_usage(
                component=component,
                forecaster=forecaster,
                ticker=ticker,
                regime=regime,
            )
            self._record_model_event(
                component,
                "snapshot_restore",
                ticker=ticker,
                regime=regime,
                n_obs=int(len(series)),
            )
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[TS_MODEL] %s snapshot restore skipped: %s", component.upper(), exc)
            return False

    def _maybe_save_snapshot(
        self,
        *,
        component: str,
        model_type: str,
        fitted_obj: Any,
        series: pd.Series,
        exogenous: Optional[pd.DataFrame],
        ticker: str,
        regime: Optional[str],
        aic: Optional[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._snapshot_store is None or not ticker:
            return
        try:
            data_hash = self._hash_snapshot_inputs(series, exogenous=exogenous)
            self._snapshot_store.save(
                ticker=ticker,
                model_type=model_type,
                regime=regime,
                fitted_obj=fitted_obj,
                n_obs=int(len(series)),
                data_hash=data_hash,
                aic=float(aic) if aic is not None else float("nan"),
                metadata=metadata or {},
            )
            self._record_model_event(
                component,
                "snapshot_save",
                ticker=ticker,
                regime=regime,
                n_obs=int(len(series)),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[TS_MODEL] %s snapshot save skipped: %s", component.upper(), exc)

    def _record_restored_order_usage(
        self,
        *,
        component: str,
        forecaster: Any,
        ticker: str,
        regime: Optional[str],
    ) -> None:
        """Touch last_used for an exact snapshot restore without counting a new fit."""
        learner = self._order_learner
        if learner is None or not ticker:
            return

        model_type = ""
        order_params: Dict[str, Any] = {}
        try:
            if component == "garch":
                if getattr(forecaster, "fitted_model", None) is None:
                    return
                model_type = "GARCH"
                order_params = {
                    "p": int(getattr(forecaster, "p", 1) or 1),
                    "q": int(getattr(forecaster, "q", 1) or 1),
                    "dist": str(getattr(forecaster, "dist", "skewt") or "skewt"),
                    "mean": str(getattr(forecaster, "mean", "AR") or "AR"),
                }
            elif component == "sarimax":
                if getattr(forecaster, "fitted_model", None) is None:
                    return
                model_type = "SARIMAX"
                order_params = {
                    "order": list(getattr(forecaster, "best_order", None) or [1, 1, 1]),
                    "seasonal": list(
                        getattr(forecaster, "best_seasonal_order", None) or [0, 0, 0, 0]
                    ),
                }
            elif component == "samossa":
                if not bool(getattr(forecaster, "_fitted", False)):
                    return
                ar_lag = getattr(forecaster, "_learned_ar_lag", None)
                if ar_lag is None:
                    residual_model = getattr(forecaster, "_residual_model", None)
                    ar_lag = getattr(residual_model, "k_ar", None)
                if ar_lag is None:
                    return
                model_type = "SAMOSSA_ARIMA"
                order_params = {"ar_lag": int(ar_lag)}
            else:
                return
        except Exception:
            return

        try:
            learner.record_usage(
                ticker=ticker,
                model_type=model_type,
                regime=regime,
                order_params=order_params,
            )
        except Exception:
            logger.debug("[TS_MODEL] %s snapshot usage bookkeeping skipped", component.upper(), exc_info=True)

    def _build_monte_carlo_summary(
        self,
        results: Dict[str, Any],
        *,
        alpha: float,
        mc_paths: int,
        mc_seed: Optional[int],
    ) -> Dict[str, Any]:
        base_payload = results.get("mean_forecast")
        base_forecast = self._extract_series(base_payload)
        if base_forecast is None:
            return {"status": "SKIP", "reason": "base_forecast_missing"}
        try:
            last_price_val = float(self._last_price) if self._last_price is not None else float("nan")
        except Exception:
            last_price_val = float("nan")
        if not np.isfinite(last_price_val) or last_price_val <= 0.0:
            return {"status": "SKIP", "reason": "last_price_missing"}

        lower_ci = self._extract_series(base_payload, "lower_ci")
        upper_ci = self._extract_series(base_payload, "upper_ci")

        volatility = None
        volatility_payload = results.get("volatility_forecast")
        if isinstance(volatility_payload, dict):
            volatility = volatility_payload.get("volatility")
        if volatility is None:
            garch_payload = results.get("garch_forecast")
            if isinstance(garch_payload, dict):
                volatility = garch_payload.get("volatility")

        from forcester_ts.monte_carlo_simulator import MonteCarloSimulator

        simulator = MonteCarloSimulator()
        return simulator.simulate_price_distribution(
            base_forecast=base_forecast,
            last_price=last_price_val,
            n_paths=mc_paths,
            seed=mc_seed,
            volatility=volatility,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            alpha=alpha,
        )

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off", ""}:
                return False
        return bool(value)

    @staticmethod
    def _coerce_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, str) and not value.strip():
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _resolve_monte_carlo_options(
        self,
        *,
        mc_enabled: Optional[bool],
        mc_paths: Optional[int],
        mc_seed: Optional[int],
    ) -> tuple[bool, int, Optional[int]]:
        cfg = self.config.monte_carlo_config if isinstance(self.config.monte_carlo_config, dict) else {}
        enabled = self._coerce_bool(
            mc_enabled if mc_enabled is not None else cfg.get("enabled"),
            default=False,
        )
        paths = self._coerce_optional_int(mc_paths if mc_paths is not None else cfg.get("paths"))
        if paths is None:
            paths = self._DEFAULT_MC_PATHS
        seed = self._coerce_optional_int(mc_seed if mc_seed is not None else cfg.get("seed"))
        return enabled, int(paths), seed

    @staticmethod
    def _resolve_ticker(
        ticker: str,
        price_series: pd.Series,
        returns_series: Optional[pd.Series] = None,
    ) -> str:
        candidate = str(ticker or "").strip()
        if candidate:
            return candidate
        for series in (price_series, returns_series):
            if series is None:
                continue
            raw_name = getattr(series, "name", "")
            if isinstance(raw_name, tuple):
                raw_name = raw_name[0] if raw_name else ""
            candidate = str(raw_name or "").strip()
            normalized = candidate.replace("_", " ").strip().lower()
            if (
                candidate
                and candidate.upper() not in {"NONE", "NAN"}
                and normalized not in TimeSeriesForecaster._GENERIC_TICKER_NAMES
            ):
                return candidate
        return ""

    def fit(
        self,
        price_series: pd.Series,
        returns_series: Optional[pd.Series] = None,
        ticker: str = "",
        macro_context: Optional[pd.DataFrame] = None,
    ) -> "TimeSeriesForecaster":
        price_series = self._ensure_series(price_series)
        ticker = self._resolve_ticker(ticker, price_series, returns_series)
        self._macro_context = macro_context  # Signal Quality A: store for exog building
        cleaned_price = price_series.dropna()
        if not cleaned_price.empty:
            self._last_price = float(cleaned_price.iloc[-1])
            self._last_timestamp = pd.to_datetime(cleaned_price.index[-1])
        self._series_freq_hint = str(
            getattr(price_series.index, "freqstr", None) or price_series.attrs.get("_pm_freq_hint") or "B"
        )
        self._series_diagnostics = self._capture_series_diagnostics(price_series)
        self._model_events.clear()
        self._model_errors.clear()
        self._instrumentation.reset()
        self._instrumentation.set_dataset_metadata(
            length=len(price_series),
            start=str(price_series.index.min()),
            end=str(price_series.index.max()),
            frequency=normalize_freq(str(getattr(price_series.index, "freqstr", None) or price_series.attrs.get("_pm_freq_hint"))),
            ticker=ticker or None,
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

        # Phase 7.16: Extract regime string for OrderLearner regime conditioning.
        _ol_regime: Optional[str] = None
        if self._regime_result is not None:
            _ol_regime = self._regime_result.get("regime") or None
            if _ol_regime == "STATIC":
                _ol_regime = None

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
                        macro_context=self._macro_context,  # Signal Quality A
                    )
                    self._sarimax_exog_columns = list(exog.columns)
                    self._sarimax_exog_last_row = exog.iloc[-1] if not exog.empty else None
                    self._instrumentation.record_artifact(
                        "sarimax_exogenous",
                        {"columns": self._sarimax_exog_columns, "row_count": int(len(exog))},
                    )
                    restored_from_snapshot = self._maybe_restore_snapshot(
                        component="sarimax",
                        model_type="SARIMAX",
                        forecaster=self._sarimax,
                        series=price_series,
                        exogenous=exog,
                        ticker=ticker,
                        regime=_ol_regime,
                    )
                    meta["restored_from_snapshot"] = restored_from_snapshot
                    if not restored_from_snapshot:
                        self._sarimax.fit(
                            price_series, exogenous=exog,
                            order_learner=self._order_learner,
                            ticker=ticker,
                            regime=_ol_regime,
                        )
                        self._maybe_save_snapshot(
                            component="sarimax",
                            model_type="SARIMAX",
                            fitted_obj={
                                "fitted_model": self._sarimax.fitted_model,
                                "model": self._sarimax.model,
                                "best_order": self._sarimax.best_order,
                                "best_seasonal_order": self._sarimax.best_seasonal_order,
                                "scale_factor": getattr(self._sarimax, "_scale_factor", 1.0),
                                "fit_metadata": dict(getattr(self._sarimax, "_fit_metadata", {})),
                            },
                            series=price_series,
                            exogenous=exog,
                            ticker=ticker,
                            regime=_ol_regime,
                            aic=getattr(getattr(self._sarimax, "fitted_model", None), "aic", None),
                            metadata={
                                "best_order": list(self._sarimax.best_order or []),
                                "best_seasonal_order": list(self._sarimax.best_seasonal_order or []),
                            },
                        )
                    meta["order"] = getattr(self._sarimax, "best_order", None)
                    meta["seasonal"] = getattr(self._sarimax, "best_seasonal_order", None)
                self._record_model_event(
                    "sarimax",
                    "fit_complete",
                    order=getattr(self._sarimax, "best_order", None),
                    seasonal=getattr(self._sarimax, "best_seasonal_order", None),
                    restored=bool(meta.get("restored_from_snapshot")),
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
                    restored_from_snapshot = self._maybe_restore_snapshot(
                        component="samossa",
                        model_type="SAMOSSA",
                        forecaster=self._samossa,
                        series=price_series,
                        exogenous=None,
                        ticker=ticker,
                        regime=_ol_regime,
                    )
                    meta["restored_from_snapshot"] = restored_from_snapshot
                    if not restored_from_snapshot:
                        self._samossa.fit(
                            price_series,
                            order_learner=self._order_learner,
                            ticker=ticker,
                            regime=_ol_regime,
                        )
                        self._maybe_save_snapshot(
                            component="samossa",
                            model_type="SAMOSSA",
                            fitted_obj={
                                "reconstructed": getattr(self._samossa, "_reconstructed", None),
                                "residuals": getattr(self._samossa, "_residuals", None),
                                "residual_model": getattr(self._samossa, "_residual_model", None),
                                "scale_mean": getattr(self._samossa, "_scale_mean", 0.0),
                                "scale_std": getattr(self._samossa, "_scale_std", 1.0),
                                "evr": getattr(self._samossa, "_explained_variance_ratio", 0.0),
                                "trend_slope": getattr(self._samossa, "_trend_slope", 0.0),
                                "trend_intercept": getattr(self._samossa, "_trend_intercept", 0.0),
                                "trend_strength": getattr(self._samossa, "_trend_strength", 0.0),
                                "last_index": getattr(self._samossa, "_last_index", None),
                                "target_freq": getattr(self._samossa, "_target_freq", None),
                                "last_observed": getattr(self._samossa, "_last_observed", None),
                                "normalized_stats": dict(getattr(self._samossa, "_normalized_stats", {})),
                                "config_window_length": getattr(self._samossa.config, "window_length", None),
                                "config_n_components": getattr(self._samossa.config, "n_components", None),
                                "learned_ar_lag": getattr(self._samossa, "_learned_ar_lag", None),
                                "learned_ar_aic": getattr(self._samossa, "_learned_ar_aic", None),
                                "learned_ar_bic": getattr(self._samossa, "_learned_ar_bic", None),
                            },
                            series=price_series,
                            exogenous=None,
                            ticker=ticker,
                            regime=_ol_regime,
                            aic=getattr(self._samossa, "_learned_ar_aic", None),
                            metadata={"window_length": getattr(self._samossa.config, "window_length", None)},
                        )
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
                    restored=bool(meta.get("restored_from_snapshot")),
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

        # EXP-R5-001: auto-fit the residual model on OOS anchor residuals.
        # Only runs when enabled AND the anchor fitted successfully.
        # Any failure leaves _residual_model=None â†’ inactive artifact at forecast time.
        if self.config.residual_experiment_enabled and self._mssa is not None:
            self._fit_residual_model(price_series)

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
                        restored_from_snapshot = self._maybe_restore_snapshot(
                            component="garch",
                            model_type="GARCH",
                            forecaster=self._garch,
                            series=returns_series,
                            exogenous=None,
                            ticker=ticker,
                            regime=_ol_regime,
                        )
                        meta["restored_from_snapshot"] = restored_from_snapshot
                        if not restored_from_snapshot:
                            self._garch.fit(
                                returns_series,
                                order_learner=self._order_learner,
                                ticker=ticker,
                                regime=_ol_regime,
                            )
                            self._maybe_save_snapshot(
                                component="garch",
                                model_type="GARCH",
                                fitted_obj={
                                    "fitted_model": getattr(self._garch, "fitted_model", None),
                                    "backend": getattr(self._garch, "backend", "arch"),
                                    "scale_factor": getattr(self._garch, "_scale_factor", 1.0),
                                    "fallback_state": getattr(self._garch, "_fallback_state", None),
                                    "p": getattr(self._garch, "p", None),
                                    "q": getattr(self._garch, "q", None),
                                    "vol": getattr(self._garch, "vol", None),
                                    "dist": getattr(self._garch, "dist", None),
                                    "mean": getattr(self._garch, "mean", None),
                                },
                                series=returns_series,
                                exogenous=None,
                                ticker=ticker,
                                regime=_ol_regime,
                                aic=getattr(getattr(self._garch, "fitted_model", None), "aic", None),
                                metadata={
                                    "p": getattr(self._garch, "p", None),
                                    "q": getattr(self._garch, "q", None),
                                    "backend": getattr(self._garch, "backend", "arch"),
                                },
                            )
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
                        restored=bool(meta.get("restored_from_snapshot")),
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

    def forecast(
        self,
        *,
        steps: Optional[int] = None,
        alpha: float = 0.05,
        mc_enabled: Optional[bool] = None,
        mc_paths: Optional[int] = None,
        mc_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        horizon = int(steps) if steps is not None else self.config.forecast_horizon
        mc_enabled_resolved, mc_paths_resolved, mc_seed_resolved = self._resolve_monte_carlo_options(
            mc_enabled=mc_enabled,
            mc_paths=mc_paths,
            mc_seed=mc_seed,
        )
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
                    results["garch_forecast"] = self._enrich_garch_forecast(garch_result)
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
            ensemble_meta = ensemble["metadata"]
            self._instrumentation.record_artifact(
                "ensemble_weights", ensemble_meta.get("weights", {})
            )
            self._instrumentation.record_artifact(
                "ensemble_selection",
                {
                    "weights": dict(ensemble_meta.get("weights") or {}),
                    "model_confidence": (
                        dict(ensemble_meta.get("confidence", {}))
                        if isinstance(ensemble_meta.get("confidence"), dict)
                        else {}
                    ),
                    "selection_score": ensemble_meta.get("selection_score"),
                    "primary_model": ensemble_meta.get("primary_model"),
                    "allow_as_default": ensemble_meta.get("allow_as_default"),
                    "ensemble_status": ensemble_meta.get("ensemble_status"),
                    "ensemble_decision_reason": ensemble_meta.get("ensemble_decision_reason"),
                },
            )
            allow_as_default = bool(ensemble_meta.get("allow_as_default", True))
            if allow_as_default:
                results["mean_forecast"] = ensemble["forecast_bundle"]
                results["default_model"] = "ENSEMBLE"
            else:
                preferred_default = ensemble_meta.get("default_model") or ensemble_meta.get("primary_model")
                default_model, default_payload = self._select_default_single_forecast(
                    results,
                    preferred_model=preferred_default,
                )
                if default_payload is not None:
                    results["mean_forecast"] = default_payload
                    results["default_model"] = default_model
                    ensemble_meta["default_model"] = default_model
                else:
                    # Defensive fallback: preserve non-empty output even when no
                    # single-model forecast payload is available.
                    results["mean_forecast"] = ensemble["forecast_bundle"]
                    results["default_model"] = "ENSEMBLE"
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
                results["default_model"] = "SAMOSSA"
            else:
                results["mean_forecast"] = results.get("sarimax_forecast")
                if results["mean_forecast"] is not None:
                    results["default_model"] = "SARIMAX"

        if mc_enabled_resolved:
            self._record_model_event("monte_carlo", "forecast_start", horizon=horizon)
            with self._instrumentation.track("monte_carlo", "forecast", horizon=horizon) as meta:
                mc_result = self._build_monte_carlo_summary(
                    results,
                    alpha=alpha,
                    mc_paths=mc_paths_resolved,
                    mc_seed=mc_seed_resolved,
                )
                results["monte_carlo"] = mc_result
                meta["status"] = mc_result.get("status")
                meta["paths_used"] = mc_result.get("paths_used")
                meta["reason"] = mc_result.get("reason")
                meta["volatility_source"] = mc_result.get("volatility_source")
            self._record_model_event(
                "monte_carlo",
                "forecast_complete",
                status=results["monte_carlo"].get("status"),
                reason=results["monte_carlo"].get("reason"),
                paths_used=results["monte_carlo"].get("paths_used"),
            )

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
        # Phase 7.14-D: alias for DB persistence (database_manager reads 'detected_regime')
        results["detected_regime"] = results["regime"] if results["regime"] != "STATIC" else None
        self._instrumentation.set_dataset_metadata(detected_regime=results["detected_regime"])

        # EXP-R5-001: residual experiment artifact (research-only, no gate impact).
        # Record in instrumentation so it appears under artifacts.residual_experiment
        # in the audit JSON â€” that is where Agent B's extractor reads it from.
        results["residual_experiment"] = self._build_residual_experiment_artifact(results)
        self._instrumentation.record_artifact(
            "residual_experiment", results["residual_experiment"]
        )

        self._latest_results = results
        results["model_errors"] = dict(self._model_errors)
        results["model_events"] = list(self._model_events)
        results["instrumentation_report"] = self._instrumentation.export()
        if self._audit_dir:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            audit_path = self._audit_dir / f"forecast_audit_{timestamp}.json"
            self.save_audit_report(audit_path)
        return results

    # ------------------------------------------------------------------
    # EXP-R5-001 â€” Research experiment artifact builder (Agent A, Phase 1)
    # ------------------------------------------------------------------

    def _build_residual_experiment_artifact(
        self, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the EXP-R5-001 residual experiment artifact.

        Phase 2 contract
        ----------------
        - When ``config.residual_experiment_enabled`` is False: returns the
          canonical inactive schema immediately â€” zero impact on everything else.
        - When enabled but ``_residual_model`` is None: returns inactive schema
          with reason ``"residual_model_not_fitted"``.
        - When enabled and model fitted: produces a real residual forecast,
          returns active schema with ``y_hat_anchor`` and
          ``y_hat_residual_ensemble`` as plain ``list[float]`` (JSON-safe for
          Agent B's extractor which requires a plain list, not a pd.Series).
        - Metric fields (rmse_*, da_*, corr_anchor_residual) are always None;
          they require realized prices and are populated by Agent B (Phase 3).
        - This method never modifies ``results``, never changes default model
          selection, and never touches gate logic.
        """
        from .residual_ensemble import build_residual_ensemble, inactive_artifact

        def _attach_observability(art: Dict[str, Any]) -> Dict[str, Any]:
            model = self._residual_model
            art["phi_hat"] = getattr(model, "_phi", None) if model is not None else None
            art["intercept_hat"] = getattr(model, "_intercept", None) if model is not None else None
            art["n_train_residuals"] = getattr(model, "_n_train", None) if model is not None else None
            art["oos_n_used"] = self._residual_model_oos_n
            art["skip_reason"] = getattr(model, "_skip_reason", None) if model is not None else None
            return art

        if not self.config.residual_experiment_enabled:
            return inactive_artifact(
                reason="experiment disabled (residual_experiment_enabled=False)",
                reason_code="EXPERIMENT_DISABLED",
            )

        anchor_payload = results.get("mssa_rl_forecast")
        anchor_series = self._extract_series(anchor_payload)
        if anchor_series is None:
            return inactive_artifact(
                reason="mssa_rl anchor forecast unavailable",
                reason_code="ANCHOR_UNAVAILABLE",
            )

        if self._residual_model is None:
            return inactive_artifact(
                reason=self._residual_model_skip_reason
                or "residual_model_not_fitted (Phase 2 pending)",
                reason_code=self._residual_model_reason_code
                or "RESIDUAL_MODEL_NOT_FITTED",
            )

        try:
            residual_forecast: Optional[pd.Series] = self._residual_model.predict(
                horizon=len(anchor_series),
                last_residual=None,
                index=anchor_series.index,
            )
        except Exception as exc:
            logger.warning("[EXP-R5-001] ResidualModel.predict() failed: %s", exc)
            art = inactive_artifact(
                reason=f"residual_model.predict() failed: {exc}",
                reason_code="RESIDUAL_MODEL_PREDICT_FAILED",
            )
            return _attach_observability(art)

        # 1.2 - predicted residual path safety gates before applying correction.
        pred_vals = residual_forecast.dropna().values.astype(float)
        pred_mean = float(np.mean(pred_vals)) if len(pred_vals) > 0 else 0.0
        pred_std = float(np.std(pred_vals)) if len(pred_vals) > 0 else 0.0
        anchor_rmse_proxy = getattr(self._residual_model, "_anchor_rmse_proxy", None)
        pred_gate_code: Optional[str] = None
        pred_gate_reason: Optional[str] = None

        if abs(pred_mean) > 1e-9:
            cv = pred_std / abs(pred_mean)
            if cv < 0.10:
                pred_gate_code = "PREDICTED_RESIDUAL_TOO_CONSTANT"
                pred_gate_reason = (
                    f"predicted residual too constant (std/abs(mean)={cv:.4f} < 0.10)"
                )

        if (
            pred_gate_code is None
            and isinstance(anchor_rmse_proxy, (int, float))
            and np.isfinite(anchor_rmse_proxy)
            and float(anchor_rmse_proxy) > 0.0
            and abs(pred_mean) > 2.0 * float(anchor_rmse_proxy)
        ):
            pred_gate_code = "PREDICTED_MEAN_TOO_LARGE"
            pred_gate_reason = (
                f"predicted residual mean too large "
                f"(|mean|={abs(pred_mean):.4f} > 2*anchor_rmse_proxy={2.0*float(anchor_rmse_proxy):.4f})"
            )

        if pred_gate_code is not None:
            art = inactive_artifact(
                reason=pred_gate_reason or pred_gate_code,
                reason_code=pred_gate_code,
            )
            art = _attach_observability(art)
            logger.info(
                "[EXP-R5-001][DECISION] status=SKIPPED reason_code=%s "
                "phi=%.4f intercept=%.4f n_train=%s oos_n=%s pred_mean=%.4f pred_std=%.4f",
                pred_gate_code,
                getattr(self._residual_model, "_phi", 0.0),
                getattr(self._residual_model, "_intercept", 0.0),
                getattr(self._residual_model, "_n_train", None),
                self._residual_model_oos_n,
                pred_mean,
                pred_std,
            )
            return art

        artifact = build_residual_ensemble(anchor_series, residual_forecast)
        artifact = _attach_observability(artifact)

        # Convert pd.Series to plain list[float] for JSON-safe audit consumption.
        artifact["y_hat_anchor"] = _series_to_plain_list(artifact.get("y_hat_anchor"))
        artifact["y_hat_residual_ensemble"] = _series_to_plain_list(
            artifact.get("y_hat_residual_ensemble")
        )

        logger.info(
            "[EXP-R5-001][DECISION] status=APPLIED reason_code=%s "
            "phi=%.4f intercept=%.4f n_train=%s oos_n=%s pred_mean=%.4f pred_std=%.4f",
            artifact.get("reason_code", "OK"),
            getattr(self._residual_model, "_phi", 0.0),
            getattr(self._residual_model, "_intercept", 0.0),
            getattr(self._residual_model, "_n_train", None),
            self._residual_model_oos_n,
            pred_mean,
            pred_std,
        )
        return artifact

    def _fit_residual_model(self, price_series: pd.Series) -> None:
        """Fit ResidualModel on OOS anchor residuals (EXP-R5-001, Phase 2).

        OOS 4-step protocol:
        1. Split: train_part = price_series[:-oos_n], val_part = price_series[-oos_n:]
        2. Fit a temporary MSSARLForecaster on train_part (never replaces self._mssa).
        3. Generate oos_n-step forecast from the temporary anchor.
        4. residuals = val_part.values - oos_forecast.values (positional alignment).
        5. Fit ResidualModel on those residuals.

        On any failure (data too short, mssa fit error, etc.) self._residual_model
        stays None â€” the artifact falls back to inactive at forecast time.
        No gates or live strategy are affected.
        """
        from .residual_ensemble import ResidualModel, compute_oos_residuals

        # Reset model state first to avoid stale carry-over when a new fit fails.
        self._residual_model = None
        self._residual_model_oos_n = None
        self._residual_model_reason_code = None
        self._residual_model_skip_reason = None

        def _log_fit_decision(
            status: str,
            reason_code: str,
            detail: str,
            *,
            model: Optional[Any] = None,
            oos_used: Optional[int] = None,
        ) -> None:
            logger.info(
                "[EXP-R5-001][FIT_DECISION] status=%s reason_code=%s detail=%s "
                "phi=%s intercept=%s n_train=%s oos_n=%s",
                status,
                reason_code,
                detail,
                getattr(model, "_phi", None) if model is not None else None,
                getattr(model, "_intercept", None) if model is not None else None,
                getattr(model, "_n_train", None) if model is not None else None,
                oos_used,
            )

        # RC3: proportional OOS slice â€” scales with available data.
        # Use 25% of series for OOS validation; enforce minimum 20 points so
        # AR(1) OLS has at least 19 regression pairs for 2 parameters (12 df).
        # The hard-cap of max(horizon, 20) = 20-30 was too small: OLS on
        # ~20 residual-of-residuals points was dominated by a few large errors.
        cleaned = price_series.dropna()
        oos_n = len(cleaned) // 4
        if oos_n < 20:
            self._residual_model_reason_code = "TOO_FEW_OOS_POINTS"
            self._residual_model_skip_reason = (
                f"oos_n={oos_n} < 20 from len={len(cleaned)}"
            )
            _log_fit_decision(
                "SKIPPED",
                self._residual_model_reason_code,
                self._residual_model_skip_reason,
                oos_used=oos_n,
            )
            return
        # Need train_part >= 3Ã—oos_n: anchor model requires substantial history.
        if len(cleaned) - oos_n < 3 * oos_n:
            self._residual_model_reason_code = "TOO_FEW_TRAINING_POINTS"
            self._residual_model_skip_reason = (
                f"train_part={len(cleaned) - oos_n} < 3*oos_n={3 * oos_n}"
            )
            _log_fit_decision(
                "SKIPPED",
                self._residual_model_reason_code,
                self._residual_model_skip_reason,
                oos_used=oos_n,
            )
            return

        train_part = cleaned.iloc[:-oos_n]
        val_part = cleaned.iloc[-oos_n:]

        try:
            mssa_config = MSSARLConfig(**self.config.mssa_rl_kwargs)
            tmp_mssa = MSSARLForecaster(config=mssa_config)
            tmp_mssa.fit(train_part)
            mssa_output = tmp_mssa.forecast(steps=oos_n)
            oos_fc = self._extract_series(mssa_output)
            if oos_fc is None or len(oos_fc) != oos_n:
                self._residual_model_reason_code = "OOS_FORECAST_INVALID"
                self._residual_model_skip_reason = (
                    f"oos forecast length invalid: {None if oos_fc is None else len(oos_fc)}"
                )
                _log_fit_decision(
                    "SKIPPED",
                    self._residual_model_reason_code,
                    self._residual_model_skip_reason,
                    oos_used=oos_n,
                )
                return

            # Align positionally: mssa forecasts future dates, val_part has training dates.
            anchor_oos = pd.Series(oos_fc.values, index=val_part.index, name="anchor_oos")
            oos_residuals = compute_oos_residuals(anchor_oos, val_part, source="oos")

            model = ResidualModel()
            model.fit_on_oos_residuals(oos_residuals, source="oos")
            # One or more model gates may have fired inside fit_on_oos_residuals().
            if not model.is_fitted:
                self._residual_model_reason_code = (
                    model._reason_code or "RESIDUAL_MODEL_NOT_FITTED"
                )
                self._residual_model_skip_reason = model._skip_reason or "model_not_fitted"
                _log_fit_decision(
                    "SKIPPED",
                    self._residual_model_reason_code,
                    self._residual_model_skip_reason,
                    model=model,
                    oos_used=oos_n,
                )
                return

            self._residual_model = model
            self._residual_model_oos_n = oos_n
            self._residual_model_reason_code = model._reason_code or "OK"
            self._residual_model_skip_reason = None
            _log_fit_decision(
                "ELIGIBLE",
                self._residual_model_reason_code,
                "residual model passed fit-time gates",
                model=model,
                oos_used=oos_n,
            )
        except Exception as exc:
            self._residual_model_reason_code = "RESIDUAL_MODEL_FIT_EXCEPTION"
            self._residual_model_skip_reason = str(exc)
            _log_fit_decision(
                "SKIPPED",
                self._residual_model_reason_code,
                self._residual_model_skip_reason,
                oos_used=oos_n,
            )
            logger.warning(
                "[EXP-R5-001] _fit_residual_model() failed: %s",
                exc,
                exc_info=True,
            )

    def _build_forecast_index(self, horizon: int) -> pd.Index:
        if horizon <= 0:
            return pd.RangeIndex(0, 0)
        if self._last_timestamp is None:
            return pd.RangeIndex(start=1, stop=horizon + 1, step=1)

        freq = normalize_freq(str(self._series_freq_hint or "B"))
        try:
            future = pd.date_range(
                start=self._last_timestamp,
                periods=int(horizon) + 1,
                freq=freq,
            )
            return pd.DatetimeIndex(future[1:])
        except Exception:
            future = pd.date_range(
                start=self._last_timestamp,
                periods=int(horizon) + 1,
                freq="B",
            )
            return pd.DatetimeIndex(future[1:])

    def _enrich_garch_forecast(self, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}

        out = dict(payload)
        mean_ret = payload.get("mean_forecast")
        variance = payload.get("variance_forecast")
        if not isinstance(mean_ret, pd.Series) or mean_ret.empty:
            return out

        horizon = int(payload.get("steps") or len(mean_ret) or self.config.forecast_horizon)
        idx = self._build_forecast_index(horizon)
        ret_vals = pd.to_numeric(mean_ret, errors="coerce").to_numpy(dtype=float)
        if ret_vals.size == 0:
            return out

        start_price = float(self._last_price) if isinstance(self._last_price, (int, float)) else 100.0
        prices: list[float] = []
        cur = start_price
        for r in ret_vals:
            if not np.isfinite(r):
                r = 0.0
            cur = cur * (1.0 + float(r))
            prices.append(cur)

        forecast = pd.Series(prices, index=idx[: len(prices)], name="garch_price_forecast")
        out["forecast"] = forecast

        if isinstance(variance, pd.Series) and not variance.empty:
            sigma = np.sqrt(pd.to_numeric(variance, errors="coerce")).to_numpy(dtype=float)
            sigma = sigma[: len(prices)]
            lower = []
            upper = []
            # Approximate return CI transformed into price CI.
            z = 1.96
            for i, p in enumerate(prices[: len(sigma)]):
                s = sigma[i] if np.isfinite(sigma[i]) else 0.0
                lower.append(max(0.0, p * (1.0 - z * s)))
                upper.append(p * (1.0 + z * s))
            out["lower_ci"] = pd.Series(lower, index=forecast.index[: len(lower)], name="garch_lower_ci")
            out["upper_ci"] = pd.Series(upper, index=forecast.index[: len(upper)], name="garch_upper_ci")

            # Phase 7.14-C: Inflate CI by 1.5x when GARCH optimizer did not converge.
            # Wide CI -> SNR drops below threshold -> signal_generator.py:478 blocks signal.
            # Self-attenuating chain: bad fit -> wider CI -> lower SNR -> no trade.
            if not payload.get("convergence_ok", True):
                logger.warning(
                    "GARCH convergence_ok=False: inflating CI half-width by 1.5x "
                    "to trigger SNR gate and block low-quality signal."
                )
                lower_arr = out["lower_ci"].to_numpy(dtype=float)
                upper_arr = out["upper_ci"].to_numpy(dtype=float)
                price_arr = forecast.to_numpy(dtype=float)
                inflated_lower = price_arr - (price_arr - lower_arr) * 1.5
                inflated_upper = price_arr + (upper_arr - price_arr) * 1.5
                out["lower_ci"] = pd.Series(
                    np.maximum(0.0, inflated_lower),
                    index=forecast.index[: len(inflated_lower)],
                    name="garch_lower_ci",
                )
                out["upper_ci"] = pd.Series(
                    inflated_upper,
                    index=forecast.index[: len(inflated_upper)],
                    name="garch_upper_ci",
                )
            out["convergence_ok"] = bool(payload.get("convergence_ok", True))

        return out

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
        payload = self._instrumentation.export_json_safe()
        cohort_id = (
            os.environ.get("PMX_EVIDENCE_COHORT_ID")
            or ("production" if output_path.parent.name.lower() == "production" else output_path.parent.name)
            or "default"
        )
        payload.update(
            {
                "evidence_contract_version": 2,
                "audit_id": output_path.stem,
                "cohort_id": cohort_id,
                "event_type": "FORECAST_AUDIT",
                "semantic_admission": {
                    "accepted_for_audit_history": True,
                    "admissible_for_readiness": False,
                    "gate_eligible": False,
                    "gate_bucket": "ACCEPTED_NONELIGIBLE",
                    "reason_code": "AWAITING_TRADE_SIGNAL_CONTEXT",
                    "production_labeled": output_path.parent.name.lower() == "production",
                    "manifest_registered": False,
                    "not_quarantined": True,
                    "superseded": False,
                    "duplicate_conflict": False,
                },
                "lineage_v2": {
                    "run_id": None,
                    "forecast_id": None,
                    "signal_id": None,
                    "order_id": None,
                    "position_id": None,
                    "entry_trade_id": None,
                    "close_trade_id": None,
                    "audit_id": output_path.stem,
                    "context_type": None,
                },
            }
        )
        atomic_write_json(output_path, payload)
        self._append_audit_manifest_entry(output_path)

    def _append_audit_manifest_entry(self, audit_path: Path) -> None:
        """
        Record immutable digest metadata for forecast audits.

        This does not replace filesystem controls, but it allows the gate to
        detect silent audit tampering or partial writes before using evidence.
        """
        entry = build_manifest_entry(
            audit_path,
            source="TimeSeriesForecaster.save_audit_report",
            extra={
                "audit_id": audit_path.stem,
                "evidence_contract_version": 2,
                "cohort_id": audit_path.parent.name,
                "event_type": "FORECAST_AUDIT",
            },
        )
        if not entry:
            logger.warning("Unable to hash forecast audit artifact: %s", audit_path)
            return

        manifest_path = audit_path.parent / "forecast_audit_manifest.jsonl"
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            upsert_jsonl_record(manifest_path, entry, key_field="file")
        except Exception as exc:
            logger.warning(
                "Failed to append forecast audit manifest entry for %s: %s",
                audit_path,
                exc,
            )

    def _build_config_from_kwargs(
        self,
        *,
        forecast_horizon: Optional[int],
        sarimax_config: Optional[Dict[str, Any]],
        garch_config: Optional[Dict[str, Any]],
        samossa_config: Optional[Dict[str, Any]],
        mssa_rl_config: Optional[Dict[str, Any]],
        ensemble_config: Optional[Dict[str, Any]],
        monte_carlo_config: Optional[Dict[str, Any]],
    ) -> TimeSeriesForecasterConfig:
        config = TimeSeriesForecasterConfig()
        if forecast_horizon is not None:
            config.forecast_horizon = int(forecast_horizon)

        if sarimax_config is not None:
            config.sarimax_enabled = bool(sarimax_config.get("enabled", False))
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

        if monte_carlo_config is not None:
            config.monte_carlo_config = dict(monte_carlo_config)

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

        # Diversity gate: measure max pairwise forecast correlation before blending.
        # Computed here so the actual forecast vectors are available.
        diversity_gate = self._ensemble_diversity_gate(forecasts, weights)

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
        preselection_gate = self._preselection_default_gate()
        metadata["preselection_gate"] = preselection_gate
        metadata["diversity_gate"] = diversity_gate
        preselection_ok = bool(preselection_gate.get("allow_as_default", True))
        diversity_ok = bool(diversity_gate.get("allow_as_default", True))
        metadata["allow_as_default"] = preselection_ok and diversity_ok
        if metadata["allow_as_default"]:
            metadata.setdefault("ensemble_status", "KEEP")
            metadata.setdefault("ensemble_decision_reason", "preselection gate passed")
        elif not preselection_ok:
            reason = str(preselection_gate.get("reason", "recent RMSE ratio gate"))
            metadata["ensemble_status"] = "DISABLE_DEFAULT"
            metadata["ensemble_decision_reason"] = f"preselection gate: {reason}"
            target_default_model = preselection_gate.get("target_default_model")
            if isinstance(target_default_model, str) and target_default_model.strip():
                metadata["default_model"] = target_default_model.strip().upper()
            elif primary_model:
                metadata["default_model"] = primary_model
            self._record_model_event(
                "ensemble",
                "preselection_gate_blocked",
                reason=reason,
                recent_rmse_ratio=preselection_gate.get("recent_rmse_ratio"),
                threshold=preselection_gate.get("threshold"),
                effective_audits=preselection_gate.get("effective_n"),
            )
        else:
            # diversity gate blocked
            reason = str(diversity_gate.get("reason", "high forecast correlation"))
            metadata["ensemble_status"] = "HIGH_CORRELATION"
            metadata["ensemble_decision_reason"] = f"diversity gate: {reason}"
            if primary_model:
                metadata["default_model"] = primary_model
            self._record_model_event(
                "ensemble",
                "diversity_gate_blocked",
                reason=reason,
                max_correlation=diversity_gate.get("max_correlation"),
                threshold=diversity_gate.get("threshold"),
            )
        return {"forecast_bundle": forecast_bundle, "metadata": metadata}

    def _select_default_single_forecast(
        self,
        results: Dict[str, Any],
        *,
        preferred_model: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        order: list[str] = []
        if isinstance(preferred_model, str) and preferred_model.strip():
            order.append(preferred_model.strip().lower())
        for model in ("samossa", "sarimax", "garch", "mssa_rl"):
            if model not in order:
                order.append(model)

        for model in order:
            payload = results.get(f"{model}_forecast")
            if not isinstance(payload, dict):
                continue
            if self._extract_series(payload) is None:
                continue
            return model.upper(), payload
        return None, None

    @staticmethod
    def _extract_series(forecast_payload: Optional[Dict[str, Any]], key: str = "forecast") -> Optional[pd.Series]:
        if not isinstance(forecast_payload, dict):
            return None
        series = forecast_payload.get(key)
        if isinstance(series, pd.Series):
            return series
        # Backward compatibility: some historical GARCH payloads expose
        # `mean_forecast` instead of a price-level `forecast`.
        if key == "forecast":
            fallback = forecast_payload.get("mean_forecast")
            if isinstance(fallback, pd.Series):
                return fallback
        return None

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
        for name in ("sarimax", "garch", "samossa", "mssa_rl"):
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

    def _ensemble_diversity_gate(
        self,
        forecasts: Dict[str, Optional[Any]],
        weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """Block ensemble blending when active model forecast vectors are too correlated.

        Computes max pairwise Pearson correlation across models that carry meaningful
        weight (> 0.001).  When correlation exceeds the configured threshold the
        blended output adds no diversity benefit over the best single model.

        Returns dict with keys:
            allow_as_default  -- False when correlation is too high
            max_correlation   -- float or None
            max_pair          -- (model_a, model_b) or None
            threshold         -- configured threshold
            reason            -- human-readable verdict
        """
        cfg = self._rmse_monitor_cfg or {}
        threshold = float(cfg.get("diversity_max_correlation_threshold", 0.95))

        active: Dict[str, np.ndarray] = {}
        for m, f in forecasts.items():
            if weights.get(m, 0.0) <= 0.001:
                continue
            if f is None:
                continue
            try:
                arr = np.asarray(f.values if hasattr(f, "values") else f, dtype=float)
            except Exception:
                continue
            if len(arr) >= 2 and np.isfinite(arr).any():
                active[m] = arr

        base = {"threshold": threshold, "max_correlation": None, "max_pair": None}
        if len(active) < 2:
            base["allow_as_default"] = True
            base["reason"] = "fewer than 2 active model forecasts â€” diversity check skipped"
            return base

        models = list(active)
        max_corr = 0.0
        max_pair: Optional[tuple] = None
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                s1, s2 = active[m1], active[m2]
                n = min(len(s1), len(s2))
                if n < 2:
                    continue
                try:
                    c = float(np.corrcoef(s1[:n], s2[:n])[0, 1])
                    if np.isfinite(c) and abs(c) > max_corr:
                        max_corr = abs(c)
                        max_pair = (m1, m2)
                except Exception:
                    continue

        base["max_correlation"] = round(max_corr, 4)
        base["max_pair"] = list(max_pair) if max_pair else None
        if max_corr > threshold:
            base["allow_as_default"] = False
            pair_str = f"{max_pair[0]}/{max_pair[1]}" if max_pair else "?"
            base["reason"] = (
                f"forecast correlation {max_corr:.3f} > {threshold:.3f} ({pair_str})"
                " â€” blending adds no diversity benefit"
            )
        else:
            base["allow_as_default"] = True
            base["reason"] = f"forecast diversity sufficient (max_corr={max_corr:.3f})"
        return base

    def _preselection_default_gate(self) -> Dict[str, Any]:
        """
        Decide whether the ensemble is allowed as the default forecast source.

        This gate runs during forecast selection (pre-holdout for the current
        window) using recent audit history. When the recent RMSE ratio of
        ensemble vs best single model is above threshold, the ensemble remains
        available for diagnostics but is not allowed as the default source.
        """
        cfg = self._rmse_monitor_cfg or {}
        enabled = bool(cfg.get("strict_preselection_gate_enabled", True))
        threshold = float(cfg.get("strict_preselection_max_rmse_ratio", 1.0))
        recent_window = max(1, int(cfg.get("strict_preselection_recent_window", 5) or 5))
        min_effective = max(1, int(cfg.get("strict_preselection_min_effective_audits", 1) or 1))

        history = self._audit_history_stats(limit=max(200, recent_window))
        ratios = list(history.get("ratios") or [])
        recent = ratios[:recent_window]
        recent_ratio = float(np.mean(recent)) if recent else None
        effective_n = int(history.get("effective_n", 0) or 0)

        decision = {
            "enabled": enabled,
            "allow_as_default": True,
            "reason": "preselection gate passed",
            "threshold": threshold,
            "effective_n": effective_n,
            "recent_window": recent_window,
            "recent_rmse_ratio": recent_ratio,
            "recent_ratios": recent,
        }
        if not enabled:
            decision["reason"] = "preselection gate disabled"
            return decision
        if recent_ratio is None:
            decision["reason"] = "no recent audit RMSE ratios"
            return decision
        if effective_n < min_effective:
            decision["reason"] = (
                f"insufficient effective audits ({effective_n} < {min_effective})"
            )
            return decision
        if recent_ratio > threshold:
            decision["allow_as_default"] = False
            decision["reason"] = (
                f"recent RMSE ratio {recent_ratio:.3f} > {threshold:.3f}"
            )
            return decision
        return decision

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
        # Keep evaluate() read-only: changing ensemble weights using realized
        # labels in this method is post-hoc leakage and contaminates gate data.
        metadata = self._latest_results.get("ensemble_metadata")
        if isinstance(metadata, dict):
            metadata.setdefault("holdout_reweight_applied", False)
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
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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
