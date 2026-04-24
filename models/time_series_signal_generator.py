"""
Time Series Signal Generator - Convert forecasts to trading signals
Line Count: ~350 lines

Converts time series forecasts (SARIMAX, SAMOSSA, GARCH, MSSA-RL) into trading signals.
This is the DEFAULT signal generator, with LLM as fallback/redundancy.

Per refactoring plan:
- Time Series ensemble is PRIMARY signal source
- LLM serves as fallback when Time Series models fail or need validation
- Signals include confidence scores, risk metrics, and provenance
"""

import json
import logging
import math
import os
import secrets
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field

from etl.timestamp_utils import utc_now
from etl.env_flags import is_synthetic_mode
from utils.weather_context import extract_weather_context

from etl.time_series_forecaster import (
    TimeSeriesForecaster,
    RollingWindowValidator,
    RollingWindowCVConfig,
    TimeSeriesForecasterConfig,
)
from etl.portfolio_math import (
    DEFAULT_RISK_FREE_RATE,
    bootstrap_confidence_intervals,
    calculate_enhanced_portfolio_metrics,
    calculate_portfolio_metrics,
    payoff_asymmetry_support_metrics,
    portfolio_metrics_ngn,
    test_strategy_significance,
)
try:  # pragma: no cover - optional for deployments without barbell overlay
    from risk.barbell_policy import BarbellConfig
    from risk.barbell_sizing import (
        barbell_bucket as resolve_barbell_bucket,
        build_barbell_market_context,
        evaluate_barbell_path_risk,
    )
except Exception:  # pragma: no cover
    BarbellConfig = None  # type: ignore
    resolve_barbell_bucket = None  # type: ignore
    build_barbell_market_context = None  # type: ignore
    evaluate_barbell_path_risk = None  # type: ignore

try:
    from scripts.robustness_thresholds import (
        MIN_SIGNAL_TO_NOISE_KEY,
        load_floored_thresholds,
    )
except Exception:  # pragma: no cover - keep generator usable in direct execution
    from robustness_thresholds import (  # type: ignore
        MIN_SIGNAL_TO_NOISE_KEY,
        load_floored_thresholds,
    )

try:  # pragma: no cover - optional import for execution references
    from execution.order_manager import request_safe_price
except Exception:  # pragma: no cover - keep generator usable without execution stack
    def request_safe_price(price: float) -> float:
        return price if price and price > 0 else 1.0

try:  # pragma: no cover - optional for environments without execution stack
    from execution.lob_simulator import LOBConfig, simulate_market_order_fill
except Exception:  # pragma: no cover - graceful fallback when execution layer absent
    LOBConfig = None  # type: ignore
    simulate_market_order_fill = None  # type: ignore

# Phase 7.9: Risk mode configuration for quant validation
RISK_MODE_AVAILABLE = False
is_quant_validation_advisory = None
is_quant_validation_disabled = None
get_active_mode = None
try:
    from utils.risk_mode_loader import (
        is_quant_validation_advisory,
        is_quant_validation_disabled,
        get_active_mode,
    )
    RISK_MODE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional feature
    pass

logger = logging.getLogger(__name__)
# Dedicated gate logger — filterable separately from the main module logger.
# Records emitted here follow the schema:
#   gate=<name>  signal_id=<ts_signal_id>  ticker=<ticker>
#   value=<float|str>  threshold=<float|str>  result=PASS|FAIL|SKIP|UNKNOWN
_gate_logger = logging.getLogger("pmx.gates")


def _pmx_env_flag(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _pmx_is_intraday_interval(interval: Optional[str]) -> bool:
    if not interval:
        return False
    text = str(interval).strip().lower()
    if any(token in text for token in ("d", "wk", "mo")):
        return False
    return any(token in text for token in ("m", "h", "min", "hour"))


def _pmx_intraday_sarimax_kwargs(interval: Optional[str]) -> Dict[str, Any]:
    if not _pmx_is_intraday_interval(interval):
        return {}
    return {
        # Do not hard-code SARIMAX orders; only constrain the search grid so
        # intraday quant-validation remains tractable while still learning
        # (p,d,q,P,D,Q,s) per Documentation/SARIMAX_IMPLEMENTATION_CHECKLIST.md.
        "auto_select": True,
        "trend": "auto",
        "max_p": 1,
        "max_d": 1,
        "max_q": 1,
        # Intraday guardrail: disable seasonal terms for speed on gappy hour bars.
        "seasonal_periods": 0,
        "max_P": 0,
        "max_D": 0,
        "max_Q": 0,
        "order_search_maxiter": 60,
        "order_search_mode": "compact",
    }


@dataclass
class TimeSeriesSignal:
    """Trading signal generated from time series forecast"""
    ticker: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    entry_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    signal_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_type: str = 'ENSEMBLE'  # 'SARIMAX', 'SAMOSSA', 'GARCH', 'MSSA_RL', 'ENSEMBLE'
    forecast_horizon: int = 30
    expected_return: float = 0.0
    risk_score: float = 0.5
    reasoning: str = ''
    provenance: Dict[str, Any] = field(default_factory=dict)
    signal_type: str = 'TIME_SERIES'
    volatility: Optional[float] = None
    lower_ci: Optional[float] = None
    upper_ci: Optional[float] = None
    signal_id: Optional[str] = None  # Phase 7.13-A2: globally unique ts_signal_id string
    confidence_calibrated: Optional[float] = None  # Pure Platt-scaled probability (before raw blend)
    p_up: Optional[float] = None                # Phase 9: directional probability P(price_up)
    directional_gate_applied: bool = False      # Phase 9: True if classifier demoted action to HOLD


class TimeSeriesSignalGenerator:
    """
    Generate trading signals from time series forecasts.

    This is the DEFAULT signal generator. Converts ensemble forecasts
    (SARIMAX, SAMOSSA, GARCH, MSSA-RL) into actionable trading signals
    with confidence scores and risk metrics.
    """

    def __init__(self,
                 confidence_threshold: float = 0.55,
                 min_expected_return: float = 0.003,  # 0.3% minimum to clear costs
                 max_risk_score: float = 0.7,
                 use_volatility_filter: bool = True,
                 quant_validation_config: Optional[Dict[str, Any]] = None,
                 quant_validation_config_path: Optional[str] = None,
                 per_ticker_thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
                 cost_model: Optional[Dict[str, Any]] = None,
                 forecasting_config_path: Optional[str] = None):
        """
        Initialize Time Series signal generator.

        Args:
            confidence_threshold: Minimum confidence to generate signal (0.55 = 55%)
            min_expected_return: Minimum expected return to trigger signal (0.3%)
            max_risk_score: Maximum risk score allowed (0.7 = 70%)
            use_volatility_filter: Filter signals based on volatility forecasts
            quant_validation_config: Optional overrides for quant success helper
            quant_validation_config_path: Optional path to YAML config describing
                quant validation thresholds (defaults to config/quant_success_config.yml)
        """
        # Phase 7.10: monotonic signal ID counter for model attribution
        self._signal_counter = 0
        # Phase 7.15: per-instance 4-char hex prevents ts_signal_id collisions across
        # instances that start within the same second (parallel processing, test runs).
        self._instance_uid: str = secrets.token_hex(2)  # e.g. "a3f2"
        # Phase 7.13-A2: current ticker set at generate_signals() entry to build ts_signal_id
        self._current_ticker: str = "unknown"
        # B5 Platt scaling: stores the pure logistic-regression probability from the
        # most recent _calibrate_confidence() call (before blending with raw confidence).
        self._platt_calibrated: Optional[float] = None

        # Diagnostic toggle (env DIAGNOSTIC_MODE=1 or TS_DIAGNOSTIC_MODE=1) to force more signals.
        diag_mode = str(os.getenv("TS_DIAGNOSTIC_MODE") or os.getenv("DIAGNOSTIC_MODE") or "0") == "1"
        if diag_mode:
            confidence_threshold = 0.10
            min_expected_return = 0.0
            max_risk_score = 1.0
            use_volatility_filter = False

        self.confidence_threshold = confidence_threshold
        self.min_expected_return = min_expected_return
        self.max_risk_score = max_risk_score
        self.use_volatility_filter = use_volatility_filter
        self._diag_mode = diag_mode
        self._per_ticker_thresholds = per_ticker_thresholds or {}
        self._cost_model = cost_model or self._load_execution_cost_model()
        self._default_roundtrip_cost_bps = {
            "US_EQUITY": 10.0,
            "INTL_EQUITY": 15.0,
            "FX": 5.0,
            "CRYPTO": 25.0,
            "INDEX": 10.0,
            "UNKNOWN": 10.0,
        }
        overrides = self._cost_model.get("default_roundtrip_cost_bps")
        if isinstance(overrides, dict):
            for key, value in overrides.items():
                try:
                    self._default_roundtrip_cost_bps[str(key).upper()] = float(value)
                except (TypeError, ValueError):
                    continue
        floored_routing = load_floored_thresholds(Path("config/signal_routing_config.yml"))
        try:
            raw_snr = float(
                self._cost_model.get(
                    MIN_SIGNAL_TO_NOISE_KEY,
                    floored_routing.get(MIN_SIGNAL_TO_NOISE_KEY, 1.5),
                )
                or 0.0
            )
        except (TypeError, ValueError):
            raw_snr = float(floored_routing.get(MIN_SIGNAL_TO_NOISE_KEY, 1.5) or 1.5)
        self._min_signal_to_noise = max(raw_snr, float(floored_routing.get(MIN_SIGNAL_TO_NOISE_KEY, 1.5) or 1.5))
        self._cost_model[MIN_SIGNAL_TO_NOISE_KEY] = self._min_signal_to_noise
        self._quant_validation_config_path = (
            Path(quant_validation_config_path).expanduser()
            if quant_validation_config_path
            else Path("config/quant_success_config.yml")
        )
        self.quant_validation_config = self._load_quant_validation_config(quant_validation_config)
        self._quant_validation_enabled = bool(
            self.quant_validation_config and self.quant_validation_config.get("enabled", False)
        )
        if diag_mode:
            # Disable quant validation in diagnostic mode to avoid gating signals.
            self._quant_validation_enabled = False

        runtime_execution_mode = self._coerce_nonempty_str(
            os.getenv("EXECUTION_MODE") or os.getenv("PMX_EXECUTION_MODE")
        )
        if runtime_execution_mode is None:
            runtime_execution_mode = "paper" if _pmx_env_flag("PMX_PROOF_MODE") else "live"
        self._runtime_execution_mode = runtime_execution_mode.lower()

        runtime_run_id = self._coerce_nonempty_str(
            os.getenv("PMX_RUN_ID") or os.getenv("RUN_ID")
        )
        if runtime_run_id is None:
            runtime_run_id = f"pmx_ts_{utc_now().strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
        self._runtime_run_id = runtime_run_id

        runtime_pipeline_id = self._coerce_nonempty_str(
            os.getenv("PORTFOLIO_MAXIMIZER_PIPELINE_ID") or os.getenv("PIPELINE_ID")
        )
        if runtime_pipeline_id is None:
            runtime_pipeline_id = self._runtime_run_id
        self._runtime_pipeline_id = runtime_pipeline_id
        self._barbell_cfg = None

        # Phase 7.4 FIX: Load forecasting config to preserve ensemble_kwargs during CV
        self._forecasting_config_path = (
            Path(forecasting_config_path).expanduser()
            if forecasting_config_path
            else Path("config/forecasting_config.yml")
        )
        self._forecasting_config = self._load_forecasting_config()

        # Cache for expensive forecast-edge validation runs (rolling CV).
        # Keyed by (ticker, last_bar_ts, horizon) so repeated calls on the same
        # bar (or across near-identical windows) stay fast.
        # Include requested baseline so baseline-policy changes never reuse stale math.
        self._forecast_edge_cache: Dict[Tuple[str, str, int, str], Tuple[Dict[str, Any], Dict[str, bool]]] = {}

        # Phase 9: directional classifier gate (lazy-loaded; off by default)
        self._directional_classifier: Optional[Any] = None
        self._signal_routing_config: Optional[Dict[str, Any]] = self._load_signal_routing_config()

        logger.info(
            "Time Series Signal Generator initialized "
            "(confidence_threshold=%.2f, min_return=%.2f%%, max_risk=%.2f, quant_validation=%s)",
            confidence_threshold,
            min_expected_return * 100,
            max_risk_score,
            "on" if self._quant_validation_enabled else "off",
        )

    def _get_barbell_config(self):
        if self._barbell_cfg is not None:
            return self._barbell_cfg
        if BarbellConfig is None:
            return None
        try:
            self._barbell_cfg = BarbellConfig.from_yaml()
        except Exception:  # pragma: no cover - keep generator usable without overlay config
            self._barbell_cfg = None
        return self._barbell_cfg

    def _load_quant_validation_config(self, override: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Load quant validation configuration from override or disk."""
        if override:
            return override

        path = getattr(self, "_quant_validation_config_path", None)
        if not path or not path.exists():
            return None

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except Exception as exc:  # pragma: no cover - configuration errors are logged
            logger.warning("Unable to read quant validation config %s: %s", path, exc)
            return None

        if isinstance(payload, dict) and "quant_validation" in payload:
            return payload["quant_validation"]

    def _load_forecasting_config(self) -> Dict[str, Any]:
        """
        Load forecasting configuration to preserve ensemble_kwargs during CV.
        Phase 7.4 FIX: Prevents empty ensemble config when creating forecasters for CV.
        """
        path = getattr(self, "_forecasting_config_path", None)
        if not path or not path.exists():
            logger.warning("Forecasting config not found at %s, ensemble_kwargs will be empty", path)
            return {}

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except Exception as exc:
            logger.warning("Unable to read forecasting config %s: %s", path, exc)
            return {}

        if isinstance(payload, dict):
            if "forecasting" in payload:
                nested = payload["forecasting"]
                return nested if isinstance(nested, dict) else {}
            return payload

        return {}

    def _load_signal_routing_config(self) -> Optional[Dict[str, Any]]:
        """Phase 9: Load the full signal_routing section from signal_routing_config.yml."""
        cfg_path = Path("config/signal_routing_config.yml")
        if not cfg_path.exists():
            return None
        try:
            with cfg_path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            routing = payload.get("signal_routing") or {}
            return dict(routing) if isinstance(routing, dict) else None
        except Exception as exc:
            logger.debug("Unable to load signal routing config: %s", exc)
            return None

    def _directional_gate_enabled(self) -> bool:
        """Phase 9: Read feature flag from signal_routing_config.yml."""
        try:
            cfg = self._signal_routing_config or {}
            if not isinstance(cfg, dict):
                logger.warning(
                    "DirectionalGate: signal_routing_config is not a dict (type=%s); gate disabled",
                    type(cfg).__name__,
                )
                return False
            dc_cfg = cfg.get("directional_classifier") or {}
            if not isinstance(dc_cfg, dict):
                logger.warning(
                    "DirectionalGate: directional_classifier section is not a dict; gate disabled"
                )
                return False
            return bool(dc_cfg.get("enabled", False))
        except Exception as exc:
            logger.error("DirectionalGate: failed to read config: %s", exc)
            return False

    # Sentinel value written to self._directional_classifier after a failed import
    # so we don't retry the import on every subsequent signal call.
    _DIRECTIONAL_CLASSIFIER_FAILED = object()

    def _score_directional(self, features: Dict[str, Any]) -> Optional[float]:
        """Phase 9: Score directional probability. Returns None when gate is disabled or cold-start."""
        if not self._directional_gate_enabled():
            return None
        if self._directional_classifier is self._DIRECTIONAL_CLASSIFIER_FAILED:
            return None  # previous import failed — do not retry
        if self._directional_classifier is None:
            try:
                from forcester_ts.directional_classifier import DirectionalClassifier
                self._directional_classifier = DirectionalClassifier()
            except Exception as exc:
                logger.error(
                    "DirectionalGate: failed to import/load DirectionalClassifier: %s — "
                    "gate will be skipped for this session",
                    exc,
                )
                self._directional_classifier = self._DIRECTIONAL_CLASSIFIER_FAILED
                return None
        return self._directional_classifier.score(features)

    def _build_forecast_edge_forecaster_config(
        self,
        *,
        horizon: int,
        baseline_key: str,
        fast_intraday_cv: bool,
        interval_hint: Optional[str],
    ) -> TimeSeriesForecasterConfig:
        """Build the CV forecaster config from the same shared forecasting config as runtime."""
        root_cfg = self._forecasting_config if isinstance(self._forecasting_config, dict) else {}

        def _cfg_section(name: str) -> Dict[str, Any]:
            section = root_cfg.get(name, {})
            return section if isinstance(section, dict) else {}

        ensemble_cfg = _cfg_section("ensemble")
        regime_cfg = _cfg_section("regime_detection")
        order_learning_cfg = _cfg_section("order_learning")
        monte_carlo_cfg = _cfg_section("monte_carlo")
        sarimax_cfg = _cfg_section("sarimax")
        garch_cfg = _cfg_section("garch")
        samossa_cfg = _cfg_section("samossa")
        mssa_cfg = _cfg_section("mssa_rl")

        ensemble_kwargs = {k: v for k, v in ensemble_cfg.items() if k != "enabled"}
        regime_detection_enabled = bool(regime_cfg.get("enabled", False))
        regime_detection_kwargs = {k: v for k, v in regime_cfg.items() if k != "enabled"}
        sarimax_kwargs = {k: v for k, v in sarimax_cfg.items() if k != "enabled"}
        garch_kwargs = {k: v for k, v in garch_cfg.items() if k != "enabled"}
        samossa_kwargs = {k: v for k, v in samossa_cfg.items() if k != "enabled"}
        mssa_kwargs = {k: v for k, v in mssa_cfg.items() if k != "enabled"}

        if fast_intraday_cv:
            resolve_best_single = baseline_key == "best_single"
            sarimax_enabled = resolve_best_single or baseline_key == "sarimax"
            samossa_enabled = resolve_best_single or baseline_key == "samossa"
            mssa_enabled = resolve_best_single or baseline_key == "mssa_rl"
            garch_enabled = resolve_best_single or baseline_key == "garch"

            if sarimax_enabled:
                sarimax_kwargs.update(_pmx_intraday_sarimax_kwargs(interval_hint))
            else:
                sarimax_kwargs = {}
            if samossa_enabled:
                samossa_kwargs = {**samossa_kwargs, "forecast_horizon": int(horizon)}
            else:
                samossa_kwargs = {}
            if mssa_enabled:
                mssa_kwargs = {**mssa_kwargs, "forecast_horizon": int(horizon)}
                if "use_gpu" not in mssa_kwargs:
                    mssa_kwargs["use_gpu"] = bool(_pmx_env_flag("MSSA_RL_USE_GPU") or False)
            else:
                mssa_kwargs = {}
            if not garch_enabled:
                garch_kwargs = {}

            return TimeSeriesForecasterConfig(
                forecast_horizon=horizon,
                sarimax_enabled=sarimax_enabled,
                samossa_enabled=samossa_enabled,
                mssa_rl_enabled=mssa_enabled,
                garch_enabled=garch_enabled,
                ensemble_enabled=True,
                sarimax_kwargs=sarimax_kwargs,
                garch_kwargs=garch_kwargs,
                samossa_kwargs=samossa_kwargs,
                mssa_rl_kwargs=mssa_kwargs,
                ensemble_kwargs=ensemble_kwargs,
                regime_detection_enabled=regime_detection_enabled,
                regime_detection_kwargs=regime_detection_kwargs,
                order_learning_config=order_learning_cfg,
                monte_carlo_config=monte_carlo_cfg,
            )

        if samossa_cfg.get("enabled", True):
            samossa_kwargs = {**samossa_kwargs, "forecast_horizon": int(horizon)}
        else:
            samossa_kwargs = {}
        if mssa_cfg.get("enabled", True):
            mssa_kwargs = {**mssa_kwargs, "forecast_horizon": int(horizon)}
        else:
            mssa_kwargs = {}

        return TimeSeriesForecasterConfig(
            forecast_horizon=horizon,
            sarimax_enabled=bool(sarimax_cfg.get("enabled", False)),
            garch_enabled=bool(garch_cfg.get("enabled", True)),
            samossa_enabled=bool(samossa_cfg.get("enabled", True)),
            mssa_rl_enabled=bool(mssa_cfg.get("enabled", True)),
            ensemble_enabled=bool(ensemble_cfg.get("enabled", True)),
            sarimax_kwargs=sarimax_kwargs,
            garch_kwargs=garch_kwargs,
            samossa_kwargs=samossa_kwargs,
            mssa_rl_kwargs=mssa_kwargs,
            ensemble_kwargs=ensemble_kwargs,
            regime_detection_enabled=regime_detection_enabled,
            regime_detection_kwargs=regime_detection_kwargs,
            order_learning_config=order_learning_cfg,
            monte_carlo_config=monte_carlo_cfg,
        )

    def _load_execution_cost_model(self) -> Dict[str, Any]:
        """Load LOB/execution cost model configuration from disk when available.

        Merges two sources (lower wins to higher):
          1. config/execution_cost_model.yml  -- LOB / fill-cost model
          2. config/signal_routing_config.yml signal_routing.time_series.cost_model
             -- SNR gate + roundtrip cost overrides

        This ensures min_signal_to_noise is active even when TSSG is
        constructed directly (not via build_signal_generator()).
        """
        result: Dict[str, Any] = {}

        # Source 1: LOB execution cost model
        ecm_path = Path("config") / "execution_cost_model.yml"
        if ecm_path.exists():
            try:
                with ecm_path.open("r", encoding="utf-8") as handle:
                    payload = yaml.safe_load(handle) or {}
                if isinstance(payload, dict):
                    ecm = payload.get("execution_cost_model") or payload
                    if isinstance(ecm, dict):
                        result.update(ecm)
            except Exception as exc:  # pragma: no cover
                logger.debug("Unable to load execution cost model config: %s", exc)

        # Source 2: signal_routing_config cost_model section (has min_signal_to_noise)
        src_path = Path("config") / "signal_routing_config.yml"
        if src_path.exists():
            try:
                with src_path.open("r", encoding="utf-8") as handle:
                    src_payload = yaml.safe_load(handle) or {}
                routing_cm = (
                    (src_payload.get("signal_routing") or {})
                    .get("time_series", {})
                    .get("cost_model", {})
                )
                if isinstance(routing_cm, dict):
                    # Only pull scalar gate params; don't override LOB depth profiles
                    for key in (MIN_SIGNAL_TO_NOISE_KEY, "default_roundtrip_cost_bps"):
                        if key in routing_cm and key not in result:
                            result[key] = routing_cm[key]
                    floors = load_floored_thresholds(src_path)
                    if MIN_SIGNAL_TO_NOISE_KEY in floors:
                        result[MIN_SIGNAL_TO_NOISE_KEY] = float(floors[MIN_SIGNAL_TO_NOISE_KEY])
            except Exception as exc:  # pragma: no cover
                logger.debug("Unable to load signal routing cost model: %s", exc)

        return result

    @staticmethod
    def _normalize_forecast_payload(payload: Any) -> Optional[Dict[str, Any]]:
        """Return a dict-based payload or None if the payload is empty."""
        if payload is None:
            return None
        if isinstance(payload, dict):
            return payload if payload else None
        if isinstance(payload, pd.Series) and not payload.empty:
            return {"forecast": payload}
        return None

    def _has_payload(self, payload: Any) -> bool:
        """Safely determine whether a forecast payload contains information."""
        return self._normalize_forecast_payload(payload) is not None

    def _resolve_primary_forecast(
        self,
        forecast_bundle: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Resolve the primary forecast payload and selected source key.

        Routing contract:
        - Use `ensemble_forecast` when the forecaster default is ENSEMBLE.
        - Use `mean_forecast` when default routing points to a single model.
        - Fall back through available model payloads defensively.
        """
        default_model = str(forecast_bundle.get("default_model") or "").strip().upper()
        ensemble_meta = forecast_bundle.get("ensemble_metadata") or {}
        allow_ensemble_default = bool(ensemble_meta.get("allow_as_default", True))

        ordered_keys: List[str] = []
        if default_model == "ENSEMBLE" and allow_ensemble_default:
            ordered_keys.append("ensemble_forecast")
        ordered_keys.append("mean_forecast")
        ordered_keys.extend(
            [
                "ensemble_forecast",
                "sarimax_forecast",
                "samossa_forecast",
                "mssa_rl_forecast",
                "garch_forecast",
            ]
        )

        seen: set[str] = set()
        for key in ordered_keys:
            if key in seen:
                continue
            seen.add(key)
            normalized = self._normalize_forecast_payload(forecast_bundle.get(key))
            if normalized:
                return normalized, key
        return None, ""

    def generate_signal(self,
                       forecast_bundle: Dict[str, Any],
                       current_price: float,
                       ticker: str,
                       market_data: Optional[pd.DataFrame] = None) -> TimeSeriesSignal:
        """
        Generate trading signal from time series forecast bundle.

        Args:
            forecast_bundle: Output from TimeSeriesForecaster.forecast()
            current_price: Current market price
            ticker: Stock ticker symbol
            market_data: Optional historical data for context

        Returns:
            TimeSeriesSignal with action, confidence, and risk metrics
        """
        # Phase 7.13-A2: track current ticker for globally unique ts_signal_id construction
        self._current_ticker = ticker or "unknown"
        try:
            # Resolve forecast source selected by the forecaster routing contract.
            primary_forecast, forecast_source = self._resolve_primary_forecast(forecast_bundle)
            if primary_forecast is None:
                logger.warning(f"No primary forecast available for {ticker}, returning HOLD")
                return self._create_hold_signal(ticker, current_price, "No forecast available")

            # Get forecast target value (horizon-consistent: horizon-end by default).
            forecast_value = self._extract_forecast_value(primary_forecast)
            if forecast_value is None:
                return self._create_hold_signal(ticker, current_price, "Invalid forecast value")
            if current_price <= 0:
                return self._create_hold_signal(ticker, current_price, "Invalid current price")

            # Calculate expected return
            expected_return = (forecast_value / current_price) - 1
            if abs(expected_return) > 0.1:
                logger.debug(
                    "Clamping extreme expected return for %s (raw=%.2f%%)",
                    ticker,
                    expected_return * 100,
                )
                expected_return = float(np.clip(expected_return, -0.1, 0.1))

            thresholds = self._resolve_thresholds_for_ticker(ticker)
            confidence_threshold = thresholds["confidence_threshold"]
            min_expected_return = thresholds["min_expected_return"]
            max_risk_score = thresholds["max_risk_score"]
            execution_mode = self._coerce_nonempty_str(
                forecast_bundle.get("execution_mode")
                or os.getenv("EXECUTION_MODE")
                or os.getenv("PMX_EXECUTION_MODE")
            )
            if execution_mode is None:
                execution_mode = self._runtime_execution_mode
            execution_mode = execution_mode.lower()
            proof_mode_raw = forecast_bundle.get("proof_mode")
            if proof_mode_raw is None:
                proof_mode_raw = _pmx_env_flag("PMX_PROOF_MODE")
            proof_mode = self._to_bool(proof_mode_raw, default=False)
            run_id = self._coerce_nonempty_str(
                forecast_bundle.get("run_id")
                or os.getenv("PMX_RUN_ID")
                or os.getenv("RUN_ID")
            )
            if run_id is None:
                run_id = self._runtime_run_id
            pipeline_id = self._coerce_nonempty_str(
                forecast_bundle.get("pipeline_id")
                or os.getenv("PORTFOLIO_MAXIMIZER_PIPELINE_ID")
                or os.getenv("PIPELINE_ID")
            )
            if pipeline_id is None:
                pipeline_id = self._runtime_pipeline_id or run_id

            friction = self._estimate_roundtrip_friction(
                ticker=ticker,
                market_data=market_data,
            )
            roundtrip_cost = float(friction["roundtrip_cost_fraction"])
            gross_trade_return = abs(expected_return)
            net_trade_return = max(0.0, gross_trade_return - roundtrip_cost)
            direction = float(np.sign(expected_return))
            net_expected_return = float(direction * net_trade_return) if direction != 0 else 0.0

            # Extract volatility for risk assessment (ensure scalar)
            volatility_forecast = forecast_bundle.get('volatility_forecast') or {}
            if isinstance(volatility_forecast, dict):
                volatility = self._to_scalar(volatility_forecast.get('volatility'))
            else:
                volatility = self._to_scalar(volatility_forecast)

            # For 1-day signals, step-1 CI == terminal CI; they're identical.
            # For multi-step signals, use terminal CI so SNR reflects full horizon uncertainty.
            # Exception: when MSSA_RL is the DISABLE_DEFAULT fallback, use step-1 CI.
            # MSSA_RL's sqrt(step+1) growth over-widens CI for range-bound periods;
            # the near-term directional call is the relevant signal edge in fallback mode.
            _forecast_horizon = getattr(self, "_forecast_horizon", None) or self.config.get(
                "forecast_horizon", 30
            ) if hasattr(self, "config") and isinstance(self.config, dict) else 30
            _effective_default = str(forecast_bundle.get("default_model") or "").strip().upper()
            # Only trust default_model label when the resolver actually used mean_forecast.
            # If mean_forecast was missing and the resolver fell through to another model
            # payload, default_model is stale and applying MSSA_RL-specific logic would
            # be a wiring mismatch (softer gate applied to a different model's CI).
            _mssa_rl_fallback = (
                _effective_default == "MSSA_RL" and forecast_source == "mean_forecast"
            )
            if isinstance(_forecast_horizon, int) and _forecast_horizon <= 1:
                lower_ci, upper_ci = self._extract_ci_bounds_step1(primary_forecast)
            elif _mssa_rl_fallback:
                lower_ci, upper_ci = self._extract_ci_bounds_step1(primary_forecast)
            else:
                lower_ci, upper_ci = self._extract_ci_bounds(primary_forecast)
            snr = self._estimate_signal_to_noise(
                current_price=current_price,
                expected_return=expected_return,
                lower_ci=lower_ci,
                upper_ci=upper_ci,
            )
            # Horizon-adjusted SNR threshold (Fix D):
            # The SNR formula compares a multi-bar cumulative expected_return against
            # a multi-bar CI half-width.  Both scale with horizon, but the threshold
            # was calibrated for 5-bar short-term signals.  Applying a 5-bar threshold
            # to a 30-bar CI systematically over-blocks medium-horizon signals whose
            # expected return per bar is well above cost but whose absolute CI width
            # is wider by sqrt(horizon/ref_horizon).
            #
            # Adjustment: threshold_effective = base_threshold / sqrt(horizon / ref_horizon)
            # At horizon=5: no change.  At horizon=30: 1.5 / sqrt(6) ≈ 0.612.
            # This is not a threshold dodge — it adjusts for the geometric fact that a
            # 30-bar CI has more noise than a 5-bar CI by exactly sqrt(30/5).
            # The MSSA_RL fallback cap (1.0) is applied after the horizon adjustment.
            _snr_ref_horizon = 5  # reference horizon the 1.5 threshold was calibrated for
            _horizon_int = int(_forecast_horizon) if isinstance(_forecast_horizon, int) else 5
            _horizon_scale = float(np.sqrt(max(_horizon_int, _snr_ref_horizon) / _snr_ref_horizon))
            _base_snr_threshold = self._min_signal_to_noise
            _snr_threshold_adjusted = _base_snr_threshold / _horizon_scale if _horizon_scale > 1.0 else _base_snr_threshold
            _snr_threshold = (
                min(_snr_threshold_adjusted, 1.0)
                if _mssa_rl_fallback
                else _snr_threshold_adjusted
            )
            # Pre-assign signal_id so that all gate log records reference the same
            # ID as the final TimeSeriesSignal object.  Counter is incremented here
            # (before any gate fires) so the ID is stable whether the signal is
            # HOLD, BUY, or SELL — every traversal of this path gets an ID.
            self._signal_counter += 1
            _pending_signal_id = self._make_ts_signal_id()

            _snr_gate_blocked = False
            if snr is not None and _snr_threshold > 0 and snr < _snr_threshold:
                logger.info(
                    "[SNR_GATE] %s: SNR %.3f < threshold %.3f (adjusted from %.3f for horizon=%d, mssa_rl_fallback=%s) "
                    "— zeroing net return (CI too wide relative to expected return; signal suppressed)",
                    ticker, snr, _snr_threshold, _base_snr_threshold, _horizon_int, _mssa_rl_fallback,
                )
                self._log_gate_result(
                    "snr", _pending_signal_id, ticker,
                    value=round(snr, 6), threshold=round(_snr_threshold, 6), result="FAIL",
                    base_threshold=round(_base_snr_threshold, 6),
                    horizon=_horizon_int, mssa_rl_fallback=_mssa_rl_fallback,
                )
                net_trade_return = 0.0
                net_expected_return = 0.0
                _snr_gate_blocked = True
            elif _mssa_rl_fallback and snr is None:
                # MSSA_RL's baseline_variance should always produce a finite CI.
                # SNR=None here means zero-width or degenerate CI — block rather than
                # silently pass, which would be a threshold dodge via missing data.
                logger.warning(
                    "[SNR_GATE] %s: MSSA_RL fallback has degenerate CI (SNR=None) — "
                    "zeroing net return (baseline_variance likely zero)",
                    ticker,
                )
                self._log_gate_result(
                    "snr", _pending_signal_id, ticker,
                    value=None, threshold=round(_snr_threshold, 6), result="FAIL",
                    gate_detail="mssa_rl_degenerate_ci",
                )
                net_trade_return = 0.0
                net_expected_return = 0.0
                _snr_gate_blocked = True
            elif snr is None and self._min_signal_to_noise > 0:
                # Non-MSSA_RL path: CI unavailable, gate cannot fire, but log so the
                # absence of SNR filtering is visible in diagnostics (not a silent pass).
                logger.debug(
                    "[SNR_GATE] %s: SNR unavailable (CI missing or degenerate) — "
                    "gate skipped for model=%s",
                    ticker, _effective_default or "unknown",
                )
                self._log_gate_result(
                    "snr", _pending_signal_id, ticker,
                    value=None, threshold=round(_snr_threshold, 6), result="UNKNOWN",
                    gate_detail="ci_unavailable",
                )
            elif snr is not None and _snr_threshold > 0:
                # SNR gate evaluated and passed — emit explicit PASS so funnel audits
                # can verify this step was not skipped.
                self._log_gate_result(
                    "snr", _pending_signal_id, ticker,
                    value=round(snr, 6), threshold=round(_snr_threshold, 6), result="PASS",
                    horizon=_horizon_int, mssa_rl_fallback=_mssa_rl_fallback,
                )

            model_agreement = self._check_model_agreement(forecast_bundle)
            diagnostics = self._evaluate_diagnostics_details(forecast_bundle)

            # Phase 9: extract classifier features while all signal-time data is available.
            _clf_features = self._extract_classifier_features(
                forecast_bundle=forecast_bundle,
                current_price=current_price,
                expected_return=expected_return,
                lower_ci=lower_ci,
                upper_ci=upper_ci,
                snr=snr,
                model_agreement=model_agreement,
                market_data=market_data,
            )

            # Calculate confidence score (discriminative: net edge + uncertainty + model quality).
            _diag_score_raw = diagnostics.get("score")
            if _diag_score_raw is None:
                logger.warning(
                    "diagnostics_score missing from forecast diagnostics for ticker=%s; "
                    "using 0.0 (pessimistic fallback — penalises forecasts with absent diagnostics)",
                    ticker,
                )
                _diagnostics_score = 0.0
            else:
                _diagnostics_score = float(_diag_score_raw)
            confidence = self._calculate_confidence(
                expected_return=expected_return,
                net_trade_return=net_trade_return,
                min_expected_return=min_expected_return,
                volatility=volatility,
                model_agreement=model_agreement,
                diagnostics_score=_diagnostics_score,
                snr=snr,
                ticker=ticker,
            )

            # Calculate risk score
            risk_score = self._calculate_risk_score(
                expected_return,
                volatility,
                forecast_bundle
            )

            if snr is not None and snr < 0.5:
                risk_score = min(1.0, risk_score + 0.10)

            change_point_info = self._summarize_recent_change_points(
                forecast_bundle=forecast_bundle,
                market_data=market_data,
            )
            if change_point_info:
                # Treat fresh change-points as elevated regime uncertainty.
                recent_days = change_point_info.get("recent_change_point_days")
                if isinstance(recent_days, int) and recent_days <= 10:
                    risk_score = min(1.0, risk_score + 0.10)
                    confidence = max(0.0, confidence - 0.05)

            action, _hold_reason = self._determine_action(
                expected_return=expected_return,
                net_trade_return=net_trade_return,
                confidence=confidence,
                risk_score=risk_score,
                confidence_threshold=confidence_threshold,
                min_expected_return=min_expected_return,
                max_risk_score=max_risk_score,
            )

            # --- Structured gate log records for confidence / return / risk gates ---
            # Emit one record per gate regardless of outcome so every signal is traceable.
            self._log_gate_result(
                "confidence", _pending_signal_id, ticker,
                value=round(confidence, 6), threshold=round(confidence_threshold, 6),
                result="FAIL" if _hold_reason == "CONFIDENCE_BELOW_THRESHOLD" else "PASS",
            )
            self._log_gate_result(
                "min_return", _pending_signal_id, ticker,
                value=round(net_trade_return, 6), threshold=round(min_expected_return, 6),
                result="FAIL" if _hold_reason == "MIN_RETURN" else "PASS",
            )
            self._log_gate_result(
                "risk", _pending_signal_id, ticker,
                value=round(risk_score, 6), threshold=round(max_risk_score, 6),
                result="FAIL" if _hold_reason == "RISK_TOO_HIGH" else "PASS",
            )

            # Phase 9: directional gate (inactive by default; enabled via config).
            _p_up = self._score_directional(_clf_features)
            _directional_gate_applied = False
            if _p_up is not None and action != "HOLD":
                _dc_cfg = (self._signal_routing_config or {}).get("directional_classifier") or {}
                _threshold_buy = float(_dc_cfg.get("p_up_threshold_buy", 0.55))
                _threshold_sell = float(_dc_cfg.get("p_up_threshold_sell", 0.55))
                if action == "BUY" and _p_up < _threshold_buy:
                    logger.info(
                        "Phase 9 directional gate: demoting BUY to HOLD for %s (p_up=%.3f < %.3f)",
                        ticker, _p_up, _threshold_buy,
                    )
                    action = "HOLD"
                    _directional_gate_applied = True
                elif action == "SELL" and _p_up > (1.0 - _threshold_sell):
                    logger.info(
                        "Phase 9 directional gate: demoting SELL to HOLD for %s (p_up=%.3f > %.3f)",
                        ticker, _p_up, 1.0 - _threshold_sell,
                    )
                    action = "HOLD"
                    _directional_gate_applied = True

            # Calculate target and stop loss (market_data enables ATR-based stops)
            target_price, stop_loss = self._calculate_targets(
                current_price,
                forecast_value,
                volatility,
                action,
                market_data=market_data,
            )

            # Build reasoning
            reasoning = self._build_reasoning(
                action,
                expected_return,
                net_trade_return,
                float(friction.get("roundtrip_cost_bps", roundtrip_cost * 1e4)),
                confidence,
                risk_score,
                forecast_bundle
            )

            # Extract provenance metadata
            provenance = self._extract_provenance(
                forecast_bundle,
                selected_source=forecast_source,
            )
            weather_context = extract_weather_context(
                market_data,
                ticker=ticker,
            )
            if weather_context:
                provenance["weather_context"] = weather_context

            provenance["execution_friction"] = friction
            provenance["thresholds"] = {
                "confidence_threshold": confidence_threshold,
                "min_expected_return": min_expected_return,
                "max_risk_score": max_risk_score,
            }
            provenance["diagnostics"] = diagnostics
            if isinstance(diagnostics.get("reasons"), list) and diagnostics.get("reasons"):
                score = self._safe_float(diagnostics.get("score"))
                reasons = [r for r in diagnostics.get("reasons") if isinstance(r, str)]
                noisy = {"missing_regression_metrics", "missing_weighted_model_confidence"}
                filtered = [r for r in reasons if r not in noisy]
                if (score is not None and score < 0.45) and filtered:
                    provenance["why_low_quality"] = filtered[:5]
            provenance["model_agreement"] = model_agreement
            if snr is not None:
                provenance["decision_context_snr"] = snr
            if change_point_info:
                provenance["mssa_rl_change_points"] = change_point_info
            provenance["execution_mode"] = execution_mode
            provenance["proof_mode"] = proof_mode
            provenance["pipeline_id"] = str(pipeline_id) if pipeline_id is not None else None
            if run_id is not None:
                provenance["run_id"] = str(run_id)
            provenance["classifier_features"] = _clf_features
            if _p_up is not None:
                provenance["p_up"] = _p_up
            if _directional_gate_applied:
                provenance["directional_gate_applied"] = True
            if _snr_gate_blocked:
                provenance["snr_gate_blocked"] = True
                provenance["snr_gate_threshold"] = self._min_signal_to_noise

            # Structured HOLD reason — disambiguates why action='HOLD' was chosen.
            # Allows aggregating HOLD causes across runs without parsing log strings.
            # Codes: SNR_GATE | CONFIDENCE_BELOW_THRESHOLD | MIN_RETURN | RISK_TOO_HIGH
            #        | ZERO_EXPECTED_RETURN | DIRECTIONAL_GATE | QUANT_VALIDATION_FAIL
            if action == "HOLD":
                if _snr_gate_blocked:
                    provenance["hold_reason"] = "SNR_GATE"
                elif _directional_gate_applied:
                    provenance["hold_reason"] = "DIRECTIONAL_GATE"
                elif _hold_reason is not None:
                    provenance["hold_reason"] = _hold_reason
                else:
                    provenance["hold_reason"] = "UNKNOWN"

            provenance['decision_context'] = {
                'expected_return': expected_return,
                'expected_return_net': net_expected_return,
                'gross_trade_return': gross_trade_return,
                'net_trade_return': net_trade_return,
                'roundtrip_cost_fraction': roundtrip_cost,
                'roundtrip_cost_bps': float(friction.get("roundtrip_cost_bps", roundtrip_cost * 1e4)),
                'confidence': confidence,
                'risk_score': risk_score,
                'volatility': volatility,
                'signal_to_noise': snr,
                'execution_mode': execution_mode,
                'proof_mode': proof_mode,
                'pipeline_id': str(pipeline_id) if pipeline_id is not None else None,
                'run_id': str(run_id) if run_id is not None else None,
            }

            # Counter was pre-incremented before the first gate; use the pre-assigned ID.
            signal = TimeSeriesSignal(
                ticker=ticker,
                action=action,
                confidence=confidence,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                signal_timestamp=utc_now(),
                model_type=provenance.get('primary_model', 'ENSEMBLE'),
                forecast_horizon=forecast_bundle.get('horizon', 30),
                expected_return=expected_return,
                risk_score=risk_score,
                reasoning=reasoning,
                provenance=provenance,
                signal_type='TIME_SERIES',
                volatility=volatility,
                lower_ci=lower_ci,
                upper_ci=upper_ci,
                signal_id=_pending_signal_id,  # pre-assigned before gates fired; gate logs reference this ID
                # B5: pure Platt-scaled probability from the most recent calibration call
                confidence_calibrated=getattr(self, '_platt_calibrated', None),
                p_up=_p_up,
                directional_gate_applied=_directional_gate_applied,
            )

            quant_profile = self._build_quant_success_profile(
                ticker=ticker,
                market_data=market_data,
                signal=signal,
            )
            if quant_profile:
                status = str(quant_profile.get("status") or "").upper()
                # Attach full profile for downstream inspection.
                signal.provenance["quant_validation"] = quant_profile
                summary = f"QuantValidation={status}"
                failed = quant_profile.get("failed_criteria") or []
                if failed:
                    summary += f" ({','.join(failed)})"
                signal.reasoning = summary if not signal.reasoning else f"{signal.reasoning} | {summary}"

                # When quant validation is enabled, treat FAILED profiles as a hard
                # gate for new TS trades so that only regimes clearing the configured
                # barbell objective and hard economic/statistical blockers can open
                # positions. In diagnostic mode this gate is disabled via
                # _quant_validation_enabled.
                #
                # Phase 7.9: Added advisory mode support via risk_mode_loader.
                # In advisory mode, we log the failure but don't demote to HOLD.
                if self._quant_validation_enabled and status == "FAIL" and action != "HOLD":
                    # Check if advisory mode is enabled via risk mode config
                    advisory_mode = False
                    if RISK_MODE_AVAILABLE and is_quant_validation_advisory is not None:
                        advisory_mode = is_quant_validation_advisory()

                    if advisory_mode:
                        # Advisory mode: log warning but allow trade through
                        mode_name = get_active_mode() if get_active_mode else "research_production"
                        logger.info(
                            "[ADVISORY] Quant validation FAILED for %s (%s signal) - risk_mode=%s allows execution.",
                            ticker,
                            action,
                            mode_name,
                        )
                        signal.reasoning = f"{signal.reasoning or ''} | [ADVISORY] QuantFail (allowed in {mode_name} mode)"
                    else:
                        # Hard fail mode: demote to HOLD
                        logger.info(
                            "Quant validation FAILED for %s; demoting %s signal to HOLD to protect PnL.",
                            ticker,
                            action,
                        )
                        action = "HOLD"
                        signal.action = "HOLD"
                        signal.provenance["hold_reason"] = "QUANT_VALIDATION_FAIL"

                self._log_quant_validation(
                    ticker=ticker,
                    signal=signal,
                    quant_profile=quant_profile,
                    market_data=market_data,
                )

                # --- Structured gate records for quant gate ---
                _qv_result = status if status in {"PASS", "FAIL", "SKIPPED"} else "UNKNOWN"
                _qv_result_normalised = "SKIP" if _qv_result == "SKIPPED" else _qv_result
                _utility_score = quant_profile.get("utility_score")
                _pass_threshold = quant_profile.get("pass_threshold")
                self._log_gate_result(
                    "quant", _pending_signal_id, ticker,
                    value=round(float(_utility_score), 6) if _utility_score is not None else None,
                    threshold=round(float(_pass_threshold), 6) if _pass_threshold is not None else None,
                    result=_qv_result_normalised,
                    failed_criteria=failed or [],
                )
                # Per-criterion records: one line per component in utility_breakdown
                # so funnel audits can show exactly which criterion flipped the gate.
                for _criterion, _breakdown in (quant_profile.get("utility_breakdown") or {}).items():
                    _crit_val = _breakdown.get("raw_value")
                    _crit_thr = _breakdown.get("threshold")
                    _crit_passed = _breakdown.get("passed_threshold")
                    _crit_result = (
                        "PASS" if _crit_passed is True
                        else "FAIL" if _crit_passed is False
                        else "UNKNOWN"
                    )
                    self._log_gate_result(
                        "quant_criterion", _pending_signal_id, ticker,
                        value=round(float(_crit_val), 6) if isinstance(_crit_val, (int, float)) and _crit_val is not None else _crit_val,
                        threshold=round(float(_crit_thr), 6) if isinstance(_crit_thr, (int, float)) and _crit_thr is not None else _crit_thr,
                        result=_crit_result,
                        criterion=_criterion,
                        weight=_breakdown.get("weight"),
                    )

            if action == 'HOLD':
                logger.info(
                    "Generated HOLD signal for %s: confidence=%.2f, expected_return=%.2f%%, risk=%.2f",
                    ticker,
                    confidence,
                    expected_return * 100,
                    risk_score,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Holding %s due to thresholds (expected_return=%.4f, confidence=%.2f, risk=%.2f)",
                        ticker,
                        expected_return,
                        confidence,
                        risk_score,
                    )
            else:
                logger.info(
                    "Generated %s signal for %s: confidence=%.2f, expected_return=%.2f%%, risk=%.2f",
                    action,
                    ticker,
                    confidence,
                    expected_return * 100,
                    risk_score,
                )

            return signal

        except Exception as e:
            logger.exception("Error generating Time Series signal for %s: %s", ticker, e)
            return self._create_hold_signal(ticker, current_price, f"Error: {str(e)}")


    def _extract_forecast_value(self, forecast: Dict[str, Any]) -> Optional[float]:
        """Extract forecast value from forecast dictionary"""
        if isinstance(forecast, dict):
            # Try different keys
            if 'forecast' in forecast:
                forecast_series = forecast['forecast']
                if isinstance(forecast_series, pd.Series):
                    if forecast_series.empty:
                        return None
                    cleaned = forecast_series.dropna()
                    if cleaned.empty:
                        return None
                    # Horizon-consistent target: use the horizon-end forecast.
                    return float(cleaned.iloc[-1])
                elif isinstance(forecast_series, (list, np.ndarray)):
                    if len(forecast_series) == 0:
                        return None
                    return float(forecast_series[-1])
                elif isinstance(forecast_series, (int, float)):
                    return float(forecast_series)
            elif 'mean' in forecast:
                return float(forecast['mean'])
            elif 'value' in forecast:
                return float(forecast['value'])

        return None

    def _to_scalar(self, value: Any) -> Optional[float]:
        """Convert Series/array-like values into a scalar float."""
        if value is None:
            return None
        if isinstance(value, pd.Series):
            if value.empty:
                return None
            return float(value.iloc[0])
        if isinstance(value, (np.ndarray, list, tuple)):
            if len(value) == 0:
                return None
            return float(value[0])
        if isinstance(value, (np.generic, float, int)):
            return float(value)
        return None

    @staticmethod
    def _to_bool(value: Any, *, default: bool = False) -> bool:
        """Convert mixed env/config values to boolean."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "y"}:
                return True
            if normalized in {"0", "false", "no", "off", "n"}:
                return False
        return default

    def _resolve_thresholds_for_ticker(self, ticker: str) -> Dict[str, Any]:
        """Resolve per-ticker overrides for routing thresholds."""
        thresholds = {
            "confidence_threshold": float(self.confidence_threshold),
            "min_expected_return": float(self.min_expected_return),
            "max_risk_score": float(self.max_risk_score),
        }
        if self._diag_mode:
            return thresholds

        key = (ticker or "").strip()
        if not key:
            return thresholds

        raw = self._per_ticker_thresholds.get(key) or self._per_ticker_thresholds.get(key.upper())
        if not isinstance(raw, dict):
            return thresholds

        for field, cast in (
            ("confidence_threshold", float),
            ("min_expected_return", float),
            ("max_risk_score", float),
        ):
            if field in raw:
                try:
                    thresholds[field] = cast(raw[field])
                except (TypeError, ValueError):
                    continue
        return thresholds

    @staticmethod
    def _infer_asset_class(ticker: str) -> str:
        """Infer a coarse asset-class for cost defaults and routing diagnostics."""
        sym = (ticker or "").upper()
        if sym.endswith("=X"):
            return "FX"
        if sym.endswith("-USD") or sym in {"BTC", "ETH", "SOL"}:
            return "CRYPTO"
        if "^" in sym:
            return "INDEX"
        if any(sym.endswith(suffix) for suffix in (".NS", ".TW", ".L")):
            return "INTL_EQUITY"
        return "US_EQUITY"

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            out = float(value)
        except Exception:
            return None
        return out

    def _estimate_roundtrip_friction(
        self,
        *,
        ticker: str,
        market_data: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Estimate round-trip friction as a fraction of notional.

        Preference order:
        1) Synthetic microstructure columns (TxnCostBps, ImpactBps) when present.
        2) LOB simulation using configured depth/spread profiles when available.
        3) Bid/Ask spread when present.
        4) Asset-class default bps.
        """
        asset_class = self._infer_asset_class(ticker)
        default_bps = float(
            self._default_roundtrip_cost_bps.get(
                asset_class, self._default_roundtrip_cost_bps["UNKNOWN"]
            )
        )

        raw_lob_cfg = self._cost_model.get("lob") if isinstance(self._cost_model, dict) else {}
        lob_cfg = raw_lob_cfg or {}
        lob_enabled = bool(lob_cfg.get("enabled", False)) and simulate_market_order_fill is not None and LOBConfig is not None
        lob_config_obj: Optional[LOBConfig] = None
        if lob_enabled:
            try:
                depth_profiles = lob_cfg.get("depth_profiles")
                if depth_profiles is not None and not isinstance(depth_profiles, dict):
                    depth_profiles = None
                lob_config_obj = LOBConfig(
                    levels=int(lob_cfg.get("levels", 10)),
                    tick_size_bps=float(lob_cfg.get("tick_size_bps", 1.0)),
                    alpha=float(lob_cfg.get("alpha", 0.8)),
                    max_exhaust_levels=int(lob_cfg.get("max_exhaust_levels", 25)),
                    default_order_value=float(lob_cfg.get("default_order_value", 10_000.0)),
                    depth_profiles=depth_profiles,
                    tail_depth_multiplier=float(lob_cfg.get("tail_depth_multiplier", 1.0)),
                )
            except Exception:
                lob_config_obj = None

        if market_data is None or market_data.empty:
            return {
                "source": "default",
                "asset_class": asset_class,
                "roundtrip_cost_bps": default_bps,
                "roundtrip_cost_fraction": default_bps / 1e4,
            }

        last = market_data.iloc[-1]

        txn_bps = None
        impact_bps = 0.0
        if "TxnCostBps" in market_data.columns:
            txn_bps = self._safe_float(last.get("TxnCostBps") if hasattr(last, "get") else None)
        if txn_bps is not None and "ImpactBps" in market_data.columns:
            impact_val = self._safe_float(last.get("ImpactBps") if hasattr(last, "get") else None)
            impact_bps = impact_val if impact_val is not None else 0.0

        if txn_bps is not None:
            per_side_bps = max(0.0, float(txn_bps) + float(impact_bps))
            roundtrip_bps = 2.0 * per_side_bps
            return {
                "source": "microstructure",
                "asset_class": asset_class,
                "txn_cost_bps": float(txn_bps),
                "impact_bps": float(impact_bps),
                "roundtrip_cost_bps": float(roundtrip_bps),
                "roundtrip_cost_fraction": float(roundtrip_bps) / 1e4,
            }

        bid = self._safe_float(last.get("Bid") if hasattr(last, "get") else None)
        ask = self._safe_float(last.get("Ask") if hasattr(last, "get") else None)
        mid_price = None
        half_spread = None
        if bid is not None and ask is not None and bid > 0 and ask > bid:
            mid_price = 0.5 * (bid + ask)
            half_spread = 0.5 * (ask - bid)
        else:
            close_val = self._safe_float(last.get("Close") if hasattr(last, "get") else None)
            high_val = self._safe_float(last.get("High") if hasattr(last, "get") else None)
            low_val = self._safe_float(last.get("Low") if hasattr(last, "get") else None)
            if high_val is not None and low_val is not None and high_val > 0 and low_val > 0:
                mid_price = 0.5 * (high_val + low_val)
            elif close_val is not None:
                mid_price = close_val
            spread_val = self._safe_float(last.get("Spread") if hasattr(last, "get") else None)
            if spread_val is not None and spread_val > 0:
                half_spread = spread_val / 2.0

        depth_notional = self._safe_float(last.get("Depth") if hasattr(last, "get") else None)

        if lob_enabled and lob_config_obj is not None and mid_price is not None:
            try:
                fill = simulate_market_order_fill(
                    side="BUY",
                    mid_price=mid_price,
                    half_spread=half_spread,
                    depth_notional=depth_notional,
                    order_notional=None,
                    asset_class=asset_class,
                    config=lob_config_obj,
                )
                lob_roundtrip_bps = max(default_bps, abs(fill.mid_slippage_bps) * 2.0)
                return {
                    "source": "lob_sim",
                    "asset_class": asset_class,
                    "mid_price": mid_price,
                    "depth_notional": depth_notional,
                    "levels_consumed": fill.levels_consumed,
                    "roundtrip_cost_bps": float(lob_roundtrip_bps),
                    "roundtrip_cost_fraction": float(lob_roundtrip_bps) / 1e4,
                }
            except Exception as exc:
                logger.debug("LOB cost estimation failed, falling back: %s", exc)

        if bid is not None and ask is not None and bid > 0 and ask > bid:
            spread = ask - bid
            spread_bps = (spread / (0.5 * (bid + ask))) * 1e4 if (bid + ask) > 0 else default_bps
            roundtrip_bps = max(spread_bps, default_bps)
            return {
                "source": "bid_ask",
                "asset_class": asset_class,
                "bid": bid,
                "ask": ask,
                "spread_bps": float(spread_bps),
                "roundtrip_cost_bps": float(roundtrip_bps),
                "roundtrip_cost_fraction": float(roundtrip_bps) / 1e4,
            }

        return {
            "source": "default",
            "asset_class": asset_class,
            "roundtrip_cost_bps": default_bps,
            "roundtrip_cost_fraction": default_bps / 1e4,
        }

    def _extract_ci_bounds(
        self,
        forecast_payload: Optional[Dict[str, Any]],
    ) -> tuple[Optional[float], Optional[float]]:
        if not forecast_payload:
            return None, None
        # Use the terminal CI step (iloc[-1]) not step-1 (iloc[0]).
        # SNR must be evaluated at the actual trade horizon: a 5-day trade gated
        # against a step-1 CI is structurally too narrow and inflates SNR.
        raw_lower = forecast_payload.get("lower_ci")
        raw_upper = forecast_payload.get("upper_ci")
        lower_ci = self._to_scalar_terminal(raw_lower)
        upper_ci = self._to_scalar_terminal(raw_upper)
        if lower_ci is None or upper_ci is None:
            return lower_ci, upper_ci
        if not np.isfinite(lower_ci) or not np.isfinite(upper_ci):
            return None, None
        return float(lower_ci), float(upper_ci)

    def _extract_ci_bounds_step1(
        self,
        forecast_payload: Optional[Dict[str, Any]],
    ) -> tuple[Optional[float], Optional[float]]:
        """Return CI bounds at step-1 (iloc[0]) for single-bar / 1-day signals.

        Use this only when forecast_horizon == 1; for multi-step trades always
        use _extract_ci_bounds() (terminal step) so SNR reflects actual uncertainty.
        """
        if not forecast_payload:
            return None, None
        raw_lower = forecast_payload.get("lower_ci")
        raw_upper = forecast_payload.get("upper_ci")
        lower_ci = self._to_scalar(raw_lower)
        upper_ci = self._to_scalar(raw_upper)
        if lower_ci is None or upper_ci is None:
            return lower_ci, upper_ci
        if not np.isfinite(lower_ci) or not np.isfinite(upper_ci):
            return None, None
        return float(lower_ci), float(upper_ci)

    def _to_scalar_terminal(self, value: Any) -> Optional[float]:
        """Return the LAST element of a Series/array CI bound (terminal horizon step)."""
        if value is None:
            return None
        if isinstance(value, pd.Series):
            if value.empty:
                return None
            return float(value.iloc[-1])
        if isinstance(value, (np.ndarray, list, tuple)):
            if len(value) == 0:
                return None
            return float(value[-1])
        # Scalar — already horizon-independent
        if isinstance(value, (np.generic, float, int)):
            return float(value)
        return None

    @staticmethod
    def _estimate_signal_to_noise(
        *,
        current_price: float,
        expected_return: float,
        lower_ci: Optional[float],
        upper_ci: Optional[float],
        z_value: float = 1.96,
    ) -> Optional[float]:
        """Estimate signal-to-noise ratio using CI-implied sigma (approximate)."""
        if lower_ci is None or upper_ci is None:
            return None
        if current_price <= 0:
            return None
        width = float(upper_ci) - float(lower_ci)
        if not np.isfinite(width) or width <= 0:
            return None
        sigma_price = (width / 2.0) / max(z_value, 1e-6)
        sigma_return = sigma_price / float(current_price)
        if sigma_return <= 0 or not np.isfinite(sigma_return):
            return None
        return float(abs(expected_return) / sigma_return)

    @staticmethod
    def _summarize_recent_change_points(
        *,
        forecast_bundle: Dict[str, Any],
        market_data: Optional[pd.DataFrame],
    ) -> Optional[Dict[str, Any]]:
        """Summarize MSSA-RL change-point recency for risk adjustments."""
        payload = forecast_bundle.get("mssa_rl_forecast")
        if not isinstance(payload, dict) or not payload:
            return None
        change_points = payload.get("change_points")
        if change_points is None:
            return None
        if market_data is None or market_data.empty:
            return None
        try:
            end_ts = pd.to_datetime(market_data.index[-1])
        except Exception:
            return None

        cps: List[pd.Timestamp] = []
        if isinstance(change_points, pd.DatetimeIndex):
            cps = [pd.to_datetime(ts) for ts in change_points.to_list()]
        elif isinstance(change_points, list):
            for item in change_points:
                try:
                    cps.append(pd.to_datetime(item))
                except Exception:
                    continue
        else:
            try:
                cps = [pd.to_datetime(change_points)]
            except Exception:
                cps = []

        cps = [ts for ts in cps if ts is not None and pd.notna(ts)]
        if not cps:
            return {"count": 0, "recent_change_point_days": None}

        last_cp = max(cps)
        try:
            recent_days = int(abs((end_ts - last_cp).days))
        except Exception:
            recent_days = None

        return {
            "count": int(len(cps)),
            "last_change_point": str(last_cp),
            "recent_change_point_days": recent_days,
        }

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _calculate_confidence(
        self,
        *,
        expected_return: float,
        net_trade_return: float,
        min_expected_return: float,
        volatility: Optional[float],
        model_agreement: float,
        diagnostics_score: float,
        snr: Optional[float],
        ticker: str,
    ) -> float:
        """
        Discriminative confidence score (0.0..1.0).

        Confidence should track *actionable predictive edge*:
        - penalize tiny net edges after costs
        - downweight high-uncertainty forecasts (CI/SNR)
        - incorporate model quality/diagnostics and agreement
        """

        # 1) Edge score: 0 when net edge is tiny, 1 when it is meaningfully above threshold.
        threshold = max(float(min_expected_return), 1e-6)
        edge_ratio = float(net_trade_return) / threshold if threshold > 0 else float(net_trade_return)
        edge_score = self._clamp01(edge_ratio / 10.0)  # 10x threshold -> full credit (2% at 20bps min)

        # 2) Uncertainty score from SNR (CI-implied).
        if snr is None:
            logger.debug(
                "SNR unavailable for ticker=%s; using pessimistic snr_score=0.0 "
                "(missing CI means unknown uncertainty — consistent with P1-C diagnostics_score policy)",
                ticker,
            )
            snr_score = 0.0
        else:
            snr_score = self._clamp01((float(snr) - 0.5) / 3.0)  # 0.5 -> 0, 3.5 -> 1

        # 3) Volatility factor (optional soft penalty; don't artificially inflate).
        # Piecewise-linear between anchor points to eliminate cliff-edge jumps:
        #   vol <= 0.15 → 1.05 | 0.15-0.25 → linear 1.05-1.00 | 0.25-0.40 → 1.00
        #   0.40-0.60 → linear 0.75-0.60 | vol >= 0.60 → 0.60
        vol_factor = 1.0
        if volatility is not None and self.use_volatility_filter:
            vol = float(volatility)
            if vol >= 0.60:
                vol_factor = 0.60
            elif vol >= 0.40:
                # Linear: 0.40 → 0.75, 0.60 → 0.60
                vol_factor = 0.75 - 0.15 * (vol - 0.40) / 0.20
            elif vol <= 0.15:
                vol_factor = 1.05
            elif vol <= 0.25:
                # Linear: 0.15 → 1.05, 0.25 → 1.00
                vol_factor = 1.05 - 0.05 * (vol - 0.15) / 0.10

        # 4) Combine model-quality features with edge/uncertainty.
        # Edge is additive (not multiplicative) so marginal edges can still
        # clear confidence when model agreement/quality is strong, while
        # remaining discriminative across regimes.
        core = (
            0.20 * self._clamp01(diagnostics_score)
            + 0.20 * self._clamp01(model_agreement)
            + 0.20 * self._clamp01(snr_score)
            + 0.40 * self._clamp01(edge_score)
        )
        raw_confidence = (0.05 + 0.95 * core) * vol_factor
        # Very small gross expected_return should be conservative even if other signals look good.
        if abs(float(expected_return)) < float(min_expected_return):
            raw_confidence *= 0.75

        if not self._quant_validation_enabled:
            # Clear calibration state so _log_quant_validation never inherits stale
            # Platt values from a prior signal generated with calibration enabled.
            self._platt_calibrated = None
            self._last_raw_confidence = float(self._clamp01(raw_confidence))
            return self._clamp01(raw_confidence)

        # Calibrate confidence from realised trade outcomes in trade_executions.
        qv_cfg = getattr(self, "quant_validation_config", None) or {}
        calibration_cfg = qv_cfg.get("calibration") if isinstance(qv_cfg, dict) else {}
        calibration_cfg = calibration_cfg if isinstance(calibration_cfg, dict) else {}
        db_path = (
            calibration_cfg.get("db_path")
            or os.getenv("PORTFOLIO_DB_PATH")
            or "data/portfolio_maximizer.db"
        )
        confidence = self._calibrate_confidence(
            raw_confidence,
            ticker=ticker,
            db_path=db_path,
        )

        return self._clamp01(confidence)

    def _extract_classifier_features(
        self,
        *,
        forecast_bundle: Dict[str, Any],
        current_price: float,
        expected_return: float,
        lower_ci: Optional[float],
        upper_ci: Optional[float],
        snr: Optional[float],
        model_agreement: float,
        market_data: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Phase 9: Extract classifier feature vector for directional classifier.

        All features are computable at signal time — no lookahead.
        Missing values are represented as float('nan').
        Feature names match forcester_ts.directional_classifier._FEATURE_NAMES.
        """
        nan = float("nan")

        # --- A: Forecast geometry ---
        feat: Dict[str, Any] = {
            "ensemble_pred_return": expected_return,
            "ci_width_normalized": (
                (float(upper_ci) - float(lower_ci)) / float(current_price)
                if lower_ci is not None and upper_ci is not None and current_price > 0
                else nan
            ),
            "snr": snr if snr is not None else nan,
            "model_agreement": model_agreement,
        }

        # directional_vote_fraction: fraction of available models predicting same
        # direction as expected_return
        direction = float(np.sign(expected_return))
        model_vals = []
        for key in ("sarimax_forecast", "samossa_forecast", "mssa_rl_forecast", "garch_forecast"):
            payload = forecast_bundle.get(key)
            if isinstance(payload, dict) and payload:
                val = self._extract_forecast_value(payload)
                if val is not None and current_price > 0:
                    model_vals.append(float(np.sign(val - current_price)))
        if model_vals and direction != 0:
            feat["directional_vote_fraction"] = float(
                sum(1 for v in model_vals if v == direction) / len(model_vals)
            )
        else:
            feat["directional_vote_fraction"] = nan

        # --- B: Per-model confidence ---
        ensemble_metadata = forecast_bundle.get("ensemble_metadata") or {}
        conf_dict = ensemble_metadata.get("confidence") or {} if isinstance(ensemble_metadata, dict) else {}
        feat["garch_conf"] = float(conf_dict["garch"]) if "garch" in conf_dict else nan
        feat["samossa_conf"] = float(conf_dict["samossa"]) if "samossa" in conf_dict else nan
        feat["mssa_rl_conf"] = float(conf_dict["mssa_rl"]) if "mssa_rl" in conf_dict else nan

        # igarch_fallback_flag: 1 if GARCH fell back to EWMA
        garch_payload = forecast_bundle.get("garch_forecast") or {}
        garch_summary = garch_payload.get("summary") or {} if isinstance(garch_payload, dict) else {}
        igarch_flag = garch_summary.get("igarch_fallback") or garch_payload.get("igarch_fallback")
        feat["igarch_fallback_flag"] = 1.0 if igarch_flag else 0.0

        # samossa_evr: explained variance ratio from SAMOSSA
        samossa_payload = forecast_bundle.get("samossa_forecast") or {}
        samossa_summary = samossa_payload.get("summary") or {} if isinstance(samossa_payload, dict) else {}
        evr = samossa_summary.get("explained_variance_ratio") or samossa_payload.get("explained_variance_ratio")
        feat["samossa_evr"] = float(evr) if evr is not None else nan

        # --- C: Regime and series structure ---
        regime_features = forecast_bundle.get("regime_features") or {}
        series_diag = forecast_bundle.get("series_diagnostics") or {}
        if not isinstance(regime_features, dict):
            regime_features = {}
        if not isinstance(series_diag, dict):
            series_diag = {}

        feat["hurst_exponent"] = float(regime_features["hurst_exponent"]) if "hurst_exponent" in regime_features else nan
        feat["trend_strength"] = float(regime_features["trend_strength"]) if "trend_strength" in regime_features else nan
        feat["realized_vol_annualized"] = float(regime_features["realized_vol_annualized"]) if "realized_vol_annualized" in regime_features else nan
        feat["adf_pvalue"] = float(series_diag["adf_pvalue"]) if "adf_pvalue" in series_diag else nan

        # Regime one-hot encoding (4 regimes, all 4 included — no drop needed for LR with intercept)
        detected_regime = str(forecast_bundle.get("detected_regime") or "").upper()
        feat["regime_liquid_rangebound"] = 1.0 if "LIQUID_RANGEBOUND" in detected_regime else 0.0
        feat["regime_moderate_trending"] = 1.0 if "MODERATE_TRENDING" in detected_regime else 0.0
        feat["regime_high_vol_trending"] = 1.0 if "HIGH_VOL_TRENDING" in detected_regime else 0.0
        feat["regime_crisis"] = 1.0 if "CRISIS" in detected_regime else 0.0

        # --- D: Recent market context (from market_data) ---
        if market_data is not None and not market_data.empty and "Close" in market_data.columns:
            try:
                close = market_data["Close"].dropna()
                if len(close) >= 6:
                    price_6d_ago = float(close.iloc[-6])
                    if price_6d_ago > 0:  # guard: division-by-zero on zero/corrupt price
                        feat["recent_return_5d"] = float(close.iloc[-1] / price_6d_ago) - 1.0
                    else:
                        feat["recent_return_5d"] = nan
                else:
                    feat["recent_return_5d"] = nan
                if len(close) >= 60:
                    ret = close.pct_change().dropna()
                    vol_5 = float(ret.iloc[-5:].std()) if len(ret) >= 5 else nan
                    vol_60 = float(ret.iloc[-60:].std()) if len(ret) >= 60 else nan
                    feat["recent_vol_ratio"] = float(vol_5 / vol_60) if vol_60 and vol_60 > 0 else nan
                else:
                    feat["recent_vol_ratio"] = nan
            except Exception as exc:
                logger.debug("Market context feature extraction failed (will use NaN): %s", exc)
                feat["recent_return_5d"] = nan
                feat["recent_vol_ratio"] = nan
        else:
            feat["recent_return_5d"] = nan
            feat["recent_vol_ratio"] = nan

        # Sanitize: replace non-finite with nan
        for k, v in feat.items():
            if isinstance(v, float) and not math.isfinite(v):
                feat[k] = nan

        return feat

    def _check_model_agreement(self, forecast_bundle: Dict[str, Any]) -> float:
        """Check agreement between different models (0.0 to 1.0)"""
        forecasts = []

        # Collect forecasts from all models
        sarimax_payload = forecast_bundle.get('sarimax_forecast')
        if self._has_payload(sarimax_payload):
            sarimax_val = self._extract_forecast_value(sarimax_payload)
            if sarimax_val is not None:
                forecasts.append(sarimax_val)

        samossa_payload = forecast_bundle.get('samossa_forecast')
        if self._has_payload(samossa_payload):
            samossa_val = self._extract_forecast_value(samossa_payload)
            if samossa_val is not None:
                forecasts.append(samossa_val)

        mssa_payload = forecast_bundle.get('mssa_rl_forecast')
        if self._has_payload(mssa_payload):
            mssa_val = self._extract_forecast_value(mssa_payload)
            if mssa_val is not None:
                forecasts.append(mssa_val)

        if len(forecasts) < 2:
            logger.debug(
                "_check_model_agreement: only %d model forecast(s) available — "
                "returning pessimistic score 0.0 (cannot assess agreement; "
                "consistent with P1-C/H6 pessimistic fallback policy).",
                len(forecasts),
            )
            return 0.0  # pessimistic: cannot assess agreement with <2 models

        # Calculate coefficient of variation (lower = more agreement)
        mean_forecast = np.mean(forecasts)
        std_forecast = np.std(forecasts)

        if mean_forecast == 0:
            return 0.5

        cv = std_forecast / abs(mean_forecast)

        # Convert CV to agreement score (lower CV = higher agreement)
        # CV < 0.1 = excellent agreement (1.0)
        # CV > 0.5 = poor agreement (0.0)
        agreement = max(0.0, min(1.0, 1.0 - (cv - 0.1) / 0.4))

        return agreement

    def _evaluate_diagnostics_details(self, forecast_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate diagnostics quality and return a score + reasons (for provenance)."""
        reasons: List[str] = []

        ensemble_metadata = forecast_bundle.get("ensemble_metadata") or {}
        weights = ensemble_metadata.get("weights") if isinstance(ensemble_metadata, dict) else None
        model_conf = ensemble_metadata.get("confidence") if isinstance(ensemble_metadata, dict) else None

        weighted_conf = None
        if isinstance(weights, dict) and isinstance(model_conf, dict) and weights:
            num = 0.0
            den = 0.0
            for model_key, weight in weights.items():
                try:
                    w = float(weight)
                except (TypeError, ValueError):
                    continue
                if w <= 0:
                    continue
                c = model_conf.get(str(model_key).lower())
                try:
                    c_val = float(c)
                except (TypeError, ValueError):
                    continue
                num += w * self._clamp01(c_val)
                den += w
            if den > 0:
                weighted_conf = num / den
        if weighted_conf is None:
            reasons.append("missing_weighted_model_confidence")
            weighted_conf = 0.5

        regression_metrics = forecast_bundle.get("regression_metrics") or {}
        rmse_ratio = None
        dir_acc = None
        if isinstance(regression_metrics, dict) and regression_metrics:
            ens = regression_metrics.get("ensemble") or {}
            base = regression_metrics.get("samossa") or regression_metrics.get("sarimax") or {}
            if isinstance(ens, dict):
                try:
                    dir_acc = float(ens.get("directional_accuracy")) if ens.get("directional_accuracy") is not None else None
                except (TypeError, ValueError):
                    dir_acc = None
            try:
                ens_rmse = float(ens.get("rmse")) if isinstance(ens, dict) and ens.get("rmse") is not None else None
                base_rmse = float(base.get("rmse")) if isinstance(base, dict) and base.get("rmse") is not None else None
                if ens_rmse is not None and base_rmse and base_rmse > 0:
                    rmse_ratio = ens_rmse / base_rmse
            except (TypeError, ValueError):
                rmse_ratio = None
        else:
            reasons.append("missing_regression_metrics")

        model_errors = forecast_bundle.get("model_errors") or {}
        if isinstance(model_errors, dict) and any(model_errors.values()):
            reasons.append("model_errors_present")

        # Diagnostics score: start from weighted model confidence, then apply
        # conservative penalties/boosts from regression metrics when available.
        score = float(weighted_conf)
        if rmse_ratio is not None:
            # Penalize when worse than baseline; soft reward when better.
            if rmse_ratio > 1.10:
                reasons.append("rmse_worse_than_baseline")
                score *= 0.70
            elif rmse_ratio < 0.95:
                score = min(1.0, score + 0.05)
        if dir_acc is not None:
            if dir_acc < 0.50:
                reasons.append("directional_accuracy_below_50pct")
                score *= 0.80
            elif dir_acc > 0.55:
                score = min(1.0, score + 0.05)
        if "model_errors_present" in reasons:
            score *= 0.70

        return {
            "score": self._clamp01(score),
            "reasons": reasons,
            "weighted_model_confidence": float(weighted_conf) if weighted_conf is not None else None,
            "rmse_ratio_vs_baseline": rmse_ratio,
            "directional_accuracy": dir_acc,
        }

    def _calculate_risk_score(self,
                              expected_return: float,
                              volatility: Optional[float],
                              forecast_bundle: Dict[str, Any]) -> float:
        """
        Calculate risk score (0.0 to 1.0, higher = riskier).

        Based on:
        - Volatility forecast
        - Confidence interval width
        - Model uncertainty
        """
        risk = 0.5  # Base risk

        # Factor 1: Volatility (higher volatility = higher risk)
        if volatility is not None:
            if volatility > 0.40:  # >40% volatility
                risk += 0.3
            elif volatility > 0.25:  # >25% volatility
                risk += 0.15
            elif volatility < 0.15:  # <15% volatility
                risk -= 0.15

        # Factor 2: Confidence interval width (wider = riskier)
        primary_forecast, _ = self._resolve_primary_forecast(forecast_bundle)
        if primary_forecast:
            lower_ci, upper_ci = self._extract_ci_bounds(primary_forecast)
            forecast_price = self._extract_forecast_value(primary_forecast)

            if lower_ci is not None and upper_ci is not None and forecast_price is not None:
                ci_width_price = float(upper_ci - lower_ci)
                # Convert to a dimensionless uncertainty ratio: CI width relative to expected move.
                denom = abs(float(expected_return)) * max(float(forecast_price) / max(1.0 + float(expected_return), 1e-6), 1e-6)
                uncertainty_ratio = ci_width_price / denom if denom > 0 else float("inf")
                if uncertainty_ratio > 2.0:
                    risk += 0.2
                elif uncertainty_ratio < 0.75:
                    risk -= 0.1

        # Factor 3: Expected return magnitude (smaller moves = riskier relative to reward)
        if abs(expected_return) < 0.01:  # <1% expected return
            risk += 0.1

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, risk))

    def _determine_action(self,
                         expected_return: float,
                         net_trade_return: float,
                         confidence: float,
                         risk_score: float,
                         confidence_threshold: float,
                         min_expected_return: float,
                         max_risk_score: float) -> tuple:
        """
        Determine trading action based on forecast and risk metrics.

        Returns:
            Tuple of (action: str, hold_reason: Optional[str]) where action is
            'BUY', 'SELL', or 'HOLD'. hold_reason is None for non-HOLD actions
            and one of the following structured codes for HOLD:
              - 'CONFIDENCE_BELOW_THRESHOLD'
              - 'MIN_RETURN'
              - 'RISK_TOO_HIGH'
              - 'ZERO_EXPECTED_RETURN'
        """
        # Must meet minimum thresholds
        if confidence < confidence_threshold:
            return 'HOLD', 'CONFIDENCE_BELOW_THRESHOLD'

        # net_trade_return is always non-negative and already clears estimated friction.
        if net_trade_return + 1e-12 < min_expected_return:
            return 'HOLD', 'MIN_RETURN'

        if risk_score > max_risk_score:
            return 'HOLD', 'RISK_TOO_HIGH'

        # Determine direction
        if expected_return > 0:
            return 'BUY', None
        if expected_return < 0:
            return 'SELL', None
        return 'HOLD', 'ZERO_EXPECTED_RETURN'

    def _compute_atr(self, market_data: Optional[pd.DataFrame], period: int = 14) -> Optional[float]:
        """Compute Average True Range (ATR) from OHLC bar data.

        ATR measures market-observable noise using actual High-Low ranges, unlike
        model-implied volatility. Returns None when OHLC columns are unavailable
        or there is insufficient history.

        Args:
            market_data: DataFrame with High, Low, Close columns (date-indexed)
            period: Lookback period in bars (default 14)

        Returns:
            ATR value in price units, or None if unavailable
        """
        if market_data is None or len(market_data) < period + 1:
            return None
        required = {'High', 'Low', 'Close'}
        if not required.issubset(market_data.columns):
            return None
        high = market_data['High']
        low = market_data['Low']
        close = market_data['Close']
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1
        ).max(axis=1)
        atr = float(tr.iloc[-period:].mean())
        return atr if atr > 0 else None

    def _calculate_targets(self,
                          current_price: float,
                          forecast_price: float,
                          volatility: Optional[float],
                          action: str,
                          market_data: Optional[pd.DataFrame] = None) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate target price and stop loss.

        Stop loss uses ATR(14) when OHLC data is available (Phase 7.14-B). ATR-based
        stops adapt to market-observable noise rather than model-implied volatility, and
        carry no upper percentage cap -- intentional for high-volatility names (NVDA).
        Falls back to volatility-scaled stop when bar data is unavailable.

        Returns:
            (target_price, stop_loss)
        """
        if action == 'HOLD':
            return None, None

        # Target: forecast price
        target_price = forecast_price

        # Stop loss: ATR-based (preferred) or volatility-based fallback
        atr = self._compute_atr(market_data)
        if atr is not None and current_price > 0:
            # ATR * 2.0 positions stop below 2 average noise ranges.
            # No upper cap -- high-vol names need wider stops to avoid noise fires.
            stop_loss_pct = max((atr * 2.0) / current_price, 0.015)
        elif volatility is not None:
            # Fallback: model-implied vol, 1.5%-5% clamp
            stop_loss_pct = max(0.015, min(0.05, volatility * 0.5))
        else:
            stop_loss_pct = 0.02  # 2% default

        if action == 'BUY':
            stop_loss = current_price * (1 - stop_loss_pct)
        else:  # SELL
            stop_loss = current_price * (1 + stop_loss_pct)

        # Enforce minimum R:R of 2:1 — target distance must be ≥ 2× stop distance.
        # A signal that risked $10 to gain $0.09 (the AAPL case) is structurally
        # unprofitable at any win rate below 91%. Extend the target rather than
        # tightening the stop so ATR-based risk sizing is preserved.
        stop_dist = abs(current_price - stop_loss)
        target_dist = abs(target_price - current_price)
        if stop_dist > 0 and target_dist < 2.0 * stop_dist:
            if action == 'BUY':
                target_price = current_price + 2.0 * stop_dist
            else:
                target_price = current_price - 2.0 * stop_dist

        return target_price, stop_loss

    def _build_reasoning(self,
                        action: str,
                        expected_return: float,
                        net_trade_return: float,
                        roundtrip_cost_bps: float,
                        confidence: float,
                        risk_score: float,
                        forecast_bundle: Dict[str, Any]) -> str:
        """Build human-readable reasoning for signal"""
        model_type = (
            forecast_bundle.get("default_model")
            or forecast_bundle.get("ensemble_metadata", {}).get("default_model")
            or forecast_bundle.get("ensemble_metadata", {}).get("primary_model")
            or "ENSEMBLE"
        )

        reasoning = (
            f"Time Series {model_type} forecast indicates {action} signal. "
            f"Gross move: {expected_return:.2%}, "
            f"Net edge: {net_trade_return:.2%} (est cost: {roundtrip_cost_bps:.1f}bp), "
            f"Confidence: {confidence:.1%}, "
            f"Risk score: {risk_score:.1%}. "
        )

        if self._has_payload(forecast_bundle.get('samossa_forecast')):
            reasoning += "SAMOSSA SSA decomposition confirms trend. "

        if self._has_payload(forecast_bundle.get('mssa_rl_forecast')):
            reasoning += "MSSA-RL change-point detection active. "

        return reasoning.strip()

    def _extract_provenance(
        self,
        forecast_bundle: Dict[str, Any],
        *,
        selected_source: str = "",
    ) -> Dict[str, Any]:
        """Extract provenance metadata from forecast bundle"""
        ensemble_metadata = forecast_bundle.get("ensemble_metadata") or {}
        default_model = (
            forecast_bundle.get("default_model")
            or ensemble_metadata.get("default_model")
            or ensemble_metadata.get("primary_model")
            or "ENSEMBLE"
        )

        provenance = {
            'model_type': 'TIME_SERIES_ENSEMBLE',
            'timestamp': utc_now().isoformat(),
            'forecast_horizon': forecast_bundle.get('horizon', 30),
            'primary_model': str(default_model).upper(),
            'selected_forecast_source': selected_source or None,
        }

        # Add model-specific metadata
        if ensemble_metadata:
            provenance.update({
                'ensemble_primary_model': ensemble_metadata.get('primary_model', 'ENSEMBLE'),
                'ensemble_default_model': ensemble_metadata.get('default_model'),
                'model_weights': ensemble_metadata.get('weights', {}),
                'aic': ensemble_metadata.get('aic'),
                'bic': ensemble_metadata.get('bic')
            })

        # Add individual model flags
        provenance['models_used'] = []
        if self._has_payload(forecast_bundle.get('sarimax_forecast')):
            provenance['models_used'].append('SARIMAX')
        if self._has_payload(forecast_bundle.get('samossa_forecast')):
            provenance['models_used'].append('SAMOSSA')
        if self._has_payload(forecast_bundle.get('mssa_rl_forecast')):
            provenance['models_used'].append('MSSA_RL')
        if self._has_payload(forecast_bundle.get('garch_forecast')):
            provenance['models_used'].append('GARCH')

        return provenance

    @staticmethod
    def _log_gate_result(
        gate: str,
        signal_id: str,
        ticker: str,
        value: object,
        threshold: object,
        result: str,
        **extra: object,
    ) -> None:
        """Emit one structured gate-decision record on the ``pmx.gates`` logger.

        Schema:
          gate       — gate name: "confidence" | "min_return" | "snr" | "quant" | "quant_criterion"
          signal_id  — ts_signal_id of the signal being evaluated (pre-assigned before gates fire)
          ticker     — ticker symbol
          value      — the measured value (confidence, snr, utility_score, …)
          threshold  — the declared gate threshold
          result     — one of PASS | FAIL | SKIP | UNKNOWN
          **extra    — optional fields: criterion, gate_detail, …

        The record is intentionally not written to a separate JSONL file here;
        downstream log handlers that route "pmx.gates" records to a dedicated
        sink can be configured in logging_config.yml without code changes.
        """
        import json
        record = {
            "gate": gate,
            "signal_id": signal_id,
            "ticker": ticker,
            "value": value,
            "threshold": threshold,
            "result": result,
        }
        record.update(extra)
        _gate_logger.info("GATE %s", json.dumps(record, default=str))

    def _create_hold_signal(self,
                           ticker: str,
                           current_price: float,
                           reason: str) -> TimeSeriesSignal:
        """Create a HOLD signal"""
        timestamp = utc_now()
        return TimeSeriesSignal(
            ticker=ticker,
            action='HOLD',
            confidence=0.0,
            entry_price=current_price,
            signal_timestamp=timestamp,
            model_type='ENSEMBLE',
            expected_return=0.0,
            risk_score=0.5,
            reasoning=f"HOLD: {reason}",
            provenance={
                'model_type': 'TIME_SERIES_ENSEMBLE',
                'reason': reason,
                'timestamp': timestamp.isoformat(),
            },
            signal_type='TIME_SERIES'
        )

    def _build_quant_success_profile(
        self,
        ticker: str,
        market_data: Optional[pd.DataFrame],
        signal: TimeSeriesSignal,
    ) -> Optional[Dict[str, Any]]:
        """
        Build quantitative success profile referencing institutional criteria.
        """
        if not self._quant_validation_enabled:
            return None
        if market_data is None or market_data.empty or 'Close' not in market_data.columns:
            return None

        config = self.quant_validation_config or {}
        lookback = max(int(config.get('lookback_days', 120)), 2)
        price_series = market_data['Close'].tail(lookback).dropna()
        if len(price_series) < 2:
            return None

        prices = np.maximum(price_series.astype(float).values, 1e-6)
        log_returns = np.diff(np.log(prices))
        if log_returns.size < 2:
            return None

        action = str(getattr(signal, "action", None) or "HOLD").upper()
        if action == "SELL":
            direction = -1.0
        else:
            direction = 1.0
        strategy_returns = log_returns * direction
        decision_ctx = (signal.provenance or {}).get("decision_context") or {}
        try:
            execution_drag_hurdle = float(decision_ctx.get("roundtrip_cost_fraction"))
        except (TypeError, ValueError):
            execution_drag_hurdle = None

        try:
            metrics = calculate_enhanced_portfolio_metrics(
                returns=strategy_returns.reshape(-1, 1),
                weights=np.array([1.0]),
                risk_free_rate=float(config.get('risk_free_rate', DEFAULT_RISK_FREE_RATE)),
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Unable to compute quant metrics for %s: %s", ticker, exc)
            return None
        try:
            domain_metrics = portfolio_metrics_ngn(
                pd.Series(strategy_returns),
                execution_drag_hurdle=execution_drag_hurdle,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Unable to compute barbell-tail metrics for %s: %s", ticker, exc)
            domain_metrics = {}
        for key in (
            "omega_ratio",
            "omega_curve",
            "omega_robustness_score",
            "omega_monotonicity_ok",
            "omega_above_hurdle_margin",
            "omega_cliff_drop_ratio",
            "omega_cliff_ok",
            "omega_robustness_complete",
            "omega_ci_lower",
            "omega_ci_upper",
            "omega_right_tail_ok",
            "omega_ci_width",
            "expected_shortfall_raw",
            "expected_shortfall_to_edge",
            "es_to_edge_bounded",
            "fractional_kelly_fat_tail",
            "ngn_daily_threshold",
            "ngn_annual_hurdle_pct",
            "beats_ngn_hurdle",
            "n_wins",
            "n_losses",
            "winner_concentration_ratio",
            "trimmed_payoff_asymmetry",
            "payoff_asymmetry_support_ok",
            "payoff_asymmetry_effective",
        ):
            if key in domain_metrics:
                metrics[key] = domain_metrics[key]

        bootstrap_cfg = config.get('bootstrap', {})
        try:
            bootstrap_stats = bootstrap_confidence_intervals(
                returns=strategy_returns,
                n_bootstrap=int(bootstrap_cfg.get('n_samples', 500)),
                confidence_level=float(bootstrap_cfg.get('confidence_level', 0.95)),
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Bootstrap stats unavailable for %s: %s", ticker, exc)
            bootstrap_stats = {}

        benchmark_col = config.get('benchmark_column')
        significance = None
        if benchmark_col and benchmark_col in market_data.columns:
            benchmark_prices = market_data[benchmark_col].tail(len(price_series)).dropna()
            bench_values = np.maximum(benchmark_prices.astype(float).values, 1e-6)
            bench_returns = np.diff(np.log(bench_values))
            if bench_returns.size >= log_returns.size:
                benchmark_slice = bench_returns[-log_returns.size:]
                try:
                    significance = test_strategy_significance(strategy_returns, benchmark_slice * direction)
                except ValueError:
                    significance = None

        performance_snapshot = self._calculate_return_based_performance(strategy_returns)

        # Replace backward-looking historical-return metrics with forward-looking
        # signal-level equivalents.  Historical omega_ratio / payoff_asymmetry reflect
        # recent asset momentum vs the NGN hurdle — not whether THIS signal has positive
        # EV.  See _build_signal_level_overrides docstring for full rationale.
        _signal_overrides = self._build_signal_level_overrides(signal)
        if _signal_overrides:
            _omega_override_keys = (
                "omega_ratio", "omega_robustness_score", "omega_monotonicity_ok",
                "omega_above_hurdle_margin", "omega_cliff_drop_ratio", "omega_cliff_ok",
                "omega_robustness_complete", "omega_curve",
            )
            for _ok in _omega_override_keys:
                if _ok in _signal_overrides:
                    metrics[_ok] = _signal_overrides[_ok]
            for _pk in ("payoff_asymmetry", "avg_gain", "avg_loss"):
                if _pk in _signal_overrides:
                    performance_snapshot[_pk] = _signal_overrides[_pk]

        # Allow per-ticker overrides for success criteria so that
        # higher-volatility or structurally weaker assets (e.g. crypto,
        # certain commodities/FX) can demand a higher expected_profit
        # before passing quant validation, without penalising the entire
        # universe.
        per_ticker_cfg = (config.get('per_ticker') or {}).get(ticker, {})
        if isinstance(per_ticker_cfg, dict) and per_ticker_cfg.get('success_criteria'):
            # Merge per-ticker overrides ON TOP of the global success_criteria so that
            # global thresholds (e.g. min_terminal_directional_accuracy) are not silently
            # suppressed when a per-ticker override only specifies a subset of fields.
            global_criteria = dict(config.get('success_criteria') or {})
            global_criteria.update(per_ticker_cfg['success_criteria'])
            criteria_cfg = global_criteria
        else:
            criteria_cfg = config.get('success_criteria', {})

        capital_base = float(criteria_cfg.get('capital_base', config.get('capital_base', 25000.0)))
        allocation = min(max(float(signal.confidence or 0.0), 0.05), 1.0)
        safe_price = request_safe_price(signal.entry_price)
        position_value = capital_base * allocation
        ctx = decision_ctx
        try:
            net_trade_return = float(ctx.get("net_trade_return"))
        except (TypeError, ValueError):
            net_trade_return = max(0.0, abs(float(signal.expected_return or 0.0)))
        expected_profit = position_value * net_trade_return
        path_metrics = self._collect_barbell_path_metrics(
            ticker=ticker,
            signal=signal,
            market_data=market_data,
            position_value=position_value,
        )
        # Track detected_regime for the CRISIS barbell block (Gap 4)
        _prov = signal.provenance if isinstance(signal.provenance, dict) else {}
        _detected_regime_for_gate = str(_prov.get("detected_regime") or "").upper()
        if path_metrics:
            metrics.update(
                {
                    "barbell_bucket": path_metrics.get("barbell_bucket"),
                    "roundtrip_cost_to_edge": path_metrics.get("roundtrip_cost_to_edge"),
                    "gap_risk_to_edge": path_metrics.get("gap_risk_to_edge"),
                    "funding_to_edge": path_metrics.get("funding_to_edge"),
                    "liquidity_to_depth": path_metrics.get("liquidity_to_depth"),
                    "leverage": path_metrics.get("leverage"),
                    "barbell_path_risk_ok": path_metrics.get("barbell_path_risk_ok"),
                }
            )
        metrics["detected_regime"] = _detected_regime_for_gate
        structural_gates = self._evaluate_success_criteria(
            criteria_cfg=criteria_cfg,
            metrics=metrics,
            performance_snapshot=performance_snapshot,
            significance=significance,
            expected_profit=expected_profit,
            position_value=position_value,
            action=action,
        )
        drift_criteria = dict(structural_gates) if isinstance(structural_gates, dict) else {}

        # Forecast-edge validation (optional): measure incremental edge using
        # rolling-window forecast regression metrics instead of drift proxy.
        validation_mode = str(config.get("validation_mode") or "drift_proxy").lower()
        edge_block: Dict[str, Any] = {}
        if validation_mode == "forecast_edge":
            edge_block, edge_criteria = self._evaluate_forecast_edge(
                price_series=price_series,
                signal=signal,
                config=config,
                criteria_cfg=criteria_cfg,
            )
            if edge_criteria:
                # Edge criteria are the primary gate; keep drift-proxy criteria
                # available in metrics but do not require them unless explicitly requested.
                include_drift = bool(criteria_cfg.get("include_drift_proxy_criteria", False))
                structural_gates = dict(edge_criteria)
                if "expected_profit" in drift_criteria:
                    structural_gates = {"expected_profit": drift_criteria.get("expected_profit"), **structural_gates}
                for extra_key in ("significance", "information_ratio"):
                    if extra_key in drift_criteria:
                        structural_gates[extra_key] = drift_criteria[extra_key]
                if include_drift:
                    structural_gates.update(
                        {f"drift_{k}": v for k, v in drift_criteria.items() if k not in {"expected_profit", "significance", "information_ratio"}}
                    )

        structural_gates = self._canonicalize_criteria(structural_gates)

        # Gap 4 — CRISIS regime barbell block (anti-omega failure mode 4):
        # Liquidity and path risk are most dangerous in CRISIS regimes because
        # gap risk spikes, spreads widen, and depth collapses simultaneously.
        # When the regime is CRISIS AND path risk is failing, inject a hard
        # synthetic gate that cannot be configured away.  This is distinct from
        # the configurable bucket_hard_gate_criteria — it is a structural
        # invariant that a barbell system must not accept new speculative
        # exposure when the market regime is CRISIS and path risk is uncleared.
        if (
            "CRISIS" in _detected_regime_for_gate
            and not bool(metrics.get("barbell_path_risk_ok", True))
            and action != "HOLD"
        ):
            structural_gates["crisis_regime_path_risk_block"] = False

        viz_cfg = config.get('visualization') or {}
        visualization_result = None
        if viz_cfg.get('enabled'):
            visualization_result = self._render_quant_validation_plot(
                ticker=ticker,
                market_data=market_data,
                output_dir=viz_cfg.get('output_dir', 'visualizations/quant_validation'),
                max_points=int(viz_cfg.get('max_points', lookback)),
            )

        scoring_mode_raw = str(
            config.get('scoring_mode')
            or config.get("objective_mode")
            or 'domain_utility'
        ).lower()
        scoring_mode = (
            "domain_utility"
            if scoring_mode_raw in {"weighted", "domain_utility", "two_layer_utility"}
            else scoring_mode_raw
        )
        pass_threshold = float(config.get('pass_threshold', 0.60))
        strict_weight_coverage = self._to_bool(
            config.get("strict_weight_coverage"),
            default=True,
        )
        domain_utility = self._build_domain_utility(
            metrics=metrics,
            performance_snapshot=performance_snapshot,
            edge_block=edge_block,
            criteria_cfg=criteria_cfg,
            config=config,
            expected_profit=expected_profit,
            position_value=position_value,
        )
        utility_score = domain_utility.get("utility_score")
        total_weight = float(domain_utility.get("total_weight") or 0.0)
        weight_validation: Dict[str, Any] = dict(domain_utility.get("weight_validation") or {})
        weight_validation["strict_weight_coverage"] = strict_weight_coverage
        config_warnings = list(domain_utility.get("config_warnings") or [])
        config_warnings.extend(self._ignored_success_threshold_warnings(criteria_cfg))
        hard_gate_criteria, hard_gate_warnings = self._resolve_hard_gate_criteria(
            config=config,
            criteria_cfg=criteria_cfg,
            structural_gates=structural_gates,
            bucket=str(metrics.get("barbell_bucket") or ""),
        )
        config_warnings.extend(hard_gate_warnings)
        if scoring_mode_raw == "weighted":
            config_warnings.append("deprecated_scoring_mode:weighted")

        utility_breakdown = dict(domain_utility.get("utility_breakdown") or {})
        utility_pass = bool(utility_score is not None and float(utility_score) >= pass_threshold)
        criteria = dict(structural_gates)
        if action != "HOLD":
            criteria["domain_utility"] = utility_pass

        hard_gate_set = set(hard_gate_criteria)
        hard_gate_results = {
            name: bool(passed)
            for name, passed in structural_gates.items()
            if name in hard_gate_set
        }
        soft_gate_results = {
            name: bool(passed)
            for name, passed in structural_gates.items()
            if name not in hard_gate_set
        }
        hard_failed = sorted(name for name, passed in hard_gate_results.items() if not passed)
        soft_failed = sorted(name for name, passed in soft_gate_results.items() if not passed)
        hard_gate_pass = all(hard_gate_results.values()) if hard_gate_results else True
        soft_gate_pass = all(soft_gate_results.values()) if soft_gate_results else True

        if action == "HOLD":
            # HOLD is non-actionable for trade-health metrics; keep as SKIPPED.
            status = "SKIPPED"
        elif not criteria:
            status = "SKIPPED"
        elif structural_gates.get("expected_profit") is False:
            # Expected-profit floor is an economic viability gate; failing it is always FAIL.
            status = "FAIL"
        elif scoring_mode == "all_pass":
            # Legacy: every criterion must pass.
            status = "PASS" if all(criteria.values()) else "FAIL"
        else:
            # Hard gate: negative expected_profit is always FAIL regardless of score.
            if expected_profit < 0:
                status = "FAIL"
            elif not hard_gate_pass:
                status = "FAIL"
            elif strict_weight_coverage and not bool(weight_validation.get("coverage_ok", True)):
                status = "FAIL"
            elif utility_score is None:
                status = "FAIL" if strict_weight_coverage else "SKIPPED"
            else:
                status = "PASS" if utility_pass else "FAIL"

        failed: List[str] = []
        if status == "FAIL":
            if scoring_mode == "all_pass":
                failed = sorted(name for name, passed in criteria.items() if not passed)
            else:
                failed = list(hard_failed)
                if not utility_pass:
                    failed.append("domain_utility")
                if strict_weight_coverage and not bool(weight_validation.get("coverage_ok", True)):
                    failed.append("weight_coverage")
                failed = sorted(set(failed))

        return {
            'status': status,
            'metrics': {
                'annual_return': metrics.get('annual_return'),
                'sharpe_ratio': metrics.get('sharpe_ratio'),
                'sortino_ratio': metrics.get('sortino_ratio'),
                'max_drawdown': metrics.get('max_drawdown'),
                'volatility': metrics.get('volatility'),
                'expected_shortfall': metrics.get('expected_shortfall'),
                'omega_ratio': metrics.get('omega_ratio'),
                'omega_curve': metrics.get('omega_curve'),
                'omega_robustness_score': metrics.get('omega_robustness_score'),
                'omega_monotonicity_ok': metrics.get('omega_monotonicity_ok'),
                'omega_above_hurdle_margin': metrics.get('omega_above_hurdle_margin'),
                # Gap 1: cliff-drop guard
                'omega_cliff_drop_ratio': metrics.get('omega_cliff_drop_ratio'),
                'omega_cliff_ok': metrics.get('omega_cliff_ok'),
                # Gap 2: right-tail bootstrap CI
                'omega_ci_lower': metrics.get('omega_ci_lower'),
                'omega_ci_upper': metrics.get('omega_ci_upper'),
                'omega_right_tail_ok': metrics.get('omega_right_tail_ok'),
                'omega_ci_width': metrics.get('omega_ci_width'),
                # Gap 3: left-tail ES relative to edge
                'expected_shortfall_raw': metrics.get('expected_shortfall_raw'),
                'expected_shortfall_to_edge': metrics.get('expected_shortfall_to_edge'),
                'es_to_edge_bounded': metrics.get('es_to_edge_bounded'),
                'payoff_asymmetry': performance_snapshot.get('payoff_asymmetry'),
                'trimmed_payoff_asymmetry': performance_snapshot.get('trimmed_payoff_asymmetry'),
                'winner_concentration_ratio': performance_snapshot.get('winner_concentration_ratio'),
                'payoff_asymmetry_support_ok': performance_snapshot.get('payoff_asymmetry_support_ok'),
                'payoff_asymmetry_effective': performance_snapshot.get('payoff_asymmetry_effective'),
                'win_rate': performance_snapshot.get('win_rate'),
                'profit_factor': performance_snapshot.get('profit_factor'),
                'gap_risk_to_edge': metrics.get('gap_risk_to_edge'),
                'liquidity_to_depth': metrics.get('liquidity_to_depth'),
                'barbell_path_risk_ok': metrics.get('barbell_path_risk_ok'),
                # Gap 4: CRISIS regime block
                'detected_regime': metrics.get('detected_regime'),
                'crisis_regime_path_risk_block': structural_gates.get('crisis_regime_path_risk_block'),
            },
            'diagnostics': {
                'win_rate': performance_snapshot.get('win_rate'),
                'sharpe_ratio': metrics.get('sharpe_ratio'),
                'sortino_ratio': metrics.get('sortino_ratio'),
                'directional_accuracy': edge_block.get('directional_accuracy'),
                'barbell_bucket': metrics.get('barbell_bucket'),
                'gap_risk_to_edge': metrics.get('gap_risk_to_edge'),
                'liquidity_to_depth': metrics.get('liquidity_to_depth'),
                'roundtrip_cost_to_edge': metrics.get('roundtrip_cost_to_edge'),
                'funding_to_edge': metrics.get('funding_to_edge'),
                'barbell_path_risk_ok': metrics.get('barbell_path_risk_ok'),
            },
            'forecast_edge': edge_block,
            'bootstrap': bootstrap_stats,
            'criteria': criteria,
            'structural_gates': structural_gates,
            'hard_gate_criteria': sorted(hard_gate_results.keys()),
            'soft_gate_criteria': sorted(soft_gate_results.keys()),
            'hard_failed_criteria': hard_failed,
            'soft_failed_criteria': soft_failed,
            'failed_criteria': failed,
            'utility_breakdown': utility_breakdown,
            'utility_score': utility_score,
            'weight_validation': weight_validation,
            'scoring': {
                'mode': scoring_mode,
                'pass_threshold': pass_threshold,
                'normalized_score': utility_score,
                'utility_score': utility_score,
                'utility_pass': utility_pass,
                'structural_pass': bool(structural_gates) and all(structural_gates.values()),
                'structural_hard_pass': hard_gate_pass,
                'structural_soft_pass': soft_gate_pass,
                'total_weight': total_weight,
            },
            'config_warnings': sorted(set(config_warnings)),
            'significance': significance,
            'lookback_bars': int(len(price_series)),
            'expected_profit': expected_profit,
            'capital_base': capital_base,
            'position_value': position_value,
            'estimated_shares': position_value / safe_price if safe_price else None,
            'performance_snapshot': performance_snapshot,
            'visualization': {'path': visualization_result} if visualization_result else {},
            'execution_mode': (signal.provenance or {}).get("execution_mode"),
            'proof_mode': self._to_bool((signal.provenance or {}).get("proof_mode"), default=False),
            'run_id': (
                (signal.provenance or {}).get("run_id")
                or ((signal.provenance or {}).get("decision_context") or {}).get("run_id")
            ),
        }

    def _evaluate_forecast_edge(
        self,
        *,
        price_series: pd.Series,
        signal: TimeSeriesSignal,
        config: Dict[str, Any],
        criteria_cfg: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        """Compute rolling CV regression metrics and evaluate forecast-edge criteria."""
        cv_cfg = config.get("forecast_edge_cv") if isinstance(config, dict) else None
        cv_cfg = cv_cfg if isinstance(cv_cfg, dict) else {}

        horizon = int(cv_cfg.get("horizon") or getattr(signal, "forecast_horizon", 30) or 30)
        horizon = max(horizon, 1)
        min_train_size = int(cv_cfg.get("min_train_size") or 180)
        step_size = int(cv_cfg.get("step_size") or horizon)
        max_folds = cv_cfg.get("max_folds")
        try:
            max_folds_int = int(max_folds) if max_folds is not None else None
        except (TypeError, ValueError):
            max_folds_int = None
        baseline_model = str(
            criteria_cfg.get("baseline_model")
            or cv_cfg.get("baseline_model")
            or "samossa"
        ).lower()

        symbol = str(getattr(signal, "ticker", "") or "").upper()
        last_ts = ""
        try:
            last_ts = pd.to_datetime(price_series.index[-1]).isoformat()
        except Exception:
            last_ts = str(price_series.index[-1]) if len(price_series.index) else ""
        baseline_key = str(baseline_model or "samossa").strip().lower().replace("-", "_")
        if baseline_key not in {"sarimax", "samossa", "mssa_rl", "garch", "best_single"}:
            baseline_key = "samossa"

        cache_key = (symbol, last_ts, horizon, baseline_key)
        cached = self._forecast_edge_cache.get(cache_key)
        if cached is not None:
            # Cache stores only edge_payload (expensive CV run result).
            # Criteria are recomputed fresh from the cached payload so that
            # threshold changes and code fixes (e.g. Fix C CI-coverage override,
            # max_rmse_ratio_vs_baseline config updates) take effect immediately
            # without requiring a cache flush or process restart.
            cached_payload = cached if isinstance(cached, dict) else cached[0]
            fresh_criteria = self._edge_criteria_from_payload(cached_payload, criteria_cfg)
            return cached_payload, fresh_criteria

        edge_payload: Dict[str, Any] = {
            "mode": "forecast_edge",
            "baseline_model": baseline_key,
            "horizon": horizon,
        }
        edge_criteria: Dict[str, bool] = {}

        try:
            returns_series = price_series.pct_change().dropna()
            interval_hint = os.getenv("YFINANCE_INTERVAL")
            if not interval_hint:
                try:
                    interval_hint = price_series.index.inferred_freq  # type: ignore[assignment]
                except Exception:
                    interval_hint = None

            fast_cv_flag = _pmx_env_flag("PMX_FAST_FORECAST_EDGE_CV")
            if fast_cv_flag is None:
                fast_cv_flag = True
            fast_intraday_cv = bool(fast_cv_flag and _pmx_is_intraday_interval(interval_hint))

            cv_max_folds = max_folds_int
            if fast_intraday_cv:
                cv_max_folds = 1

            forecaster_config = self._build_forecast_edge_forecaster_config(
                horizon=horizon,
                baseline_key=baseline_key,
                fast_intraday_cv=fast_intraday_cv,
                interval_hint=interval_hint,
            )

            validator = RollingWindowValidator(
                forecaster_config=forecaster_config,
                cv_config=RollingWindowCVConfig(
                    min_train_size=int(min_train_size),
                    horizon=int(horizon),
                    step_size=int(step_size),
                    max_folds=cv_max_folds,
                ),
            )
            report = validator.run(
                price_series=price_series,
                returns_series=returns_series,
                ticker=symbol,
            )
            aggregate = report.get("aggregate_metrics") or {}
            fold_count = int(report.get("fold_count") or 0)
            ens = aggregate.get("ensemble") or {}
            if baseline_key == "best_single":
                _candidates = {k: v for k, v in aggregate.items() if k != "ensemble" and v}
                base = (
                    min(_candidates.values(), key=lambda m: float(m.get("rmse") or float("inf")))
                    if _candidates
                    else {}
                )
            else:
                base = aggregate.get(baseline_key) or {}
            edge_payload.update(
                {
                    "fold_count": fold_count,
                    "ensemble": ens,
                    "baseline": base,
                }
            )

            rmse_ratio = None
            try:
                ens_rmse = float(ens.get("rmse")) if isinstance(ens, dict) and ens.get("rmse") is not None else None
                base_rmse = float(base.get("rmse")) if isinstance(base, dict) and base.get("rmse") is not None else None
                if ens_rmse is not None and base_rmse and base_rmse > 0:
                    rmse_ratio = ens_rmse / base_rmse
            except (TypeError, ValueError):
                rmse_ratio = None
            edge_payload["rmse_ratio_vs_baseline"] = rmse_ratio

            dir_acc = None
            try:
                dir_acc = float(ens.get("directional_accuracy")) if isinstance(ens, dict) and ens.get("directional_accuracy") is not None else None
            except (TypeError, ValueError):
                dir_acc = None
            edge_payload["directional_accuracy"] = dir_acc
            terminal_dir_acc = None
            try:
                terminal_dir_acc = (
                    float(ens.get("terminal_directional_accuracy"))
                    if isinstance(ens, dict) and ens.get("terminal_directional_accuracy") is not None
                    else None
                )
            except (TypeError, ValueError):
                terminal_dir_acc = None
            edge_payload["terminal_directional_accuracy"] = terminal_dir_acc

            # Extract CI coverage for both ensemble and baseline — needed for the
            # RMSE override (Fix C) and for the domain_utility CI component.
            ens_ci_cov = None
            base_ci_cov = None
            try:
                if isinstance(ens, dict) and ens.get("terminal_ci_coverage") is not None:
                    ens_ci_cov = float(ens["terminal_ci_coverage"])
                if isinstance(base, dict) and base.get("terminal_ci_coverage") is not None:
                    base_ci_cov = float(base["terminal_ci_coverage"])
            except (TypeError, ValueError):
                pass
            edge_payload["terminal_ci_coverage"] = ens_ci_cov
            edge_payload["baseline_ci_coverage"] = base_ci_cov
        except Exception as exc:  # pragma: no cover - best-effort metric
            edge_payload["error"] = str(exc)

        if len(self._forecast_edge_cache) >= 256:
            self._forecast_edge_cache.clear()
        # Store only the payload (computed from expensive CV runs).
        # Criteria are NOT cached — they must be recomputed fresh each call so
        # that threshold/code changes take effect without a process restart.
        self._forecast_edge_cache[cache_key] = edge_payload
        edge_criteria = self._edge_criteria_from_payload(edge_payload, criteria_cfg)
        return edge_payload, edge_criteria

    @staticmethod
    def _build_signal_level_overrides(signal: Any) -> Dict[str, Any]:
        """Replace backward-looking historical daily-return metrics with forward-looking
        signal-level equivalents derived from this signal's trade parameters.

        Root cause fixed:  omega_ratio and payoff_asymmetry were previously computed
        from ``strategy_returns = log_returns * direction`` — i.e. 365 days of raw
        market price changes.  This measures *recent asset momentum* vs the NGN hurdle,
        not signal quality.  AAPL returning 15 % over 120 days naturally gives
        omega < 1.0 at the 28 % NGN annual hurdle even when the specific signal has
        strong positive expected value.

        Replacements
        ------------
        omega_ratio (tau=0):
            ``confidence * upside / (1 - confidence) * stop``
            Threshold 1.0 = positive expected-value requirement.
            tau=0 is the per-signal EV gate; the portfolio NGN hurdle is a
            separate portfolio-level goal, not a per-trade filter.

        payoff_asymmetry:
            ``upside_pct / stop_pct``  (forward R:R ratio)
            Replaces historical avg_win / |avg_loss| from daily bar returns.
            Threshold ~0.60: don't risk more than ~1.67× potential reward.
            The two gates together (omega>1 AND R:R>0.60) require both positive
            expected value AND a meaningful risk/reward setup.
        """
        try:
            entry = float(getattr(signal, 'entry_price', None) or 0)
            target = float(getattr(signal, 'target_price', None) or 0)
            stop = float(getattr(signal, 'stop_loss', None) or 0)
            conf = float(getattr(signal, 'confidence', None) or 0)
            action = str(getattr(signal, 'action', 'HOLD')).upper()

            if action not in ('BUY', 'SELL') or entry <= 0 or target <= 0 or stop <= 0:
                return {}
            if not (0.0 < conf < 1.0):
                return {}

            if action == 'BUY':
                upside_pct = (target - entry) / entry
                stop_pct = (entry - stop) / entry
            else:  # SELL
                upside_pct = (entry - target) / entry
                stop_pct = (stop - entry) / entry

            upside_pct = max(upside_pct, 0.0)
            stop_pct = max(stop_pct, 1e-4)

            if upside_pct == 0.0:
                return {}

            # Bernoulli synthetic distribution (total trade returns, not daily):
            #   n_wins samples of +upside_pct, n_losses samples of -stop_pct
            # omega(tau=0) from this = conf*upside / (1-conf)*stop = EV ratio
            n = 120  # satisfies omega_ratio's >=10 observation requirement
            n_wins = max(1, round(n * conf))
            n_losses = max(1, n - n_wins)
            synthetic_returns = np.concatenate([
                np.full(n_wins, upside_pct),
                np.full(n_losses, -stop_pct),
            ])

            from etl.portfolio_math import omega_ratio as _omega_ratio, omega_robustness_summary

            # tau=0: forward EV gate (not DAILY_NGN_THRESHOLD which is for portfolio health)
            fwd_omega = _omega_ratio(pd.Series(synthetic_returns), threshold=0.0)
            if isinstance(fwd_omega, float) and math.isnan(fwd_omega):
                fwd_omega = None

            # Robustness from the forward distribution.  omega_robustness_summary
            # evaluates across tau=0/NGN_hurdle/cost-adjusted.  With total-return
            # outcomes (0.011 >> 0.00108 NGN daily threshold), the curve is stable,
            # which correctly reflects that this trade's return magnitude is robust
            # against realistic hurdle escalation.
            ctx = (getattr(signal, 'provenance', None) or {}).get("decision_context") or {}
            roundtrip_cost: Optional[float] = None
            try:
                rc = ctx.get("roundtrip_cost_fraction")
                if rc is not None:
                    roundtrip_cost = max(float(rc), 0.0)
            except (TypeError, ValueError):
                pass

            fwd_robustness = omega_robustness_summary(
                pd.Series(synthetic_returns),
                execution_drag_hurdle=roundtrip_cost,
            )

            # Forward payoff_asymmetry = R:R ratio (replaces historical avg_win/|avg_loss|)
            fwd_payoff = upside_pct / stop_pct

            result: Dict[str, Any] = {
                "payoff_asymmetry": fwd_payoff,
                "avg_gain": upside_pct,
                "avg_loss": -stop_pct,
            }
            if fwd_omega is not None:
                result["omega_ratio"] = fwd_omega
            # Merge robustness keys (only non-None values to preserve strict_weight_coverage logic)
            for k, v in fwd_robustness.items():
                if v is not None:
                    result[k] = v
            return result
        except Exception as exc:  # pragma: no cover
            logger.debug("Signal-level override failed: %s", exc)
            return {}

    @staticmethod
    def _calculate_return_based_performance(returns: np.ndarray) -> Dict[str, float]:
        """Return simplified performance stats used for config thresholds."""
        if returns.size == 0:
            return {'win_rate': 0.0, 'profit_factor': 0.0, 'gross_profit': 0.0, 'gross_loss': 0.0,
                    'avg_gain': 0.0, 'avg_loss': 0.0, 'payoff_asymmetry': 0.0,
                    'n_wins': 0, 'n_losses': 0, 'winner_concentration_ratio': 0.0,
                    'trimmed_payoff_asymmetry': 0.0, 'payoff_asymmetry_support_ok': False,
                    'payoff_asymmetry_effective': 0.0}

        positive = returns[returns > 0]
        negative = returns[returns < 0]
        gross_profit = float(positive.sum()) if positive.size else 0.0
        gross_loss = abs(float(negative.sum())) if negative.size else 0.0
        if gross_loss > 1e-8:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 0.0
        avg_gain = float(positive.mean()) if positive.size else 0.0
        avg_loss = float(negative.mean()) if negative.size else 0.0
        if positive.size and negative.size and abs(avg_loss) > 1e-8:
            payoff_asymmetry = avg_gain / abs(avg_loss)
        else:
            payoff_asymmetry = float('inf') if positive.size else 0.0

        support = payoff_asymmetry_support_metrics(pd.Series(returns))

        return {
            'win_rate': float(positive.size / returns.size),
            'profit_factor': float(profit_factor),
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_gain': avg_gain,
            'avg_loss': avg_loss,
            'payoff_asymmetry': float(payoff_asymmetry),
            'n_wins': int(support.get('n_wins', positive.size)),
            'n_losses': int(support.get('n_losses', negative.size)),
            'winner_concentration_ratio': float(support.get('winner_concentration_ratio', 0.0)),
            'trimmed_payoff_asymmetry': float(support.get('trimmed_payoff_asymmetry', 0.0)),
            'payoff_asymmetry_support_ok': bool(support.get('payoff_asymmetry_support_ok', False)),
            'payoff_asymmetry_effective': float(support.get('payoff_asymmetry_effective', 0.0)),
        }

    @staticmethod
    def _canonical_criterion_key(raw_key: str) -> str:
        """Normalize criterion aliases so weighted scoring and schema keys stay aligned."""
        key = str(raw_key or "").strip().lower().replace("-", "_")
        prefix = ""
        if key.startswith("drift_"):
            prefix = "drift_"
            key = key[len(prefix):]

        aliases = {
            "rmse_ratio": "rmse_ratio_vs_baseline",
            "rmse": "rmse_ratio_vs_baseline",
            "directional_accuracy": "terminal_directional_accuracy",
        }
        return prefix + aliases.get(key, key)

    def _canonicalize_criteria(self, criteria: Dict[str, bool]) -> Dict[str, bool]:
        if not isinstance(criteria, dict):
            return {}
        normalized: Dict[str, bool] = {}
        for raw_key, passed in criteria.items():
            key = self._canonical_criterion_key(str(raw_key))
            if not key:
                continue
            if key in normalized:
                # Conservative merge for duplicate aliases: both must pass.
                normalized[key] = bool(normalized[key]) and bool(passed)
            else:
                normalized[key] = bool(passed)
        return normalized

    @staticmethod
    def _edge_criteria_from_payload(
        edge_payload: Dict[str, Any],
        criteria_cfg: Optional[Dict[str, Any]],
    ) -> Dict[str, bool]:
        """Derive pass/fail criteria from a (possibly cached) edge_payload.

        Separated from _evaluate_forecast_edge() so that criteria are always
        recomputed with the current code/config thresholds — even when edge_payload
        was loaded from the in-process cache.  The CV run (which produces edge_payload)
        is expensive; the threshold checks here are O(1) and must not be stale.
        """
        edge_criteria: Dict[str, bool] = {}
        try:
            rmse_ratio = edge_payload.get("rmse_ratio_vs_baseline")
            terminal_dir_acc = edge_payload.get("terminal_directional_accuracy")
            ens_ci_cov = edge_payload.get("terminal_ci_coverage")
            base_ci_cov = edge_payload.get("baseline_ci_coverage")

            # Fix C — CI-coverage override for rmse_ratio_vs_baseline:
            # Ensemble CI coverage materially better than baseline → RMSE regression
            # does not block the signal (better tail-risk management wins).
            _ci_cov_delta = (
                (ens_ci_cov - base_ci_cov)
                if ens_ci_cov is not None and base_ci_cov is not None
                else None
            )
            _ci_cov_override_threshold = float(
                (criteria_cfg or {}).get("ci_coverage_rmse_override_delta", 0.20)
            )
            _rmse_overridden_by_ci = (
                _ci_cov_delta is not None
                and _ci_cov_delta >= _ci_cov_override_threshold
            )

            max_rmse_ratio = (criteria_cfg or {}).get("max_rmse_ratio_vs_baseline")
            if max_rmse_ratio is not None:
                try:
                    thr = float(max_rmse_ratio)
                    passes_rmse = rmse_ratio is not None and rmse_ratio <= thr
                    edge_criteria["rmse_ratio_vs_baseline"] = passes_rmse or _rmse_overridden_by_ci
                except (TypeError, ValueError):
                    pass
            min_dir_acc = (criteria_cfg or {}).get("min_terminal_directional_accuracy") or (
                criteria_cfg or {}).get("min_directional_accuracy")
            if min_dir_acc is not None:
                try:
                    thr = float(min_dir_acc)
                    edge_criteria["terminal_directional_accuracy"] = (
                        terminal_dir_acc is not None and terminal_dir_acc >= thr
                    )
                except (TypeError, ValueError):
                    pass
            if "rmse_ratio_vs_baseline" not in edge_criteria and rmse_ratio is not None:
                # Conservative default when no explicit criteria provided.
                edge_criteria["rmse_ratio_vs_baseline"] = rmse_ratio <= 1.10 or _rmse_overridden_by_ci
            if "terminal_directional_accuracy" not in edge_criteria and terminal_dir_acc is not None:
                edge_criteria["terminal_directional_accuracy"] = terminal_dir_acc >= 0.50
        except Exception:  # pragma: no cover
            pass
        return edge_criteria

    @staticmethod
    def _default_domain_utility_weights() -> Dict[str, float]:
        """Default weights for the asymmetry-first domain utility scorer.

        Weight rationale (barbell objective, Nigeria jurisdiction):
        - expected_profit (0.18): primary economic gate — signal must clear transaction
          costs and produce meaningful dollar edge. Slightly reduced because it
          already remains a hard gate by default.
        - omega_ratio (0.24): primary barbell outcome metric; if this rises above 1.0,
          the realized return distribution is beating the NGN hurdle.
        - payoff_asymmetry (0.16): avg_win / |avg_loss|. This is the structural engine
          behind the current 2.65x realized winner/loser profile and should be scored
          directly instead of only being inferred via profit factor.
        - profit_factor (0.10): retains realized dollar win/loss context, but with lower
          weight because it mixes payoff shape with hit-rate frequency.
        - terminal_directional_accuracy (0.10): did the forecast call the 30-bar
          direction correctly?  Reduced from 0.18 because it is only meaningful when
          n_obs >= 30 and the 0.50 boundary already has zero normalized score.
        - terminal_ci_coverage (0.10): did the actual terminal price land inside the
          CI?  This directly measures whether the model's uncertainty quantification
          is trustworthy for stop/target placement — the barbell's tail-control axis.
          A model with better CI coverage supports more accurate position sizing even
          if its point-forecast RMSE is worse.
        - max_drawdown (0.06): loss-control gate; kept explicit so upside does not win
          by simply ignoring the path of losses.
        - expected_shortfall (0.06): tail-loss below NGN daily threshold (~0.108%/day);
          uses domain-specific hurdle, not generic -0.02 USD floor.

        Weights sum to 1.0.  terminal_ci_coverage is only injected when available
        (forecast_edge mode populates it from OOS fold CI coverage data).  When absent,
        the weight resolver distributes proportionally among present components.
        """
        return {
            "expected_profit": 0.18,
            "omega_ratio": 0.20,
            "omega_robustness_score": 0.10,
            "payoff_asymmetry": 0.14,
            "profit_factor": 0.10,
            "terminal_directional_accuracy": 0.10,
            "terminal_ci_coverage": 0.10,
            "max_drawdown": 0.04,
            "expected_shortfall": 0.04,
        }

    @staticmethod
    def _resolve_success_threshold(
        criteria_cfg: Optional[Dict[str, Any]],
        primary_key: str,
        *,
        aliases: Tuple[str, ...] = (),
    ) -> Tuple[Any, Optional[str]]:
        if not isinstance(criteria_cfg, dict):
            return None, None
        if primary_key in criteria_cfg:
            return criteria_cfg.get(primary_key), None
        for alias in aliases:
            if alias in criteria_cfg:
                return criteria_cfg.get(alias), alias
        return None, None

    @staticmethod
    def _effective_expected_profit_floor(
        criteria_cfg: Optional[Dict[str, Any]],
        *,
        position_value: Optional[float],
    ) -> float:
        if not isinstance(criteria_cfg, dict):
            return 1.0

        try:
            abs_floor = float(criteria_cfg.get("min_expected_profit", 1.0))
        except (TypeError, ValueError):
            abs_floor = 1.0

        try:
            pct_floor = float(criteria_cfg.get("min_expected_profit_pct", 0.0))
        except (TypeError, ValueError):
            pct_floor = 0.0

        try:
            trade_notional = float(position_value) if position_value is not None else 0.0
        except (TypeError, ValueError):
            trade_notional = 0.0

        pct_floor_abs = trade_notional * pct_floor if pct_floor > 0 else 0.0
        positive_floors = [value for value in (abs_floor, pct_floor_abs) if value > 0]
        return float(min(positive_floors)) if positive_floors else 1.0

    @staticmethod
    def _ignored_success_threshold_warnings(criteria_cfg: Optional[Dict[str, Any]]) -> List[str]:
        if not isinstance(criteria_cfg, dict):
            return []
        ignored_keys = (
            "min_sharpe",
            "min_sortino",
            "min_win_rate",
            "min_annual_return",
        )
        return [f"ignored_success_threshold:{key}" for key in ignored_keys if key in criteria_cfg]

    @staticmethod
    def _resolve_bucket_threshold(
        criteria_cfg: Optional[Dict[str, Any]],
        key: str,
        bucket: str,
        default: Optional[float] = None,
    ) -> Optional[float]:
        if not isinstance(criteria_cfg, dict):
            return default
        raw = criteria_cfg.get(key)
        if isinstance(raw, dict):
            raw = raw.get(bucket) if bucket in raw else raw.get("default", default)
        if raw is None:
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    def _collect_barbell_path_metrics(
        self,
        *,
        ticker: str,
        signal: TimeSeriesSignal,
        market_data: Optional[pd.DataFrame],
        position_value: float,
    ) -> Dict[str, Any]:
        cfg = self._get_barbell_config()
        if (
            cfg is None
            or build_barbell_market_context is None
            or evaluate_barbell_path_risk is None
            or resolve_barbell_bucket is None
        ):
            return {}

        provenance = signal.provenance if isinstance(signal.provenance, dict) else {}
        decision_ctx = provenance.get("decision_context") if isinstance(provenance.get("decision_context"), dict) else {}
        signal_payload = {
            "expected_return_net": decision_ctx.get("net_trade_return", getattr(signal, "expected_return", None)),
            "forecast_horizon": getattr(signal, "forecast_horizon", None),
            "roundtrip_cost_bps": decision_ctx.get("roundtrip_cost_bps"),
            "position_value": position_value,
            "leverage": decision_ctx.get("leverage") or provenance.get("leverage"),
        }
        try:
            context = build_barbell_market_context(
                signal_payload=signal_payload,
                market_data=market_data,
                detected_regime=str(provenance.get("detected_regime") or "").strip().upper() or None,
            )
        except Exception:
            return {}

        assessed = evaluate_barbell_path_risk(context=context, cfg=cfg)
        diagnostics = dict(assessed.get("diagnostics") or {})
        diagnostics["barbell_path_risk_ok"] = bool(assessed.get("barbell_path_risk_ok", True))
        diagnostics["path_risk_checks"] = dict(assessed.get("path_risk_checks") or {})
        diagnostics["barbell_bucket"] = resolve_barbell_bucket(ticker, cfg)
        return diagnostics

    @staticmethod
    def _default_hard_gate_criteria() -> Tuple[str, ...]:
        """Criteria that remain hard blockers under the barbell objective by default."""
        return ("expected_profit", "significance", "information_ratio")

    def _resolve_hard_gate_criteria(
        self,
        *,
        config: Optional[Dict[str, Any]],
        criteria_cfg: Optional[Dict[str, Any]],
        structural_gates: Dict[str, bool],
        bucket: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """Resolve hard-gate criteria without letting soft forecast diagnostics veto payoff asymmetry."""
        raw_config = config if isinstance(config, dict) else {}
        configured = raw_config.get("hard_gate_criteria")
        warnings: List[str] = []
        if configured is None and raw_config.get("structural_hard_gate_criteria") is not None:
            configured = raw_config.get("structural_hard_gate_criteria")
            warnings.append("deprecated_config_key:structural_hard_gate_criteria")

        if isinstance(configured, (list, tuple, set)):
            requested = [self._canonical_criterion_key(str(item)) for item in configured]
        else:
            requested = [self._canonical_criterion_key(item) for item in self._default_hard_gate_criteria()]

        hard_gate_keys: List[str] = []
        for key in requested:
            if key and key in structural_gates and key not in hard_gate_keys:
                hard_gate_keys.append(key)

        # expected_profit remains a non-negotiable economic viability gate whenever present.
        if "expected_profit" in structural_gates and "expected_profit" not in hard_gate_keys:
            hard_gate_keys.insert(0, "expected_profit")

        # crisis_regime_path_risk_block is a structural invariant: cannot be
        # removed from hard-gate set via config.  CRISIS + bad path risk = hard FAIL.
        if (
            "crisis_regime_path_risk_block" in structural_gates
            and "crisis_regime_path_risk_block" not in hard_gate_keys
        ):
            hard_gate_keys.append("crisis_regime_path_risk_block")

        if isinstance(criteria_cfg, dict):
            bucket_cfg = criteria_cfg.get("bucket_hard_gate_criteria")
            requested_bucket = str(bucket or "").strip().lower()
            if isinstance(bucket_cfg, dict) and requested_bucket:
                bucket_requested = bucket_cfg.get(requested_bucket) or []
                if isinstance(bucket_requested, (list, tuple, set)):
                    for raw_key in bucket_requested:
                        key = self._canonical_criterion_key(str(raw_key))
                        if key and key in structural_gates and key not in hard_gate_keys:
                            hard_gate_keys.append(key)

        return sorted(hard_gate_keys), warnings

    @staticmethod
    def _normalize_domain_utility_component(
        *,
        name: str,
        raw_value: Any,
        threshold: Any,
    ) -> Tuple[Optional[float], bool]:
        key = str(name or "").strip().lower()

        try:
            raw = float(raw_value)
        except (TypeError, ValueError):
            return None, False

        if math.isnan(raw):
            return None, False

        if key in {"omega_ratio", "profit_factor", "payoff_asymmetry", "payoff_asymmetry_effective"}:
            try:
                thr = float(threshold)
            except (TypeError, ValueError):
                thr = 1.0
            thr = max(thr, 1e-6)
            if math.isinf(raw):
                return (1.0 if raw > 0 else 0.0), bool(raw > 0)
            passed = raw >= thr
            normalized = math.log1p(max(raw, 0.0) / thr) / math.log1p(3.0)
            return float(np.clip(normalized, 0.0, 1.0)), passed

        if key == "omega_robustness_score":
            try:
                thr = float(threshold)
            except (TypeError, ValueError):
                thr = 0.45
            thr = float(np.clip(thr, 0.0, 0.999999))
            passed = raw >= thr
            denom = max(1.0 - thr, 1e-6)
            normalized = (raw - thr) / denom
            return float(np.clip(normalized, 0.0, 1.0)), passed

        if key == "expected_profit":
            try:
                thr = float(threshold)
            except (TypeError, ValueError):
                thr = 1.0
            thr = max(thr, 1e-6)
            passed = raw >= thr
            if raw <= 0:
                return 0.0, False
            normalized = math.log1p(raw / thr) / math.log1p(4.0)
            return float(np.clip(normalized, 0.0, 1.0)), passed

        if key == "terminal_directional_accuracy":
            try:
                thr = float(threshold)
            except (TypeError, ValueError):
                thr = 0.5
            thr = float(np.clip(thr, 0.0, 0.999999))
            passed = raw >= thr
            denom = max(1.0 - thr, 1e-6)
            normalized = (raw - thr) / denom
            return float(np.clip(normalized, 0.0, 1.0)), passed

        if key == "max_drawdown":
            try:
                thr = float(threshold)
            except (TypeError, ValueError):
                thr = 0.25
            thr = max(thr, 1e-6)
            passed = raw <= thr
            normalized = (thr - raw) / thr
            return float(np.clip(normalized, 0.0, 1.0)), passed

        if key == "expected_shortfall":
            try:
                thr = float(threshold)
            except (TypeError, ValueError):
                thr = -0.02
            if math.isnan(thr):
                thr = -0.02
            upper = 0.0
            lower = thr if thr < upper else -0.02
            span = max(upper - lower, 1e-6)
            passed = raw >= lower
            normalized = (raw - lower) / span
            return float(np.clip(normalized, 0.0, 1.0)), passed

        if key == "terminal_ci_coverage":
            # CI coverage is a fraction [0,1] measuring how often realized terminal
            # price fell inside the forecast CI.  Normalise linearly between the
            # threshold (→0) and 1.0 (→1).  Scores below threshold are clamped to 0.
            try:
                thr = float(threshold)
            except (TypeError, ValueError):
                thr = 0.25
            thr = float(np.clip(thr, 0.0, 0.999))
            passed = raw >= thr
            denom = max(1.0 - thr, 1e-6)
            normalized = (raw - thr) / denom
            return float(np.clip(normalized, 0.0, 1.0)), passed

        return None, False

    def _build_domain_utility(
        self,
        *,
        metrics: Dict[str, float],
        performance_snapshot: Dict[str, float],
        edge_block: Dict[str, Any],
        criteria_cfg: Optional[Dict[str, Any]],
        config: Dict[str, Any],
        expected_profit: float,
        position_value: Optional[float],
    ) -> Dict[str, Any]:
        expected_profit_floor = self._effective_expected_profit_floor(
            criteria_cfg,
            position_value=position_value,
        )
        terminal_da_threshold, terminal_da_alias = self._resolve_success_threshold(
            criteria_cfg,
            "min_terminal_directional_accuracy",
            aliases=("min_directional_accuracy",),
        )
        if terminal_da_threshold is None:
            terminal_da_threshold = 0.50

        expected_shortfall_threshold = None
        if isinstance(criteria_cfg, dict):
            expected_shortfall_threshold = criteria_cfg.get("min_expected_shortfall")
        if expected_shortfall_threshold is None:
            # Domain-specific default: 10× the NGN daily hurdle rate.
            #
            # The NGN daily hurdle (~0.108%/day) is the minimum acceptable RETURN,
            # not an ES floor.  Expected shortfall (CVaR 5%) for US equities is
            # typically 1-3% per day (e.g., AAPL: ~1-1.5% from 365-day window).
            # Using -DAILY_NGN_THRESHOLD directly (-0.108%) would block every real
            # equity signal — a threshold miscalibration, not a domain improvement.
            #
            # 10× multiplier rationale:
            #   - Tail losses should not exceed 10× the daily return hurdle in a
            #     worst-5% scenario.  At hurdle 0.108%/day: threshold = -1.08%/day.
            #   - This is stricter than the US-convention -2% (0.92pp tighter) but
            #     remains reachable for investment-grade US equities (AAPL ES ≈ -1%).
            #   - Signals with ES worse than -1.08% represent excessive tail risk
            #     relative to the NGN benchmark even if omega_ratio passes.
            #   - Configurable via min_expected_shortfall in success_criteria.
            try:
                from etl.portfolio_math import DAILY_NGN_THRESHOLD as _NGN_DAILY
                expected_shortfall_threshold = -10.0 * float(_NGN_DAILY)
            except Exception:
                expected_shortfall_threshold = -0.0108  # fallback: 10× (31%/252)

        validation_mode = str(config.get("validation_mode") or "drift_proxy").lower()

        # CI coverage is a primary barbell component — it measures whether the model's
        # uncertainty bounds actually contain the realized outcome.  Good CI coverage
        # enables correct position sizing and tail-risk management even when point
        # forecast RMSE is worse than the baseline.  Extract from edge_block when
        # available (forecast_edge mode populates it from OOS fold metrics).
        terminal_ci_coverage = edge_block.get("terminal_ci_coverage")

        utility_values = {
            "expected_profit": expected_profit,
            "omega_ratio": metrics.get("omega_ratio"),
            "omega_robustness_score": metrics.get("omega_robustness_score"),
            "payoff_asymmetry": performance_snapshot.get("payoff_asymmetry"),
            "profit_factor": performance_snapshot.get("profit_factor"),
            "max_drawdown": metrics.get("max_drawdown"),
            "expected_shortfall": metrics.get("expected_shortfall"),
        }
        if terminal_ci_coverage is not None:
            utility_values["terminal_ci_coverage"] = terminal_ci_coverage

        utility_thresholds = {
            "expected_profit": expected_profit_floor,
            "omega_ratio": (criteria_cfg or {}).get("min_omega_ratio", 1.0),
            "omega_robustness_score": (criteria_cfg or {}).get("min_omega_robustness_score", 0.45),
            "payoff_asymmetry": (criteria_cfg or {}).get("min_payoff_asymmetry", 1.25),
            "profit_factor": (criteria_cfg or {}).get("min_profit_factor", 1.0),
            "max_drawdown": (criteria_cfg or {}).get("max_drawdown", 0.25),
            "expected_shortfall": expected_shortfall_threshold,
        }
        if terminal_ci_coverage is not None:
            # CI coverage threshold: ensemble must cover at least 25% of realized
            # terminal prices within its CI bounds.  Below this the CI is too narrow
            # to be trusted for stop/target placement (barbell tail-risk management).
            utility_thresholds["terminal_ci_coverage"] = float(
                (criteria_cfg or {}).get("min_terminal_ci_coverage", 0.25)
            )
        if validation_mode == "forecast_edge" or edge_block.get("terminal_directional_accuracy") is not None:
            utility_values["terminal_directional_accuracy"] = edge_block.get("terminal_directional_accuracy")
            utility_thresholds["terminal_directional_accuracy"] = terminal_da_threshold

        configured_weights = None
        weights_source = "utility_weights"
        config_warnings: List[str] = []
        if isinstance(config, dict):
            configured_weights = config.get("utility_weights")
            if configured_weights is None and isinstance(config.get("criterion_weights"), dict):
                configured_weights = config.get("criterion_weights")
                weights_source = "criterion_weights"
                config_warnings.append("deprecated_config_key:criterion_weights")
        if terminal_da_alias:
            config_warnings.append(f"deprecated_success_threshold:{terminal_da_alias}")

        resolved_weights, missing_keys, unused_keys = self._resolve_weighted_scoring_weights(
            criteria={key: True for key in utility_values},
            configured_weights=configured_weights,
        )

        missing_value_keys: List[str] = []
        utility_breakdown: Dict[str, Dict[str, Any]] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for key, raw_value in utility_values.items():
            normalized_score, passed_threshold = self._normalize_domain_utility_component(
                name=key,
                raw_value=raw_value,
                threshold=utility_thresholds.get(key),
            )
            if normalized_score is None:
                missing_value_keys.append(key)
                weighted_contribution = 0.0
            else:
                weight = float(resolved_weights.get(key, 0.0))
                weighted_contribution = weight * normalized_score
                total_weight += weight
                weighted_sum += weighted_contribution

            utility_breakdown[key] = {
                "raw_value": raw_value,
                "threshold": utility_thresholds.get(key),
                "normalized_score": normalized_score,
                "weight": float(resolved_weights.get(key, 0.0)),
                "weighted_contribution": float(weighted_contribution),
                "passed_threshold": bool(passed_threshold),
            }

        utility_score = (weighted_sum / total_weight) if total_weight > 0 else None
        coverage_ok = not missing_keys and not missing_value_keys
        return {
            "utility_breakdown": utility_breakdown,
            "utility_score": utility_score,
            "total_weight": total_weight,
            "weight_validation": {
                "strict_weight_coverage": True,
                "missing_weight_keys": missing_keys,
                "unused_weight_keys": unused_keys,
                "missing_component_values": missing_value_keys,
                "coverage_ok": coverage_ok,
                "weights_source": weights_source,
            },
            "config_warnings": config_warnings,
        }

    def _resolve_weighted_scoring_weights(
        self,
        *,
        criteria: Dict[str, bool],
        configured_weights: Any,
    ) -> Tuple[Dict[str, float], List[str], List[str]]:
        """
        Resolve weights for weighted scoring with strict key-coverage diagnostics.

        Returns:
            (resolved_weights_by_criterion, missing_criterion_weights, unused_weight_keys)
        """
        defaults = self._default_domain_utility_weights()
        normalized_config: Dict[str, float] = {}
        if isinstance(configured_weights, dict):
            for raw_key, raw_weight in configured_weights.items():
                key = self._canonical_criterion_key(str(raw_key))
                if not key:
                    continue
                try:
                    weight = float(raw_weight)
                except (TypeError, ValueError):
                    continue
                if weight <= 0:
                    continue
                normalized_config[key] = weight

        criteria_keys = [self._canonical_criterion_key(k) for k in criteria]
        criteria_keys = [k for k in criteria_keys if k]

        resolved: Dict[str, float] = {}
        missing: List[str] = []
        for key in criteria_keys:
            if key in normalized_config:
                resolved[key] = float(normalized_config[key])
                continue
            if key in defaults:
                resolved[key] = float(defaults[key])
            else:
                resolved[key] = 0.0
            if isinstance(configured_weights, dict):
                missing.append(key)

        unused = sorted(
            key for key in normalized_config.keys() if key not in set(criteria_keys)
        )
        return resolved, sorted(set(missing)), unused

    @staticmethod
    def _evaluate_success_criteria(
        criteria_cfg: Optional[Dict[str, Any]],
        metrics: Dict[str, float],
        performance_snapshot: Dict[str, float],
        significance: Optional[Dict[str, Any]],
        expected_profit: float,
        position_value: Optional[float] = None,
        action: str = "HOLD",
    ) -> Dict[str, bool]:
        """Evaluate metrics vs configuration thresholds."""
        if not isinstance(criteria_cfg, dict) or not criteria_cfg:
            return {}

        results: Dict[str, bool] = {}

        action_upper = str(action or "HOLD").upper()
        if action_upper != "HOLD" and ('min_expected_profit' in criteria_cfg or 'min_expected_profit_pct' in criteria_cfg):
            abs_floor = float(criteria_cfg.get('min_expected_profit', 1.0))
            pct_floor = float(criteria_cfg.get('min_expected_profit_pct', 0.0))
            # PASS if expected_profit meets EITHER the absolute OR the relative floor.
            abs_pass = expected_profit >= abs_floor
            if pct_floor > 0:
                # Relative floor is per-trade notional, not global capital base.
                try:
                    trade_notional = float(position_value) if position_value is not None else 0.0
                except (TypeError, ValueError):
                    trade_notional = 0.0
                rel_floor_dollars = trade_notional * pct_floor
                rel_pass = rel_floor_dollars > 0 and expected_profit >= rel_floor_dollars
            else:
                rel_pass = False
            results['expected_profit'] = abs_pass or rel_pass

        if criteria_cfg.get('require_significance'):
            if significance is None:
                results['statistical_significance'] = False
            else:
                results['statistical_significance'] = bool(significance.get('significant'))

        if 'min_information_ratio' in criteria_cfg and significance is not None:
            ir = significance.get('information_ratio')
            if ir is not None:
                results['information_ratio'] = float(ir) >= float(criteria_cfg['min_information_ratio'])

        bucket = str(metrics.get("barbell_bucket") or "").strip().lower()
        bucket_cfg = criteria_cfg.get("bucket_hard_gate_criteria")
        requested_bucket_gates = []
        if isinstance(bucket_cfg, dict) and bucket:
            raw_requested = bucket_cfg.get(bucket) or []
            if isinstance(raw_requested, (list, tuple, set)):
                requested_bucket_gates = [str(item).strip().lower().replace("-", "_") for item in raw_requested if str(item).strip()]

        if "expected_shortfall" in requested_bucket_gates:
            try:
                es_value = metrics.get("expected_shortfall")
                es_threshold = float(criteria_cfg.get("min_expected_shortfall", -0.0108))
                results["expected_shortfall"] = es_value is not None and float(es_value) >= es_threshold
            except (TypeError, ValueError):
                results["expected_shortfall"] = False

        if "gap_risk_to_edge" in requested_bucket_gates:
            ratio = metrics.get("gap_risk_to_edge")
            threshold = TimeSeriesSignalGenerator._resolve_bucket_threshold(
                criteria_cfg,
                "max_gap_risk_to_edge",
                bucket,
                default=None,
            )
            results["gap_risk_to_edge"] = (
                ratio is not None and threshold is not None and float(ratio) <= float(threshold)
            )

        if "liquidity_to_depth" in requested_bucket_gates:
            ratio = metrics.get("liquidity_to_depth")
            threshold = TimeSeriesSignalGenerator._resolve_bucket_threshold(
                criteria_cfg,
                "max_liquidity_to_depth",
                bucket,
                default=None,
            )
            results["liquidity_to_depth"] = (
                ratio is not None and threshold is not None and float(ratio) <= float(threshold)
            )

        if "barbell_path_risk_ok" in requested_bucket_gates:
            results["barbell_path_risk_ok"] = bool(metrics.get("barbell_path_risk_ok"))

        return results

    def _render_quant_validation_plot(
        self,
        ticker: str,
        market_data: pd.DataFrame,
        output_dir: str,
        max_points: int,
    ) -> Optional[str]:
        """Persist optional visualization artifact for dashboards."""
        try:
            from etl.visualizer import TimeSeriesVisualizer  # pylint: disable=import-outside-toplevel
            import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        except Exception as exc:  # pragma: no cover
            logger.debug("Quant validation visualization skipped: %s", exc)
            return None

        subset = market_data[['Close']].tail(max(max_points, 2))
        if subset.empty:
            return None

        visualizer = TimeSeriesVisualizer()
        fig = visualizer.plot_time_series_overview(subset, title=f"{ticker} Quant Validation")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        target = output_path / f"{ticker}_quant_validation.png"
        fig.savefig(target, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return str(target)

    def _calibrate_confidence(
        self,
        raw_confidence: float,
        ticker: str,
        db_path: Optional[str] = None,
    ) -> float:
        """Calibrate raw confidence using realized win/loss outcomes.

        Priority: (1) quant_validation.jsonl outcome pairs, (2) trade_executions DB.
        Also stores the pure Platt probability in self._platt_calibrated before
        blending so it can be written to the confidence_calibrated DB column.
        """
        self._platt_calibrated = None
        self._last_raw_confidence: Optional[float] = None
        if raw_confidence is None:
            return 0.50

        raw_conf = float(raw_confidence)
        # Record pre-blend raw confidence so JSONL entries store the value that
        # LR predict_proba is evaluated on (fixes train/predict distribution mismatch).
        self._last_raw_confidence = raw_conf

        # 1. JSONL-based (conf, outcome) pairs — populated by update_platt_outcomes.py.
        pairs_conf, pairs_win = self._load_jsonl_outcome_pairs(limit=2000)
        source = "jsonl"

        _j_n = len(pairs_conf)
        _j_w = int(sum(pairs_win)) if pairs_win else 0
        _j_l = _j_n - _j_w
        db_file = Path(
            db_path
            or os.getenv("PORTFOLIO_DB_PATH")
            or "data/portfolio_maximizer.db"
        )
        if _j_n < 30 or _j_w < 5 or _j_l < 5:
            if _j_n >= 30 and _j_l < 5:
                # JSONL has enough total pairs but is class-imbalanced (very few losses).
                # AUGMENT with DB non-mechanical pairs instead of replacing — preserves
                # the JSONL signal mass while adding the minority class from live trades.
                # Replacing would discard 30+ valid JSONL pairs in favour of DB-only.
                _aug_conf, _aug_win = self._load_realized_outcome_pairs(
                    db_file=db_file,
                    ticker="",
                    limit=200,
                )
                pairs_conf = pairs_conf + _aug_conf
                pairs_win = pairs_win + _aug_win
                source = "jsonl+db_augmented"
            else:
                # 2. Ticker-local DB fallback — JSONL is too small.
                pairs_conf, pairs_win = self._load_realized_outcome_pairs(
                    db_file=db_file,
                    ticker=ticker,
                    limit=1200,
                )
                source = "db_local"

                _db_n = len(pairs_conf)
                _db_w = int(sum(pairs_win)) if pairs_win else 0
                _db_l = _db_n - _db_w
                if (_db_n < 30 or _db_w < 5 or _db_l < 5) and ticker:
                    # 3. Global DB fallback.
                    pairs_conf, pairs_win = self._load_realized_outcome_pairs(
                        db_file=db_file,
                        ticker="",
                        limit=2000,
                    )
                    source = "db_global"

        n = len(pairs_conf)
        wins = int(sum(pairs_win))
        losses = int(n - wins)
        # Minimum pairs raised to 43: 70/30 split gives 30 train + 13 holdout.
        # At 30 pairs (old floor) we only get 9 holdout samples — insufficient
        # to distinguish a miscalibrated model from random variation.
        if n < 43 or wins < 5 or losses < 5:
            logger.debug(
                "Outcome calibration skipped (n=%d wins=%d losses=%d source=%s); using raw.",
                n, wins, losses, source,
            )
            return float(max(0.05, min(0.95, raw_conf)))

        try:
            import numpy as _np  # pylint: disable=import-outside-toplevel
            x = _np.array(pairs_conf, dtype=float).reshape(-1, 1)
            y = _np.array(pairs_win, dtype=float)

            # LEAK-01 fix: time-based 70/30 train/holdout split.
            # Pairs are chronologically ordered (DB: by rowid; JSONL: by file order).
            # Training uses only the first 70%; holdout validates calibration quality.
            # Floor at 30 train samples ensures the logistic regression is not noise-fit.
            split_idx = max(int(len(x) * 0.70), 30)
            if split_idx >= len(x):
                # Insufficient data for a meaningful split -- skip calibration
                logger.debug(
                    "Platt calibration skipped -- need >%d pairs for 70/30 split (have %d)",
                    split_idx, len(x),
                )
                return float(max(0.05, min(0.95, raw_conf)))

            x_train, x_holdout = x[:split_idx], x[split_idx:]
            y_train, y_holdout = y[:split_idx], y[split_idx:]

            # PLATT-BUG3: class-imbalance guard — chronological ordering means recent
            # losses cluster in the holdout slice, leaving training single-class.
            # sklearn LR raises ValueError on single-class input; guard prevents silent
            # raw-conf fallback masking the root cause.
            _train_classes = set(int(v) for v in y_train)
            if len(_train_classes) < 2:
                logger.warning(
                    "Platt calibration skipped -- training slice is single-class "
                    "(classes=%s, n_train=%d, n_total=%d source=%s); using raw.",
                    _train_classes, len(y_train), n, source,
                )
                return float(max(0.05, min(0.95, raw_conf)))

            from sklearn.linear_model import LogisticRegression  # pylint: disable=import-outside-toplevel
            clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
            clf.fit(x_train, y_train)

            # Evaluate on holdout to detect overfitting / poisoned training data.
            # Guard raised to 0.55 and minimum holdout size raised to 10:
            # a 0.50 floor with n_holdout=9 allows effectively-random calibration
            # to pass and corrupt the blended confidence.
            if len(x_holdout) >= 10:
                holdout_preds = clf.predict(x_holdout)
                holdout_acc = float((holdout_preds == y_holdout).mean())
                self._calibration_holdout_accuracy = holdout_acc
                if holdout_acc < 0.55:
                    logger.warning(
                        "Platt holdout accuracy %.2f below 0.55 (n_holdout=%d) -- "
                        "falling back to raw confidence to avoid miscalibration.",
                        holdout_acc, len(x_holdout),
                    )
                    return float(max(0.05, min(0.95, raw_conf)))

            calibrated = float(clf.predict_proba([[raw_conf]])[0][1])
            # Phase 7.31: NaN/inf guard — LogisticRegression can produce non-finite
            # values on degenerate inputs; fall back to clamped raw_conf if so.
            if not math.isfinite(calibrated):
                logger.warning(
                    "Platt calibration returned non-finite %.6f; "
                    "falling back to raw_conf=%.4f",
                    calibrated, raw_conf,
                )
                return float(max(0.05, min(0.95, raw_conf)))
            # Store pure Platt probability for the confidence_calibrated DB column.
            self._platt_calibrated = float(max(0.05, min(0.95, calibrated)))
            qv_cfg = getattr(self, "quant_validation_config", None) or {}
            calibration_cfg = qv_cfg.get("calibration") if isinstance(qv_cfg, dict) else {}
            calibration_cfg = calibration_cfg if isinstance(calibration_cfg, dict) else {}
            try:
                raw_weight = float(calibration_cfg.get("raw_weight", 0.80))
            except (TypeError, ValueError):
                raw_weight = 0.80
            raw_weight = max(0.0, min(1.0, raw_weight))
            # Dynamic raw_weight ramp: as pairs accumulate, reduce raw_weight so Platt
            # has meaningful influence. At 43 pairs (minimum floor) Platt has 20% weight;
            # at 100+ pairs it grows to 50%. Config raw_weight is the MAXIMUM (early-data cap).
            # Formula: ramp = 0.80 - 0.30 * min(1.0, (n - 43) / 57)
            #   n=43  → ramp=0.80 (20% Platt — same as before)
            #   n=100 → ramp=0.50 (50% Platt — meaningful correction)
            #   n=200 → ramp=0.50 (capped — prevent over-correction)
            ramp_raw_weight = 0.80 - 0.30 * min(1.0, max(0.0, (n - 43) / 57.0))
            # Apply ramp only when it is more restrictive than the config value
            raw_weight = min(raw_weight, ramp_raw_weight)
            try:
                max_downside = float(calibration_cfg.get("max_downside_adjustment", 0.15))
            except (TypeError, ValueError):
                max_downside = 0.15
            max_downside = max(0.0, min(1.0, max_downside))
            try:
                max_upside = float(calibration_cfg.get("max_upside_adjustment", 0.10))
            except (TypeError, ValueError):
                max_upside = 0.10
            max_upside = max(0.0, min(1.0, max_upside))
            blended = (raw_weight * raw_conf) + ((1.0 - raw_weight) * calibrated)
            # Symmetric correction window: prevent both over-deflation and over-inflation.
            blended = max(blended, raw_conf - max_downside)
            blended = min(blended, raw_conf + max_upside)
            logger.debug(
                (
                    "Outcome calibration: raw=%.3f model=%.3f blended=%.3f "
                    "(n=%d wins=%d losses=%d ticker=%s raw_weight=%.2f source=%s)"
                ),
                raw_conf, calibrated, blended, n, wins, losses, ticker, raw_weight, source,
            )
            return float(max(0.05, min(0.95, blended)))
        except Exception as exc:
            logger.warning("Outcome calibration fit failed (%s); using raw confidence.", exc)
            return float(max(0.05, min(0.95, raw_conf)))

    def _load_jsonl_outcome_pairs(
        self,
        *,
        limit: int = 2000,
    ) -> Tuple[List[float], List[float]]:
        """Load (confidence, win) pairs from quant_validation.jsonl outcome entries.

        Only entries that have both 'confidence' and 'outcome' fields are used.
        'outcome' is written by update_platt_outcomes.py after a trade closes.
        """
        pairs_conf: List[float] = []
        pairs_win: List[float] = []
        config = self.quant_validation_config or {}
        logging_cfg = config.get("logging") or {}
        if not logging_cfg.get("enabled"):
            return pairs_conf, pairs_win

        log_dir = Path(logging_cfg.get("log_dir", "logs/signals"))
        log_file = log_dir / logging_cfg.get("filename", "quant_validation.jsonl")
        if not log_file.exists():
            return pairs_conf, pairs_win

        try:
            lines = log_file.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            logger.debug("_load_jsonl_outcome_pairs: cannot read %s (%s)", log_file, exc)
            return pairs_conf, pairs_win

        # Iterate newest-first so limit applies to most recent data
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            action = str(entry.get("action", "")).upper()
            if action and action not in {"BUY", "SELL"}:
                continue
            # Exclude synthetic execution entries using the canonical helper so
            # the JSONL tier matches the execution/DB provenance contract.
            exec_mode = str(entry.get("execution_mode") or "").lower()
            data_source_hint = entry.get("data_source")
            if data_source_hint is None and isinstance(entry.get("market_context"), dict):
                data_source_hint = entry["market_context"].get("data_source")
            if is_synthetic_mode(execution_mode=exec_mode or None, data_source=data_source_hint):
                continue
            # Prefer raw_confidence (pre-blend) so LR trains on the same distribution
            # that predict_proba is evaluated on.  Fall back to blended 'confidence'
            # for backward-compatible reads of older JSONL entries (Phase 7.16-C1).
            conf_raw = entry.get("raw_confidence") if entry.get("raw_confidence") is not None else entry.get("confidence")
            outcome = entry.get("outcome")
            if conf_raw is None or outcome is None:
                continue
            if not isinstance(outcome, dict):
                continue
            win_raw = outcome.get("win")
            if win_raw is None:
                continue
            # Exclude mechanical exits: stop-loss and time-based (max_holding) exits
            # are directionally uninformative — the trade was terminated by a risk
            # guard, not by the model's prediction being confirmed or refuted.
            # Including them poisons the logistic regression with irrelevant labels.
            exit_reason = str(outcome.get("exit_reason") or entry.get("exit_reason") or "").lower()
            if exit_reason in {"stop_loss", "max_holding", "time_exit", "forced_exit"}:
                continue
            try:
                pairs_conf.append(float(conf_raw))
                pairs_win.append(1.0 if win_raw else 0.0)
            except (TypeError, ValueError):
                continue
            if len(pairs_conf) >= limit:
                break

        return pairs_conf, pairs_win

    def _load_realized_outcome_pairs(
        self,
        *,
        db_file: Path,
        ticker: str,
        limit: int,
    ) -> Tuple[List[float], List[float]]:
        pairs_conf: List[float] = []
        pairs_win: List[float] = []
        if not db_file.exists():
            return pairs_conf, pairs_win

        query = [
            "SELECT",
            "  COALESCE(confidence_calibrated, effective_confidence, base_confidence) AS conf,",
            "  realized_pnl",
            "FROM trade_executions",
            "WHERE realized_pnl IS NOT NULL",
            "  AND action IN ('BUY', 'SELL')",
            "  AND COALESCE(confidence_calibrated, effective_confidence, base_confidence) IS NOT NULL",
            "  AND is_close = 1",
            "  AND is_diagnostic = 0",
            "  AND is_synthetic = 0",
            # Exclude mechanical exits: stop_loss and max_holding exits are directionally
            # uninformative and poison the Platt logistic regression with irrelevant labels.
            # UPPER() required: DB stores uppercase ('STOP_LOSS', 'TIME_EXIT') but the
            # original filter used lowercase — SQLite NOT IN is case-sensitive, so the
            # filter was silently passing all mechanical exits through.
            "  AND UPPER(COALESCE(exit_reason, '')) NOT IN ('STOP_LOSS', 'MAX_HOLDING', 'TIME_EXIT', 'FORCED_EXIT')",
        ]
        params: List[Any] = []
        ticker_norm = str(ticker or "").strip().upper()
        if ticker_norm:
            query.append("  AND UPPER(ticker) = ?")
            params.append(ticker_norm)
        query.append("ORDER BY id DESC")
        query.append("LIMIT ?")
        params.append(int(max(limit, 50)))

        conn = None
        try:
            conn = sqlite3.connect(str(db_file), timeout=2.0)
            cur = conn.cursor()
            cols = {
                row[1]
                for row in cur.execute("PRAGMA table_info(trade_executions)").fetchall()
            }
            if "is_contaminated" in cols:
                query.append("  AND is_contaminated = 0")
            cur.execute("\n".join(query), params)
            rows = cur.fetchall()
        except Exception as exc:
            logger.debug("Outcome calibration: unable to read %s (%s)", db_file, exc)
            return pairs_conf, pairs_win
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        for row in rows:
            if not row or len(row) < 2:
                continue
            conf_raw, pnl_raw = row[0], row[1]
            try:
                conf_val = float(conf_raw)
                pnl_val = float(pnl_raw)
            except (TypeError, ValueError):
                continue
            if pnl_val == 0:
                continue
            pairs_conf.append(conf_val)
            pairs_win.append(1.0 if pnl_val > 0 else 0.0)

        return pairs_conf, pairs_win

    def _log_quant_validation(
        self,
        ticker: str,
        signal: TimeSeriesSignal,
        quant_profile: Dict[str, Any],
        market_data: Optional[pd.DataFrame],
    ) -> None:
        """Persist quant validation output for downstream troubleshooting."""
        config = self.quant_validation_config or {}
        logging_cfg = config.get('logging') or {}
        if not logging_cfg.get('enabled'):
            return

        log_dir = Path(logging_cfg.get('log_dir', 'logs/signals'))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / logging_cfg.get('filename', 'quant_validation.jsonl')

        visualization_path = (
            (quant_profile.get('visualization') or {}).get('path')
            if isinstance(quant_profile, dict)
            else None
        )

        market_context: Dict[str, Any] = {
            'rows': int(market_data.shape[0]) if market_data is not None else 0,
            'start': None,
            'end': None,
            'data_source': None,
        }
        if market_data is not None and hasattr(market_data, 'index') and len(market_data.index) > 0:
            start_idx = market_data.index[0]
            end_idx = market_data.index[-1]
            market_context['start'] = (
                start_idx.isoformat()
                if hasattr(start_idx, 'isoformat')
                else str(start_idx)
            )
            market_context['end'] = (
                end_idx.isoformat()
                if hasattr(end_idx, 'isoformat')
                else str(end_idx)
            )
        if hasattr(market_data, 'attrs'):
            market_context['data_source'] = market_data.attrs.get('source')

        signal_provenance = signal.provenance if isinstance(signal.provenance, dict) else {}
        signal_ctx = signal_provenance.get("decision_context") if isinstance(signal_provenance, dict) else {}
        signal_ctx = signal_ctx if isinstance(signal_ctx, dict) else {}
        pipeline_id = self._coerce_nonempty_str(
            quant_profile.get("pipeline_id")
            or signal_provenance.get("pipeline_id")
            or signal_ctx.get("pipeline_id")
            or os.environ.get("PORTFOLIO_MAXIMIZER_PIPELINE_ID")
            or os.environ.get("PIPELINE_ID")
        )
        if pipeline_id is None:
            pipeline_id = self._runtime_pipeline_id
        execution_mode = (
            quant_profile.get("execution_mode")
            or signal_provenance.get("execution_mode")
            or signal_ctx.get("execution_mode")
            or os.environ.get("EXECUTION_MODE")
            or os.environ.get("PMX_EXECUTION_MODE")
        )
        execution_mode = self._coerce_nonempty_str(execution_mode) or self._runtime_execution_mode
        execution_mode = execution_mode.lower()
        proof_mode = quant_profile.get("proof_mode")
        if proof_mode is None:
            proof_mode = signal_provenance.get("proof_mode")
        if proof_mode is None:
            proof_mode = signal_ctx.get("proof_mode")
        run_id = self._coerce_nonempty_str(
            quant_profile.get("run_id")
            or signal_provenance.get("run_id")
            or signal_ctx.get("run_id")
            or os.environ.get("PMX_RUN_ID")
            or os.environ.get("RUN_ID")
        )
        if run_id is None:
            run_id = self._runtime_run_id or pipeline_id
        if pipeline_id is None:
            pipeline_id = run_id or self._runtime_pipeline_id

        quant_profile = dict(quant_profile or {})
        quant_profile.setdefault("execution_mode", execution_mode)
        quant_profile.setdefault("proof_mode", self._to_bool(proof_mode, default=False))
        quant_profile.setdefault("pipeline_id", pipeline_id)
        quant_profile.setdefault("run_id", run_id)

        entry = {
            'timestamp': utc_now().isoformat(),
            'signal_id': signal.signal_id,
            'pipeline_id': str(pipeline_id) if pipeline_id is not None else None,
            'run_id': str(run_id) if run_id is not None else None,
            'ticker': ticker,
            'action': signal.action,
            'execution_mode': execution_mode,
            'proof_mode': self._to_bool(proof_mode, default=False),
            'confidence': signal.confidence,
            'confidence_calibrated': signal.confidence_calibrated,  # Phase 7.13-B1: Platt-scaled probability
            # Pre-blend raw confidence — used by _load_jsonl_outcome_pairs so LR trains
            # and predicts on the same (raw) distribution (fixes mismatch, Phase 7.16-C1).
            'raw_confidence': getattr(self, '_last_raw_confidence', None),
            'expected_return': signal.expected_return,
            'risk_score': signal.risk_score,
            'volatility': signal.volatility,
            'status': quant_profile.get('status'),
            'failed_criteria': quant_profile.get('failed_criteria'),
            'position_value': quant_profile.get('position_value'),
            'estimated_shares': quant_profile.get('estimated_shares'),
            'visualization_path': visualization_path,
            'quant_validation': quant_profile,
            'market_context': market_context,
            'classifier_features': (
                (signal.provenance or {}).get("classifier_features")
                if isinstance(signal.provenance, dict) else None
            ),
            'p_up': signal.p_up,
            'directional_gate_applied': signal.directional_gate_applied,
            # Structured HOLD reason — enables aggregating HOLD causes without log parsing.
            # Populated for all action='HOLD' signals. None for BUY/SELL.
            'hold_reason': (signal.provenance or {}).get("hold_reason") if isinstance(signal.provenance, dict) else None,
            'snr_gate_blocked': (signal.provenance or {}).get("snr_gate_blocked") if isinstance(signal.provenance, dict) else None,
        }

        try:
            with log_file.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(entry, default=self._json_serializer) + "\n")
        except Exception as exc:  # pragma: no cover - logging must not break signals
            logger.warning("Unable to persist quant validation log for %s: %s", ticker, exc)

    def _make_ts_signal_id(self) -> str:
        """Build a globally unique ts_signal_id string (Phase 7.13-A2).

        Format: ts_{ticker}_{datetime_seconds}_{instance_uid}_{counter:04d}
        Example: ts_AAPL_20260224T231500Z_a3f2_0001

        Globally unique because:
        - _current_ticker distinguishes signals across ticker instances
        - datetime_seconds (YYYYMMDDTHHMMSSz) is unique per second
        - _instance_uid (4-char random hex per instance, Phase 7.15) prevents
          collisions between instances that start within the same second
        - counter monotonically increments within the instance
        """
        # Extract the compact datetime-seconds component from the run_id.
        # run_id format: pmx_ts_20260224T231500Z_<pid> -> extract 20260224T231500Z
        run_id = (self._runtime_run_id or "unknown")
        if "_" in run_id:
            parts = run_id.split("_")
            # parts: ["pmx", "ts", "20260224T231500Z", "<pid>"]
            # Take the third segment (index 2) which is the timestamp
            dt_part = parts[2] if len(parts) >= 3 else parts[-1]
        else:
            dt_part = run_id
        # Keep up to 16 chars (YYYYMMDDTHHMMSSz) -- no truncation that loses seconds
        dt_part = dt_part[:16]
        ticker_safe = (self._current_ticker or "unknown").upper().replace("-", "")[:6]
        uid = getattr(self, "_instance_uid", "0000")
        return f"ts_{ticker_safe}_{dt_part}_{uid}_{self._signal_counter:04d}"

    @staticmethod
    def _coerce_nonempty_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Serializer for numpy/pandas objects when dumping JSON."""
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, (np.ndarray, list, tuple)):
            return list(obj)
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return str(obj)

    def generate_signals_batch(self,
                               forecast_bundles: Dict[str, Dict[str, Any]],
                               current_prices: Dict[str, float],
                               market_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, TimeSeriesSignal]:
        """
        Generate signals for multiple tickers.

        Args:
            forecast_bundles: Dict mapping ticker to forecast bundle
            current_prices: Dict mapping ticker to current price
            market_data: Optional dict mapping ticker to market data

        Returns:
            Dict mapping ticker to TimeSeriesSignal
        """
        signals = {}

        for ticker, forecast_bundle in forecast_bundles.items():
            current_price = current_prices.get(ticker)
            if current_price is None:
                logger.warning(f"No current price for {ticker}, skipping")
                continue

            market_data_ticker = market_data.get(ticker) if market_data else None

            signal = self.generate_signal(
                forecast_bundle,
                current_price,
                ticker,
                market_data_ticker
            )

            signals[ticker] = signal

        return signals


# Validation
assert TimeSeriesSignalGenerator.generate_signal.__doc__ is not None
assert TimeSeriesSignalGenerator.generate_signals_batch.__doc__ is not None

logger.info("Time Series Signal Generator module loaded successfully")
