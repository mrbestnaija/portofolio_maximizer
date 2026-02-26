"""
signal_generator_factory.py -- Single authoritative constructor for TimeSeriesSignalGenerator.

Phase 7.15: Eliminates config drift between run_etl_pipeline.py, run_auto_trader.py
(via SignalRouter), run_barbell_pnl_evaluation.py, and backtesting/candidate_simulator.py.

Previously each script loaded signal_routing_config.yml differently and passed a different
subset of the 9 __init__ parameters, silently diverging backtest vs live signal behavior:

  run_etl_pipeline.py          -- passed 4 params (missing per_ticker, cost_model, ...)
  SignalRouter (auto_trader)   -- passed 6 params (missing quant_validation_config, ...)
  run_barbell_pnl_evaluation   -- passed 5 params (missing per_ticker, cost_model, ...)
  candidate_simulator.py       -- passed 6 params (missing quant_validation_config, ...)

After Phase 7.15 all callers use build_signal_generator() which ensures all 9 params
are always extracted from the canonical config path.

Config resolution order:
  1. config/signal_routing_config.yml  signal_routing.time_series  (base)
  2. ts_cfg_overrides                                               (highest priority, merged last)

Usage:
    from models.signal_generator_factory import build_signal_generator
    gen = build_signal_generator()                        # plain production use
    gen = build_signal_generator(ts_cfg_overrides=cfg)   # with proof-mode overrides
    gen = build_signal_generator(                         # test / no-log
        quant_validation_config={"logging": {"enabled": False}}
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Canonical config location (relative to repo root or CWD)
_DEFAULT_ROUTING_CONFIG = Path("config/signal_routing_config.yml")
_DEFAULT_FORECASTING_CONFIG = Path("config/forecasting_config.yml")


def _load_ts_section(config_path: Path) -> Dict[str, Any]:
    """Load and return the signal_routing.time_series dict from a YAML file.

    Returns an empty dict on any error so callers always get a valid dict.
    """
    if not config_path.exists():
        return {}
    try:
        import yaml  # local import — yaml may not be installed in all test envs
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        routing = raw.get("signal_routing") or {}
        ts = routing.get("time_series") or {}
        return dict(ts)
    except Exception as exc:
        logger.warning("signal_generator_factory: failed to load %s: %s", config_path, exc)
        return {}


def _float_or(val: Any, default: float) -> float:
    """Convert val to float, falling back to default when val is None or invalid.

    Unlike ``float(val or default)`` this does NOT convert 0.0 to default, which
    would be wrong for min_expected_return=0.0 (a deliberate "no floor" setting).
    """
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def build_signal_generator(
    *,
    config_path: Optional[Path] = None,
    ts_cfg_overrides: Optional[Dict[str, Any]] = None,
    quant_validation_config: Optional[Dict[str, Any]] = None,
    quant_validation_config_path: Optional[str] = None,
    forecasting_config_path: Optional[Path] = None,
) -> "TimeSeriesSignalGenerator":  # type: ignore[name-defined]  # noqa: F821
    """Construct a fully-configured TimeSeriesSignalGenerator from canonical YAML.

    Covers all 9 TimeSeriesSignalGenerator.__init__ parameters:
      confidence_threshold, min_expected_return, max_risk_score,
      use_volatility_filter, quant_validation_config, quant_validation_config_path,
      per_ticker_thresholds, cost_model, forecasting_config_path.

    Parameters
    ----------
    config_path:
        Path to signal_routing_config.yml.  Defaults to config/signal_routing_config.yml
        relative to CWD.  Pass an explicit path in tests or non-standard environments.
    ts_cfg_overrides:
        Dict of time_series-section overrides merged *on top of* the loaded YAML.
        Used by run_auto_trader.py to inject proof-mode strict thresholds without
        re-loading the YAML.  Keys must be valid signal_routing.time_series fields.
        None values in the dict fall back to YAML base or constructor defaults.
    quant_validation_config:
        Passed through verbatim to TimeSeriesSignalGenerator.  Use
        ``{"logging": {"enabled": False}}`` in tests to suppress JSONL writes.
    quant_validation_config_path:
        Optional path string to a YAML file describing quant validation thresholds.
        Passed through to TimeSeriesSignalGenerator; defaults to
        config/quant_success_config.yml when None.
    forecasting_config_path:
        Passed through to TimeSeriesSignalGenerator for ensemble/regime config
        preservation during cross-validation.  Defaults to
        config/forecasting_config.yml if that file exists.

    Returns
    -------
    TimeSeriesSignalGenerator
        Fully configured instance with all 9 __init__ params populated.
    """
    # Deferred import — keeps factory importable even before models/ is on sys.path
    from models.time_series_signal_generator import TimeSeriesSignalGenerator  # noqa: PLC0415

    # --- 1. Load base config -------------------------------------------------
    resolved_config_path = config_path or _DEFAULT_ROUTING_CONFIG
    ts_cfg = _load_ts_section(resolved_config_path)

    # --- 2. Apply caller overrides (proof-mode strict thresholds, etc.) ------
    if ts_cfg_overrides:
        ts_cfg = {**ts_cfg, **ts_cfg_overrides}

    # --- 3. Extract all constructor params -----------------------------------
    # _float_or handles None values that may appear in override dicts (e.g.
    # candidate_simulator guardrails that were built with `float(x or default)`).
    confidence_threshold = _float_or(ts_cfg.get("confidence_threshold"), 0.55)
    min_expected_return = _float_or(ts_cfg.get("min_expected_return"), 0.003)
    max_risk_score = _float_or(ts_cfg.get("max_risk_score"), 0.7)
    use_volatility_filter = bool(ts_cfg.get("use_volatility_filter", True))

    per_ticker = ts_cfg.get("per_ticker")
    per_ticker_thresholds: Optional[Dict[str, Dict[str, Any]]] = (
        per_ticker if isinstance(per_ticker, dict) else None
    )

    cost_model_raw = ts_cfg.get("cost_model")
    cost_model: Optional[Dict[str, Any]] = (
        cost_model_raw if isinstance(cost_model_raw, dict) else None
    )

    # --- 4. Resolve forecasting_config_path ----------------------------------
    if forecasting_config_path is None:
        if _DEFAULT_FORECASTING_CONFIG.exists():
            forecasting_config_path = _DEFAULT_FORECASTING_CONFIG

    forecasting_config_path_str: Optional[str] = (
        str(forecasting_config_path) if forecasting_config_path is not None else None
    )

    # --- 5. Construct and return (all 9 params) ------------------------------
    logger.debug(
        "build_signal_generator: confidence=%.2f min_return=%.4f max_risk=%.2f "
        "per_ticker=%s cost_model=%s quant_validation=%s qv_path=%s forecasting_cfg=%s",
        confidence_threshold,
        min_expected_return,
        max_risk_score,
        bool(per_ticker_thresholds),
        bool(cost_model),
        bool(quant_validation_config),
        bool(quant_validation_config_path),
        forecasting_config_path_str,
    )

    return TimeSeriesSignalGenerator(
        confidence_threshold=confidence_threshold,
        min_expected_return=min_expected_return,
        max_risk_score=max_risk_score,
        use_volatility_filter=use_volatility_filter,
        per_ticker_thresholds=per_ticker_thresholds,
        cost_model=cost_model,
        quant_validation_config=quant_validation_config,
        quant_validation_config_path=quant_validation_config_path,
        forecasting_config_path=forecasting_config_path_str,
    )
