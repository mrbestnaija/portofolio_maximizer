"""
Risk Mode Configuration Loader

Loads risk filter settings from config/risk_mode.yml and provides
easy access to the active mode's settings.

Phase 7.9: Introduced to support configurable risk filter modes.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

logger = logging.getLogger(__name__)

# Cache for loaded config
_risk_mode_config: Optional[Dict[str, Any]] = None
_active_mode_settings: Optional[Dict[str, Any]] = None


def _find_config_path() -> Path:
    """Find the risk_mode.yml config file."""
    # Try relative to this file first
    module_dir = Path(__file__).parent.parent
    config_path = module_dir / "config" / "risk_mode.yml"

    if config_path.exists():
        return config_path

    # Try current working directory
    cwd_path = Path.cwd() / "config" / "risk_mode.yml"
    if cwd_path.exists():
        return cwd_path

    # Fallback - return expected path even if doesn't exist
    return config_path


def load_risk_mode_config(force_reload: bool = False) -> Dict[str, Any]:
    """
    Load the risk mode configuration from YAML.

    Args:
        force_reload: If True, reload from disk even if cached

    Returns:
        Full configuration dict including all modes
    """
    global _risk_mode_config

    if _risk_mode_config is not None and not force_reload:
        return _risk_mode_config

    config_path = _find_config_path()

    if not config_path.exists():
        logger.warning(f"Risk mode config not found at {config_path}, using defaults")
        _risk_mode_config = _get_default_config()
        return _risk_mode_config

    try:
        with open(config_path, 'r') as f:
            _risk_mode_config = yaml.safe_load(f)
        logger.info(f"Loaded risk mode config from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load risk mode config: {e}, using defaults")
        _risk_mode_config = _get_default_config()

    return _risk_mode_config


def get_active_mode() -> str:
    """
    Get the currently active risk mode.

    Priority:
    1. DIAGNOSTIC_MODE=1 env var -> "diagnostic"
    2. RISK_MODE env var -> that value
    3. active_mode from config file
    4. Default: "research_production"
    """
    # Check for legacy DIAGNOSTIC_MODE first (backwards compatibility)
    if os.getenv("DIAGNOSTIC_MODE", "0") == "1":
        return "diagnostic"

    # Check for RISK_MODE env var override
    env_mode = os.getenv("RISK_MODE", "").lower()
    if env_mode in ("strict", "research_production", "diagnostic"):
        return env_mode

    # Fall back to config file setting
    config = load_risk_mode_config()
    return config.get("active_mode", "research_production")


def get_active_mode_settings(force_reload: bool = False) -> Dict[str, Any]:
    """
    Get the settings for the currently active mode.

    Args:
        force_reload: If True, reload config from disk

    Returns:
        Dict of settings for the active mode
    """
    global _active_mode_settings

    if _active_mode_settings is not None and not force_reload:
        # Check if mode changed via env var
        current_mode = get_active_mode()
        if _active_mode_settings.get("_mode_name") == current_mode:
            return _active_mode_settings

    config = load_risk_mode_config(force_reload)
    active_mode = get_active_mode()

    modes = config.get("modes", {})
    if active_mode not in modes:
        logger.warning(f"Mode '{active_mode}' not found, falling back to research_production")
        active_mode = "research_production"

    _active_mode_settings = modes.get(active_mode, _get_default_mode_settings())
    _active_mode_settings["_mode_name"] = active_mode

    logger.info(f"Active risk mode: {active_mode}")
    return _active_mode_settings


def get_counter_trend_config() -> Dict[str, Any]:
    """Get counter-trend filter configuration."""
    settings = get_active_mode_settings()
    return settings.get("counter_trend_filter", {
        "mode": "advisory",
        "max_warnings": 4
    })


def get_regime_filter_config() -> Dict[str, Any]:
    """Get regime filter configuration."""
    settings = get_active_mode_settings()
    return settings.get("regime_filter", {
        "mode": "confidence_gated",
        "min_regime_confidence": 0.70,
        "bearish_buy_block": False,
        "high_vol_block": False
    })


def get_confidence_config() -> Dict[str, Any]:
    """Get confidence threshold configuration."""
    settings = get_active_mode_settings()
    return settings.get("confidence", {
        "min_signal_confidence": 0.45,
        "warning_penalty_pct": 2,
        "failure_penalty_pct": 10
    })


def get_position_sizing_config() -> Dict[str, Any]:
    """Get position sizing configuration."""
    settings = get_active_mode_settings()
    return settings.get("position_sizing", {
        "max_position_pct": 5,
        "max_short_pct": 2,
        "confidence_floor": 0.30
    })


def get_quant_validation_config() -> Dict[str, Any]:
    """Get quant validation configuration."""
    settings = get_active_mode_settings()
    return settings.get("quant_validation", {
        "enabled": True,
        "mode": "hard_fail",
        "min_directional_accuracy": 0.42,
        "max_rmse_ratio": 1.15
    })


def is_quant_validation_advisory() -> bool:
    """Check if quant validation is in advisory mode (log but don't block)."""
    config = get_quant_validation_config()
    return config.get("mode", "hard_fail") == "advisory"


def is_quant_validation_disabled() -> bool:
    """Check if quant validation is completely disabled."""
    config = get_quant_validation_config()
    return config.get("mode", "hard_fail") == "disabled" or not config.get("enabled", True)


def get_transaction_costs() -> Dict[str, float]:
    """Get transaction cost estimates by asset class."""
    settings = get_active_mode_settings()
    return settings.get("transaction_costs", {
        "US_EQUITY": 1.5,
        "INTL_EQUITY": 3.0,
        "FX": 1.0,
        "CRYPTO": 8.0,
        "INDEX": 1.5,
        "UNKNOWN": 3.0
    })


def is_counter_trend_advisory() -> bool:
    """Check if counter-trend filter is in advisory mode (log but don't block)."""
    config = get_counter_trend_config()
    return config.get("mode", "advisory") == "advisory"


def is_counter_trend_disabled() -> bool:
    """Check if counter-trend filter is completely disabled."""
    config = get_counter_trend_config()
    return config.get("mode", "advisory") == "disabled"


def is_regime_filter_disabled() -> bool:
    """Check if regime filter is completely disabled."""
    config = get_regime_filter_config()
    return config.get("mode", "confidence_gated") == "disabled"


def should_block_on_regime(regime_confidence: float) -> bool:
    """
    Determine if a trade should be blocked based on regime confidence.

    Args:
        regime_confidence: Confidence level of detected regime (0-1)

    Returns:
        True if the trade should be blocked, False otherwise
    """
    config = get_regime_filter_config()
    mode = config.get("mode", "confidence_gated")

    if mode == "disabled":
        return False
    elif mode == "hard_fail":
        return True
    elif mode == "confidence_gated":
        min_conf = config.get("min_regime_confidence", 0.70)
        return regime_confidence >= min_conf

    return False


def _get_default_config() -> Dict[str, Any]:
    """Get default configuration when file is missing."""
    return {
        "active_mode": "research_production",
        "modes": {
            "research_production": _get_default_mode_settings()
        }
    }


def _get_default_mode_settings() -> Dict[str, Any]:
    """Get default mode settings (research_production)."""
    return {
        "description": "Research mode with balanced risk controls",
        "counter_trend_filter": {
            "mode": "advisory",
            "max_warnings": 4
        },
        "regime_filter": {
            "mode": "confidence_gated",
            "min_regime_confidence": 0.70,
            "bearish_buy_block": False,
            "high_vol_block": False
        },
        "confidence": {
            "min_signal_confidence": 0.45,
            "warning_penalty_pct": 2,
            "failure_penalty_pct": 10
        },
        "position_sizing": {
            "max_position_pct": 5,
            "max_short_pct": 2,
            "confidence_floor": 0.30
        },
        "quant_validation": {
            "enabled": True,
            "min_directional_accuracy": 0.42,
            "max_rmse_ratio": 1.15
        },
        "transaction_costs": {
            "US_EQUITY": 1.5,
            "INTL_EQUITY": 3.0,
            "FX": 1.0,
            "CRYPTO": 8.0,
            "INDEX": 1.5,
            "UNKNOWN": 3.0
        }
    }


# Module-level initialization
def init():
    """Initialize the risk mode loader on import."""
    load_risk_mode_config()
    mode = get_active_mode()
    logger.debug(f"Risk mode loader initialized with mode: {mode}")


# Auto-initialize on import
try:
    init()
except Exception:
    pass  # Silently fail if config not available during import
