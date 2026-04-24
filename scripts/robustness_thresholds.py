"""
Shared, read-only threshold definitions for the robustness automation pipeline.

This module is the single source of truth for analytics-facing thresholds used
by the automation scripts. It imports the hard-gate values from
scripts/capital_readiness_check.py and reads the ensemble lift threshold from
config/forecaster_monitoring.yml so downstream scripts do not duplicate them.

Library-style helper only: no CLI entrypoint.
Keep the import graph one-way. This module may depend on gate/config sources,
but those sources must not import back into the automation pipeline helpers.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

try:  # pragma: no cover - PyYAML is a project dependency but keep bootstrap resilient
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.capital_readiness_check import (  # type: ignore[import-not-found]
        R3_MIN_PROFIT_FACTOR as _R3_MIN_PROFIT_FACTOR,
        R3_MIN_TRADES as _R3_MIN_TRADES,
        R3_MIN_WIN_RATE as _R3_MIN_WIN_RATE,
        R4_MAX_BRIER as _R4_MAX_BRIER,
    )
except ModuleNotFoundError:
    from capital_readiness_check import (  # type: ignore[import-not-found]
        R3_MIN_PROFIT_FACTOR as _R3_MIN_PROFIT_FACTOR,
        R3_MIN_TRADES as _R3_MIN_TRADES,
        R3_MIN_WIN_RATE as _R3_MIN_WIN_RATE,
        R4_MAX_BRIER as _R4_MAX_BRIER,
    )

try:
    from etl.domain_objective import (
        DOMAIN_OBJECTIVE_VERSION,
        MIN_OMEGA_VS_HURDLE,
        MIN_TAKE_PROFIT_FREQUENCY,
        TARGET_AMPLITUDE_MULTIPLIER,
        TAKE_PROFIT_FILTER_THRESHOLD_FALLBACK,
        SYSTEM_OBJECTIVE,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script bootstrap
    from domain_objective import (  # type: ignore[import-not-found]
        DOMAIN_OBJECTIVE_VERSION,
        MIN_OMEGA_VS_HURDLE,
        MIN_TAKE_PROFIT_FREQUENCY,
        TARGET_AMPLITUDE_MULTIPLIER,
        TAKE_PROFIT_FILTER_THRESHOLD_FALLBACK,
        SYSTEM_OBJECTIVE,
    )

CAPITAL_READINESS_CHECK_PATH = ROOT / "scripts" / "capital_readiness_check.py"
FORECASTER_MONITORING_PATH = ROOT / "config" / "forecaster_monitoring.yml"

# Centralized advisory values used only for visualization / recommendation.
POLICY_WR_FLOOR = 0.25
BREAK_EVEN_PROFIT_FACTOR = 1.0
WEAK_MAX_WIN_RATE = 0.30
WEAK_MAX_PROFIT_FACTOR = 1.00
WEAK_MIN_TRADES = 5

# PENDING_CALIBRATION: the 1.5 SNR floor is a conservative holdover from the
# pre-DCR world where EWMA CI was inflated for convergence_ok=False.
PENDING_CALIBRATION = "post_dcr_snr_distribution_required_78142bb"

LINKAGE_MIN_MATCHED_KEY = "linkage_min_matched"
LINKAGE_MIN_RATIO_KEY = "linkage_min_ratio"
MIN_SIGNAL_TO_NOISE_KEY = "min_signal_to_noise"
MIN_OMEGA_RATIO_KEY = "min_omega_ratio"
MIN_PAYOFF_ASYMMETRY_KEY = "min_payoff_asymmetry"
MIN_TAKE_PROFIT_FREQUENCY_LIVE_KEY = "min_take_profit_frequency_live"
MIN_TARGET_AMPLITUDE_HIT_RATE_KEY = "min_target_amplitude_hit_rate"

R3_MIN_TRADES = _R3_MIN_TRADES
R3_MIN_WIN_RATE = _R3_MIN_WIN_RATE
R3_MIN_PROFIT_FACTOR = _R3_MIN_PROFIT_FACTOR
R4_MAX_BRIER = _R4_MAX_BRIER


def _extract_yaml_float(path: Path, key: str, default: float) -> float:
    """
    Extract a simple scalar float value from YAML-like text without requiring
    PyYAML. This is sufficient for the flat scalar settings we use here.
    """
    if not path.exists():
        return default
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*([-+]?\d+(?:\.\d+)?)\s*$")
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = pattern.match(line)
            if match:
                return float(match.group(1))
    except Exception:
        return default
    return default


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                h.update(chunk)
    except Exception:
        return None
    return h.hexdigest()


MIN_LIFT_FRACTION = _extract_yaml_float(FORECASTER_MONITORING_PATH, "min_lift_fraction", 0.25)


def _recursive_find_scalar(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload.get(key)
        for value in payload.values():
            found = _recursive_find_scalar(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _recursive_find_scalar(value, key)
            if found is not None:
                return found
    return None


def _load_yaml_payload(path: Path) -> dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def load_floored_thresholds(config_path: Path | str | None) -> dict[str, Any]:
    """Load gate thresholds while enforcing hard floors."""
    path = Path(config_path) if config_path else None
    payload = _load_yaml_payload(path) if path is not None else {}
    floor_warnings: list[str] = []

    raw_linkage_min_matched = _recursive_find_scalar(payload, LINKAGE_MIN_MATCHED_KEY)
    raw_linkage_min_ratio = _recursive_find_scalar(payload, LINKAGE_MIN_RATIO_KEY)
    raw_min_signal_to_noise = _recursive_find_scalar(payload, MIN_SIGNAL_TO_NOISE_KEY)
    raw_min_omega_ratio = _recursive_find_scalar(payload, MIN_OMEGA_RATIO_KEY)
    raw_min_payoff_asymmetry = _recursive_find_scalar(payload, MIN_PAYOFF_ASYMMETRY_KEY)
    raw_min_take_profit_frequency_live = _recursive_find_scalar(payload, MIN_TAKE_PROFIT_FREQUENCY_LIVE_KEY)
    raw_min_target_amplitude_hit_rate = _recursive_find_scalar(payload, MIN_TARGET_AMPLITUDE_HIT_RATE_KEY)

    def _coerce(raw: Any, default: float, floor: float, key: str) -> float:
        try:
            value = float(raw)
        except Exception:
            value = default
        if raw is not None and value < floor:
            floor_warnings.append(f"{key}_raised_to_floor({value}->{floor})")
        return max(value, floor)

    return {
        LINKAGE_MIN_MATCHED_KEY: int(_coerce(raw_linkage_min_matched, 10.0, 10.0, LINKAGE_MIN_MATCHED_KEY)),
        LINKAGE_MIN_RATIO_KEY: float(_coerce(raw_linkage_min_ratio, 0.8, 0.8, LINKAGE_MIN_RATIO_KEY)),
        MIN_SIGNAL_TO_NOISE_KEY: float(_coerce(raw_min_signal_to_noise, 1.5, 1.5, MIN_SIGNAL_TO_NOISE_KEY)),
        MIN_OMEGA_RATIO_KEY: float(_coerce(raw_min_omega_ratio, MIN_OMEGA_VS_HURDLE, MIN_OMEGA_VS_HURDLE, MIN_OMEGA_RATIO_KEY)),
        MIN_PAYOFF_ASYMMETRY_KEY: float(_coerce(raw_min_payoff_asymmetry, TARGET_AMPLITUDE_MULTIPLIER, TARGET_AMPLITUDE_MULTIPLIER, MIN_PAYOFF_ASYMMETRY_KEY)),
        MIN_TAKE_PROFIT_FREQUENCY_LIVE_KEY: float(
            _coerce(raw_min_take_profit_frequency_live, 0.05, 0.05, MIN_TAKE_PROFIT_FREQUENCY_LIVE_KEY)
        ),
        MIN_TARGET_AMPLITUDE_HIT_RATE_KEY: float(
            _coerce(raw_min_target_amplitude_hit_rate, 0.10, 0.10, MIN_TARGET_AMPLITUDE_HIT_RATE_KEY)
        ),
        "pending_calibration": PENDING_CALIBRATION,
        "system_objective": SYSTEM_OBJECTIVE,
        "domain_objective_version": DOMAIN_OBJECTIVE_VERSION,
        "take_profit_filter_threshold_fallback": TAKE_PROFIT_FILTER_THRESHOLD_FALLBACK,
        "gate_floor_bypass_active": False,
        "floor_warnings": floor_warnings,
        "source_path": str(path) if path is not None else None,
    }


def threshold_source_info() -> dict[str, Any]:
    return {
        "source_paths": {
            "capital_readiness_check": str(CAPITAL_READINESS_CHECK_PATH),
            "forecaster_monitoring": str(FORECASTER_MONITORING_PATH),
        },
        "source_hashes": {
            "capital_readiness_check": _sha256_file(CAPITAL_READINESS_CHECK_PATH),
            "forecaster_monitoring": _sha256_file(FORECASTER_MONITORING_PATH),
        },
    }


def threshold_map() -> dict[str, Any]:
    return {
        "r3_min_trades": R3_MIN_TRADES,
        "r3_min_win_rate": R3_MIN_WIN_RATE,
        "r3_min_profit_factor": R3_MIN_PROFIT_FACTOR,
        "r4_max_brier": R4_MAX_BRIER,
        "min_omega_ratio": MIN_OMEGA_VS_HURDLE,
        "min_payoff_asymmetry": TARGET_AMPLITUDE_MULTIPLIER,
        "min_take_profit_frequency_live": 0.05,
        "min_target_amplitude_hit_rate": 0.10,
        "policy_wr_floor": POLICY_WR_FLOOR,
        "break_even_profit_factor": BREAK_EVEN_PROFIT_FACTOR,
        "weak_max_win_rate": WEAK_MAX_WIN_RATE,
        "weak_max_profit_factor": WEAK_MAX_PROFIT_FACTOR,
        "weak_min_trades": WEAK_MIN_TRADES,
        "min_lift_fraction": MIN_LIFT_FRACTION,
        "pending_calibration": PENDING_CALIBRATION,
        "system_objective": SYSTEM_OBJECTIVE,
        "domain_objective_version": DOMAIN_OBJECTIVE_VERSION,
        "forecaster_monitoring_path": str(FORECASTER_MONITORING_PATH),
        **threshold_source_info(),
    }


def locate_pending_calibration_target() -> dict[str, str]:
    """Return the threshold key that still needs post-DCR recalibration."""
    return {
        MIN_SIGNAL_TO_NOISE_KEY: PENDING_CALIBRATION,
    }
