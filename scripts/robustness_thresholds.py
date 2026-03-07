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

import re
import sys
from pathlib import Path
from typing import Any

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
    from scripts.telemetry_adapter import sha256_file
except ModuleNotFoundError:
    from telemetry_adapter import sha256_file

CAPITAL_READINESS_CHECK_PATH = ROOT / "scripts" / "capital_readiness_check.py"
FORECASTER_MONITORING_PATH = ROOT / "config" / "forecaster_monitoring.yml"

# Centralized advisory values used only for visualization / recommendation.
POLICY_WR_FLOOR = 0.25
BREAK_EVEN_PROFIT_FACTOR = 1.0
WEAK_MAX_WIN_RATE = 0.30
WEAK_MAX_PROFIT_FACTOR = 1.00
WEAK_MIN_TRADES = 5

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
    digest, _ = sha256_file(path)
    return digest


MIN_LIFT_FRACTION = _extract_yaml_float(FORECASTER_MONITORING_PATH, "min_lift_fraction", 0.25)


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
        "policy_wr_floor": POLICY_WR_FLOOR,
        "break_even_profit_factor": BREAK_EVEN_PROFIT_FACTOR,
        "weak_max_win_rate": WEAK_MAX_WIN_RATE,
        "weak_max_profit_factor": WEAK_MAX_PROFIT_FACTOR,
        "weak_min_trades": WEAK_MIN_TRADES,
        "min_lift_fraction": MIN_LIFT_FRACTION,
        "forecaster_monitoring_path": str(FORECASTER_MONITORING_PATH),
        **threshold_source_info(),
    }
