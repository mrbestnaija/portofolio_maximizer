"""
etl/paths.py — Central path constants for Portfolio Maximizer (Phase 7.13-C1).

Import these constants instead of hardcoding paths in each script.
The ``PORTFOLIO_DB_PATH`` environment variable overrides ``DB_PATH`` at runtime.

Usage::

    from etl.paths import DB_PATH, QUANT_VALIDATION_JSONL, FORECAST_AUDITS_CACHE

All paths are resolved relative to the project root (parent of this file's package).
"""
from __future__ import annotations

import os
from pathlib import Path

# Project root: two levels up from etl/paths.py  →  .../portfolio_maximizer_v45/
ROOT: Path = Path(__file__).resolve().parent.parent

# Primary SQLite database.  Override with PORTFOLIO_DB_PATH env var at runtime.
DB_PATH: Path = Path(os.environ.get("PORTFOLIO_DB_PATH", "") or str(ROOT / "data" / "portfolio_maximizer.db"))

# Quant-validation JSONL (written by TimeSeriesSignalGenerator, read by update_platt_outcomes.py).
QUANT_VALIDATION_JSONL: Path = ROOT / "logs" / "signals" / "quant_validation.jsonl"

# Directory where per-ticker forecast audit JSON files live.
FORECAST_AUDITS_DIR: Path = ROOT / "logs" / "forecast_audits"

# Latest summary JSON written by production_audit_gate.py / check_forecast_audits.py.
FORECAST_AUDITS_CACHE: Path = ROOT / "logs" / "forecast_audits_cache" / "latest_summary.json"

# Convenience log directories.
LOGS_DIR: Path = ROOT / "logs"
RUN_AUDIT_DIR: Path = ROOT / "logs" / "run_audit"
AUTOMATION_DIR: Path = ROOT / "logs" / "automation"
