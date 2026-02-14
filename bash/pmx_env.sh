#!/usr/bin/env bash
#
# Portfolio Maximizer WSL environment defaults (per Documentation/RUNTIME_GUARDRAILS.md)
# Safe to source before runs; does not override values already set by caller.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Live-data guard: keep synthetic providers OFF unless explicitly enabled.
unset ENABLE_SYNTHETIC_PROVIDER ENABLE_SYNTHETIC_DATA_SOURCE SYNTHETIC_ONLY

# Default execution mode for live evaluation.
export EXECUTION_MODE="${EXECUTION_MODE:-live}"

# Forecast audit trail + governance gate config.
export TS_FORECAST_AUDIT_DIR="${TS_FORECAST_AUDIT_DIR:-${ROOT_DIR}/logs/forecast_audits}"
export TS_FORECAST_MONITOR_CONFIG="${TS_FORECAST_MONITOR_CONFIG:-${ROOT_DIR}/config/forecaster_monitoring.yml}"

# Dashboard persistence controls.
export DASHBOARD_PERSIST="${DASHBOARD_PERSIST:-1}"
export DASHBOARD_KEEP_ALIVE="${DASHBOARD_KEEP_ALIVE:-1}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"

# Production safety: long-only unless explicitly overridden.
export PMX_LONG_ONLY="${PMX_LONG_ONLY:-1}"
