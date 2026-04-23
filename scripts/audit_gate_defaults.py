"""Shared defaults for forecast/audit gate tooling."""

from __future__ import annotations

FORECAST_AUDIT_MAX_FILES_DEFAULT = 500
# Outcome linkage must scan further back than RMSE analysis — use a larger window.
FORECAST_AUDIT_OUTCOME_MAX_FILES_DEFAULT = 10000

