"""
Shared pytest fixtures for repository-wide test hygiene.
"""

from __future__ import annotations

import os

import pytest

# Guard import-time side effects from test modules that instantiate forecasters
# before fixtures execute.
os.environ["TS_FORECAST_AUDIT_DIR"] = ""


@pytest.fixture(autouse=True)
def _disable_forecast_audit_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Prevent tests from writing into logs/forecast_audits by default.

    Production gate evidence should come from runtime pipelines, not unit/integration
    test executions. Tests that need a specific audit dir can still override this.
    """
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", "")
